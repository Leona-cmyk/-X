from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.build_manifest import build_manifest
from src.constants import TRAINABLE_CLASSES
from src.features import read_operation_csv
from src.preprocessing import TimeSeriesStandardScaler


@dataclass
class WindowRecord:
    path: str
    accident: str
    severity: float
    start: int
    end: int
    sample_id: str


def load_trainable_manifest(max_files_per_class: int | None = None, classes: list[str] | None = None) -> pd.DataFrame:
    manifest = build_manifest()
    classes = classes or TRAINABLE_CLASSES
    manifest = manifest[(manifest["use_for_training"]) & (manifest["accident"].isin(classes))].copy()
    if max_files_per_class and max_files_per_class > 0:
        manifest = (
            manifest.sort_values(["accident", "severity"])
            .groupby("accident", group_keys=False)
            .head(max_files_per_class)
            .reset_index(drop=True)
        )
    return manifest.reset_index(drop=True)


def split_manifest_by_file(
    manifest: pd.DataFrame,
    val_ratio: float = 0.15,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = random.Random(seed)
    train_parts, val_parts, test_parts = [], [], []
    for _, group in manifest.groupby("accident"):
        rows = group.to_dict("records")
        rng.shuffle(rows)
        n = len(rows)
        n_test = max(1, int(round(n * test_ratio)))
        n_val = max(1, int(round(n * val_ratio)))
        test_parts.extend(rows[:n_test])
        val_parts.extend(rows[n_test : n_test + n_val])
        train_parts.extend(rows[n_test + n_val :])
    return pd.DataFrame(train_parts), pd.DataFrame(val_parts), pd.DataFrame(test_parts)


def build_class_maps(manifest: pd.DataFrame) -> tuple[dict[str, int], dict[int, str]]:
    labels = sorted(manifest["accident"].unique())
    class_to_idx = {label: i for i, label in enumerate(labels)}
    idx_to_class = {i: label for label, i in class_to_idx.items()}
    return class_to_idx, idx_to_class


class WindowedNPPADDataset(Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        class_to_idx: dict[str, int],
        scaler: TimeSeriesStandardScaler,
        window_size: int = 256,
        stride: int = 128,
        early_ratio: float = 1.0,
        random_early_ratio: bool = False,
        early_choices: list[float] | None = None,
        augment: bool = False,
        gaussian_noise_std: float = 0.0,
        sensor_dropout: float = 0.0,
        time_mask_ratio: float = 0.0,
        magnitude_scale_std: float = 0.0,
        max_windows_per_file: int | None = None,
    ) -> None:
        self.manifest = manifest.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.scaler = scaler
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.early_ratio = float(early_ratio)
        self.random_early_ratio = random_early_ratio
        self.early_choices = early_choices or [0.1, 0.2, 0.3, 0.5, 1.0]
        self.augment = augment
        self.gaussian_noise_std = gaussian_noise_std
        self.sensor_dropout = sensor_dropout
        self.time_mask_ratio = time_mask_ratio
        self.magnitude_scale_std = magnitude_scale_std
        self.records: list[WindowRecord] = []
        self._cache: dict[str, np.ndarray] = {}
        self._build_records(max_windows_per_file=max_windows_per_file)

    def _build_records(self, max_windows_per_file: int | None) -> None:
        for row in self.manifest.to_dict("records"):
            path = row["path"]
            df = read_operation_csv(path)
            usable_len = max(1, int(len(df) * self.early_ratio))
            end_limit = min(len(df), usable_len)
            starts = list(range(0, max(1, end_limit - self.window_size + 1), max(1, self.stride)))
            if not starts:
                starts = [0]
            if max_windows_per_file and len(starts) > max_windows_per_file:
                idx = np.linspace(0, len(starts) - 1, max_windows_per_file).round().astype(int)
                starts = [starts[i] for i in idx]
            for start in starts:
                end = min(start + self.window_size, end_limit)
                self.records.append(
                    WindowRecord(
                        path=path,
                        accident=row["accident"],
                        severity=float(row["severity"]),
                        start=start,
                        end=end,
                        sample_id=f"{row['accident']}_{Path(path).stem}_{start}_{end}",
                    )
                )

    def __len__(self) -> int:
        return len(self.records)

    def _load_scaled(self, path: str) -> np.ndarray:
        if path not in self._cache:
            self._cache[path] = self.scaler.transform_frame(read_operation_csv(path))
        return self._cache[path]

    def __getitem__(self, index: int):
        rec = self.records[index]
        arr = self._load_scaled(rec.path)[rec.start : rec.end]
        if len(arr) < self.window_size:
            pad = np.zeros((self.window_size - len(arr), arr.shape[1]), dtype=np.float32)
            arr = np.vstack([arr.astype(np.float32), pad])
        else:
            arr = arr[: self.window_size].astype(np.float32)
        full_arr = arr.copy()
        used_ratio = self.early_ratio
        if self.random_early_ratio:
            used_ratio = float(random.choice(self.early_choices))
        if used_ratio < 1.0:
            visible = max(1, int(self.window_size * used_ratio))
            arr[visible:] = 0.0
        if self.augment:
            arr = self._augment(arr)
        return {
            "x": torch.from_numpy(arr),
            "x_full": torch.from_numpy(full_arr),
            "early_ratio": torch.tensor(used_ratio, dtype=torch.float32),
            "y_class": torch.tensor(self.class_to_idx[rec.accident], dtype=torch.long),
            "y_severity": torch.tensor(rec.severity, dtype=torch.float32),
            "metadata": {
                "path": rec.path,
                "accident": rec.accident,
                "severity": rec.severity,
                "start": rec.start,
                "end": rec.end,
                "sample_id": rec.sample_id,
            },
        }

    def _augment(self, arr: np.ndarray) -> np.ndarray:
        out = arr.copy()
        if self.gaussian_noise_std > 0:
            out += np.random.normal(0, self.gaussian_noise_std, out.shape).astype(np.float32)
        if self.sensor_dropout > 0:
            mask = np.random.rand(out.shape[1]) < self.sensor_dropout
            out[:, mask] = 0.0
        if self.time_mask_ratio > 0:
            width = max(1, int(out.shape[0] * self.time_mask_ratio))
            start = np.random.randint(0, max(1, out.shape[0] - width + 1))
            out[start : start + width] = 0.0
        if self.magnitude_scale_std > 0:
            scale = np.random.normal(1.0, self.magnitude_scale_std, (1, out.shape[1])).astype(np.float32)
            out *= scale
        return out.astype(np.float32)


NPPADWindowDataset = WindowedNPPADDataset
