from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.constants import SENSOR_DESCRIPTIONS
from src.features import DROP_COLUMNS, read_operation_csv
from src.nppad_paths import SCALER_PATH, SENSOR_META_PATH


@dataclass
class TimeSeriesStandardScaler:
    sensor_cols: list[str]
    mean_: np.ndarray
    scale_: np.ndarray

    @classmethod
    def fit_from_files(cls, paths: list[str | Path], sensor_cols: list[str] | None = None) -> "TimeSeriesStandardScaler":
        sums = None
        sums_sq = None
        count = 0
        inferred_cols = sensor_cols
        for path in paths:
            df = read_operation_csv(path)
            if inferred_cols is None:
                inferred_cols = [col for col in df.columns if col not in DROP_COLUMNS]
            arr = df[inferred_cols].to_numpy(dtype=np.float32)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            if sums is None:
                sums = arr.sum(axis=0, dtype=np.float64)
                sums_sq = np.square(arr, dtype=np.float64).sum(axis=0)
            else:
                sums += arr.sum(axis=0, dtype=np.float64)
                sums_sq += np.square(arr, dtype=np.float64).sum(axis=0)
            count += arr.shape[0]
        if inferred_cols is None or sums is None or sums_sq is None or count == 0:
            raise ValueError("No rows found while fitting scaler.")
        mean = sums / count
        var = np.maximum(sums_sq / count - mean**2, 0.0)
        scale = np.sqrt(var)
        scale[scale < 1e-8] = 1.0
        return cls(sensor_cols=list(inferred_cols), mean_=mean.astype(np.float32), scale_=scale.astype(np.float32))

    def transform_frame(self, df: pd.DataFrame) -> np.ndarray:
        clean = df.copy()
        for col in self.sensor_cols:
            if col not in clean.columns:
                clean[col] = 0.0
        arr = clean[self.sensor_cols].to_numpy(dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return (arr - self.mean_) / self.scale_

    def save(self, path: str | Path = SCALER_PATH) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path = SCALER_PATH) -> "TimeSeriesStandardScaler":
        with open(path, "rb") as f:
            return pickle.load(f)


def write_sensor_metadata(sensor_cols: list[str], output_path: Path = SENSOR_META_PATH) -> None:
    sensors = []
    for col in sensor_cols:
        name, unit = SENSOR_DESCRIPTIONS.get(col, (f"{col} 参数", "待补充"))
        sensors.append({"name": col, "zh_name": name, "unit": unit})
    output_path.write_text(json.dumps(sensors, ensure_ascii=False, indent=2), encoding="utf-8")


def load_sensor_metadata(path: Path = SENSOR_META_PATH) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))

