from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.dataset import build_class_maps, load_trainable_manifest
from src.features import read_operation_csv, summarize_series
from src.models.deep_timeseries import MultiTaskTemporalNet
from src.models.nucleoguard_mtsformer import NucleoGuardMTSFormer
from src.nppad_paths import ADVANCED_MODEL_PATH, MODEL_ROOT, SCALER_PATH
from src.preprocessing import TimeSeriesStandardScaler


def load_baseline_bundle():
    with open(MODEL_ROOT / "baseline_classifier.pkl", "rb") as f:
        classifier = pickle.load(f)
    with open(MODEL_ROOT / "severity_regressor.pkl", "rb") as f:
        regressor = pickle.load(f)
    return classifier, regressor


def predict_baseline(csv_path: str | Path, early_ratio: float = 1.0, add_noise: bool = False) -> dict:
    start = time.perf_counter()
    classifier_bundle, regressor_bundle = load_baseline_bundle()
    clf = classifier_bundle["model"]
    reg = regressor_bundle["model"]
    feature_cols = classifier_bundle["feature_cols"]
    df = read_operation_csv(csv_path)
    if early_ratio < 1.0:
        df = df.iloc[: max(5, int(len(df) * early_ratio))].copy()
    if add_noise:
        numeric = [c for c in df.columns if c != "TIME"]
        df[numeric] = df[numeric] + np.random.normal(0, df[numeric].std().replace(0, 1) * 0.01, df[numeric].shape)
    row = summarize_series(df)
    X = pd.DataFrame([{col: row.get(col, 0.0) for col in feature_cols}])
    probs = clf.predict_proba(X)[0]
    pred_idx = int(np.argmax(probs))
    elapsed = (time.perf_counter() - start) * 1000
    return {
        "model": "Baseline RandomForest",
        "classes": list(clf.classes_),
        "probabilities": probs,
        "predicted_class": str(clf.classes_[pred_idx]),
        "confidence": float(probs[pred_idx]),
        "severity": float(reg.predict(X)[0]),
        "inference_ms": elapsed,
        "attention": None,
        "sensor_importance": None,
        "dataframe": df,
    }


def advanced_available() -> bool:
    return ADVANCED_MODEL_PATH.exists() and SCALER_PATH.exists()


def load_advanced_model(device: str = "cpu"):
    checkpoint = torch.load(ADVANCED_MODEL_PATH, map_location=device)
    if checkpoint.get("model_name") == "NucleoGuard-MTSFormer":
        model = NucleoGuardMTSFormer(
            num_sensors=len(checkpoint["sensor_cols"]),
            num_classes=len(checkpoint["idx_to_class"]),
            sensor_dim=checkpoint.get("sensor_dim", 32),
            model_dim=checkpoint.get("hidden_dim", 96),
            patch_size=checkpoint.get("patch_size", 8),
            patch_stride=checkpoint.get("patch_stride", 4),
            num_heads=checkpoint.get("num_heads", 4),
            num_layers=checkpoint.get("num_layers", 2),
            dropout=checkpoint.get("dropout", 0.15),
        )
    else:
        model = MultiTaskTemporalNet(
            num_sensors=len(checkpoint["sensor_cols"]),
            num_classes=len(checkpoint["idx_to_class"]),
            hidden_dim=checkpoint.get("hidden_dim", 96),
        )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    scaler = TimeSeriesStandardScaler.load(SCALER_PATH)
    idx_to_class = {int(k): v for k, v in checkpoint["idx_to_class"].items()}
    return model, scaler, idx_to_class, checkpoint


def make_advanced_input(csv_path: str | Path, scaler: TimeSeriesStandardScaler, window_size: int, early_ratio: float):
    df = read_operation_csv(csv_path)
    if early_ratio < 1.0:
        df = df.iloc[: max(5, int(len(df) * early_ratio))].copy()
    arr = scaler.transform_frame(df)
    if len(arr) < window_size:
        arr = np.vstack([arr, np.zeros((window_size - len(arr), arr.shape[1]), dtype=np.float32)])
    else:
        arr = arr[:window_size]
    return torch.from_numpy(arr.astype(np.float32)).unsqueeze(0), df


def predict_advanced(csv_path: str | Path, early_ratio: float = 1.0, add_noise: bool = False, device: str = "cpu") -> dict:
    if not advanced_available():
        raise FileNotFoundError("高级模型尚未训练，请运行 python scripts/train_advanced.py")
    start = time.perf_counter()
    model, scaler, idx_to_class, checkpoint = load_advanced_model(device=device)
    x, df = make_advanced_input(csv_path, scaler, checkpoint.get("window_size", 256), early_ratio)
    if add_noise:
        x = x + torch.randn_like(x) * 0.01
    with torch.no_grad():
        out = model(x.to(device))
    probs = out["probabilities"].cpu().numpy()[0]
    pred_idx = int(np.argmax(probs))
    elapsed = (time.perf_counter() - start) * 1000
    return {
        "model": "Advanced Temporal Transformer",
        "model_name": checkpoint.get("model_name", "Advanced Temporal Transformer"),
        "classes": [idx_to_class[i] for i in range(len(idx_to_class))],
        "probabilities": probs,
        "predicted_class": idx_to_class[pred_idx],
        "confidence": float(probs[pred_idx]),
        "severity": float(out["severity"].cpu().numpy()[0]),
        "inference_ms": elapsed,
        "attention": out["temporal_attention"].cpu().numpy()[0],
        "sensor_importance": None,
        "dataframe": df,
    }


def default_class_maps():
    manifest = load_trainable_manifest()
    return build_class_maps(manifest)
