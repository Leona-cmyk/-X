from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.constants import ACCIDENT_NAMES, ACCIDENT_PHENOMENA, SENSOR_DESCRIPTIONS
from src.features import read_operation_csv
from src.inference import advanced_available, load_advanced_model, make_advanced_input, predict_baseline
from src.nppad_paths import EXPLANATION_ROOT, REPORT_ROOT


def baseline_sensor_importance(top_k: int = 20) -> pd.DataFrame:
    path = REPORT_ROOT / "feature_importance.csv"
    if not path.exists():
        return pd.DataFrame(columns=["sensor", "importance", "direction"])
    df = pd.read_csv(path)
    df["sensor"] = df["feature"].str.split("__").str[0]
    grouped = df.groupby("sensor", as_index=False)["importance"].sum().sort_values("importance", ascending=False)
    grouped["direction"] = "贡献增强"
    return grouped.head(top_k)


def advanced_gradient_importance(csv_path: str | Path, early_ratio: float = 1.0, top_k: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not advanced_available():
        raise FileNotFoundError("高级模型尚未训练，请运行 python scripts/train_advanced.py")
    model, scaler, idx_to_class, checkpoint = load_advanced_model()
    x, df = make_advanced_input(csv_path, scaler, checkpoint.get("window_size", 256), early_ratio)
    x.requires_grad_(True)
    out = model(x)
    pred_idx = int(out["logits"].argmax(dim=-1).item())
    out["logits"][0, pred_idx].backward()
    grad = x.grad.detach().abs().numpy()[0]
    sensor_scores = grad.mean(axis=0)
    imp = pd.DataFrame({"sensor": scaler.sensor_cols, "importance": sensor_scores})
    imp = imp.sort_values("importance", ascending=False).head(top_k)
    imp["direction"] = imp["sensor"].apply(lambda s: _direction(df[s]) if s in df.columns else "波动增强")

    attention = out["temporal_attention"].detach().numpy()[0]
    time = df["TIME"].to_numpy() if "TIME" in df.columns else np.arange(len(df))
    if len(time) < len(attention):
        time = np.pad(time, (0, len(attention) - len(time)), mode="edge")
    att = pd.DataFrame({"time": time[: len(attention)], "attention": attention})
    sample_id = Path(csv_path).parent.name + "_" + Path(csv_path).stem
    imp.to_csv(EXPLANATION_ROOT / f"{sample_id}_sensor_importance.csv", index=False, encoding="utf-8-sig")
    att.to_csv(EXPLANATION_ROOT / f"{sample_id}_temporal_attention.csv", index=False, encoding="utf-8-sig")
    return imp, att


def _direction(series: pd.Series) -> str:
    if len(series) < 3:
        return "波动增强"
    delta = float(series.iloc[-1] - series.iloc[0])
    std = float(series.std())
    if abs(delta) < max(std * 0.2, 1e-6):
        return "波动增强"
    return "上升" if delta > 0 else "下降"


def generate_chinese_explanation(predicted_class: str, confidence: float, severity: float, sensors: list[str], time_hint: str = "早期窗口") -> str:
    sensor_text = "、".join(
        f"{SENSOR_DESCRIPTIONS.get(s, (s, ''))[0]}({s})" for s in sensors[:3]
    ) or "关键运行参数"
    phenomenon = ACCIDENT_PHENOMENA.get(predicted_class, "关键热工水力参数出现与该类事故一致的耦合偏离。")
    return (
        f"模型在{time_hint}内判断为{ACCIDENT_NAMES.get(predicted_class, predicted_class)}"
        f"（{predicted_class}），置信度 {confidence:.1%}，预测严重度 {severity:.1f}。"
        f"主要依据是 {sensor_text} 的异常贡献最高；{phenomenon}"
    )


def explain_sample(model_type: str, csv_path: str | Path, prediction: dict, early_ratio: float = 1.0) -> dict:
    if (model_type.startswith("Advanced") or "NucleoGuard" in model_type or "MTSFormer" in model_type) and advanced_available():
        sensor_imp, attention = advanced_gradient_importance(csv_path, early_ratio=early_ratio)
    else:
        sensor_imp = baseline_sensor_importance()
        attention = pd.DataFrame(columns=["time", "attention"])
    sensors = sensor_imp["sensor"].head(5).tolist() if not sensor_imp.empty else []
    summary = generate_chinese_explanation(
        prediction["predicted_class"],
        prediction["confidence"],
        prediction["severity"],
        sensors,
        time_hint=f"事故进程前 {int(early_ratio * 100)}%",
    )
    return {"sensor_importance": sensor_imp, "attention": attention, "summary": summary}
