from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import WindowedNPPADDataset, load_trainable_manifest, split_manifest_by_file
from src.inference import advanced_available, load_advanced_model
from src.nppad_paths import REPORT_ROOT


def top2_accuracy(y_true, probs):
    if len(y_true) == 0:
        return 0.0
    top2 = np.argsort(probs, axis=1)[:, -2:]
    return float(np.mean([yt in top2[i] for i, yt in enumerate(y_true)]))


def eval_ratio(ratio: float, noise: float = 0.0, mask_rate: float = 0.0, time_mask: float = 0.0):
    model, scaler, idx_to_class, checkpoint = load_advanced_model()
    manifest = load_trainable_manifest(max_files_per_class=checkpoint.get("args", {}).get("max_files_per_class", 8))
    class_to_idx = checkpoint["class_to_idx"]
    _, _, test_m = split_manifest_by_file(manifest, seed=checkpoint.get("args", {}).get("seed", 42))
    ds = WindowedNPPADDataset(test_m, class_to_idx, scaler, checkpoint.get("window_size", 128), checkpoint.get("window_size", 128), ratio, 2)
    y_true, y_pred, sev_true, sev_pred, confs, probs_all = [], [], [], [], [], []
    with torch.no_grad():
        for item in ds:
            x = item["x"].unsqueeze(0)
            if noise:
                x = x + torch.randn_like(x) * noise
            if mask_rate:
                mask = torch.rand(x.shape[-1]) < mask_rate
                x[:, :, mask] = 0
            if time_mask:
                width = max(1, int(x.shape[1] * time_mask))
                start = 0 if x.shape[1] == width else torch.randint(0, x.shape[1] - width + 1, (1,)).item()
                x[:, start : start + width, :] = 0
            out = model(x)
            probs = out["probabilities"][0].numpy()
            y_true.append(int(item["y_class"]))
            y_pred.append(int(probs.argmax()))
            confs.append(float(probs.max()))
            probs_all.append(probs)
            sev_true.append(float(item["y_severity"]))
            sev_pred.append(float(out["severity"][0]))
    probs_all = np.asarray(probs_all)
    return {
        "early_ratio": ratio,
        "noise": noise,
        "mask_rate": mask_rate,
        "time_mask": time_mask,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "severity_mae": mean_absolute_error(sev_true, sev_pred),
        "severity_rmse": mean_squared_error(sev_true, sev_pred) ** 0.5,
        "top2_accuracy": top2_accuracy(y_true, probs_all),
        "average_confidence": float(np.mean(confs)),
        "ece_simple": float(abs(np.mean(confs) - accuracy_score(y_true, y_pred))),
    }


def main():
    if not advanced_available():
        print("高级模型尚未训练，请运行 python scripts/train_advanced.py")
        return
    early = [eval_ratio(r) for r in [0.1, 0.2, 0.3, 0.5, 1.0]]
    clean = eval_ratio(1.0)
    robust = [
        {"condition": "clean", **clean},
        {"condition": "noise_0.01", **eval_ratio(1.0, noise=0.01)},
        {"condition": "noise_0.05", **eval_ratio(1.0, noise=0.05)},
        {"condition": "sensor_missing_5%", **eval_ratio(1.0, mask_rate=0.05)},
        {"condition": "sensor_missing_10%", **eval_ratio(1.0, mask_rate=0.10)},
        {"condition": "time_mask_10%", **eval_ratio(1.0, time_mask=0.10)},
    ]
    clean_acc = max(clean["accuracy"], 1e-8)
    for row in robust:
        row["robustness_score"] = row["accuracy"] / clean_acc
    pd.DataFrame(early).to_csv(REPORT_ROOT / "advanced_early_diagnosis_curve.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(robust).to_csv(REPORT_ROOT / "advanced_robustness.csv", index=False, encoding="utf-8-sig")

    baseline_metrics = {}
    baseline_path = REPORT_ROOT / "metrics.json"
    if baseline_path.exists():
        baseline_metrics = json.loads(baseline_path.read_text(encoding="utf-8"))
    adv = early[-1]
    compare = pd.DataFrame(
        [
            {"model": "Baseline RandomForest", "accuracy": baseline_metrics.get("accuracy"), "macro_f1": baseline_metrics.get("macro_f1"), "severity_mae": baseline_metrics.get("severity_mae")},
            {"model": "NucleoGuard-MTSFormer", "accuracy": adv["accuracy"], "macro_f1": adv["macro_f1"], "severity_mae": adv["severity_mae"], "top2_accuracy": adv["top2_accuracy"], "ece_simple": adv["ece_simple"]},
        ]
    )
    compare.to_csv(REPORT_ROOT / "model_compare.csv", index=False, encoding="utf-8-sig")
    print(compare.to_string(index=False))


if __name__ == "__main__":
    main()
