from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.constants import ACCIDENT_NAMES
from src.dataset import load_trainable_manifest
from src.explain import explain_sample
from src.inference import advanced_available, predict_advanced
from src.nppad_paths import EXPLANATION_ROOT
from src.ui.theme import risk_level


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sample", default="demo", help="demo or path to a csv sample")
    p.add_argument("--early-ratio", "--early_ratio", type=float, default=0.3)
    return p.parse_args()


def main():
    args = parse_args()
    if not advanced_available():
        print("高级模型尚未训练，请先运行 python scripts/train_advanced.py")
        return
    if args.sample == "demo":
        manifest = load_trainable_manifest(max_files_per_class=1, classes=["LLB"])
        if manifest.empty:
            manifest = load_trainable_manifest(max_files_per_class=1)
        sample_path = manifest.iloc[0]["path"]
    else:
        sample_path = args.sample
    pred = predict_advanced(sample_path, early_ratio=args.early_ratio)
    exp = explain_sample(pred.get("model_name", pred["model"]), sample_path, pred, early_ratio=args.early_ratio)
    risk, _ = risk_level(pred["severity"])
    top_idx = pred["probabilities"].argsort()[-2:][::-1]
    topk = [
        {"class": pred["classes"][i], "name": ACCIDENT_NAMES.get(pred["classes"][i], pred["classes"][i]), "prob": float(pred["probabilities"][i])}
        for i in top_idx
    ]
    result = {
        "pred_class": pred["predicted_class"],
        "pred_class_name": ACCIDENT_NAMES.get(pred["predicted_class"], pred["predicted_class"]),
        "confidence": pred["confidence"],
        "topk": topk,
        "severity": pred["severity"],
        "risk_level": risk,
        "temporal_attention": exp["attention"].to_dict("records"),
        "sensor_importance": exp["sensor_importance"].head(20).to_dict("records"),
        "explanation": exp["summary"],
    }
    output = EXPLANATION_ROOT / "demo_explanation.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2)[:2000])
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()

