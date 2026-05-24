from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import WindowedNPPADDataset, build_class_maps, load_trainable_manifest, split_manifest_by_file
from src.inference import advanced_available, predict_advanced, predict_baseline
from src.nppad_paths import MODEL_ROOT, REPORT_ROOT
from src.preprocessing import TimeSeriesStandardScaler


def main():
    manifest = load_trainable_manifest(max_files_per_class=4, classes=["FLB", "LOCA"])
    assert not manifest.empty, "NPPAD trainable manifest is empty."
    sample = manifest.iloc[0]["path"]
    assert Path(sample).exists(), f"Sample missing: {sample}"
    assert (MODEL_ROOT / "baseline_classifier.pkl").exists(), "Baseline classifier missing."
    assert (MODEL_ROOT / "severity_regressor.pkl").exists(), "Severity regressor missing."
    assert (REPORT_ROOT / "manifest.csv").exists(), "Manifest report missing."

    pred = predict_baseline(sample, early_ratio=0.3)
    print("baseline:", pred["predicted_class"], f"{pred['confidence']:.3f}", f"severity={pred['severity']:.2f}")

    train_m, _, _ = split_manifest_by_file(manifest)
    scaler = TimeSeriesStandardScaler.fit_from_files(train_m["path"].tolist())
    class_to_idx, _ = build_class_maps(manifest)
    ds = WindowedNPPADDataset(train_m, class_to_idx, scaler, window_size=64, stride=64, early_ratio=0.5, max_windows_per_file=1)
    item = ds[0]
    assert item["x"].shape[0] == 64
    assert item["x"].shape[1] == len(scaler.sensor_cols)
    print("window:", tuple(item["x"].shape), item["metadata"]["sample_id"])

    if advanced_available():
        adv = predict_advanced(sample, early_ratio=0.3)
        print("advanced:", adv["predicted_class"], f"{adv['confidence']:.3f}", f"severity={adv['severity']:.2f}")
    else:
        print("advanced: not trained, fallback OK")


if __name__ == "__main__":
    main()
