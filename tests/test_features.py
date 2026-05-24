from __future__ import annotations

import math

from src.dataset import load_trainable_manifest
from src.features import read_operation_csv, summarize_series


def test_csv_read_and_summary_has_no_nan():
    manifest = load_trainable_manifest(max_files_per_class=1, classes=["FLB"])
    sample = manifest.iloc[0]["path"]
    df = read_operation_csv(sample)
    assert not df.empty
    summary = summarize_series(df)
    assert summary
    assert all(math.isfinite(v) for v in summary.values())
