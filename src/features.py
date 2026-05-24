from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DROP_COLUMNS = {"TIME"}


def read_operation_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    return df.interpolate(limit_direction="both").fillna(0.0)


def summarize_series(df: pd.DataFrame) -> dict[str, float]:
    features: dict[str, float] = {}
    signal_cols = [col for col in df.columns if col not in DROP_COLUMNS]
    for col in signal_cols:
        s = df[col].astype(float)
        values = s.to_numpy()
        features[f"{col}__mean"] = float(np.mean(values))
        features[f"{col}__std"] = float(np.std(values))
        features[f"{col}__min"] = float(np.min(values))
        features[f"{col}__max"] = float(np.max(values))
        features[f"{col}__last"] = float(values[-1])
        features[f"{col}__delta"] = float(values[-1] - values[0])
    features["n_steps"] = float(len(df))
    return features


def build_feature_table(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for record in manifest.to_dict("records"):
        df = read_operation_csv(record["path"])
        row = summarize_series(df)
        row.update(
            {
                "path": record["path"],
                "accident": record["accident"],
                "severity": record["severity"],
                "use_for_training": record["use_for_training"],
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)

