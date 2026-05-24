from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.nppad_paths import DATA_ROOT, REPORT_ROOT


SINGLE_SAMPLE_CLASSES = {"ATWS", "LACP", "LOF", "Normal", "SP", "TT"}


def parse_severity(path: Path) -> float:
    try:
        return float(path.stem)
    except ValueError:
        return -1.0


def build_manifest(data_root: Path = DATA_ROOT) -> pd.DataFrame:
    rows = []
    for class_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        for csv_path in sorted(class_dir.glob("*.csv"), key=lambda p: (len(p.stem), p.stem)):
            rows.append(
                {
                    "path": str(csv_path),
                    "accident": class_dir.name,
                    "severity": parse_severity(csv_path),
                    "use_for_training": class_dir.name not in SINGLE_SAMPLE_CLASSES,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    manifest = build_manifest()
    output = REPORT_ROOT / "manifest.csv"
    manifest.to_csv(output, index=False, encoding="utf-8-sig")

    summary = (
        manifest.groupby(["accident", "use_for_training"])
        .size()
        .reset_index(name="csv_count")
        .sort_values(["use_for_training", "accident"], ascending=[False, True])
    )
    summary_output = REPORT_ROOT / "class_summary.csv"
    summary.to_csv(summary_output, index=False, encoding="utf-8-sig")

    print(f"Manifest: {output}")
    print(f"Class summary: {summary_output}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
