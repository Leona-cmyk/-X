from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from src.nppad_paths import REPORT_ROOT
from src.ui.components import metric_card, plot_confusion_matrix, plot_early_curve, section_header
from src.ui.theme import apply_theme


st.set_page_config(page_title="模型性能评估", layout="wide")
apply_theme()
st.title("模型性能评估")


def load_json(name: str) -> dict:
    path = REPORT_ROOT / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


baseline = load_json("metrics.json")
advanced = load_json("advanced_metrics.json")

section_header("Baseline vs Advanced 总体对比")
rows = [
    {
        "model": "Baseline RandomForest",
        "accuracy": baseline.get("accuracy"),
        "macro_f1": baseline.get("macro_f1"),
        "severity_mae": baseline.get("severity_mae"),
        "top2_acc": None,
        "inference_ms": None,
        "robustness_score": None,
    }
]
if advanced:
    rows.append({"model": "Advanced Temporal Transformer", **advanced})
else:
    st.info("高级模型结果尚未生成。运行 `python scripts/train_advanced.py` 和 `python scripts/evaluate_advanced.py` 后会自动显示。")
compare = pd.DataFrame(rows)
st.dataframe(compare, use_container_width=True)

c1, c2, c3 = st.columns(3)
with c1:
    metric_card("Baseline Accuracy", f"{baseline.get('accuracy', 0):.2%}")
with c2:
    metric_card("Baseline Macro F1", f"{baseline.get('macro_f1', 0):.3f}")
with c3:
    metric_card("Baseline Severity MAE", f"{baseline.get('severity_mae', 0):.2f}")

section_header("混淆矩阵")
cm_path = REPORT_ROOT / ("advanced_confusion_matrix.csv" if (REPORT_ROOT / "advanced_confusion_matrix.csv").exists() else "classification_report.csv")
if (REPORT_ROOT / "advanced_confusion_matrix.csv").exists():
    cm = pd.read_csv(REPORT_ROOT / "advanced_confusion_matrix.csv", index_col=0)
else:
    # Rebuild baseline confusion matrix display from saved image is less useful; show report table and prompt advanced.
    st.warning("当前没有 CSV 格式混淆矩阵。Baseline 图片仍在 outputs/figures 中；运行 advanced 后此处显示交互热力图。")
    cm = pd.DataFrame()
if not cm.empty:
    normalize = st.toggle("归一化显示", value=False)
    st.plotly_chart(plot_confusion_matrix(cm, normalize=normalize), use_container_width=True)

section_header("早期诊断曲线")
early_path = REPORT_ROOT / "advanced_early_diagnosis_curve.csv"
if early_path.exists():
    early = pd.read_csv(early_path)
    metric = st.selectbox("指标", [c for c in ["accuracy", "macro_f1", "severity_mae"] if c in early.columns])
    st.plotly_chart(plot_early_curve(early, metric), use_container_width=True)
else:
    st.info("请运行 `python scripts/evaluate_advanced.py` 生成早期诊断曲线。")

section_header("类别性能、鲁棒性与消融实验")
cols = st.columns(3)
with cols[0]:
    p = REPORT_ROOT / "advanced_classification_report.csv"
    if p.exists():
        rep = pd.read_csv(p, index_col=0)
        f1 = rep[rep.index.str.len() <= 6]["f1-score"].sort_values()
        st.bar_chart(f1)
    else:
        st.info("暂无 advanced per-class F1。")
with cols[1]:
    p = REPORT_ROOT / "advanced_robustness.csv"
    if p.exists():
        st.dataframe(pd.read_csv(p), use_container_width=True)
    else:
        st.info("暂无鲁棒性实验结果。")
with cols[2]:
    p = REPORT_ROOT / "advanced_ablation.csv"
    if p.exists():
        st.dataframe(pd.read_csv(p), use_container_width=True)
    else:
        st.info("暂无消融实验结果。")

