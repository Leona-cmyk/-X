from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.constants import ACCIDENT_NAMES
from src.explain import explain_sample
from src.inference import advanced_available, predict_advanced, predict_baseline
from src.nppad_paths import REPORT_ROOT
from src.ui.components import (
    diagnosis_summary,
    plot_attention_timeline,
    plot_sensor_importance,
    plot_timeseries,
    section_header,
)
from src.ui.theme import apply_theme


st.set_page_config(page_title="可解释性分析", layout="wide")
apply_theme()
st.title("可解释性分析")

manifest = pd.read_csv(REPORT_ROOT / "manifest.csv")
trainable = manifest[manifest["use_for_training"]].copy()

with st.sidebar:
    model_label = st.radio("解释模型", ["Baseline RandomForest", "Advanced Temporal Transformer"])
    selected_class = st.selectbox("事故类型", sorted(trainable["accident"].unique()), format_func=lambda x: f"{x} · {ACCIDENT_NAMES.get(x, x)}")
    class_files = trainable[trainable["accident"] == selected_class].copy()
    class_files["sample"] = class_files["path"].apply(lambda p: Path(p).name)
    selected_sample = st.selectbox("样本", class_files["sample"].tolist())
    early_ratio = st.select_slider("早期比例", options=[0.1, 0.2, 0.3, 0.5, 1.0], value=0.5, format_func=lambda x: f"{int(x*100)}%")

selected_path = class_files[class_files["sample"] == selected_sample]["path"].iloc[0]
if model_label.startswith("Advanced") and advanced_available():
    pred = predict_advanced(selected_path, early_ratio=early_ratio)
else:
    if model_label.startswith("Advanced"):
        st.warning("高级模型尚未训练，当前展示 baseline 解释。")
    pred = predict_baseline(selected_path, early_ratio=early_ratio)
exp = explain_sample(pred["model"], selected_path, pred, early_ratio=early_ratio)

section_header("物理证据链")
diagnosis_summary(exp["summary"])

c1, c2 = st.columns([0.45, 0.55])
with c1:
    st.plotly_chart(plot_sensor_importance(exp["sensor_importance"], top_k=18), use_container_width=True)
with c2:
    st.plotly_chart(plot_attention_timeline(exp["attention"]), use_container_width=True)

section_header("关键传感器叠加曲线")
df = pred["dataframe"]
sensors = exp["sensor_importance"]["sensor"].head(6).tolist() if not exp["sensor_importance"].empty else ["P", "TAVG", "QMWT"]
st.plotly_chart(plot_timeseries(df, [s for s in sensors if s in df.columns], early_ratio=early_ratio), use_container_width=True)

section_header("模型结构图示")
st.markdown(
    """
```mermaid
flowchart LR
  A["多传感器时间序列"] --> B["1D-CNN 局部突变特征"]
  B --> C["Transformer 全局时序依赖"]
  C --> D["Attention Pooling 时间证据聚合"]
  D --> E["事故分类"]
  D --> F["严重度回归"]
  D --> G["XAI: 时间注意力 + 传感器贡献"]
```
"""
)

