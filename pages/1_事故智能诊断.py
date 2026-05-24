from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.constants import ACCIDENT_NAMES, SENSOR_DESCRIPTIONS
from src.explain import explain_sample
from src.inference import advanced_available, predict_advanced, predict_baseline
from src.nppad_paths import REPORT_ROOT
from src.ui.components import (
    diagnosis_summary,
    metric_card,
    plot_probability_bars,
    plot_radar_probabilities,
    plot_sensor_importance,
    plot_timeseries,
    risk_badge,
    section_header,
)
from src.ui.theme import apply_theme


st.set_page_config(page_title="事故智能诊断", layout="wide")
apply_theme()
st.title("事故智能诊断驾驶舱")


@st.cache_data
def load_manifest():
    return pd.read_csv(REPORT_ROOT / "manifest.csv")


manifest = load_manifest()
trainable = manifest[manifest["use_for_training"]].copy()

with st.sidebar:
    st.header("诊断控制台")
    model_label = st.radio("选择模型", ["Baseline RandomForest", "Advanced Temporal Transformer"])
    if model_label.startswith("Advanced") and not advanced_available():
        st.warning("高级模型尚未训练，请运行 `python scripts/train_advanced.py`。当前可切回 baseline。")
    selected_class = st.selectbox("事故类型", sorted(trainable["accident"].unique()), format_func=lambda x: f"{x} · {ACCIDENT_NAMES.get(x, x)}")
    class_files = trainable[trainable["accident"] == selected_class].copy()
    class_files["sample"] = class_files["path"].apply(lambda p: Path(p).name)
    selected_sample = st.selectbox("样本", class_files["sample"].tolist())
    early_ratio = st.select_slider("早期诊断比例", options=[0.1, 0.2, 0.3, 0.5, 1.0], value=0.3, format_func=lambda x: f"{int(x*100)}%")
    add_noise = st.checkbox("加入小幅噪声模拟", value=False)

selected_path = class_files[class_files["sample"] == selected_sample]["path"].iloc[0]

try:
    if model_label.startswith("Advanced") and advanced_available():
        pred = predict_advanced(selected_path, early_ratio=early_ratio, add_noise=add_noise)
    else:
        pred = predict_baseline(selected_path, early_ratio=early_ratio, add_noise=add_noise)
except Exception as exc:
    st.error(f"诊断失败：{exc}")
    st.stop()

exp = explain_sample(pred["model"], selected_path, pred, early_ratio=early_ratio)
top2 = pd.DataFrame({"class": pred["classes"], "prob": pred["probabilities"]}).sort_values("prob", ascending=False).head(2)

k1, k2, k3, k4, k5, k6 = st.columns(6)
with k1:
    metric_card("真实事故", selected_class, help_text=ACCIDENT_NAMES.get(selected_class, ""))
with k2:
    metric_card("模型诊断", pred["predicted_class"], help_text=ACCIDENT_NAMES.get(pred["predicted_class"], ""))
with k3:
    metric_card("诊断置信度", f"{pred['confidence']:.1%}", help_text=f"Top-2: {', '.join(top2['class'])}")
with k4:
    metric_card("预测严重度", f"{pred['severity']:.1f}")
with k5:
    risk_badge(pred["severity"])
with k6:
    metric_card("推理耗时", f"{pred['inference_ms']:.1f} ms")

section_header("事故概率主视觉")
c1, c2 = st.columns([0.62, 0.38])
with c1:
    st.plotly_chart(plot_probability_bars(pred["classes"], pred["probabilities"], pred["predicted_class"]), use_container_width=True)
with c2:
    st.plotly_chart(plot_radar_probabilities(pred["classes"], pred["probabilities"]), use_container_width=True)

section_header("事故演化时间轴")
df = pred["dataframe"]
time_col = df["TIME"] if "TIME" in df.columns else pd.Series(range(len(df)))
cut_time = time_col.iloc[max(0, min(len(time_col) - 1, int(len(time_col) * early_ratio) - 1))]
st.markdown(
    f"""
<div class='ng-card'>
初始稳态 → 异常萌芽 → <b>t≈{cut_time:.0f}s 早期诊断窗口</b> → 高置信事故判定 → 严重度评估与处置建议
</div>
""",
    unsafe_allow_html=True,
)

section_header("关键运行参数")
default_sensors = exp["sensor_importance"]["sensor"].head(6).tolist() if not exp["sensor_importance"].empty else ["P", "TAVG", "WRCA", "WFWA", "QMWT"]
available = [c for c in df.columns if c != "TIME"]
sensors = st.multiselect(
    "选择传感器",
    available,
    default=[s for s in default_sensors if s in available][:6],
    format_func=lambda x: f"{x} · {SENSOR_DESCRIPTIONS.get(x, (x, ''))[0]}",
)
st.plotly_chart(plot_timeseries(df, sensors, early_ratio=early_ratio), use_container_width=True)

section_header("模型诊断证据链")
left, right = st.columns([0.58, 0.42])
with left:
    diagnosis_summary(exp["summary"])
with right:
    st.plotly_chart(plot_sensor_importance(exp["sensor_importance"]), use_container_width=True)

