from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.constants import ACCIDENT_NAMES, SENSOR_DESCRIPTIONS
from src.ui.theme import ACCIDENT_COLORS, PLOTLY_TEMPLATE, risk_level


def _apply_layout(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=height)
    return fig


def metric_card(label: str, value: str, delta: str | None = None, help_text: str | None = None) -> None:
    delta_html = f"<div class='ng-help'>{delta}</div>" if delta else ""
    help_html = f"<div class='ng-help'>{help_text}</div>" if help_text else ""
    st.markdown(
        f"<div class='ng-card'><div class='ng-title'>{label}</div><div class='ng-value'>{value}</div>{delta_html}{help_html}</div>",
        unsafe_allow_html=True,
    )


def risk_badge(severity: float) -> None:
    level, color = risk_level(severity)
    st.markdown(
        f"<span class='ng-badge' style='background:{color}22;color:{color};'>风险等级：{level}</span>",
        unsafe_allow_html=True,
    )


def section_header(title: str) -> None:
    st.markdown(f"<div class='ng-section'>{title}</div>", unsafe_allow_html=True)


def diagnosis_summary(text: str) -> None:
    st.markdown(f"<div class='ng-summary'>{text}</div>", unsafe_allow_html=True)
    st.code(text, language="text")


def plot_probability_bars(classes: list[str], probabilities, predicted_class: str | None = None) -> go.Figure:
    df = pd.DataFrame({"事故类别": classes, "概率": probabilities})
    df["名称"] = df["事故类别"].map(lambda x: ACCIDENT_NAMES.get(x, x))
    df = df.sort_values("概率", ascending=True)
    colors = [ACCIDENT_COLORS.get(c, "#60a5fa") if c == predicted_class else "rgba(96,165,250,.45)" for c in df["事故类别"]]
    fig = go.Figure(
        go.Bar(
            x=df["概率"],
            y=df["事故类别"] + " · " + df["名称"],
            orientation="h",
            marker_color=colors,
            hovertemplate="事故：%{y}<br>概率：%{x:.2%}<extra></extra>",
        )
    )
    fig.update_xaxes(tickformat=".0%")
    fig.update_layout(title="全类别事故概率分布")
    return _apply_layout(fig, height=430)


def plot_timeseries(df: pd.DataFrame, sensors: list[str], early_ratio: float = 1.0, highlight: tuple[float, float] | None = None) -> go.Figure:
    x = df["TIME"] if "TIME" in df.columns else np.arange(len(df))
    fig = go.Figure()
    for sensor in sensors:
        if sensor in df.columns:
            name, unit = SENSOR_DESCRIPTIONS.get(sensor, (sensor, ""))
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=df[sensor],
                    mode="lines",
                    name=f"{sensor} {name}",
                    hovertemplate=f"时间：%{{x}} s<br>{sensor}：%{{y:.3f}} {unit}<extra></extra>",
                )
            )
    if len(x) > 0:
        cutoff = x.iloc[min(len(x) - 1, max(0, int(len(x) * early_ratio) - 1))] if hasattr(x, "iloc") else x[min(len(x) - 1, int(len(x) * early_ratio))]
        fig.add_vline(x=cutoff, line_color="#facc15", line_dash="dash", annotation_text=f"早期截断 {int(early_ratio * 100)}%")
    if highlight:
        fig.add_vrect(x0=highlight[0], x1=highlight[1], fillcolor="rgba(239,68,68,.16)", line_width=0)
    fig.update_layout(title="关键运行参数演化")
    return _apply_layout(fig, height=430)


def plot_confusion_matrix(cm: pd.DataFrame, normalize: bool = False) -> go.Figure:
    data = cm.astype(float)
    if normalize:
        data = data.div(data.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    fig = go.Figure(
        go.Heatmap(
            z=data.to_numpy(),
            x=data.columns,
            y=data.index,
            colorscale="Tealgrn",
            hovertemplate="真实：%{y}<br>预测：%{x}<br>数值：%{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(title="混淆矩阵")
    return _apply_layout(fig, height=560)


def plot_early_curve(df: pd.DataFrame, metric: str = "accuracy") -> go.Figure:
    fig = go.Figure()
    if metric in df.columns:
        fig.add_trace(go.Scatter(x=df["early_ratio"], y=df[metric], mode="lines+markers", name=metric))
    fig.update_layout(title="早期诊断性能曲线", xaxis_title="事故进程比例", yaxis_title=metric)
    return _apply_layout(fig, height=360)


def plot_sensor_importance(df: pd.DataFrame, top_k: int = 15) -> go.Figure:
    if df.empty:
        df = pd.DataFrame({"sensor": ["暂无"], "importance": [0]})
    top = df.sort_values("importance", ascending=False).head(top_k).iloc[::-1]
    labels = [f"{s} · {SENSOR_DESCRIPTIONS.get(s, (s, ''))[0]}" for s in top["sensor"]]
    fig = go.Figure(go.Bar(x=top["importance"], y=labels, orientation="h", marker_color="#2dd4bf"))
    fig.update_layout(title="关键传感器贡献")
    return _apply_layout(fig, height=420)


def plot_attention_timeline(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        df = pd.DataFrame({"time": [0], "attention": [0]})
    fig = go.Figure(
        go.Scatter(
            x=df["time"],
            y=df["attention"],
            mode="lines",
            fill="tozeroy",
            line_color="#facc15",
            hovertemplate="时间：%{x}<br>注意力：%{y:.4f}<extra></extra>",
        )
    )
    fig.update_layout(title="Temporal Attention 时间关注度", xaxis_title="时间", yaxis_title="注意力权重")
    return _apply_layout(fig, height=320)


def plot_radar_probabilities(classes: list[str], probabilities) -> go.Figure:
    labels = [f"{c}" for c in classes]
    values = list(probabilities) + [probabilities[0]]
    labels = labels + [labels[0]]
    fig = go.Figure(go.Scatterpolar(r=values, theta=labels, fill="toself", line_color="#38bdf8"))
    fig.update_layout(title="事故概率雷达图", polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(tickformat=".0%")))
    return _apply_layout(fig, height=420)

