from __future__ import annotations

import streamlit as st


ACCIDENT_COLORS = {
    "FLB": "#42d392",
    "LLB": "#7dd3fc",
    "LOCA": "#fb7185",
    "LOCAC": "#f97316",
    "LR": "#a78bfa",
    "MD": "#facc15",
    "RI": "#38bdf8",
    "RW": "#e879f9",
    "SGATR": "#2dd4bf",
    "SGBTR": "#22c55e",
    "SLBIC": "#ef4444",
    "SLBOC": "#f59e0b",
}

PLOTLY_TEMPLATE = {
    "layout": {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(3,7,18,0.35)",
        "font": {"color": "#dbeafe", "family": "Microsoft YaHei, Segoe UI, Arial"},
        "xaxis": {"gridcolor": "rgba(148,163,184,0.16)", "zerolinecolor": "rgba(148,163,184,0.18)"},
        "yaxis": {"gridcolor": "rgba(148,163,184,0.16)", "zerolinecolor": "rgba(148,163,184,0.18)"},
        "legend": {"orientation": "h", "y": 1.08, "x": 0},
        "margin": {"l": 40, "r": 25, "t": 45, "b": 40},
    }
}

DARK_THEME = """
<style>
html, body, [class*="css"] {
  font-family: "Microsoft YaHei", "Segoe UI", Arial, sans-serif;
}
.stApp {
  background:
    radial-gradient(circle at 15% 0%, rgba(20, 184, 166, 0.14), transparent 32%),
    linear-gradient(135deg, #020617 0%, #08111f 54%, #0b1220 100%);
  color: #e5eefb;
}
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(15, 23, 42, .96), rgba(2, 6, 23, .98));
  border-right: 1px solid rgba(148, 163, 184, .18);
}
h1, h2, h3 { letter-spacing: 0; color: #f8fafc; }
.ng-card {
  border: 1px solid rgba(125, 211, 252, .18);
  border-radius: 8px;
  background: linear-gradient(180deg, rgba(15, 23, 42, .86), rgba(15, 23, 42, .56));
  box-shadow: 0 14px 35px rgba(0, 0, 0, .22);
  padding: 16px 18px;
  margin-bottom: 12px;
}
.ng-title {
  font-size: 14px;
  color: #93c5fd;
  margin-bottom: 8px;
}
.ng-value {
  font-size: 28px;
  line-height: 1.15;
  font-weight: 700;
  color: #f8fafc;
}
.ng-help {
  margin-top: 6px;
  color: #94a3b8;
  font-size: 12px;
}
.ng-badge {
  display: inline-block;
  padding: 5px 10px;
  border-radius: 999px;
  font-weight: 700;
  border: 1px solid rgba(255,255,255,.18);
}
.ng-section {
  margin: 18px 0 10px;
  padding-left: 10px;
  border-left: 3px solid #38bdf8;
  color: #bfdbfe;
  font-weight: 700;
}
.ng-summary {
  border-left: 3px solid #2dd4bf;
  background: rgba(15, 23, 42, .72);
  padding: 14px 16px;
  border-radius: 8px;
  color: #dbeafe;
}
div[data-testid="stMetric"] {
  background: rgba(15, 23, 42, .62);
  border: 1px solid rgba(148, 163, 184, .16);
  border-radius: 8px;
  padding: 12px;
}
</style>
"""


def apply_theme() -> None:
    st.markdown(DARK_THEME, unsafe_allow_html=True)


def risk_level(severity: float) -> tuple[str, str]:
    if severity < 20:
        return "低", "#22c55e"
    if severity < 50:
        return "中", "#facc15"
    if severity < 80:
        return "高", "#f97316"
    return "极高", "#ef4444"

