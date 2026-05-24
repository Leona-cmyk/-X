from __future__ import annotations

import pandas as pd
import streamlit as st

from src.constants import ACCIDENT_NAMES, ACCIDENT_PHENOMENA
from src.nppad_paths import REPORT_ROOT
from src.ui.components import section_header
from src.ui.theme import apply_theme


st.set_page_config(page_title="数据集与工程说明", layout="wide")
apply_theme()
st.title("数据集与工程说明")

section_header("NPPAD 数据集")
st.markdown(
    """
NPPAD 是面向压水堆核电站事故诊断的公开多变量时间序列数据集，基于 PCTRAN 仿真生成，覆盖 LOCA、SGTR、SLB、LLB、FLB、控制棒动作、慢化剂稀释等典型工况。数据包含稳压器压力、主泵流量、堆芯功率、冷却剂温度、蒸汽发生器压力、辐射监测等关键运行参数，可用于事故类型分类、严重程度评估、早期诊断和可解释 AI 分析。
"""
)

section_header("事故类别字典")
manifest_path = REPORT_ROOT / "class_summary.csv"
if manifest_path.exists():
    summary = pd.read_csv(manifest_path)
else:
    summary = pd.DataFrame()
rows = []
for code, name in ACCIDENT_NAMES.items():
    rows.append({"缩写": code, "中文名": name, "典型物理现象": ACCIDENT_PHENOMENA.get(code, "待补充"), "样本数": int(summary[summary["accident"] == code]["csv_count"].sum()) if not summary.empty else None})
st.dataframe(pd.DataFrame(rows), use_container_width=True)

section_header("工程架构")
st.markdown(
    """
```mermaid
flowchart TB
  A["数据层: NPPAD CSV / MDB"] --> B["特征层: 统计特征 + 窗口化序列"]
  B --> C["模型层: Baseline RF + Temporal Transformer"]
  C --> D["解释层: Feature Importance / Attention / Gradient"]
  D --> E["可视化层: NucleoGuard AI 驾驶舱"]
```
"""
)

section_header("项目创新点")
st.markdown(
    """
- 早期诊断：支持仅使用事故进程前 10%、20%、30%、50% 数据进行推理。
- 多任务学习：同一模型同时输出事故类别和严重程度。
- 可解释 AI：展示时间关注度、关键传感器贡献和中文物理证据链。
- 鲁棒性评估：支持噪声扰动和传感器遮蔽实验。
- 高端可视化驾驶舱：把模型结果转化为答辩友好的工业安全界面。
"""
)

section_header("一键运行命令")
st.code(
    """
pip install -r requirements_project.txt
python scripts/train_baseline.py
python scripts/train_advanced.py --epochs 3 --max-files-per-class 8
python scripts/evaluate_advanced.py
streamlit run app.py
""",
    language="powershell",
)

