from __future__ import annotations

import streamlit as st

from src.ui.components import metric_card, section_header
from src.ui.theme import apply_theme


st.set_page_config(
    page_title="NucleoGuard AI｜核电站复杂事故早期智能诊断系统",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()

st.title("NucleoGuard AI｜核电站复杂事故早期智能诊断系统")
st.caption("面向压水堆复杂瞬态事故的多任务时空注意力诊断与可解释 AI 原型平台")

c1, c2, c3, c4 = st.columns(4)
with c1:
    metric_card("数据基础", "NPPAD", help_text="PCTRAN 仿真的多变量时间序列事故数据")
with c2:
    metric_card("任务闭环", "分类 + 严重度", help_text="同时判断事故类型和事故强度")
with c3:
    metric_card("模型路线", "RF / Transformer", help_text="baseline 与 advanced 双模型对照")
with c4:
    metric_card("解释能力", "XAI Evidence", help_text="传感器贡献、时间注意力、物理证据链")

section_header("系统故事线")
st.markdown(
    """
传统核电站事故诊断依赖阈值报警和人工研判。面对 LOCA、SGTR、LLB、SLB 等复杂瞬态事故，早期信号弱、参数耦合强、误判代价高。

本项目基于 NPPAD 多变量时间序列数据，构建多任务时空注意力模型，同时完成事故类型识别和严重程度回归；通过 temporal attention、sensor importance 和 baseline 特征重要性解释模型判断依据；再通过高端可视化驾驶舱呈现事故概率、关键参数演化、风险等级、早期诊断曲线和物理证据链。
"""
)

section_header("页面导航")
st.markdown(
    """
- **事故智能诊断**：选择模型、事故样本和早期比例，查看实时诊断、风险等级、概率分布和自动报告。
- **模型性能评估**：比较 baseline 与 advanced，查看混淆矩阵、早期诊断曲线、鲁棒性和消融结果。
- **可解释性分析**：展示 temporal attention、sensor importance、关键传感器曲线和物理证据链。
- **数据集与工程说明**：说明 NPPAD、事故类别字典、工程架构和一键运行命令。
"""
)

st.info("Baseline 模型已保持可用；Advanced 模型若尚未训练，页面会提示运行 `python scripts/train_advanced.py`。")
