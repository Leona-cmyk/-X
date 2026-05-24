# NucleoGuard AI

核电站复杂事故早期智能诊断系统。项目基于 NPPAD 多变量时间序列数据，构建 baseline 统计特征模型与 advanced 多任务时空注意力模型，用于事故类型识别、严重程度回归、早期预警和可解释 AI 展示。

## 数据准备

将 NPPAD GitHub 仓库解压到项目根目录，保持如下结构：

```text
NuclearPowerPlantAccidentData-main/
  Operation_csv_data/
    LOCA/
    SGBTR/
    ...
```

## 安装依赖

```powershell
pip install -r requirements_project.txt
```

## 训练 Baseline

```powershell
python scripts/train_baseline.py
```

输出：

- `outputs/models/baseline_classifier.pkl`
- `outputs/models/severity_regressor.pkl`
- `outputs/reports/metrics.json`
- `outputs/figures/baseline_confusion_matrix.png`
- `outputs/figures/baseline_feature_importance.png`

## 训练 Advanced

快速验证：

```powershell
python scripts/train_advanced.py --epochs 1 --max-files-per-class 2 --window-size 64 --stride 64 --max-windows-per-file 1
```

正式训练：

```powershell
python scripts/train_advanced.py --epochs 20 --max-files-per-class 0 --window-size 256 --stride 128
python scripts/evaluate_advanced.py
```

输出：

- `outputs/models/advanced_model.pt`
- `outputs/models/scaler.pkl`
- `outputs/reports/advanced_metrics.json`
- `outputs/reports/advanced_early_diagnosis_curve.csv`
- `outputs/reports/advanced_robustness.csv`
- `outputs/reports/advanced_ablation.csv`

## 启动网站

```powershell
streamlit run app.py
```

页面包括：

- 事故智能诊断
- 模型性能评估
- 可解释性分析
- 数据集与工程说明

## 技术创新点

- 早期诊断：支持 10%、20%、30%、50%、100% 事故进程推理。
- 多任务学习：事故分类与严重度回归共享时序表示。
- 时空建模：1D-CNN 提取局部突变，Transformer 捕捉长程耦合。
- 可解释 AI：Temporal Attention、传感器梯度贡献、baseline 全局重要性。
- 指挥舱可视化：概率分布、风险等级、时间线、关键传感器曲线和自动中文诊断报告。

## 模型创新点

Advanced 模型命名为 **NucleoGuard-MTSFormer**，全称为 Multi-task Spatio-Temporal Transformer for Nuclear Power Plant Accident Diagnosis。

1. 从静态统计特征升级为多变量时序端到端建模，输入为 `[batch, time, sensors]`。
2. Sensor Embedding 将传感器身份编码与实时数值融合，让模型区分不同物理测点。
3. Temporal Patch Embedding 将长序列切成时间片段，捕捉事故早期微弱局部突变。
4. Local Temporal Conv Block 使用轻量深度可分离卷积建模短期依赖和突变斜率。
5. Spatio-Temporal Transformer 建模长时间事故演化和传感器耦合。
6. Multi-task Learning 同时优化事故分类和严重度预测。
7. Early Consistency Loss 约束早期窗口与完整窗口预测分布一致，提升早期诊断稳定性。
8. Robust Augmentation 使用高斯噪声、传感器 dropout、时间遮蔽和幅值扰动提升可靠性。
9. Temporal Attention + Sensor Importance 提供可解释诊断证据链。
10. Baseline、早期诊断、鲁棒性和消融表共同支撑答辩论证。

当前真实结果会写入：

- `outputs/reports/advanced_metrics.json`
- `outputs/reports/model_compare.csv`
- `outputs/reports/advanced_early_diagnosis_curve.csv`
- `outputs/reports/advanced_robustness.csv`
- `outputs/reports/advanced_ablation.csv`

如果 advanced 暂未达到 baseline 的 99% 准确率，不能伪造指标。应在答辩中说明：baseline 使用全序列统计特征，信息高度压缩且任务较容易；advanced 面向更真实的早期窗口、噪声和传感器缺失场景，当前优先证明端到端时序建模、解释性和早期预警能力。后续优化方向包括扩大训练窗口数、延长训练轮次、增大模型维度、使用全量文件、加入学习率 warmup 和类别专用阈值校准。

## 竞赛答辩亮点

项目形成“数据层-模型层-解释层-可视化层”的完整 AI+核安全工程闭环。它不是单纯模型准确率展示，而是把事故早期信号、模型置信度、严重度、关键传感器证据和核工程物理逻辑整合为可演示的智能运维原型。
