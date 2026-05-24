# 核+X 项目实施方案

## 项目名称

面向核电站复杂事故早期诊断的可解释时空注意力模型

## 选择理由

在 Gemini 给出的三套方案中，本项目优先选择“反应堆复杂事故智能诊断”。它比核医学影像方向更容易获取和处理数据，也比 PINNs 中子通量求解方向更容易在短周期内做出可运行原型。项目使用公开 NPPAD 数据集，任务形式清晰：输入多传感器时间序列，输出事故类型和事故严重程度，并通过可解释性方法展示模型依据。

## 核心价值

核电站事故早期阶段往往表现微弱，传统阈值报警容易滞后，也难以解释根因。本项目将核电站多源传感器数据转化为 AI 诊断任务，构建一个既能分类事故类型、又能定位关键传感器贡献的智能辅助诊断系统。比赛展示时可以形成“数据-模型-解释-演示系统”的完整闭环。

## 数据需求

优先数据源：

- NPPAD: Nuclear Power Plant Accident Data
- Figshare: https://springernature.figshare.com/articles/dataset/NPPAD_An_Open_Time-series_Dataset_Covering_Various_Accidents_for_Nuclear_Power_Plants/21299880
- GitHub: https://github.com/thu-inet/NuclearPowerPlantAccidentData

你需要下载的数据：

- `Operation_csv_data`
- 如空间允许，也下载完整压缩包或完整 GitHub 仓库
- 暂时不需要 `Dose_csv_data`，后续可作为扩展亮点

数据规模参考：

- NPPAD 公开数据约 775 MB
- 数据覆盖压水堆多类事故和正常工况
- 每个样本为多变量时间序列，文献中常见设置为约 97 个运行参数

## 第一阶段目标

做出一个可运行的最小闭环：

1. 读取 NPPAD 的运行参数 CSV 数据
2. 自动识别事故类别和严重程度标签
3. 构建滑动时间窗样本
4. 训练基线模型：Random Forest 或 LightGBM
5. 训练深度模型：1D-CNN + Transformer Encoder
6. 输出准确率、F1、混淆矩阵
7. 输出特征重要性或注意力热力图
8. 做一个 Streamlit 演示页面

## 推荐技术路线

数据预处理：

- 合并不同事故文件夹下的 CSV
- 提取事故类型作为分类标签
- 提取文件编号或元信息作为严重程度标签
- 对传感器特征做 Z-score 标准化
- 使用固定长度滑动窗口，例如 30、60、120 个时间步

模型设计：

- 基线模型：Random Forest / XGBoost / LightGBM
- 主模型：Temporal CNN + Transformer Encoder
- 输出头一：事故类型分类
- 输出头二：严重程度回归或分级分类

可解释性：

- 基线模型使用 SHAP 特征重要性
- 深度模型使用输入梯度、遮挡实验或注意力权重
- 重点解释压力、温度、流量、液位等关键核电运行参数对事故判定的贡献

展示系统：

- 左侧选择事故样本和时间窗
- 中间展示多传感器曲线
- 右侧展示模型诊断结果、置信度、关键传感器排名
- 底部展示混淆矩阵和模型性能对比

## 预期创新点

- 面向核电站事故早期阶段的短窗口诊断，而不是完整事故过程后验分类
- 多任务学习：同时输出事故类别和严重程度
- 可解释 AI：把模型判断映射回核电站物理参数变化
- 可交互演示：让评委直观看到事故演化、模型判断和解释依据

## 交付物

- 可运行 Python 项目
- 数据预处理脚本
- 模型训练脚本
- 结果图表：混淆矩阵、F1 对比、关键传感器贡献图
- Streamlit 可视化演示
- 竞赛论文/研究报告初稿
- 答辩 PPT 提纲

## 当前已完成进展

- 已读取 `NuclearPowerPlantAccidentData-main/Operation_csv_data`
- 已生成数据清单：`outputs/reports/manifest.csv`
- 已筛选 12 个样本量充足的事故类别用于第一版训练
- 已训练 Random Forest 事故分类基线模型
- 已训练 Random Forest 严重程度回归基线模型
- 已生成混淆矩阵和全局特征重要性图
- 已完成 Streamlit 交互式诊断原型

第一版实验结果：

- 训练样本：908
- 测试样本：303
- 事故分类准确率：99.01%
- Macro F1：0.9901
- 严重程度预测 MAE：2.64
- 严重程度预测 R2：0.9792

当前原型文件：

- `src/build_manifest.py`：生成数据清单与类别统计
- `src/features.py`：读取 CSV 并提取时间序列统计特征
- `src/train_baseline.py`：训练分类和严重程度回归模型
- `app.py`：交互式事故诊断演示页面
- `outputs/figures/baseline_confusion_matrix.png`：分类混淆矩阵
- `outputs/figures/baseline_feature_importance.png`：关键特征重要性

## 需要你提供或下载

请优先下载 NPPAD 的 `Operation_csv_data`。如果网速和空间允许，直接下载完整 GitHub 仓库或 Figshare 完整数据包。

下载完成后，把数据放到：

```text
D:\竞赛材料汇总\核+X\data\nppad\
```

我拿到数据后可以继续完成：

- 项目代码结构
- 数据读取与探索
- 第一版基线模型
- 可视化演示页面
