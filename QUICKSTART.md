# 5 分钟跑通版本

1. 确认数据存在：

```powershell
Test-Path .\NuclearPowerPlantAccidentData-main\Operation_csv_data
```

2. 安装依赖：

```powershell
pip install -r requirements_project.txt
```

3. 运行 smoke test：

```powershell
python scripts/smoke_test.py
```

4. 训练一个极小 advanced 模型：

```powershell
python scripts/train_advanced.py --epochs 1 --max-files-per-class 2 --window-size 64 --stride 64 --max-windows-per-file 1
```

5. 启动网站：

```powershell
streamlit run app.py
```

如果只想演示 baseline，可跳过第 4 步。Advanced 页面会自动提示模型尚未训练。

