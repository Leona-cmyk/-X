from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "NuclearPowerPlantAccidentData-main" / "Operation_csv_data"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
MODEL_ROOT = OUTPUT_ROOT / "models"
FIGURE_ROOT = OUTPUT_ROOT / "figures"
REPORT_ROOT = OUTPUT_ROOT / "reports"
EXPLANATION_ROOT = REPORT_ROOT / "explanations"
SENSOR_META_PATH = REPORT_ROOT / "sensors.json"
ADVANCED_MODEL_PATH = MODEL_ROOT / "advanced_model.pt"
SCALER_PATH = MODEL_ROOT / "scaler.pkl"


for path in (OUTPUT_ROOT, MODEL_ROOT, FIGURE_ROOT, REPORT_ROOT, EXPLANATION_ROOT):
    path.mkdir(parents=True, exist_ok=True)
