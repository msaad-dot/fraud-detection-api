from pathlib import Path

# ------------------------
# Project paths
# ------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "xgboost.pkl"
SCALER_PATH = BASE_DIR / "artifacts" / "standard_scaler.pkl"

# ------------------------
# Model metadata
# ------------------------
MODEL_NAME = "xgboost_fraud_detector"
MODEL_VERSION = "1.0.0"

# ------------------------
# Inference configuration
# ------------------------
THRESHOLD = 0.5

FEATURE_ORDER = [
    "Time",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9",
    "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19",
    "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28",
    "Amount"
]