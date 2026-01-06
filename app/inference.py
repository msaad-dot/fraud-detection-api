import joblib # type: ignore
import pandas as pd  # pyright: ignore[reportMissingModuleSource]
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "xgboost.pkl"
SCALER_PATH = BASE_DIR / "artifacts" / "standard_scaler.pkl"


model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

FEATURE_ORDER = [
    "Time",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9",
    "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19",
    "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28",
    "Amount"
]

#temp should be moved to config file later
THRESHOLD = 0.5

def predict_fraud(features: dict) -> dict:
    data = pd.DataFrame([features])

    # 2. Scale ONLY Time and Amount
    data[["Time", "Amount"]] = scaler.transform(
        data[["Time", "Amount"]]
    )

    # Reorder columns to match training schema
    data = data[FEATURE_ORDER]

    probability = model.predict_proba(data)[0][1]

    return {
        "fraud_probability": float(probability),
        "threshold": THRESHOLD,
        "is_fraud": bool(probability >= THRESHOLD)
    }