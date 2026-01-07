import joblib # type: ignore
import pandas as pd  # pyright: ignore[reportMissingModuleSource]
from pathlib import Path
from .config import (
    MODEL_PATH,
    SCALER_PATH,
    THRESHOLD,
    FEATURE_ORDER
)

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

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