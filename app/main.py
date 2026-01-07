from fastapi import FastAPI, HTTPException
from .schemas import Transaction
from .inference import predict_fraud

from .config import (
    MODEL_NAME,
    MODEL_VERSION,
    THRESHOLD,
    FEATURE_ORDER
)

app = FastAPI(
    title="Fraud Detection API",
    version="1.0.0"
)

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
def predict(transaction: Transaction):
    try:
        return predict_fraud(transaction.model_dump())

    except KeyError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required feature: {str(e)}"
        )

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Internal model inference error"
        )

@app.get("/model-info")
def model_info():
    return {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "threshold": THRESHOLD,
        "num_features": len(FEATURE_ORDER)
    }

