from fastapi import FastAPI
from .schemas import Transaction
from .inference import predict_fraud


app = FastAPI(
    title="Fraud Detection API",
    version="1.0.0"
)

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
def predict(transaction: Transaction):
     return predict_fraud(transaction.model_dump())
