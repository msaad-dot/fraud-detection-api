# Fraud Detection API — FastAPI Inference Service

A production-style machine learning inference API for detecting fraudulent credit card transactions.
The API serves a trained XGBoost fraud detection model via FastAPI, containerized with Docker and automatically deployed via a GitHub Actions CI/CD pipeline.

---

## Problem Context

Credit card fraud detection is a highly imbalanced classification problem, where missing a fraudulent transaction is often more costly than flagging a legitimate one.

In real production systems, models are rarely used directly. They must be:
- Validated
- Properly preprocessed
- Safely deployed behind an API

This project focuses on the **serving and inference layer** of a fraud detection system.

---

## Key Features

- FastAPI-based inference service
- XGBoost fraud detection model
- Strict request validation using Pydantic schemas
- Feature name and order alignment with training schema
- Partial preprocessing during inference (Time & Amount scaling)
- Clear error handling with meaningful HTTP responses
- Model metadata endpoint for observability
- JSON-safe prediction outputs
- Dockerized for consistent deployment anywhere
- CI/CD pipeline via GitHub Actions → auto-builds and pushes to Docker Hub on every push

---

## Project Structure

```
fraud-detection-api/
├── app/
│   ├── main.py        # API endpoints
│   ├── schemas.py     # Request validation schemas
│   ├── inference.py   # Model inference logic
│   ├── config.py      # Centralized configuration
│   └── __init__.py
├── models/
│   └── xgboost.pkl
├── artifacts/
│   └── standard_scaler.pkl
├── .github/
│   └── workflows/
│       └── deploy.yml  # CI/CD pipeline
├── Dockerfile
└── .gitignore
```

---

## API Endpoints

### `GET /health`
Returns API health status.

### `GET /model-info`
Returns model metadata:
- Model name and version
- Decision threshold
- Number of features

### `POST /predict`
Accepts a full feature vector (Time, Amount, V1–V28) and returns a fraud probability and decision.

**Example Request:**
```json
{
  "Time": 50000,
  "Amount": 120.75,
  "V1": 0.12,
  "V2": -0.45,
  "V3": 0.33,
  "V4": -0.08,
  "V5": 0.21,
  "V6": -0.15,
  "V7": 0.04,
  "V8": -0.02,
  "V9": 0.11,
  "V10": -0.19,
  "V11": 0.07,
  "V12": -0.13,
  "V13": 0.00,
  "V14": -0.22,
  "V15": 0.09,
  "V16": -0.17,
  "V17": 0.05,
  "V18": -0.06,
  "V19": 0.18,
  "V20": -0.03,
  "V21": 0.14,
  "V22": -0.20,
  "V23": 0.02,
  "V24": -0.09,
  "V25": 0.00,
  "V26": -0.11,
  "V27": 0.08,
  "V28": -0.04
}
```

---

## Running with Docker

Pull and run directly from Docker Hub — no setup required:

```bash
docker pull saaddot/fraud-detection-api
docker run -p 8000:8000 saaddot/fraud-detection-api
```

Then open: `http://localhost:8000/health`

Or build locally:
```bash
docker build -t fraud-detection-api .
docker run -p 8000:8000 fraud-detection-api
```

---

## CI/CD Pipeline

Every push to `main` triggers a GitHub Actions workflow that:

1. Checks out the code on a fresh Ubuntu runner
2. Logs into Docker Hub using repository secrets
3. Builds the Docker image
4. Pushes it to Docker Hub as `saaddot/fraud-detection-api:latest`

```
Push to main
     ↓
GitHub Actions (Ubuntu)
     ↓
Build Docker image
     ↓
Push to Docker Hub ← anyone can pull and run
```

---

## Design Decisions

- The API expects the **same feature schema used during training** to avoid inference drift
- PCA features (V1–V28) are assumed to be computed upstream
- Only Time and Amount are scaled during inference
- Feature order is explicitly enforced before prediction
- Inference logic is separated from API routing and configuration

---

## Tech Stack

Python · FastAPI · Pydantic · XGBoost · Scikit-learn · Docker · GitHub Actions

---

## Related Projects

- [fraud-detection-ml](https://github.com/msaad-dot/fraud-detection-ml) — Model training pipeline
- [drift-monitoring](https://github.com/msaad-dot/drift-monitoring) — Data drift monitoring

---

## Author

Mohamed Saad — [GitHub](https://github.com/msaad-dot)
