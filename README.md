# Churn MLOps — Production-Grade ML Pipeline

End-to-end MLOps pipeline for telecom customer churn prediction. Built to production standards with reproducibility, automated quality checks, observability, and cloud deployment.

> **Status:** 🚧 In active development — Phase 4 of 8 complete

---

## 🎯 Project Goal

Predict which telecom customers will churn in the next 30 days, so retention teams can intervene early. Treat the entire pipeline as a production system: every component must be reproducible, observable, and recoverable.

This project is the foundation of my **production ML reliability** specialization — the engineer who keeps AI systems alive at 3am.

---

## 🛠️ Tech Stack

### Foundation
- **Python 3.11.9** locked via `pyenv` + `.python-version`
- **pip-tools** for reproducible dependency locking
- **pre-commit** + **ruff** + **black** + **mypy** for automated code quality

### ML & Data
- **pandas** + **numpy** for data manipulation
- **Pandera** for schema validation
- **LightGBM** for gradient boosting
- **imbalanced-learn** for class imbalance (SMOTE)
- **Optuna** for hyperparameter tuning

### MLOps Stack
- **DVC** for data + model versioning ✅
- **MLflow** for experiment tracking + model registry ✅
- **FastAPI** + **Uvicorn** for model serving ✅
- **Docker** + **docker-compose** for containerization *(Phase 5)*
- **Prefect** for pipeline orchestration *(Phase 6)*
- **GitHub Actions** + **CML** for CI/CD *(Phase 6)*

### Cloud & Monitoring
- **GCP Cloud Run** for serverless deployment *(Phase 7)*
- **GCP Cloud Storage** as DVC remote *(Phase 7)*
- **Terraform** for Infrastructure-as-Code *(Phase 7)*
- **Evidently** for drift detection *(Phase 8)*
- **Prometheus** + **Grafana** for live monitoring *(Phase 8)*

---

## 🏗️ Architecture (Coming Soon)

Architecture diagram will be added in Phase 8.

**Pipeline flow:**

~~~
Data → Validation → Feature Engineering → Training → Registry
                                                        ↓
Cloud Storage ← Drift Detection ← Monitoring ← Serving API
~~~
---

## 🚀 Quick Start

### Prerequisites
- macOS or Linux
- `pyenv` installed
- Python 3.11.9 available via pyenv
- Git

### Setup

```bash
# Clone the repo
git clone https://github.com/5Malav/churn-mlops.git
cd churn-mlops

# Set Python version + create virtualenv
pyenv local 3.11.9
python -m venv .venv
source .venv/bin/activate

# Install dependencies (pinned versions)
pip install pip-tools
pip-sync requirements.txt

# Install pre-commit hooks
pre-commit install

# Verify everything works
pre-commit run --all-files
```

---
## 📁 Project Structure

~~~
churn-mlops/
├── data/
│   ├── raw/          # Untouched original data (sacred, DVC-tracked)
│   ├── interim/      # Intermediate cleaning steps
│   ├── processed/    # Modeling-ready data
│   └── external/     # Third-party data
├── models/           # Trained model artifacts (DVC-tracked)
├── src/
│   └── churn_mlops/
│       ├── data/     # Data loading + validation
│       ├── features/ # Feature engineering
│       ├── models/   # Training + prediction
│       ├── api/      # FastAPI service (app + schemas)
│       └── utils/    # Helpers + logging
├── tests/            # Pytest tests (incl. API tests)
├── notebooks/        # Jupyter exploration only
├── configs/          # Hydra YAML configs
├── scripts/          # One-off CLI scripts (incl. DVC stages)
├── dvc.yaml          # DVC pipeline definition (validate → train)
├── dvc.lock          # Pinned hashes for reproducibility
├── pyproject.toml    # Single config for all dev tools
├── requirements.in   # Top-level dependencies
└── requirements.txt  # Locked dependency versions
~~~
---

## 📊 Phase Progress

- [x] **Phase 0:** Foundation (pyenv, venv, Python 3.11.9)
- [x] **Phase 1:** Reproducible training ✅
  - [x] Pre-commit hooks (ruff, black, ruff-format, end-of-file-fixer)
  - [x] Pandera schema validation (21 columns, happy + sad path tested)
  - [x] LightGBM training with SMOTE class balancing
  - [x] MLflow experiment tracking (params, metrics, model artifacts)
  - [x] Reproducibility verified (3 runs, identical metrics)
- [x] **Phase 2:** Hyperparameter tuning with Optuna ✅
  - [x] Stratified subsampling for fast iteration (50K subset, 0.0003% class drift)
  - [x] Pluggable balancing strategies (SMOTE / class_weight / none)
  - [x] 30-trial Optuna study with TPE sampler, optimizing F1
  - [x] class_weight chosen over SMOTE: same F1, ~500x faster training
- [x] **Phase 3:** Data + model versioning with DVC ✅
  - [x] DVC tracks 594K-row dataset (Git stores 93-byte pointers, not 80MB)
  - [x] Local DVC remote for offsite data + model storage
  - [x] Production model exported as versioned artifact (churn_model.txt)
  - [x] 2-stage reproducible pipeline (validate → train) via dvc.yaml
  - [x] `dvc repro` rebuilds the exact model; `dvc.lock` pins all hashes
- [x] **Phase 4:** Prediction API with FastAPI ✅
  - [x] Prediction module (loads model, predicts one customer)
  - [x] Pydantic request/response validation (the input bouncer)
  - [x] FastAPI app: `/`, `/health`, `/predict` endpoints
  - [x] Graceful error handling (clean 500, no leaked tracebacks)
  - [x] 4 automated tests (health, predict, validation, error path)
- [ ] **Phase 5:** Containerization with Docker
- [ ] **Phase 6:** Pipeline automation with Prefect + CI/CD
- [ ] **Phase 7:** GCP Cloud Run deployment with Terraform
- [ ] **Phase 8:** Monitoring + drift detection + polish

---
## 🎯 v1.2 Status — Optuna-Tuned on Real Data

The training pipeline is end-to-end functional, **Optuna-tuned**, on the full 594K-row Telco dataset.

~~~bash
# Run the full training pipeline
python -m src.churn_mlops.models.train
~~~

**v1.2 metrics (594K training rows, Optuna-tuned hyperparameters, class_weight balancing):**

| Metric    | Value  | Interpretation |
|-----------|--------|----------------|
| Accuracy  | 0.8296 | 83% of all predictions correct |
| Precision | 0.5837 | When model predicts "churn", 58% are real churners |
| Recall    | 0.8487 | Model catches **85%** of actual churners ⭐ |
| F1 Score  | 0.6917 | Balanced metric of precision + recall |
| ROC AUC   | 0.9143 | Strong ranking ability across 118K test samples |

**Performance gains vs v1.1:**

| Improvement | v1.1 | v1.2 | Change |
|-------------|------|------|--------|
| F1 Score    | 0.6823 | 0.6917 | **+0.94%** |
| ROC AUC     | 0.9090 | 0.9143 | **+0.53%** |
| Training time | ~92 min | ~11 sec | **~500x faster** ⚡ |

**Tuning methodology:** 30-trial Optuna study on a stratified 50K subsample, optimizing F1 with TPE sampler. Winning hyperparameters (learning_rate=0.0101, num_leaves=169, min_child_samples=63, feature_fraction=0.5665, bagging_fraction=0.6944) were then applied to the full dataset. The combination of class_weight balancing (instead of SMOTE) + Optuna-tuned params delivered both higher accuracy AND dramatically faster training.

**View experiments in MLflow:**

~~~bash
mlflow ui --backend-store-uri file:./mlruns
# Open http://localhost:5000
~~~

---

## 🌐 Prediction API (Phase 4)

The trained model is served as a REST API with FastAPI.

~~~bash
# Start the API server
uvicorn src.churn_mlops.api.app:app --reload

# Open interactive docs in your browser
# http://localhost:8000/docs
~~~

**Endpoints:**

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | Welcome message |
| GET | `/health` | Health check (for monitoring) |
| POST | `/predict` | Send a customer → get churn probability + prediction |

**Example prediction request:**

~~~bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"gender":"Female","SeniorCitizen":0,"Partner":"No","Dependents":"No","tenure":2,"PhoneService":"Yes","MultipleLines":"No","InternetService":"Fiber optic","OnlineSecurity":"No","OnlineBackup":"No","DeviceProtection":"No","TechSupport":"No","StreamingTV":"No","StreamingMovies":"No","Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check","MonthlyCharges":70.70,"TotalCharges":151.65}'
~~~

**Response:**

~~~json
{"churn_probability": 0.78, "churn_prediction": 1}
~~~

Input is validated with Pydantic (malformed requests get a clean 422 error).
Internal failures return a clean 500 without leaking tracebacks.
Run the API tests with `pytest tests/test_api.py -v`.

---

## 🔄 Reproducing This Project

This project uses DVC for full data and model reproducibility:

~~~bash
# 1. Clone the repo
git clone https://github.com/5Malav/churn-mlops.git
cd churn-mlops

# 2. Set up environment
pip-sync requirements.txt

# 3. Pull DVC-tracked data and model from remote
dvc pull

# 4. Reproduce the entire pipeline (validate → train)
dvc repro
~~~

The `dvc.lock` file pins exact hashes of all dependencies and outputs, so `dvc repro` produces a bit-for-bit identical model. The pipeline is defined in `dvc.yaml` as two stages: **validate** (Pandera schema check) and **train** (LightGBM with Optuna-tuned hyperparameters).

> **Note:** The DVC remote in this project is a local folder (development setup). Production deployment (Phase 7) will migrate to Google Cloud Storage.

---

## 📊 Version History

| Version | Date | Dataset | Key Achievement |
|---------|------|---------|-----------------|
| v1.0 | May 2026 | 30 rows (sample) | Pipeline correctness verified end-to-end |
| **v1.1** | **May 2026** | **594K rows (full)** | **Real-data baseline: ROC AUC 0.91, Recall 0.85** |
| **v1.2** | **May 2026** | **594K rows + Optuna-tuned** | **Tuned model: F1 0.69, ROC AUC 0.91, training 500x faster than v1.1** |
| **v1.3** | **May 2026** | **594K rows + DVC pipeline** | **Full data + model versioning, reproducible `dvc repro` pipeline** |
| **v1.4** | **May 2026** | **594K rows + REST API** | **Model served via FastAPI: /predict, /health, validated input, graceful errors** |

---

## 👤 Author

**Malav Joshi**
Ahmedabad, India · [GitHub](https://github.com/5Malav)

Building toward production ML reliability — the engineer who keeps AI systems alive.

---

## 📜 License

MIT
