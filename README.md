# Churn MLOps — Production-Grade ML Pipeline

End-to-end MLOps pipeline for telecom customer churn prediction. Built to production standards with reproducibility, automated quality checks, observability, and cloud deployment.

> **Status:** 🚧 In active development — Phase 6 of 8 complete

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
- **Docker** + **docker-compose** for containerization ✅
- **Prefect** for pipeline orchestration ✅
- **GitHub Actions** for CI/CD ✅

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
│       ├── pipeline/ # Prefect orchestration flow
│       └── utils/    # Helpers + logging
├── tests/            # Pytest tests (incl. API tests + conftest)
├── notebooks/        # Jupyter exploration only
├── configs/          # Hydra YAML configs
├── scripts/          # One-off CLI scripts (incl. DVC stages)
├── .github/workflows/ # GitHub Actions CI pipeline
├── Dockerfile        # Container recipe (lean, runtime-only deps)
├── .dockerignore     # Keeps build context lean
├── docker-compose.yml # One-command container orchestration
├── dvc.yaml          # DVC pipeline definition (validate → train)
├── dvc.lock          # Pinned hashes for reproducibility
├── pyproject.toml    # Single config for all dev tools
├── requirements.in   # Top-level dev dependencies
├── requirements.txt  # Locked dev dependency versions
├── requirements-api.in  # Lean runtime-only dependencies (for Docker)
└── requirements-api.txt # Locked runtime dependency versions
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
- [x] **Phase 5:** Containerization with Docker ✅
  - [x] Dockerfile with layer-cached builds (deps before code)
  - [x] Lean runtime-only image (816MB; dev/exploration tools excluded)
  - [x] Runtime deps pinned to training versions (no train/serve drift)
  - [x] System libs handled (libgomp1 for LightGBM)
  - [x] docker-compose for one-command orchestration + auto-restart
- [x] **Phase 6:** Pipeline automation with Prefect + CI/CD ✅
  - [x] GitHub Actions CI: ruff lint + black format + pytest on every push
  - [x] Parallel CI job builds the Docker image from scratch
  - [x] Lazy model loading + test fixture so CI runs without the DVC model
  - [x] Prefect flow orchestrates validate → train as tracked tasks
  - [x] Automatic retries on task failure (self-healing pipeline)
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

## 🐳 Running with Docker (Phase 5)

The API is fully containerized for portable, reproducible deployment.

~~~bash
# Build and run with docker-compose (one command)
docker compose up

# Or build and run manually
docker build -t churn-api:latest .
docker run -p 8000:8000 churn-api:latest

# API is then live at http://localhost:8000/docs
~~~

**Container design notes:**
- **Lean image (816MB):** the production image installs only runtime dependencies (`requirements-api.txt`) — dev/exploration tools like ydata-profiling, seaborn, and pytest are intentionally excluded for a smaller, faster, more secure image.
- **No train/serve drift:** runtime dependencies (pandas, numpy, lightgbm) are pinned to the exact versions used during training, so the model behaves identically in the container.
- **System libraries:** `libgomp1` is installed for LightGBM's OpenMP runtime (the slim base image omits it).
- **Layer caching:** dependencies are installed before code is copied, so code changes trigger fast rebuilds.
- **Auto-restart:** `docker-compose.yml` uses `restart: unless-stopped` so the service recovers from crashes.

---

## 🤖 CI/CD & Orchestration (Phase 6)

### Continuous Integration (GitHub Actions)

Every push to `main` triggers an automated pipeline (`.github/workflows/ci.yml`) with two parallel jobs:

- **test job:** lints with ruff, checks formatting with black, runs the full pytest suite
- **docker job:** builds the production Docker image from scratch to catch any Dockerfile breakage

Because the model is DVC-tracked (absent on the CI runner), both jobs generate a lightweight stand-in model so the full pipeline runs anywhere. The app uses **lazy model loading** — the module imports cleanly without the model file present, which makes it testable in CI.

### Pipeline Orchestration (Prefect)

The training pipeline is wrapped as a Prefect flow (`src/churn_mlops/pipeline/flow.py`):

~~~bash
# Run the orchestrated pipeline
python -m src.churn_mlops.pipeline.flow
~~~

- **Tasks:** `validate_data` → `train_model`, each tracked with its own state (Completed/Failed)
- **Automatic retries:** each task retries up to 2 times with a delay on failure — the pipeline self-heals from transient errors instead of dying
- **Observability:** Prefect tracks every run, task state, and timing; a monitoring dashboard is available via `prefect server start`

This is the foundation for scheduled retraining and failure-alerting in production.

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
| **v1.5** | **May 2026** | **594K rows + Docker** | **Containerized: lean 816MB image, compose, pinned runtime deps, portable anywhere** |
| **v1.6** | **June 2026** | **594K rows + CI/CD** | **GitHub Actions CI (lint/test/build) + Prefect orchestration with retries** |

---

## 👤 Author

**Malav Joshi**
Ahmedabad, India · [GitHub](https://github.com/5Malav)

Building toward production ML reliability — the engineer who keeps AI systems alive.

---

## 📜 License

MIT
