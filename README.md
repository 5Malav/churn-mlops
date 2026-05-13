# Churn MLOps — Production-Grade ML Pipeline

End-to-end MLOps pipeline for telecom customer churn prediction. Built to production standards with reproducibility, automated quality checks, observability, and cloud deployment.

> **Status:** 🚧 In active development — Phase 1 of 8

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
- **DVC** for data versioning *(coming Phase 3)*
- **MLflow** for experiment tracking + model registry *(Phase 2)*
- **FastAPI** + **Uvicorn** for model serving *(Phase 4)*
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
│   ├── raw/          # Untouched original data (sacred)
│   ├── interim/      # Intermediate cleaning steps
│   ├── processed/    # Modeling-ready data
│   └── external/     # Third-party data
├── src/
│   └── churn_mlops/
│       ├── data/     # Data loading + validation
│       ├── features/ # Feature engineering
│       ├── models/   # Training + prediction
│       ├── api/      # FastAPI service
│       └── utils/    # Helpers + logging
├── tests/            # Pytest tests
├── notebooks/        # Jupyter exploration only
├── configs/          # Hydra YAML configs
├── scripts/          # One-off CLI scripts
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
- [ ] **Phase 2:** Hyperparameter tuning with Optuna
- [ ] **Phase 3:** Data + model versioning with DVC
- [ ] **Phase 4:** Prediction API with FastAPI
- [ ] **Phase 5:** Containerization with Docker
- [ ] **Phase 6:** Pipeline automation with Prefect + CI/CD
- [ ] **Phase 7:** GCP Cloud Run deployment with Terraform
- [ ] **Phase 8:** Monitoring + drift detection + polish

---
## 🎯 v1.1 Status — Trained on Real Data

The training pipeline is end-to-end functional on the **full 594K-row Telco dataset**.

~~~bash
# Run the full training pipeline
python -m src.churn_mlops.models.train
~~~

**v1.1 metrics (594K training rows):**

| Metric    | Value  | Interpretation |
|-----------|--------|----------------|
| Accuracy  | 0.8216 | 82% of all predictions correct |
| Precision | 0.5696 | When model predicts "churn", 57% are real churners |
| Recall    | 0.8506 | Model catches **85%** of actual churners ⭐ |
| F1 Score  | 0.6823 | Balanced metric of precision + recall |
| ROC AUC   | 0.9090 | Strong ranking ability across 118K test samples |

**Class distribution (real-world imbalance preserved):**

| Class | Count | Percentage |
|-------|-------|------------|
| No churn (0) | 460,377 | 77.5% |
| Churn (1)    | 133,817 | 22.5% |

**View experiments in MLflow:**

~~~bash
mlflow ui --backend-store-uri file:./mlruns
# Open http://localhost:5000
~~~

---

## 📊 Version History

| Version | Date | Dataset | Key Achievement |
|---------|------|---------|-----------------|
| v1.0 | May 2026 | 30 rows (sample) | Pipeline correctness verified end-to-end |
| **v1.1** | **May 2026** | **594K rows (full)** | **Real-data baseline: ROC AUC 0.91, Recall 0.85** |

**Engineering note:** SMOTE balancing took ~90 minutes at full scale due to O(n²) nearest-neighbor search across 107K minority samples. For Phase 2 hyperparameter tuning (Optuna), the pipeline will be modified to support stratified subsampling for fast iteration — standard MLOps practice for compute-bound experimentation. Final tuned model will be retrained on full data for the v1.2 release.

---

## 👤 Author

**Malav Joshi**
Ahmedabad, India · [GitHub](https://github.com/5Malav)

Building toward production ML reliability — the engineer who keeps AI systems alive.

---

## 📜 License

MIT
