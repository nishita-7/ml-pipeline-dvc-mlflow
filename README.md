# 🩺 Diabetes Prediction — ML Pipeline with DVC & MLflow

A production-style, end-to-end machine learning pipeline for predicting diabetes onset using the **Pima Indians Diabetes Dataset**. Built with MLOps best practices: versioned data and models via **DVC**, experiment tracking via **MLflow**, and remote storage on **DagsHub**.

---

## 📁 Project Structure

```
machinelearningpipeline/
├── data/
│   ├── raw/
│   │   └── data.csv              # Original dataset (DVC-tracked)
│   └── processed/
│       └── data.csv              # Preprocessed output (DVC-tracked)
├── models/
│   └── model.pkl                 # Trained Random Forest model (DVC-tracked)
├── src/
│   ├── __init__.py
│   ├── preprocess.py             # Stage 1: Data preprocessing
│   ├── train.py                  # Stage 2: Model training + hyperparameter tuning
│   └── evaluate.py               # Stage 3: Model evaluation
├── dvc.yaml                      # Pipeline stage definitions
├── dvc.lock                      # Locked dependency hashes (auto-generated)
├── params.yaml                   # Centralized hyperparameters & config
├── requirements.txt              # Python dependencies
└── README.md
```

---

## 📊 Dataset

The **Pima Indians Diabetes Dataset** contains diagnostic measurements for 768 female patients. The goal is to predict whether a patient has diabetes (`Outcome = 1`) or not (`Outcome = 0`).

| Feature | Description |
|---|---|
| `Pregnancies` | Number of times pregnant |
| `Glucose` | Plasma glucose concentration (2-hour oral glucose test) |
| `BloodPressure` | Diastolic blood pressure (mm Hg) |
| `SkinThickness` | Triceps skin fold thickness (mm) |
| `Insulin` | 2-hour serum insulin (mu U/ml) |
| `BMI` | Body mass index (kg/m²) |
| `DiabetesPedigreeFunction` | Diabetes likelihood based on family history |
| `Age` | Age in years |
| `Outcome` | **Target variable** — 1 = diabetic, 0 = non-diabetic |

---

## ⚙️ Pipeline Overview

The pipeline consists of three sequential stages, orchestrated by DVC. Re-running `dvc repro` will only execute stages whose inputs have changed.

```
data/raw/data.csv
       │
       ▼
 [1] preprocess.py  ──►  data/processed/data.csv
                                  │
                                  ▼
                        [2] train.py  ──►  models/model.pkl
                                                  │
                                                  ▼
                                        [3] evaluate.py  ──►  MLflow metrics
```

### Stage 1 — Preprocess (`src/preprocess.py`)
Reads the raw CSV and writes a cleaned version to the `data/processed/` directory. Input and output paths are controlled via `params.yaml`.

### Stage 2 — Train (`src/train.py`)
- Splits data 80/20 (train/test)
- Runs **GridSearchCV** over a `RandomForestClassifier` to find the best hyperparameter combination
- Logs the best model, hyperparameters, accuracy, confusion matrix, and classification report to **MLflow**
- Registers the final model as `"Best Model"` in the MLflow Model Registry
- Saves the trained model to `models/model.pkl`

**Hyperparameter search space:**

| Parameter | Values Searched |
|---|---|
| `n_estimators` | 100, 200 |
| `max_depth` | 5, 10, None |
| `min_samples_split` | 2, 5 |
| `min_samples_leaf` | 1, 2 |

### Stage 3 — Evaluate (`src/evaluate.py`)
Loads the saved model and runs inference on the full dataset. Logs the final accuracy metric back to MLflow.

---

## 🔧 Configuration

All tunable parameters live in `params.yaml`. Edit this file to change paths or model settings — no code changes needed.

```yaml
preprocess:
  input:  data/raw/data.csv
  output: data/processed/data.csv

train:
  data:          data/raw/data.csv
  model:         models/model.pkl
  random_state:  42
  n_estimators:  100
  max_depth:     5
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Git

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd machinelearningpipeline

# Install dependencies
pip install -r requirements.txt
```

### Running the Pipeline

```bash
# Run all pipeline stages (only re-runs changed stages)
dvc repro

# Force re-run everything
dvc repro --force
```

### Viewing Experiments

Experiments are tracked on DagsHub. Open the MLflow UI to compare runs, parameters, and metrics:

```bash
# Or visit directly:
# https://dagshub.com/krishnaik06/machinelearningpipeline.mlflow
mlflow ui
```

---

## 📦 Adding / Modifying Pipeline Stages

Use `dvc stage add` to register new stages. Examples:

```bash
# Preprocess stage
dvc stage add -n preprocess \
    -p preprocess.input,preprocess.output \
    -d src/preprocess.py -d data/raw/data.csv \
    -o data/processed/data.csv \
    python src/preprocess.py

# Train stage
dvc stage add -n train \
    -p train.data,train.model,train.random_state,train.n_estimators,train.max_depth \
    -d src/train.py -d data/raw/data.csv \
    -o models/model.pkl \
    python src/train.py

# Evaluate stage
dvc stage add -n evaluate \
    -d src/evaluate.py -d models/model.pkl -d data/raw/data.csv \
    python src/evaluate.py
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| [DVC](https://dvc.org/) | Data & model versioning, pipeline orchestration |
| [MLflow](https://mlflow.org/) | Experiment tracking, model registry |
| [DagsHub](https://dagshub.com/) | Remote storage + hosted MLflow tracking server |
| [scikit-learn](https://scikit-learn.org/) | RandomForestClassifier, GridSearchCV, metrics |
| [pandas](https://pandas.pydata.org/) | Data loading and preprocessing |

---

## 📋 Requirements

```
dvc
dagshub
scikit-learn
mlflow
dvc-s3
```

Install with:

```bash
pip install -r requirements.txt
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes and run `dvc repro` to validate the pipeline
4. Commit both code and DVC lock files: `git add dvc.lock && git commit`
5. Open a pull request

---

## ⚠️ Security Note

MLflow tracking credentials are currently set as environment variables directly in the source code. Before sharing or open-sourcing this project, move these to a `.env` file or a secrets manager and add `.env` to `.gitignore`.

---

## 📄 License

This project is open-source. See `LICENSE` for details.
