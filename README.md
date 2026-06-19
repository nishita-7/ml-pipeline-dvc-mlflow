# 🩺 Diabetes Prediction — ML Pipeline with DVC & MLflow

An end-to-end machine learning pipeline for predicting diabetes onset using the **Pima Indians Diabetes Dataset**. Built with MLOps best practices: versioned data and models via **DVC**, experiment tracking via **MLflow**, and remote storage on **DagsHub**.

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
├── .env                          # MLflow credentials (never commit this)
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

The pipeline has three sequential stages orchestrated by DVC. Only stages whose inputs have changed are re-run.

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
- Reads raw CSV from `data/raw/`
- Replaces biologically impossible zero values in columns like `Glucose`, `BMI`, `Insulin` with column medians
- Saves cleaned data to `data/processed/data.csv`

### Stage 2 — Train (`src/train.py`)
- Splits data 80/20 (fixed `random_state` for reproducibility)
- Runs **GridSearchCV** over a `RandomForestClassifier`
- Logs hyperparameters, accuracy, confusion matrix, and classification report to **MLflow**
- Saves the best model to `models/model.pkl`

**Hyperparameter search space:**

| Parameter | Values Searched |
|---|---|
| `n_estimators` | 100, 200 |
| `max_depth` | 5, 10, None |
| `min_samples_split` | 2, 5 |
| `min_samples_leaf` | 1, 2 |

### Stage 3 — Evaluate (`src/evaluate.py`)
- Loads the saved model
- Evaluates on the same held-out test split used during training
- Logs accuracy, F1 score, ROC-AUC, precision, and recall to MLflow

---

## 🔧 Configuration

All tunable parameters live in `params.yaml`. Edit this file to change paths or model settings — no code changes needed.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Git
- A [DagsHub](https://dagshub.com) account

---

### Step 1 — Create a DagsHub repository

1. Go to [dagshub.com](https://dagshub.com) and log in
2. Click **New Repository**
3. Name it and create it
4. On the repo page, click **Remote** → copy both the Git and DVC remote URLs

---

### Step 2 — Clone & set up locally

```bash
git clone https://dagshub.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

Create a virtual environment and install dependencies:

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```

---

### Step 3 — Set up your `.env` file

Create a `.env` file in the project root with your DagsHub MLflow credentials:

```
MLFLOW_TRACKING_URI=https://dagshub.com/<your-username>/<your-repo-name>.mlflow
MLFLOW_TRACKING_USERNAME=<your-username>
MLFLOW_TRACKING_PASSWORD=<your-dagshub-token>
```

> Your DagsHub token is found under **Settings → Access Tokens** on DagsHub.

---

### Step 4 — Initialize DVC & track raw data

```bash
dvc init
dvc add data/raw/data.csv
git add data/raw/data.csv.dvc data/raw/.gitignore
```

---

### Step 5 — Set DVC remote storage to DagsHub

```bash
dvc remote add origin https://dagshub.com/<your-username>/<your-repo-name>.dvc
dvc remote modify origin --local auth basic
dvc remote modify origin --local user <your-username>
dvc remote modify origin --local password <your-dagshub-token>
```

---

### Step 6 — Register pipeline stages

```bash
# Stage 1
dvc stage add -n preprocess \
    -p preprocess.input,preprocess.output \
    -d src/preprocess.py -d data/raw/data.csv \
    -o data/processed/data.csv \
    python src/preprocess.py

# Stage 2
dvc stage add -n train \
    -p train.data,train.model,train.random_state,train.n_estimators,train.max_depth \
    -d src/train.py -d data/processed/data.csv \
    -o models/model.pkl \
    python src/train.py

# Stage 3
dvc stage add -n evaluate \
    -p train.random_state,evaluate.data,evaluate.model \
    -d src/evaluate.py -d data/processed/data.csv -d models/model.pkl \
    python src/evaluate.py
```

> **Windows (PowerShell):** Replace `\` with a backtick `` ` ``

---

### Step 7 — First commit

```bash
git add .
git commit -m "initial project setup"
```

---

### Step 8 — Run the pipeline

```bash
dvc repro
```

You should see all three stages execute in order:

```
Running stage 'preprocess'...
Running stage 'train'...
Running stage 'evaluate'...
```

---

### Step 9 — Push everything

```bash
# Push data & model to DagsHub DVC storage
dvc push

# Commit the generated lock file
git add dvc.lock
git commit -m "first pipeline run"

# Push code to DagsHub
git push -u origin main
```

---

### Step 10 — View results on DagsHub

| Tab | What you'll see |
|---|---|
| **Repository** | All your code and config files |
| **Files** | `data/raw/data.csv` and `models/model.pkl` (DVC-managed) |
| **Experiments** | MLflow runs with metrics, params, and artifacts |

---

## 🔁 Re-running After Changes

DVC is smart — it only re-runs stages whose inputs changed:

```bash
# Edit params.yaml (e.g. change n_estimators) then:
dvc repro

# Force re-run everything regardless
dvc repro --force

# Check what DVC thinks has changed
dvc status
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| [DVC](https://dvc.org/) | Data & model versioning, pipeline orchestration |
| [MLflow](https://mlflow.org/) | Experiment tracking, metrics logging |
| [DagsHub](https://dagshub.com/) | Remote Git + DVC storage + MLflow server |
| [scikit-learn](https://scikit-learn.org/) | RandomForestClassifier, GridSearchCV, metrics |
| [pandas](https://pandas.pydata.org/) | Data loading and preprocessing |
| [python-dotenv](https://pypi.org/project/python-dotenv/) | Loading credentials from `.env` |

---

## 📄 License

This project is open-source. See `LICENSE` for details.