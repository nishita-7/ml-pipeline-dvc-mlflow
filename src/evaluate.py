import pandas as pd
import pickle
import yaml
import os
import logging
from sklearn.metrics import (
    accuracy_score, f1_score,
    roc_auc_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
import mlflow

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

with open("params.yaml") as f:
    params = yaml.safe_load(f)["train"]

def evaluate(data_path, model_path):
    data = pd.read_csv(data_path)
    X = data.drop(columns=["Outcome"])
    y = data["Outcome"]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, random_state=params["random_state"]
    )

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    predictions = model.predict(X_test)

    accuracy  = accuracy_score(y_test, predictions)
    f1        = f1_score(y_test, predictions)
    roc_auc   = roc_auc_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall    = recall_score(y_test, predictions)

    logger.info(f"Accuracy  : {accuracy:.4f}")
    logger.info(f"F1 Score  : {f1:.4f}")
    logger.info(f"ROC-AUC   : {roc_auc:.4f}")
    logger.info(f"Precision : {precision:.4f}")
    logger.info(f"Recall    : {recall:.4f}")

    with mlflow.start_run():
        mlflow.log_metric("accuracy",  accuracy)
        mlflow.log_metric("f1_score",  f1)
        mlflow.log_metric("roc_auc",   roc_auc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall",    recall)

        cm = confusion_matrix(y_test, predictions)
        cr = classification_report(y_test, predictions)
        mlflow.log_text(str(cm), "confusion_matrix.txt")
        mlflow.log_text(cr, "classification_report.txt")

        mlflow.log_artifact("params.yaml")

if __name__ == "__main__":
    evaluate(params["data"], params["model"])