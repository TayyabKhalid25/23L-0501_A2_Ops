import os
import subprocess
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
import mlflow

def get_git_commit_hash():
    """Returns current git commit hash."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
        return commit
    except Exception:
        return "unknown_commit"

def get_dvc_data_hash(dvc_file="data/raw.dvc"):
    """Reads DVC md5 hash for the dataset."""
    try:
        if os.path.exists(dvc_file):
            with open(dvc_file, "r") as f:
                content = yaml.safe_load(f)
                return content["outs"][0]["md5"]
    except Exception:
        pass
    return "unknown_dvc_hash"

def log_evaluation_plots_and_metrics(model, test_ds, run_name, artifacts_dir="artifacts", threshold=0.5):
    """
    Evaluates model on test_ds, logs metrics (accuracy, precision, recall, f1, roc_auc),
    generates and logs Confusion Matrix and ROC Curve to MLflow.
    """
    os.makedirs(artifacts_dir, exist_ok=True)
    
    y_true_list = []
    y_pred_prob_list = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true_list.extend(labels.numpy().flatten())
        y_pred_prob_list.extend(preds.flatten())

    y_true = np.array(y_true_list)
    y_probs = np.array(y_pred_prob_list)
    y_pred = (y_probs >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_probs)
    except Exception:
        auc = 0.5

    # Log metrics to MLflow
    mlflow.log_metric("test_accuracy", acc)
    mlflow.log_metric("test_precision", prec)
    mlflow.log_metric("test_recall", rec)
    mlflow.log_metric("test_f1", f1)
    mlflow.log_metric("test_roc_auc", auc)

    # 1. Confusion Matrix Plot
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Real (0)', 'AI (1)'],
                yticklabels=['Real (0)', 'AI (1)'])
    plt.title(f"Confusion Matrix: {run_name} (Threshold: {threshold})")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    cm_path = os.path.join(artifacts_dir, f"{run_name}_confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()
    mlflow.log_artifact(cm_path, artifact_path="evaluation_plots")

    # 2. ROC Curve Plot
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})', color='darkorange', lw=2)
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve: {run_name}')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    roc_path = os.path.join(artifacts_dir, f"{run_name}_roc_curve.png")
    plt.savefig(roc_path, dpi=200)
    plt.close()
    mlflow.log_artifact(roc_path, artifact_path="evaluation_plots")

    metrics_dict = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": auc,
        "y_true": y_true,
        "y_probs": y_probs
    }
    return metrics_dict
