"""
src/register_and_test_champion.py

Task 3.6: Model Registry Management and Champion Model Inference.
Allows explicit selection of which MLflow runs to register as Version 1 and Version 2
after inspecting the experiment comparison results from Tasks 2.7 & 3.5.

Usage:
  1. List all available runs and metrics:
     python src/register_and_test_champion.py --list

  2. Register your chosen best runs (e.g., best Custom CNN as v1, best MobileNetV2 as v2 & champion):
     python src/register_and_test_champion.py --v1-run <cnn_run_name> --v2-run <mobilenet_run_name>
"""

import os
import sys
import argparse
from pathlib import Path

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

from src.data_pipeline import TARGET_SIZE

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
MODEL_NAME = "VisionGuard-Classifier"
EXPERIMENT_NAME = "VisionGuard-AI-vs-Real"


def setup_mlflow():
    load_dotenv(override=True)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient()


def list_logged_runs():
    exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if not exp:
        print(f"[ERROR] Experiment '{EXPERIMENT_NAME}' not found.")
        return

    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    records = []
    for _, row in runs.iterrows():
        records.append({
            "Run Name": row.get("tags.mlflow.runName"),
            "Run ID": row["run_id"],
            "Model Type": row.get("tags.model_type"),
            "Test Acc": round(float(row["metrics.test_accuracy"]), 4) if pd.notna(row.get("metrics.test_accuracy")) else None,
            "Test F1": round(float(row["metrics.test_f1"]), 4) if pd.notna(row.get("metrics.test_f1")) else None,
            "Test AUC": round(float(row["metrics.test_roc_auc"]), 4) if pd.notna(row.get("metrics.test_roc_auc")) else None,
            "Best Val Acc": round(float(row["metrics.best_val_accuracy"]), 4) if pd.notna(row.get("metrics.best_val_accuracy")) else None
        })

    df = pd.DataFrame(records)
    print("\n" + "=" * 95)
    print(f"AVAILABLE RUNS IN DAGSHUB EXPERIMENT: '{EXPERIMENT_NAME}'")
    print("=" * 95)
    print(df.to_markdown(index=False))
    print("=" * 95 + "\n")


def register_selected_models(client, v1_run_name, v2_run_name):
    print("=" * 75)
    print("TASK 3.6: DAGSHUB MODEL REGISTRY CONFIGURATION")
    print("=" * 75)

    try:
        client.create_registered_model(MODEL_NAME, description="VisionGuard AI vs Real Image Binary Classifier")
        print(f"[OK] Created Registered Model: '{MODEL_NAME}'")
    except Exception as e:
        print(f"[INFO] Registered model '{MODEL_NAME}' exists or retrieved: {e}")

    exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])

    v1_df = runs[runs["tags.mlflow.runName"] == v1_run_name]
    v2_df = runs[runs["tags.mlflow.runName"] == v2_run_name]

    if v1_df.empty:
        raise ValueError(f"Could not find V1 run '{v1_run_name}' in MLflow. Run with --list to check run names.")
    if v2_df.empty:
        raise ValueError(f"Could not find V2 run '{v2_run_name}' in MLflow. Run with --list to check run names.")

    run_v1 = v1_df.iloc[0]["run_id"]
    run_v2 = v2_df.iloc[0]["run_id"]

    # Register Version 1
    v1 = client.create_model_version(
        name=MODEL_NAME,
        source=f"runs:/{run_v1}/model",
        run_id=run_v1,
        description=f"Version 1: Best Baseline Custom CNN ({v1_run_name})"
    )
    print(f"[OK] Registered Model Version {v1.version} from '{v1_run_name}' (Run ID: {run_v1})")

    # Register Version 2 (Champion)
    v2 = client.create_model_version(
        name=MODEL_NAME,
        source=f"runs:/{run_v2}/model",
        run_id=run_v2,
        description=f"Version 2: Champion Model ({v2_run_name})"
    )
    v2_version = str(v2.version)
    print(f"[OK] Registered Model Version {v2_version} from '{v2_run_name}' (Run ID: {run_v2})")

    # Assign alias 'champion' to Version 2
    try:
        client.set_registered_model_alias(name=MODEL_NAME, alias="champion", version=v2_version)
        print(f"[OK] Successfully assigned alias 'champion' -> Version {v2_version} ('{v2_run_name}')")
    except Exception as e:
        print(f"[INFO] Alias API note: {e}. Transitioning Version {v2_version} to Production...")
        client.transition_model_version_stage(name=MODEL_NAME, version=v2_version, stage="Production")


def run_inference_by_alias(v2_run_name):
    print("\n" + "=" * 75)
    print("TASK 3.6: INFERENCE USING CHAMPION MODEL LOADED BY ALIAS")
    print("=" * 75)

    model_uri = f"models:/{MODEL_NAME}@champion"
    print(f"Loading champion model from URI: '{model_uri}'...")

    try:
        model = mlflow.tensorflow.load_model(model_uri)
        print("[OK] Successfully downloaded and loaded champion model from DagsHub Model Registry!")
    except Exception as e:
        print(f"[NOTE] Remote registry URI load note: {e}")
        local_candidate = f"models/{v2_run_name}.keras"
        fallback_path = local_candidate if os.path.exists(local_candidate) else "models/champion_model.keras"
        print(f"Falling back to local DVC model artifact '{fallback_path}'...")
        model = keras.models.load_model(fallback_path)
        print(f"[OK] Loaded '{fallback_path}' successfully!")

    sample_img_path = "data/raw/ai/Midjourney-L-2068f65.jpg"
    if not os.path.exists(sample_img_path):
        sample_img_path = list(Path("data/raw").glob("*/*.*"))[0].as_posix()

    print(f"\n---> Running classification on sample image: '{sample_img_path}'")
    raw_bytes = tf.io.read_file(sample_img_path)
    img = tf.image.decode_image(raw_bytes, channels=3, expand_animations=False)
    img_resized = tf.image.resize(img, TARGET_SIZE, method="bilinear")
    img_batch = np.expand_dims(img_resized.numpy(), axis=0)

    prob = float(model.predict(img_batch, verbose=0)[0][0])
    threshold = 0.279  # Optimal threshold calibrated in Task 2.5
    label = "AI-Generated (Fake)" if prob >= threshold else "Authentic (Real)"
    confidence = prob if prob >= threshold else (1.0 - prob)

    print("-" * 75)
    print("PREDICTION RESULT:")
    print(f"  • Image Path         : {sample_img_path}")
    print(f"  • Raw Sigmoid Output : {prob:.4f}")
    print(f"  • Decision Threshold : {threshold:.3f}")
    print(f"  • Predicted Class    : {label}")
    print(f"  • Confidence Score   : {confidence * 100:.2f}%")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(description="Task 3.6: Register chosen best models & run inference by alias")
    parser.add_argument("--list", action="store_true", help="List all logged runs and their metrics before registering")
    parser.add_argument("--v1-run", type=str, help="Run name of best Custom CNN to register as Version 1")
    parser.add_argument("--v2-run", type=str, help="Run name of best Transfer/Tuned model to register as Version 2 (champion)")
    args = parser.parse_args()

    client = setup_mlflow()

    if args.list or not (args.v1_run and args.v2_run):
        list_logged_runs()
        if not (args.v1_run and args.v2_run):
            print("To register your chosen best runs after inspecting the table above, run:")
            print("  .\\.venv\\Scripts\\python src/register_and_test_champion.py --v1-run <best_cnn_run> --v2-run <best_mobilenet_run>")
        return

    register_selected_models(client, args.v1_run, args.v2_run)
    run_inference_by_alias(args.v2_run)


if __name__ == "__main__":
    main()
