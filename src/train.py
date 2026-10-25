import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import mlflow
import mlflow.tensorflow
from dotenv import load_dotenv

from src.data_pipeline import get_datasets, RANDOM_SEED, BATCH_SIZE, TARGET_SIZE
from src.models import build_baseline_cnn, build_transfer_mobilenet
from src.utils import get_git_commit_hash, get_dvc_data_hash, log_evaluation_plots_and_metrics

# Silence redundant TF logs and enable file store fallback
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

# ==========================================
# Global Constants 
# ==========================================
EPOCHS_BASELINE = 10
EPOCHS_TRANSFER_STAGE1 = 8
EPOCHS_TRANSFER_STAGE2 = 6
LR_BASELINE = 1e-3
LR_TRANSFER_STAGE1 = 1e-4
LR_TRANSFER_STAGE2 = 1e-5
UNFREEZE_LAYERS = 20
EXPERIMENT_NAME = "VisionGuard-AI-vs-Real"

class MLflowEpochCallback(keras.callbacks.Callback):
    """Logs epoch metrics to MLflow per step."""
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for metric, val in logs.items():
            mlflow.log_metric(f"epoch_{metric}", float(val), step=epoch + 1)

def print_layer_table(model):
    """Prints a formatted Markdown table of output shape and parameter count per layer."""
    records = []
    total_params = 0
    trainable_params = 0

    for layer in model.layers:
        params = layer.count_params()
        total_params += params
        if layer.trainable:
            trainable_params += params
        records.append({
            "Layer Name": layer.name,
            "Layer Type": layer.__class__.__name__,
            "Output Shape": str(layer.output.shape),
            "Param Count": f"{params:,}"
        })

    df = pd.DataFrame(records)
    print("\n" + "=" * 75)
    print(f"TASK 2.2 DELIVERABLE: {model.name} LAYER-BY-LAYER SPECIFICATION TABLE")
    print("=" * 75)
    print(df.to_markdown(index=False))
    print("-" * 75)
    print(f"Total Parameters     : {total_params:,}")
    print(f"Trainable Parameters : {trainable_params:,}")
    print(f"Non-Trainable Params : {total_params - trainable_params:,}")
    print("=" * 75 + "\n")

def plot_training_curves(history, run_name, artifacts_dir="artifacts"):
    """Plots and logs Loss and Accuracy curves highlighting the best epoch."""
    epochs = range(1, len(history.history['loss']) + 1)
    best_epoch = int(np.argmin(history.history['val_loss']) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss Curve
    axes[0].plot(epochs, history.history['loss'], label='Train Loss', marker='o')
    axes[0].plot(epochs, history.history['val_loss'], label='Val Loss', marker='s')
    axes[0].axvline(best_epoch, color='red', linestyle='--', label=f'Best Epoch ({best_epoch})')
    axes[0].set_title(f"Loss Curves: {run_name}")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy Curve
    axes[1].plot(epochs, history.history['accuracy'], label='Train Accuracy', marker='o')
    axes[1].plot(epochs, history.history['val_accuracy'], label='Val Accuracy', marker='s')
    axes[1].axvline(best_epoch, color='red', linestyle='--', label=f'Best Epoch ({best_epoch})')
    axes[1].set_title(f"Accuracy Curves: {run_name}")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    curve_path = os.path.join(artifacts_dir, f"{run_name}_training_curves.png")
    plt.savefig(curve_path, dpi=200)
    plt.close()
    mlflow.log_artifact(curve_path, artifact_path="training_plots")
    return best_epoch

def setup_remote_tracking():
    """Always configures MLflow remote tracking via .env credentials."""
    load_dotenv(override=True)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    username = os.getenv("MLFLOW_TRACKING_USERNAME")
    password = os.getenv("MLFLOW_TRACKING_PASSWORD")
    
    if not tracking_uri:
        raise ValueError("MLFLOW_TRACKING_URI not set in .env! Please check your .env file.")
    
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    os.environ["MLFLOW_TRACKING_USERNAME"] = username or ""
    os.environ["MLFLOW_TRACKING_PASSWORD"] = password or ""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"[OK] MLflow tracking connected to DagsHub: {tracking_uri}")

def run_baseline_cnn(train_ds, val_ds, test_ds, git_hash, dvc_hash, is_autolog=False):
    run_name = "baseline_cnn_autolog" if is_autolog else "baseline_cnn_manual"
    print(f"\n{'='*70}\n[1/2] RUNNING BASELINE CNN ({'AUTOLOG' if is_autolog else 'MANUAL'})\n{'='*70}")

    if is_autolog:
        mlflow.tensorflow.autolog(log_models=True)

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("git_commit_hash", git_hash)
        mlflow.set_tag("dvc_data_hash", dvc_hash)
        mlflow.set_tag("random_seed", str(RANDOM_SEED))
        mlflow.set_tag("model_type", "baseline_cnn")
        mlflow.set_tag("logging_mode", "autolog" if is_autolog else "manual")

        mlflow.log_param("architecture", "Custom_4Block_CNN")
        mlflow.log_param("epochs", EPOCHS_BASELINE)
        mlflow.log_param("learning_rate", LR_BASELINE)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("loss", "binary_crossentropy")

        # Log split CSVs as artifacts
        for split_file in ["train_split.csv", "val_split.csv", "test_split.csv"]:
            p = os.path.join("artifacts", split_file)
            if os.path.exists(p):
                mlflow.log_artifact(p, artifact_path="splits")

        model = build_baseline_cnn(input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3), learning_rate=LR_BASELINE)
        
        # Print summary and layer-by-layer deliverable table
        model.summary()
        print_layer_table(model)

        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6, verbose=1)
        ]
        if not is_autolog:
            callbacks.append(MLflowEpochCallback())

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS_BASELINE,
            callbacks=callbacks,
            verbose=1
        )

        best_epoch = plot_training_curves(history, run_name)
        mlflow.log_metric("best_epoch", best_epoch)

        eval_metrics = log_evaluation_plots_and_metrics(model, test_ds, run_name)
        print(f"[OK] Baseline Test Results -> Accuracy: {eval_metrics['accuracy']:.4f}, AUC: {eval_metrics['roc_auc']:.4f}")

        if not is_autolog:
            os.makedirs("models", exist_ok=True)
            model.save(f"models/{run_name}.keras")
            mlflow.tensorflow.log_model(model, artifact_path="model")

def print_and_save_trainable_layers(model, stage_name, artifacts_dir="artifacts"):
    """Lists, prints, and saves all trainable layers for a given training stage."""
    os.makedirs(artifacts_dir, exist_ok=True)
    records = []
    
    for layer in model.layers:
        # If layer is a nested Model (e.g. MobileNetV2 base_model), check sublayers
        if hasattr(layer, 'layers'):
            for sublayer in layer.layers:
                if sublayer.trainable and len(sublayer.weights) > 0:
                    records.append({
                        "Scope": layer.name,
                        "Layer Name": sublayer.name,
                        "Type": sublayer.__class__.__name__,
                        "Trainable Params": f"{sublayer.count_params():,}"
                    })
        else:
            if layer.trainable and len(layer.weights) > 0:
                records.append({
                    "Scope": "Head",
                    "Layer Name": layer.name,
                    "Type": layer.__class__.__name__,
                    "Trainable Params": f"{layer.count_params():,}"
                })

    df = pd.DataFrame(records)
    print("\n" + "=" * 75)
    print(f"TASK 2.3 DELIVERABLE: {stage_name.upper()} TRAINABLE LAYERS LIST")
    print("=" * 75)
    if not df.empty:
        print(df.to_markdown(index=False))
    else:
        print("No trainable layers detected.")
    print("=" * 75 + "\n")

    # Save to text artifact
    out_file = os.path.join(artifacts_dir, f"{stage_name}_trainable_layers.txt")
    with open(out_file, "w") as f:
        f.write(f"=== {stage_name.upper()} TRAINABLE LAYERS ===\n\n")
        f.write(df.to_string(index=False) if not df.empty else "None")
    mlflow.log_artifact(out_file, artifact_path="model_specifications")
    return df

def run_transfer_learning(train_ds, val_ds, test_ds, git_hash, dvc_hash, is_autolog=False):
    run_name = "mobilenetv2_transfer_autolog" if is_autolog else "mobilenetv2_transfer_manual"
    print(f"\n{'='*70}\n[2/2] RUNNING TRANSFER LEARNING (MOBILENETV2)\n{'='*70}")

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("git_commit_hash", git_hash)
        mlflow.set_tag("dvc_data_hash", dvc_hash)
        mlflow.set_tag("random_seed", str(RANDOM_SEED))
        mlflow.set_tag("model_type", "transfer_learning_mobilenetv2")
        mlflow.set_tag("logging_mode", "autolog" if is_autolog else "manual")

        mlflow.log_param("backbone", "MobileNetV2")
        mlflow.log_param("weights", "ImageNet")
        mlflow.log_param("stage1_epochs", EPOCHS_TRANSFER_STAGE1)
        mlflow.log_param("stage2_epochs", EPOCHS_TRANSFER_STAGE2)
        mlflow.log_param("stage1_lr", LR_TRANSFER_STAGE1)
        mlflow.log_param("stage2_lr", LR_TRANSFER_STAGE2)
        mlflow.log_param("unfreeze_layers", UNFREEZE_LAYERS)

        # -------------------------------------------------------------
        # Stage 1: Feature Extraction (Backbone Frozen)
        # -------------------------------------------------------------
        print("\n---> Stage 1: Feature Extraction (MobileNetV2 Backbone Frozen)...")
        model, base_model = build_transfer_mobilenet(
            input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3),
            learning_rate=LR_TRANSFER_STAGE1,
            trainable_layers=0
        )

        # Deliverable 1: List of trainable layers in Stage 1
        print_and_save_trainable_layers(model, stage_name="stage1_feature_extraction")

        callbacks_s1 = [
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1),
            MLflowEpochCallback()
        ]
        history1 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS_TRANSFER_STAGE1,
            callbacks=callbacks_s1,
            verbose=1
        )

        # Deliverable 2: Training curve for Stage 1
        plot_training_curves(history1, run_name=f"{run_name}_stage1")

        # -------------------------------------------------------------
        # Stage 2: Fine-Tuning (Unfreeze Top Layers)
        # -------------------------------------------------------------
        print(f"\n---> Stage 2: Fine-Tuning (Unfreezing top {UNFREEZE_LAYERS} layers with lr={LR_TRANSFER_STAGE2})...")
        base_model.trainable = True
        for layer in base_model.layers[:-UNFREEZE_LAYERS]:
            layer.trainable = False

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LR_TRANSFER_STAGE2),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        )

        # Deliverable 3: List of trainable layers in Stage 2
        print_and_save_trainable_layers(model, stage_name="stage2_finetuning")

        callbacks_s2 = [
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-7, verbose=1),
            MLflowEpochCallback()
        ]
        history2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS_TRANSFER_STAGE2,
            callbacks=callbacks_s2,
            verbose=1
        )

        # Deliverable 4: Training curve for Stage 2
        plot_training_curves(history2, run_name=f"{run_name}_stage2")

        # Final Evaluation on isolated test set
        eval_metrics = log_evaluation_plots_and_metrics(model, test_ds, run_name)
        print(f"[OK] Transfer Learning Final Test Results -> Accuracy: {eval_metrics['accuracy']:.4f}, AUC: {eval_metrics['roc_auc']:.4f}")

        # Save champion model weights locally
        os.makedirs("models", exist_ok=True)
        champion_path = "models/champion_model.keras"
        model.save(champion_path)
        print(f"[OK] Champion weights saved to '{champion_path}'")
        mlflow.tensorflow.log_model(model, artifact_path="model")

def main():
    parser = argparse.ArgumentParser(description="VisionGuard Training Pipeline")
    parser.add_argument(
        "--logging",
        choices=["manual", "autolog"],
        default="manual",
        help="Logging mode for Task 3.3/3.4 (default: manual)"
    )
    args = parser.parse_args()

    # 1. Connect directly to DagsHub remote MLflow using .env
    setup_remote_tracking()

    # 2. Load cached deterministic data splits
    train_ds, val_ds, test_ds, _, _, _ = get_datasets()
    git_hash = get_git_commit_hash()
    dvc_hash = get_dvc_data_hash()

    is_autolog = (args.logging == "autolog")

    # 3. Sequentially train Baseline CNN, then Transfer Learning model
    run_baseline_cnn(train_ds, val_ds, test_ds, git_hash, dvc_hash, is_autolog=is_autolog)
    run_transfer_learning(train_ds, val_ds, test_ds, git_hash, dvc_hash, is_autolog=is_autolog)

    print("\n" + "=" * 70)
    print(f"ALL TRAINING COMPLETED SUCCESSFULLY IN '{args.logging.upper()}' MODE!")
    print("=" * 70)

if __name__ == "__main__":
    main()
