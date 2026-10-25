"""
src/run_experiment_variations.py

Task 3.5: Structured Experiment Comparison across Both Model Families (6 Distinct Configurations).
Since manual and autolog runs share identical hyperparameters, we run 2 additional
meaningful single-variable changes for EACH architecture so that we have:
  • 3 Distinct Custom CNN Configurations:
      1. baseline_cnn_manual          (Already logged: lr=1e-3, augmentation=True)
      2. baseline_cnn_low_lr          (Change: Lower learning rate lr=1e-4, augmentation=True)
      3. baseline_cnn_no_augmentation (Change: No data augmentation, lr=1e-3)
  • 3 Distinct MobileNetV2 Configurations:
      4. mobilenetv2_transfer_manual  (Already logged: unfrozen_layers=20, lr=1e-4 -> 1e-5)
      5. mobilenetv2_fully_frozen     (Change: Frozen backbone, unfrozen_layers=0, lr=1e-4)
      6. mobilenetv2_high_lr          (Change: Aggressive learning rate lr=1e-3, unfrozen_layers=20)

Every variation logs hyperparameters, per-epoch metrics, test metrics, plots, and the
trained Keras model to DagsHub MLflow so any run can be selected for the Model Registry.
"""

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import tensorflow as tf
from tensorflow import keras
import mlflow
import mlflow.tensorflow

from src.data_pipeline import (
    get_datasets, decode_and_resize, RANDOM_SEED, TARGET_SIZE, BATCH_SIZE
)
from src.models import build_baseline_cnn, build_transfer_mobilenet
from src.utils import get_git_commit_hash, get_dvc_data_hash, log_evaluation_plots_and_metrics
from src.train import (
    setup_remote_tracking, MLflowEpochCallback, plot_training_curves,
    EPOCHS_BASELINE, EXPERIMENT_NAME
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"


def build_non_augmented_train_ds(train_df, batch_size=BATCH_SIZE, target_size=TARGET_SIZE):
    """Builds a training dataset with shuffling but WITHOUT random augmentation."""
    filepaths = train_df["filepath"].values
    labels = train_df["numeric_label"].values

    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    ds = ds.shuffle(buffer_size=len(train_df), seed=RANDOM_SEED, reshuffle_each_iteration=True)
    ds = ds.map(
        lambda x, y: decode_and_resize(x, y, target_size),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def run_cnn_variation(train_ds, val_ds, test_ds, run_name, lr, augmentation, epochs=EPOCHS_BASELINE, note=""):
    print(f"\n{'='*75}\n[CUSTOM CNN VARIATION] {run_name}\n{note}\n{'='*75}")
    git_hash = get_git_commit_hash()
    dvc_hash = get_dvc_data_hash()

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("git_commit_hash", git_hash)
        mlflow.set_tag("dvc_data_hash", dvc_hash)
        mlflow.set_tag("random_seed", str(RANDOM_SEED))
        mlflow.set_tag("model_type", "baseline_cnn")
        mlflow.set_tag("logging_mode", "manual")
        mlflow.set_tag("variation_note", note)

        mlflow.log_param("architecture", "Custom_4Block_CNN")
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("augmentation", augmentation)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("loss", "binary_crossentropy")

        model = build_baseline_cnn(
            input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3),
            learning_rate=lr
        )

        callbacks = [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-6, verbose=1),
            MLflowEpochCallback()
        ]

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )

        best_epoch = plot_training_curves(history, run_name)
        mlflow.log_metric("best_epoch", best_epoch)

        eval_metrics = log_evaluation_plots_and_metrics(model, test_ds, run_name)
        f1_val = eval_metrics.get('f1_score', eval_metrics.get('f1', 0.0))
        print(
            f"[OK] {run_name} -> Test Acc: {eval_metrics['accuracy']:.4f}, "
            f"F1: {f1_val:.4f}, AUC: {eval_metrics['roc_auc']:.4f}"
        )

        os.makedirs("models", exist_ok=True)
        model.save(f"models/{run_name}.keras")
        mlflow.tensorflow.log_model(model, artifact_path="model")


def run_mobilenet_variation(train_ds, val_ds, test_ds, run_name, lr, trainable_layers, epochs=6, note=""):
    print(f"\n{'='*75}\n[MOBILENETV2 VARIATION] {run_name}\n{note}\n{'='*75}")
    git_hash = get_git_commit_hash()
    dvc_hash = get_dvc_data_hash()

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("git_commit_hash", git_hash)
        mlflow.set_tag("dvc_data_hash", dvc_hash)
        mlflow.set_tag("random_seed", str(RANDOM_SEED))
        mlflow.set_tag("model_type", "transfer_learning_mobilenetv2")
        mlflow.set_tag("logging_mode", "manual")
        mlflow.set_tag("variation_note", note)

        mlflow.log_param("backbone", "MobileNetV2")
        mlflow.log_param("weights", "ImageNet")
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("unfreeze_layers", trainable_layers)
        mlflow.log_param("augmentation", True)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", BATCH_SIZE)

        model, _ = build_transfer_mobilenet(
            input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3),
            learning_rate=lr,
            trainable_layers=trainable_layers
        )

        callbacks = [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-6, verbose=1),
            MLflowEpochCallback()
        ]

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )

        best_epoch = plot_training_curves(history, run_name)
        mlflow.log_metric("best_epoch", best_epoch)

        eval_metrics = log_evaluation_plots_and_metrics(model, test_ds, run_name)
        f1_val = eval_metrics.get('f1_score', eval_metrics.get('f1', 0.0))
        print(
            f"[OK] {run_name} -> Test Acc: {eval_metrics['accuracy']:.4f}, "
            f"F1: {f1_val:.4f}, AUC: {eval_metrics['roc_auc']:.4f}"
        )

        os.makedirs("models", exist_ok=True)
        model.save(f"models/{run_name}.keras")
        mlflow.tensorflow.log_model(model, artifact_path="model")


def summarize_six_runs():
    """Fetches the 6 distinct manual runs from DagsHub MLflow and outputs a comparison table."""
    exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if not exp:
        return

    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    target_runs = [
        "baseline_cnn_manual",
        "baseline_cnn_low_lr",
        "baseline_cnn_no_augmentation",
        "mobilenetv2_transfer_manual",
        "mobilenetv2_fully_frozen",
        "mobilenetv2_high_lr"
    ]

    records = []
    for r_name in target_runs:
        subset = runs[runs["tags.mlflow.runName"] == r_name]
        if not subset.empty:
            row = subset.iloc[0]
            records.append({
                "Run Name": r_name,
                "Run ID": row["run_id"][:8],
                "Family": "Custom CNN" if "baseline" in r_name else "MobileNetV2",
                "Test Accuracy": round(float(row.get("metrics.test_accuracy", 0.0)), 4),
                "Test Precision": round(float(row.get("metrics.test_precision", 0.0)), 4),
                "Test Recall": round(float(row.get("metrics.test_recall", 0.0)), 4),
                "Test F1": round(float(row.get("metrics.test_f1", 0.0)), 4),
                "Test ROC-AUC": round(float(row.get("metrics.test_roc_auc", 0.0)), 4)
            })

    if records:
        df_comp = pd.DataFrame(records)
        os.makedirs("artifacts", exist_ok=True)
        out_csv = "artifacts/experiment_comparison_6runs.csv"
        df_comp.to_csv(out_csv, index=False)

        print("\n" + "=" * 95)
        print("TASK 3.5 DELIVERABLE: 6-RUN EXPERIMENT COMPARISON TABLE (3 CUSTOM CNN + 3 MOBILENETV2)")
        print("=" * 95)
        print(df_comp.to_markdown(index=False))
        print("=" * 95)
        print(f"[OK] Saved 6-run comparison table to '{out_csv}'")


def main():
    print("=" * 80)
    print("TASK 3.5: RUNNING 4 ADDITIONAL VARIATIONS (2 CUSTOM CNN + 2 MOBILENETV2)")
    print("=" * 80)

    setup_remote_tracking()
    train_ds_aug, val_ds, test_ds, train_df, _, _ = get_datasets()
    train_ds_no_aug = build_non_augmented_train_ds(train_df)

    # ---------------------------------------------------------
    # A. Custom CNN Variations (Adding 2 runs -> 3 total CNN runs)
    # ---------------------------------------------------------
    # Run 2 for Custom CNN: Lower Learning Rate (1e-4 vs 1e-3)
    run_cnn_variation(
        train_ds_aug, val_ds, test_ds,
        run_name="baseline_cnn_low_lr",
        lr=1e-4,
        augmentation=True,
        epochs=10,
        note="Meaningful Change: Lower learning rate (1e-4 vs baseline 1e-3)"
    )

    # Run 3 for Custom CNN: Disabled Data Augmentation (augmentation=False vs True)
    run_cnn_variation(
        train_ds_no_aug, val_ds, test_ds,
        run_name="baseline_cnn_no_augmentation",
        lr=1e-3,
        augmentation=False,
        epochs=10,
        note="Meaningful Change: Data augmentation disabled (False vs baseline True)"
    )

    # ---------------------------------------------------------
    # B. MobileNetV2 Variations (Adding 2 runs -> 3 total MobileNetV2 runs)
    # ---------------------------------------------------------
    # Run 2 for MobileNetV2: Fully Frozen Backbone (0 unfrozen layers vs 20)
    run_mobilenet_variation(
        train_ds_aug, val_ds, test_ds,
        run_name="mobilenetv2_fully_frozen",
        lr=1e-4,
        trainable_layers=0,
        epochs=6,
        note="Meaningful Change: Backbone completely frozen (0 unfrozen layers vs 20)"
    )

    # Run 3 for MobileNetV2: High Learning Rate (1e-3 vs 1e-4/1e-5)
    run_mobilenet_variation(
        train_ds_aug, val_ds, test_ds,
        run_name="mobilenetv2_high_lr",
        lr=1e-3,
        trainable_layers=20,
        epochs=6,
        note="Meaningful Change: Aggressive learning rate (1e-3 vs 1e-4/1e-5) with 20 unfrozen layers"
    )

    # Print comparison summary of all 6 runs
    summarize_six_runs()


if __name__ == "__main__":
    main()
