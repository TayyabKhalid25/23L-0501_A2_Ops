import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import mlflow
import mlflow.tensorflow
from dotenv import load_dotenv

# Silence redundant TF logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

class MLflowMetricLogger(tf.keras.callbacks.Callback):
    """Logs training and validation metrics to MLflow at the end of each epoch."""
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for metric_name, metric_val in logs.items():
            mlflow.log_metric(metric_name, float(metric_val), step=epoch + 1)

def build_model():
    model = models.Sequential([
        layers.Input(shape=(28, 28)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

def train_and_log(lr: float, batch_size: int, epochs: int, run_name: str, tracking_mode: str):
    print(f"\n---> Starting Run: {run_name} (lr={lr}, mode={tracking_mode})")
    
    # 1. Load MNIST Toy Dataset
    (x_train, y_train), (x_val, y_val) = tf.keras.datasets.mnist.load_data()
    # Normalize pixel values
    x_train = x_train.astype("float32") / 255.0
    x_val = x_val.astype("float32") / 255.0
    
    # Use smaller subset for lightning-fast toy execution
    train_subset_size = 5000
    val_subset_size = 1000
    x_train, y_train = x_train[:train_subset_size], y_train[:train_subset_size]
    x_val, y_val = x_val[:val_subset_size], y_val[:val_subset_size]

    with mlflow.start_run(run_name=run_name):
        # 2. Log Hyperparameters
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("optimizer", "adam")
        mlflow.log_param("architecture", "2-layer-dense-128")
        mlflow.set_tag("environment", tracking_mode)
        mlflow.set_tag("task", "toy_lab_3.1")

        # 3. Build & Compile Model
        model = build_model()
        model.compile(
            optimizer=optimizers.Adam(learning_rate=lr),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # 4. Train with custom callback logging metrics per epoch
        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[MLflowMetricLogger()],
            verbose=1
        )

        # 5. Plot and Log Loss Curves Artifact
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, epochs + 1), history.history['loss'], label='Train Loss', marker='o')
        plt.plot(range(1, epochs + 1), history.history['val_loss'], label='Val Loss', marker='s')
        plt.title(f"Loss Curve - {run_name} (lr={lr})")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        
        plot_path = f"{run_name}_loss_curve.png"
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        
        mlflow.log_artifact(plot_path, artifact_path="figures")
        if os.path.exists(plot_path):
            os.remove(plot_path)

        # 6. Log the Model
        mlflow.tensorflow.log_model(model, artifact_path="model")
        print(f"✓ Run {run_name} completed and logged successfully!")

def main():
    parser = argparse.ArgumentParser(description="MLflow Toy Experiment")
    parser.add_argument("--remote", action="store_true", help="Log to remote DagsHub server instead of local mlruns")
    args = parser.parse_args()

    if args.remote:
        load_dotenv(override=True)
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        username = os.getenv("MLFLOW_TRACKING_USERNAME")
        password = os.getenv("MLFLOW_TRACKING_PASSWORD")
        if not tracking_uri:
            raise ValueError("MLFLOW_TRACKING_URI is not set in .env!")
        
        # Configure DagsHub remote auth
        os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
        os.environ["MLFLOW_TRACKING_USERNAME"] = username or ""
        os.environ["MLFLOW_TRACKING_PASSWORD"] = password or ""
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("Toy-Experiment-DagsHub")
        print(f"Tracking configured for DagsHub Remote: {tracking_uri}")
        mode = "dagshub_remote"
        
        # Run one toy run for DagsHub (Step 2.3)
        train_and_log(lr=0.01, batch_size=64, epochs=5, run_name="toy_remote_dagshub_lr_0.01", tracking_mode=mode)
    else:
        # Local mlruns
        local_dir = os.path.abspath("mlruns")
        mlflow.set_tracking_uri(f"file:///{local_dir}".replace("\\", "/"))
        mlflow.set_experiment("Toy-Experiment-Local")
        print(f"Tracking configured for Local MLflow: {local_dir}")
        mode = "local"

        # Run 1: lr=0.01
        train_and_log(lr=0.01, batch_size=64, epochs=5, run_name="toy_run_lr_0.01", tracking_mode=mode)
        # Run 2: lr=0.001
        train_and_log(lr=0.001, batch_size=64, epochs=5, run_name="toy_run_lr_0.001", tracking_mode=mode)

if __name__ == "__main__":
    main()
