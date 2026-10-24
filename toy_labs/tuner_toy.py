import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def build_model(hp):
    """Builds a tunable sequential neural network for Fashion-MNIST."""
    model = keras.Sequential()
    model.add(layers.Input(shape=(28, 28)))
    model.add(layers.Flatten())

    # Tunable dense units: 32 to 128 with step of 32
    units = hp.Int('units', min_value=32, max_value=128, step=32)
    model.add(layers.Dense(units=units, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    # Tunable learning rate
    lr = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def main():
    print("=" * 65)
    print("Task 2.7 Toy Lab: Keras Tuner on Fashion-MNIST")
    print("=" * 65)

    # 1. Load Fashion-MNIST dataset
    (x_train, y_train), (x_val, y_val) = keras.datasets.fashion_mnist.load_data()
    # Normalize images
    x_train = x_train.astype("float32") / 255.0
    x_val = x_val.astype("float32") / 255.0

    # Use a subset for fast tuning
    x_train, y_train = x_train[:6000], y_train[:6000]
    x_val, y_val = x_val[:1500], y_val[:1500]

    # 2. Instantiate RandomSearch Tuner
    tuner = kt.RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=5,
        executions_per_trial=1,
        directory='toy_labs/tuner_results',
        project_name='fashion_mnist_tuning',
        overwrite=True
    )

    tuner.search_space_summary()
    print("\n---> Commencing hyperparameter search (5 trials)...")

    # 3. Execute Search
    tuner.search(
        x_train,
        y_train,
        epochs=4,
        validation_data=(x_val, y_val),
        batch_size=64,
        verbose=1
    )

    print("\n" + "=" * 65)
    print("Hyperparameter Search Results Summary")
    print("=" * 65)
    tuner.results_summary()

    # 4. Generate structured deliverable table of trials
    trials = tuner.oracle.get_best_trials(num_trials=5)
    records = []
    for rank, trial in enumerate(trials, 1):
        records.append({
            "Rank": rank,
            "Trial ID": trial.trial_id,
            "Units": trial.hyperparameters.get('units'),
            "Learning Rate": trial.hyperparameters.get('learning_rate'),
            "Best Val Accuracy": round(trial.score, 4) if trial.score is not None else "N/A"
        })

    df_results = pd.DataFrame(records)
    print("\n" + "=" * 65)
    print("Deliverable Table: Toy Tuner Summary")
    print("=" * 65)
    print(df_results.to_markdown(index=False))

    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("\nBest Hyperparameters Found:")
    print(f" - Dense Units : {best_hp.get('units')}")
    print(f" - Learning Rate: {best_hp.get('learning_rate')}")
    print("=" * 65)

if __name__ == "__main__":
    main()
