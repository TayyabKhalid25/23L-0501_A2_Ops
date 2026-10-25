import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

RANDOM_SEED = 42
TARGET_SIZE = (224, 224)
BATCH_SIZE = 32

def create_deterministic_splits(audit_csv="artifacts/dataset_audit_full.csv", artifacts_dir="artifacts"):
    """
    Creates deterministic 70/15/15 stratified train/val/test splits
    and saves split CSVs to artifacts directory.
    """
    os.makedirs(artifacts_dir, exist_ok=True)
    df = pd.read_csv(audit_csv)
    
    # Filter out corrupted images if any and ensure valid label mapping
    df = df[df['class'].isin(['AI', 'Real'])].copy()
    
    # Encode label: AI -> 1, Real -> 0
    df['numeric_label'] = df['class'].map({'Real': 0, 'AI': 1})
    
    # Stratified split: 70% train, 30% temp
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=df['numeric_label']
    )
    
    # Split temp into 15% validation and 15% test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=temp_df['numeric_label']
    )
    
    train_path = os.path.join(artifacts_dir, "train_split.csv")
    val_path = os.path.join(artifacts_dir, "val_split.csv")
    test_path = os.path.join(artifacts_dir, "test_split.csv")
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Created Deterministic Splits (Seed: {RANDOM_SEED}):")
    print(f"  - Train Set : {len(train_df)} images ({train_df['class'].value_counts().to_dict()})")
    print(f"  - Val Set   : {len(val_df)} images ({val_df['class'].value_counts().to_dict()})")
    print(f"  - Test Set  : {len(test_df)} images ({test_df['class'].value_counts().to_dict()})")
    return train_df, val_df, test_df

def decode_and_resize(filepath, label, target_size=TARGET_SIZE):
    """Reads image from disk, decodes to 3-channel RGB, and resizes."""
    img_bytes = tf.io.read_file(filepath)
    # expand_animations=False ensures single frame from GIF/WebP if any
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    # Ensure explicit static shape for Keras layers
    img = tf.image.resize(img, target_size, method='bilinear')
    img = tf.ensure_shape(img, [target_size[0], target_size[1], 3])
    return img, tf.cast(label, tf.float32)

def build_augmentation_layer():
    """Data augmentation sequential block applied ONLY to training data."""
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ], name="data_augmentation")

def build_dataset(df, is_training=False, batch_size=BATCH_SIZE, target_size=TARGET_SIZE):
    """Constructs an optimized tf.data.Dataset pipeline."""
    filepaths = df['filepath'].values
    labels = df['numeric_label'].values

    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    
    if is_training:
        ds = ds.shuffle(buffer_size=len(df), seed=RANDOM_SEED, reshuffle_each_iteration=True)
    
    # Map decoding and resizing with parallel calls
    ds = ds.map(
        lambda x, y: decode_and_resize(x, y, target_size),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    ds = ds.batch(batch_size)
    
    if is_training:
        aug_layer = build_augmentation_layer()
        ds = ds.map(lambda x, y: (aug_layer(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    
    # Prefetch in memory for optimal GPU/CPU overlap
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

def get_datasets(batch_size=BATCH_SIZE, target_size=TARGET_SIZE, artifacts_dir="artifacts"):
    train_path = os.path.join(artifacts_dir, "train_split.csv")
    val_path = os.path.join(artifacts_dir, "val_split.csv")
    test_path = os.path.join(artifacts_dir, "test_split.csv")

    if not (os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path)):
        train_df, val_df, test_df = create_deterministic_splits(artifacts_dir=artifacts_dir)
    else:
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)

    train_ds = build_dataset(train_df, is_training=True, batch_size=batch_size, target_size=target_size)
    val_ds = build_dataset(val_df, is_training=False, batch_size=batch_size, target_size=target_size)
    test_ds = build_dataset(test_df, is_training=False, batch_size=batch_size, target_size=target_size)

    return train_ds, val_ds, test_ds, train_df, val_df, test_df

def plot_augmented_batch(train_ds, save_path="artifacts/augmented_batch_sample.png"):
    """Fetches one batch of augmented images and saves a 3x3 grid."""
    for images, labels in train_ds.take(1):
        plt.figure(figsize=(10, 10))
        for i in range(min(9, len(images))):
            ax = plt.subplot(3, 3, i + 1)
            # Clip pixel values between 0 and 255 for display
            img_np = np.clip(images[i].numpy(), 0, 255).astype("uint8")
            plt.imshow(img_np)
            label_name = "AI" if int(labels[i].numpy()) == 1 else "Real"
            plt.title(f"Label: {label_name}", fontsize=11)
            plt.axis("off")
        plt.suptitle("Task 2.1: Augmented Training Batch Sample (224x224)", fontsize=13)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"[OK] Saved augmented batch figure to '{save_path}'")
        break

if __name__ == "__main__":
    train_ds, val_ds, test_ds, tr_df, v_df, te_df = get_datasets()
    plot_augmented_batch(train_ds)
