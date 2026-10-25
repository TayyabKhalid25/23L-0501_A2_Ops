import os
from pathlib import Path
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from PIL import Image

def generate_eda(data_dir="data/raw", artifacts_dir="artifacts"):
    os.makedirs(artifacts_dir, exist_ok=True)
    audit_file = Path(artifacts_dir) / "dataset_audit_full.csv"

    # If audit CSV exists, load it; otherwise notify user
    if not audit_file.exists():
        print("Please run scripts/audit_dataset.py first to generate the metadata!")
        return

    df = pd.read_csv(audit_file)
    print(f"Loaded audit records for {len(df)} images.")

    sns.set_theme(style="whitegrid")

    # 1. Random Sample Grid (4 AI vs 4 Real)
    classes = df['class'].unique()
    fig, axes = plt.subplots(len(classes), 4, figsize=(14, 4 * len(classes)))
    if len(classes) == 1:
        axes = [axes]

    for row_idx, cls_name in enumerate(classes):
        cls_df = df[df['class'] == cls_name]
        sample_paths = cls_df['filepath'].sample(n=min(4, len(cls_df)), random_state=42).tolist()
        for col_idx, img_path in enumerate(sample_paths):
            ax = axes[row_idx][col_idx] if len(classes) > 1 else axes[col_idx]
            try:
                img = Image.open(img_path)
                ax.imshow(img)
                ax.set_title(f"{cls_name} ({img.width}x{img.height})", fontsize=10)
                ax.axis('off')
            except Exception as e:
                ax.set_title(f"Error loading: {e}", fontsize=8)
                ax.axis('off')

    plt.suptitle("Sample Image Grid: AI vs. Real", fontsize=14, y=0.98)
    sample_grid_path = os.path.join(artifacts_dir, "eda_sample_grid.png")
    plt.tight_layout()
    plt.savefig(sample_grid_path, dpi=200)
    plt.close()
    print(f"✓ Saved sample image grid: {sample_grid_path}")

    # 2. Separate Histograms: Width, Height, and File Size per Class
    df['file_size_kb'] = df['file_size_bytes'] / 1024.0

    # 2a. Width Distribution
    plt.figure(figsize=(7, 5))
    sns.histplot(data=df, x='width', hue='class', kde=True, bins=20, palette='Set2')
    plt.title("Image Width Distribution by Class", fontsize=12)
    plt.xlabel("Width (pixels)", fontsize=10)
    plt.ylabel("Count", fontsize=10)
    plt.tight_layout()
    width_path = os.path.join(artifacts_dir, "eda_width_distribution.png")
    plt.savefig(width_path, dpi=200)
    plt.close()
    print(f"✓ Saved width histogram: {width_path}")

    # 2b. Height Distribution
    plt.figure(figsize=(7, 5))
    sns.histplot(data=df, x='height', hue='class', kde=True, bins=20, palette='Set2')
    plt.title("Image Height Distribution by Class", fontsize=12)
    plt.xlabel("Height (pixels)", fontsize=10)
    plt.ylabel("Count", fontsize=10)
    plt.tight_layout()
    height_path = os.path.join(artifacts_dir, "eda_height_distribution.png")
    plt.savefig(height_path, dpi=200)
    plt.close()
    print(f"✓ Saved height histogram: {height_path}")

    # 2c. File Size Distribution (KB)
    plt.figure(figsize=(7, 5))
    sns.histplot(data=df, x='file_size_kb', hue='class', kde=True, bins=20, palette='Set2')
    plt.title("File Size Distribution by Class", fontsize=12)
    plt.xlabel("File Size (KB)", fontsize=10)
    plt.ylabel("Count", fontsize=10)
    plt.tight_layout()
    filesize_path = os.path.join(artifacts_dir, "eda_filesize_distribution.png")
    plt.savefig(filesize_path, dpi=200)
    plt.close()
    print(f"✓ Saved file size histogram: {filesize_path}")

if __name__ == "__main__":
    generate_eda()
