import os
import hashlib
from pathlib import Path
from PIL import Image
import pandas as pd

def compute_sha256(filepath):
    """Computes SHA-256 hash of a file's raw bytes."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def audit_dataset(data_dir="data/raw"):
    print(f"Scanning dataset in '{data_dir}'...")
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Error: Directory '{data_dir}' not found! Please place your images inside '{data_dir}'.")
        return

    records = []
    seen_hashes = {}
    corrupted_files = []
    duplicates = []

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

    for root, _, files in os.walk(data_path):
        for file in files:
            file_path = Path(root) / file
            if file_path.suffix.lower() not in valid_extensions:
                continue

            # Infer class label: check if 'ai' or 'real' in path components
            parts = [p.lower() for p in file_path.parts]
            if "ai" in parts:
                label = "AI"
            elif "real" in parts:
                label = "Real"
            else:
                label = file_path.parent.name

            file_size = os.path.getsize(file_path)

            try:
                with Image.open(file_path) as img:
                    img.verify()  # Verify image integrity
                
                # Reopen to read metadata after verify()
                with Image.open(file_path) as img:
                    width, height = img.size
                    mode = img.mode
                    img_format = img.format

                # Compute hash for duplicate detection
                file_hash = compute_sha256(file_path)
                if file_hash in seen_hashes:
                    duplicates.append((str(file_path), seen_hashes[file_hash]))
                    is_duplicate = True
                else:
                    seen_hashes[file_hash] = str(file_path)
                    is_duplicate = False

                records.append({
                    "filepath": str(file_path),
                    "class": label,
                    "file_size_bytes": file_size,
                    "width": width,
                    "height": height,
                    "color_mode": mode,
                    "format": img_format,
                    "is_duplicate": is_duplicate,
                    "sha256": file_hash
                })

            except Exception as e:
                corrupted_files.append((str(file_path), str(e)))

    if not records and not corrupted_files:
        print(f"No image files found in '{data_dir}'.")
        return

    df = pd.DataFrame(records)
    
    # Save full audit records for artifact logging
    os.makedirs("artifacts", exist_ok=True)
    df.to_csv("artifacts/dataset_audit_full.csv", index=False)
    print(f"\nAudit complete! Total images examined: {len(records) + len(corrupted_files)}")
    print(f"Saved complete audit trail to 'artifacts/dataset_audit_full.csv'")

    print("\n" + "=" * 65)
    print("TASK 1.2: DATASET AUDIT SUMMARY TABLE")
    print("=" * 65)
    print(f"Total Corrupted / Unreadable Files: {len(corrupted_files)}")
    print(f"Total Exact Byte-Level Duplicates : {len(duplicates)}")
    print("-" * 65)

    if not df.empty:
        summary_rows = []
        for cls_name, grp in df.groupby("class"):
            summary_rows.append({
                "Category / Class": cls_name,
                "Count": len(grp),
                "Width (Min / Med / Max)": f"{grp['width'].min()} / {int(grp['width'].median())} / {grp['width'].max()}",
                "Height (Min / Med / Max)": f"{grp['height'].min()} / {int(grp['height'].median())} / {grp['height'].max()}",
                "File Size KB (Min / Med / Max)": f"{round(grp['file_size_bytes'].min()/1024, 1)} / {round(grp['file_size_bytes'].median()/1024, 1)} / {round(grp['file_size_bytes'].max()/1024, 1)}",
                "Formats": ", ".join(grp['format'].dropna().unique()),
                "Color Modes": ", ".join(grp['color_mode'].dropna().unique())
            })

        # Also add Overall (all images regardless of class) to be 100% comprehensive
        summary_rows.append({
            "Category / Class": "Overall (All)",
            "Count": len(df),
            "Width (Min / Med / Max)": f"{df['width'].min()} / {int(df['width'].median())} / {df['width'].max()}",
            "Height (Min / Med / Max)": f"{df['height'].min()} / {int(df['height'].median())} / {df['height'].max()}",
            "File Size KB (Min / Med / Max)": f"{round(df['file_size_bytes'].min()/1024, 1)} / {round(df['file_size_bytes'].median()/1024, 1)} / {round(df['file_size_bytes'].max()/1024, 1)}",
            "Formats": ", ".join(df['format'].dropna().unique()),
            "Color Modes": ", ".join(df['color_mode'].dropna().unique())
        })

        summary_df = pd.DataFrame(summary_rows)
        print(summary_df.to_markdown(index=False))
    print("=" * 65)

if __name__ == "__main__":
    audit_dataset()
