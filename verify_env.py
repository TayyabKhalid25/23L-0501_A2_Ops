import sys
import platform
import importlib.metadata

print("=" * 60)
print("VisionGuard Environment & Library Verification")
print("=" * 60)
print(f"OS Platform        : {platform.system()} {platform.release()} ({platform.architecture()[0]})")
print(f"Python Version     : {sys.version.split()[0]}")
print(f"Python Executable  : {sys.executable}")
print("-" * 60)

packages = [
    ("tensorflow", "TensorFlow"),
    ("keras-tuner", "Keras Tuner"),
    ("mlflow", "MLflow"),
    ("dagshub", "DagsHub"),
    ("dvc", "DVC"),
    ("scikit-learn", "Scikit-Learn"),
    ("pandas", "Pandas"),
    ("numpy", "NumPy"),
    ("pillow", "Pillow"),
    ("matplotlib", "Matplotlib"),
    ("seaborn", "Seaborn"),
    ("flask", "Flask"),
    ("requests", "Requests"),
    ("python-dotenv", "Python-Dotenv"),
]

for pkg_name, display_name in packages:
    try:
        ver = importlib.metadata.version(pkg_name)
        print(f"{display_name:<20}: {ver}")
    except importlib.metadata.PackageNotFoundError:
        print(f"{display_name:<20}: NOT INSTALLED")

print("-" * 60)
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Available GPUs     : {[gpu.name for gpu in gpus]}")
    else:
        print("Available GPUs     : None detected (Running in CPU mode)")
except Exception as e:
    print(f"GPU Detection Error: {e}")
print("=" * 60)
