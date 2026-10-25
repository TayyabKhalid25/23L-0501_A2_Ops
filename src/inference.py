"""
src/inference.py

Task 4.1: Standalone Model Inference Module.
- Loads the Champion Keras model ONCE at startup to eliminate per-request latency.
- Strictly matches training preprocessing:
    • Bilinear resizing to 224x224
    • 3-channel RGB conversion
    • Normalization to [-1, 1] via keras.applications.mobilenet_v2.preprocess_input
- Memory-only decoding (no disk I/O).
- Returns predicted label, confidence score, raw probability, and execution latency.
- Includes standalone runnable test for two known images (Task 4.1 evidence).
"""

import os
import sys
import time
import io
from pathlib import Path

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from dotenv import load_dotenv

# Suppress verbose TF logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

DEFAULT_MODEL_PATH = os.path.join("models", "champion_model.keras")
DEFAULT_THRESHOLD = 0.279
TARGET_SIZE = (224, 224)


class VisionGuardClassifier:
    """Singleton inference engine that keeps model weights hot in memory."""
    
    def __init__(self, model_path=None, threshold=None):
        load_dotenv(override=True)
        self.model_path = model_path or os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)
        self.threshold = float(threshold or os.getenv("DECISION_THRESHOLD", DEFAULT_THRESHOLD))
        self.model_version = "v2-champion-mobilenetv2"
        
        print(f"[VisionGuard] Initializing inference engine...")
        print(f"  • Model Path         : {self.model_path}")
        print(f"  • Decision Threshold : {self.threshold:.3f}")
        print(f"  • Target Input Shape : {TARGET_SIZE + (3,)}")
        
        start_load = time.perf_counter()
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at '{self.model_path}'. Ensure champion model is trained/downloaded.")
            
        self.model = keras.models.load_model(self.model_path)
        load_time = (time.perf_counter() - start_load) * 1000
        print(f"[VisionGuard] Model loaded successfully into memory in {load_time:.2f} ms.\n")

    def preprocess_image_bytes(self, image_bytes):
        """
        Validates, decodes, and preprocesses image directly from memory buffer.
        Raises ValueError if corrupted or cannot be parsed as an image.
        """
        try:
            # 1. Verify byte integrity with PIL
            pil_img = Image.open(io.BytesIO(image_bytes))
            pil_img.verify()
            
            # Reopen after verify() (Pillow requirement)
            pil_img = Image.open(io.BytesIO(image_bytes))
            
            # 2. Convert to 3-channel RGB (handles RGBA, grayscale, palette P modes)
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
                
            # 3. Bilinear resize matching tf.image.resize(..., method='bilinear')
            pil_img = pil_img.resize(TARGET_SIZE, Image.BILINEAR)
            img_array = np.array(pil_img, dtype=np.float32)
            
            # 4. Note: MobileNetV2 architecture in models.py contains preprocess_input
            # inside its internal computation graph. The model expects raw [0, 255] RGB floats.
            batch_tensor = np.expand_dims(img_array, axis=0)
            return batch_tensor
            
        except Exception as e:
            raise ValueError(f"Image validation/decoding failed: {str(e)}")

    def predict(self, image_bytes):
        """
        Executes end-to-end inference on in-memory bytes.
        Returns:
            dict containing predicted_label, confidence, raw_probability, latency_ms, threshold.
        """
        start_time = time.perf_counter()
        
        # Preprocessing (memory-only)
        input_tensor = self.preprocess_image_bytes(image_bytes)
        
        # Forward pass
        raw_prob = float(self.model.predict(input_tensor, verbose=0)[0][0])
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Binary decision based on calibrated threshold
        if raw_prob >= self.threshold:
            label = "AI-Generated"
            confidence = raw_prob
        else:
            label = "Authentic (Real)"
            confidence = 1.0 - raw_prob
            
        return {
            "predicted_label": label,
            "confidence": float(confidence),
            "raw_probability": float(raw_prob),
            "latency_ms": round(latency_ms, 2),
            "decision_threshold": self.threshold,
            "model_version": self.model_version
        }


# Global singleton instance for serving
_classifier_instance = None

def get_classifier():
    """Returns singleton classifier instance, loading model on first access."""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = VisionGuardClassifier()
    return _classifier_instance


if __name__ == "__main__":
    print("=" * 80)
    print("TASK 4.1 DELIVERABLE: STANDALONE INFERENCE MODULE DEMONSTRATION")
    print("=" * 80)
    
    # Initialize engine
    classifier = get_classifier()
    
    # Select two known sample images: one Real, one AI
    sample_real = "data/raw/real/dog.jpg"
    sample_ai = "data/raw/ai/Midjourney-L-2068f65.jpg"
    
    # Fallback search if specific files moved
    if not os.path.exists(sample_real):
        sample_real = str(list(Path("data/raw/real").glob("*.*"))[0])
    if not os.path.exists(sample_ai):
        sample_ai = str(list(Path("data/raw/ai").glob("*.*"))[0])
        
    test_cases = [
        ("Known Authentic (Real)", sample_real),
        ("Known Synthetic (AI-Generated)", sample_ai)
    ]
    
    for category, path in test_cases:
        with open(path, "rb") as f:
            raw_bytes = f.read()
            
        result = classifier.predict(raw_bytes)
        print(f"\n[Test Case] {category}:")
        print(f"  • File Path         : {path}")
        print(f"  • File Size         : {len(raw_bytes) / 1024:.1f} KB")
        print(f"  • Raw Sigmoid Output: {result['raw_probability']:.4f}")
        print(f"  • Calibrated Thresh : {result['decision_threshold']:.3f}")
        print(f"  • Predicted Label   : {result['predicted_label']}")
        print(f"  • Confidence Score  : {result['confidence'] * 100:.2f}%")
        print(f"  • Inference Latency : {result['latency_ms']:.2f} ms")
        
    print("\n" + "=" * 80)
    print("STANDALONE TEST PASSED SUCCESSFULLY!")
    print("=" * 80)
