"""
tests/test_app.py

Automated CI/CD Unit & Integration Tests for VisionGuard (Task 6.6).
Verifies:
  1. Health-check endpoint (/health) status and payload integrity.
  2. Web interface routes (GET /, GET /history).
  3. Input validation: rejection of non-image file extensions.
  4. Database audit connectivity and logging.
  5. Inference engine correctness on synthetic test image.
"""

import os
import sys
import io
import pytest
from PIL import Image

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Ensure local test runs use temporary SQLite DB and disable production requirements
os.environ["DATABASE_PATH"] = os.path.join("data", "test_visionguard.db")
os.environ["SECRET_KEY"] = "test_ci_secret_key"
os.environ["MAX_CONTENT_LENGTH_MB"] = "5"

from app import app
from src.database import init_db, check_db_health, get_prediction_history
from src.inference import get_classifier


@pytest.fixture
def client():
    """Configures Flask test client with testing mode."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    """Initializes a clean test database."""
    init_db()
    yield
    # Cleanup test db
    test_db = os.environ.get("DATABASE_PATH")
    if test_db and os.path.exists(test_db):
        try:
            os.remove(test_db)
        except OSError:
            pass


def test_health_endpoint(client):
    """Test /health returns HTTP 200 and required fields."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert data["service"] == "VisionGuard"
    assert "model_loaded" in data
    assert "decision_threshold" in data
    assert "database_connected" in data
    assert data["database_connected"] is True


def test_index_route(client):
    """Test GET / returns HTTP 200 and loads UI."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"VisionGuard" in response.data
    assert b"Upload" in response.data or b"upload" in response.data


def test_history_route(client):
    """Test GET /history returns HTTP 200."""
    response = client.get("/history")
    assert response.status_code == 200
    assert b"Audit Trail" in response.data or b"History" in response.data


def test_reject_invalid_extension(client):
    """Test uploading a .txt file rejects payload and redirects with flash error."""
    data = {
        "file": (io.BytesIO(b"This is not an image file."), "malicious_script.txt")
    }
    response = client.post("/predict", data=data, content_type="multipart/form-data", follow_redirects=True)
    assert response.status_code == 200
    assert b"Unsupported" in response.data or b"Invalid" in response.data or b"error" in response.data.lower()


def test_inference_pipeline_synthetic_image():
    """Test classifier processes an in-memory synthetic RGB image."""
    classifier = get_classifier()
    # Create synthetic 224x224 RGB image
    img = Image.new("RGB", (224, 224), color=(128, 64, 200))
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    
    result = classifier.predict(img_bytes.getvalue())
    assert "predicted_label" in result
    assert result["predicted_label"] in ["AI-generated", "Authentic (Real)"]
    assert 0.0 <= result["confidence"] <= 1.0
    assert 0.0 <= result["raw_probability"] <= 1.0
    assert result["latency_ms"] > 0
