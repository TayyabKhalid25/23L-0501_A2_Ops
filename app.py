"""
app.py

VisionGuard Flask Web Application (Tasks 4 & 5).
Provides:
  - In-memory inference serving the Champion MobileNetV2 model.
  - Strict input validation (type checking, 5MB limit, corruption detection, zero disk footprint).
  - SQLite database query trail logging every inference request.
  - Health-check endpoint (/health) for production readiness monitoring.
  - Environment-based configuration running with debug=False.
"""

import os
import sys
import base64
import mimetypes
from pathlib import Path

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from dotenv import load_dotenv

from src.inference import get_classifier
from src.database import init_db, log_prediction, get_prediction_history, check_db_health

# Load environment configuration
load_dotenv(override=True)

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "visionguard_ops_secret_key_23L-0501")

# Configure upload limits (Task 4.2 / 4.4: 5 MB limit from environment)
MAX_SIZE_MB = int(os.getenv("MAX_CONTENT_LENGTH_MB", "5"))
app.config["MAX_CONTENT_LENGTH"] = MAX_SIZE_MB * 1024 * 1024

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}

# Initialize database and warm up model singleton at boot
init_db()
classifier = get_classifier()


def validate_file(file_storage):
    """
    Validates uploaded file against extension, MIME type, and empty payload.
    Returns: (is_valid, error_code, error_message, sanitized_filename)
    """
    if not file_storage or file_storage.filename == "":
        return False, 400, "No file was selected for upload.", ""

    raw_name = file_storage.filename
    sanitized_name = secure_filename(raw_name)
    if not sanitized_name:
        sanitized_name = "uploaded_image"

    ext = Path(sanitized_name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return (
            False, 
            415, 
            f"Invalid file extension '{ext}'. Only JPEG, PNG, and WEBP formats are accepted.", 
            sanitized_name
        )

    # Validate MIME type header
    mimetype = file_storage.mimetype or mimetypes.guess_type(sanitized_name)[0]
    if mimetype not in ALLOWED_MIME_TYPES:
        return (
            False, 
            415, 
            f"Unsupported media MIME type '{mimetype}'. Only authentic image payloads are allowed.", 
            sanitized_name
        )

    return True, 200, "OK", sanitized_name


# ==========================================
# Routes
# ==========================================

@app.route("/", methods=["GET"])
def index():
    """Renders main image upload and verification page."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Main inference route.
    Validates payload, executes memory-only inference, persists audit record to DB.
    """
    if "image" not in request.files:
        flash("No image file found in the request.", "error")
        return redirect(url_for("index"))

    file_obj = request.files["image"]
    is_valid, err_code, err_msg, safe_filename = validate_file(file_obj)
    if not is_valid:
        return render_template("error.html", error_code=err_code, error_title="Invalid Image Upload", error_message=err_msg), err_code

    try:
        # Read stream into memory (Zero disk storage footprint)
        raw_bytes = file_obj.read()
        if len(raw_bytes) == 0:
            return render_template("error.html", error_code=400, error_title="Empty File", error_message="The uploaded file contains 0 bytes."), 400

        # Execute inference through warmed singleton
        result = classifier.predict(raw_bytes)

        # Log permanent audit trail into SQLite (Task 5)
        log_prediction(
            filename=safe_filename,
            predicted_label=result["predicted_label"],
            confidence=result["confidence"],
            decision_threshold=result["decision_threshold"],
            model_version=result["model_version"],
            latency_ms=result["latency_ms"]
        )

        # Base64 encode for in-memory preview without saving to disk
        mime_type = file_obj.mimetype or "image/jpeg"
        b64_img = base64.b64encode(raw_bytes).decode("utf-8")
        data_uri = f"data:{mime_type};base64,{b64_img}"

        return render_template(
            "result.html",
            filename=safe_filename,
            predicted_label=result["predicted_label"],
            confidence=result["confidence"],
            raw_probability=result["raw_probability"],
            decision_threshold=result["decision_threshold"],
            latency_ms=result["latency_ms"],
            model_version=result["model_version"],
            image_data_uri=data_uri
        )

    except ValueError as ve:
        # Triggered when byte stream is corrupted or not a valid decodable image
        return render_template(
            "error.html",
            error_code=400,
            error_title="Corrupted / Un-decodable Image",
            error_message=f"The uploaded file could not be decoded as a valid image: {str(ve)}"
        ), 400

    except Exception as e:
        return render_template(
            "error.html",
            error_code=500,
            error_title="Internal Processing Error",
            error_message=f"An unexpected error occurred during model inference: {str(e)}"
        ), 500


@app.route("/history", methods=["GET"])
def history():
    """Renders persistent database query history (Task 5)."""
    records = get_prediction_history(limit=50)
    return render_template("history.html", records=records)


@app.route("/health", methods=["GET"])
def health():
    """
    Production health-check endpoint (Task 4.4 / 6.3).
    Returns JSON status confirming model readiness and database connectivity.
    """
    db_ok, record_count = check_db_health()
    status_code = 200 if db_ok else 503

    payload = {
        "status": "healthy" if db_ok else "unhealthy",
        "service": "VisionGuard",
        "project": os.getenv("PROJECT_NAME", "VisionGuard_A2_Ops"),
        "model_loaded": classifier.model is not None,
        "model_version": classifier.model_version,
        "decision_threshold": classifier.threshold,
        "max_upload_size_mb": MAX_SIZE_MB,
        "database_connected": db_ok,
        "database_records_logged": record_count if db_ok else 0,
        "environment": os.getenv("FLASK_ENV", "production")
    }
    return jsonify(payload), status_code


# ==========================================
# Error Handlers (Task 4.2: Friendly Error Pages)
# ==========================================

@app.errorhandler(413)
@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    return render_template(
        "error.html",
        error_code=413,
        error_title="File Size Limit Exceeded",
        error_message=f"The uploaded file exceeds the strict {MAX_SIZE_MB} MB limit allowed by VisionGuard."
    ), 413


@app.errorhandler(404)
def handle_not_found(e):
    return render_template(
        "error.html",
        error_code=404,
        error_title="Page Not Found",
        error_message="The requested URL route does not exist on this server."
    ), 404


# ==========================================
# Main Server Entrypoint (Task 4.4)
# ==========================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    host = os.getenv("HOST", "0.0.0.0")
    
    # Task 4.4 explicit requirement: Run with debug=False in production
    print("=" * 75)
    print("STARTING VISIONGUARD FLASK PRODUCTION SERVER")
    print(f"  • Address: http://127.0.0.1:{port}")
    print(f"  • Debug Mode: OFF (Mandatory Production Safety)")
    print("=" * 75)
    
    app.run(host=host, port=port, debug=False)
