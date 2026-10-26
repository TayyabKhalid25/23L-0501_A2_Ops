"""
app.py

VisionGuard Flask Web Application (Tasks 4, 5, 7 & 8).
Features:
  - When CLOUD_API_URL / CLOUD_HOST is configured:
      • Client Gateway Mode: Forwards inference to the Cloud AWS EC2 instance.
      • Parses the Cloud response and renders a distinctive, branded Client UI.
      • Fetches and displays the live Cloud Database Trail on /history.
      • Proxies /health from the Cloud server.
  - When running standalone:
      • In-memory inference serving the Champion MobileNetV2 model.
      • Strict input validation (5MB limit, corruption detection).
      • Local database persistence (SQLite/PostgreSQL).
"""

import os
import sys
import re
import time
import base64
import mimetypes
from pathlib import Path
from urllib.parse import urlparse

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from dotenv import load_dotenv
import requests

# Load environment configuration
load_dotenv(override=True)

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "visionguard_ops_secret_key_23L-0501")

# Configure upload limits (Task 4.2 / 4.4: 5 MB limit)
MAX_SIZE_MB = int(os.getenv("MAX_CONTENT_LENGTH_MB", "5"))
app.config["MAX_CONTENT_LENGTH"] = MAX_SIZE_MB * 1024 * 1024

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}

# ==============================================================================
# Resolve Cloud Host vs Standalone Local Mode
# ==============================================================================
raw_cloud_url = os.getenv("CLOUD_API_URL", "").strip()
if raw_cloud_url:
    parsed = urlparse(raw_cloud_url)
    CLOUD_HOST = f"{parsed.scheme}://{parsed.netloc}"
else:
    CLOUD_HOST = os.getenv("CLOUD_HOST", "").strip()

# Lazy-load local model only if running in standalone mode
classifier = None
if not CLOUD_HOST:
    from src.database import init_db, log_prediction, get_prediction_history, check_db_health
    from src.inference import get_classifier
    init_db()
    classifier = get_classifier()
else:
    print(f"\n[VisionGuard] Running in Cloud Client Gateway Mode (Task 8).")
    print(f"  • Cloud Server Target : {CLOUD_HOST}")
    print(f"  • Predict Endpoint    : {CLOUD_HOST}/predict")
    print(f"  • Database Trail      : {CLOUD_HOST}/history\n")


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

    mimetype = file_storage.mimetype or mimetypes.guess_type(sanitized_name)[0]
    if mimetype not in ALLOWED_MIME_TYPES:
        return (
            False, 
            415, 
            f"Unsupported media MIME type '{mimetype}'. Only authentic image payloads are allowed.", 
            sanitized_name
        )

    return True, 200, "OK", sanitized_name


def parse_cloud_history_html(html_text):
    """Parses audit records from Cloud server's HTML table into structured records."""
    trs = re.findall(r'<tr>(.*?)</tr>', html_text, re.DOTALL)
    records = []
    for tr in trs[1:]:
        tds = re.findall(r'<td.*?>(.*?)</td>', tr, re.DOTALL)
        clean_tds = [re.sub(r'<.*?>', '', td).strip() for td in tds]
        if len(clean_tds) >= 8:
            try:
                conf_val = float(clean_tds[4].replace("%", "")) / 100.0
            except Exception:
                conf_val = 0.5
            records.append({
                "id": clean_tds[0].replace("#", ""),
                "timestamp": clean_tds[1],
                "filename": clean_tds[2],
                "predicted_label": "AI-Generated" if "AI" in clean_tds[3] else "Real",
                "confidence": conf_val,
                "decision_threshold": float(clean_tds[5]) if clean_tds[5] else 0.279,
                "model_version": clean_tds[6],
                "latency_ms": float(re.sub(r"[^\d.]", "", clean_tds[7]) or 0)
            })
    return records


# ==============================================================================
# Routes
# ==============================================================================

@app.route("/", methods=["GET"])
def index():
    """Renders main image upload page with cloud gateway status."""
    return render_template("index.html", is_cloud=bool(CLOUD_HOST), cloud_host=CLOUD_HOST)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Main inference route.
    If CLOUD_HOST is configured:
      1. Forwards image bytes to Cloud EC2 API.
      2. Parses prediction results from Cloud response.
      3. Renders the local Client UI with distinct styling and cloud metadata.
    """
    if "image" not in request.files:
        flash("No image file found in the request.", "error")
        return redirect(url_for("index"))

    file_obj = request.files["image"]
    is_valid, err_code, err_msg, safe_filename = validate_file(file_obj)
    if not is_valid:
        return render_template("error.html", error_code=err_code, error_title="Invalid Image Upload", error_message=err_msg), err_code

    try:
        raw_bytes = file_obj.read()
        if len(raw_bytes) == 0:
            return render_template("error.html", error_code=400, error_title="Empty File", error_message="The uploaded file contains 0 bytes."), 400

        mime_type = file_obj.mimetype or "image/jpeg"
        b64_img = base64.b64encode(raw_bytes).decode("utf-8")
        data_uri = f"data:{mime_type};base64,{b64_img}"

        # -------------------------------------------------------------
        # Mode 1: Cloud Gateway Mode (Task 8: Proxy to AWS EC2 Cloud API)
        # -------------------------------------------------------------
        if CLOUD_HOST:
            cloud_predict_url = f"{CLOUD_HOST}/predict"
            start_api = time.perf_counter()
            
            cloud_resp = requests.post(
                cloud_predict_url,
                files={"image": (safe_filename, raw_bytes, mime_type)},
                timeout=45
            )
            roundtrip_ms = round((time.perf_counter() - start_api) * 1000, 2)

            if cloud_resp.status_code != 200:
                raise RuntimeError(f"Cloud API returned HTTP {cloud_resp.status_code}: {cloud_resp.text[:200]}")

            # Check if JSON or HTML response
            if cloud_resp.headers.get("Content-Type", "").startswith("application/json"):
                data = cloud_resp.json()
                pred_label = data.get("predicted_label", "Real")
                conf = float(data.get("confidence", 0.5))
                raw_p = float(data.get("raw_probability", 0.5))
                thresh = float(data.get("decision_threshold", 0.279))
                m_version = data.get("model_version", "v2-champion-mobilenetv2")
                lat = float(data.get("latency_ms", roundtrip_ms))
            else:
                # Parse structured fields from Cloud HTML response
                v_m = re.search(r'class="verdict-status">(.*?)</span>', cloud_resp.text)
                c_m = re.search(r'class="confidence-percentage">(.*?)%?</span>', cloud_resp.text)
                p_m = re.search(r'Raw P\(AI\) Output.*?<code>(.*?)</code>', cloud_resp.text, re.DOTALL)
                t_m = re.search(r'Decision Threshold.*?<code>(.*?)</code>', cloud_resp.text, re.DOTALL)
                l_m = re.search(r'Inference Latency.*?<span class="metric-val">(.*?) ms</span>', cloud_resp.text, re.DOTALL)

                pred_label = "AI-Generated" if (v_m and "AI-GENERATED" in v_m.group(1)) else "Real"
                conf_pct = float(c_m.group(1).replace("%", "")) if c_m else 50.0
                conf = conf_pct / 100.0 if conf_pct > 1.0 else conf_pct
                raw_p = float(p_m.group(1)) if p_m else 0.5
                thresh = float(t_m.group(1)) if t_m else 0.279
                lat = float(l_m.group(1)) if l_m else roundtrip_ms
                m_version = "v2-champion-mobilenetv2"

            return render_template(
                "result.html",
                filename=safe_filename,
                predicted_label=pred_label,
                confidence=conf,
                raw_probability=raw_p,
                decision_threshold=thresh,
                latency_ms=lat,
                roundtrip_ms=roundtrip_ms,
                model_version=m_version,
                image_data_uri=data_uri,
                is_cloud=True,
                cloud_host=CLOUD_HOST
            )

        # -------------------------------------------------------------
        # Mode 2: Standalone Local Mode
        # -------------------------------------------------------------
        from src.database import log_prediction

        result = classifier.predict(raw_bytes)

        log_prediction(
            filename=safe_filename,
            predicted_label=result["predicted_label"],
            confidence=result["confidence"],
            decision_threshold=result["decision_threshold"],
            model_version=result["model_version"],
            latency_ms=result["latency_ms"]
        )

        return render_template(
            "result.html",
            filename=safe_filename,
            predicted_label=result["predicted_label"],
            confidence=result["confidence"],
            raw_probability=result["raw_probability"],
            decision_threshold=result["decision_threshold"],
            latency_ms=result["latency_ms"],
            model_version=result["model_version"],
            image_data_uri=data_uri,
            is_cloud=False
        )

    except ValueError as ve:
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
            error_title="Inference Error",
            error_message=f"An error occurred during inference: {str(e)}"
        ), 500


@app.route("/history", methods=["GET"])
def history():
    """
    Audit Trail Route.
    Fetches live records directly from the Cloud database when CLOUD_HOST is configured.
    """
    if CLOUD_HOST:
        try:
            cloud_resp = requests.get(f"{CLOUD_HOST}/history", timeout=15)
            records = parse_cloud_history_html(cloud_resp.text)
            return render_template("history.html", records=records, is_cloud=True, cloud_host=CLOUD_HOST)
        except Exception as e:
            return render_template(
                "error.html",
                error_code=502,
                error_title="Cloud Database Unreachable",
                error_message=f"Could not fetch audit history from Cloud server at {CLOUD_HOST}: {str(e)}"
            ), 502
    else:
        from src.database import get_prediction_history
        records = get_prediction_history(limit=50)
        return render_template("history.html", records=records, is_cloud=False)


@app.route("/health", methods=["GET"])
def health():
    """
    Health check Route.
    Proxies live status from Cloud server when CLOUD_HOST is set.
    """
    if CLOUD_HOST:
        try:
            cloud_resp = requests.get(f"{CLOUD_HOST}/health", timeout=10)
            return (cloud_resp.content, cloud_resp.status_code, {"Content-Type": "application/json"})
        except Exception as e:
            return jsonify({"status": "unhealthy", "error": f"Failed to reach Cloud server at {CLOUD_HOST}: {str(e)}"}), 503
    else:
        from src.database import check_db_health
        db_ok, record_count, db_engine = check_db_health()
        status_code = 200 if db_ok else 503

        payload = {
            "status": "healthy" if db_ok else "unhealthy",
            "service": "VisionGuard",
            "project": os.getenv("PROJECT_NAME", "VisionGuard_A2_Ops"),
            "model_loaded": classifier.model is not None if classifier else False,
            "model_version": classifier.model_version if classifier else "none",
            "decision_threshold": classifier.threshold if classifier else 0.279,
            "max_upload_size_mb": MAX_SIZE_MB,
            "database_connected": db_ok,
            "database_engine": db_engine,
            "database_records_logged": record_count if db_ok else 0,
            "environment": os.getenv("FLASK_ENV", "production")
        }
        return jsonify(payload), status_code


# ==============================================================================
# Error Handlers
# ==============================================================================

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


# ==============================================================================
# Main Server Entrypoint
# ==============================================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    host = os.getenv("HOST", "0.0.0.0")
    
    print("=" * 75)
    print("STARTING VISIONGUARD FLASK SERVER")
    print(f"  • Address: http://127.0.0.1:{port}")
    if CLOUD_HOST:
        print(f"  • Mode: CLOUD GATEWAY CLIENT (Target: {CLOUD_HOST})")
    else:
        print(f"  • Mode: STANDALONE LOCAL")
    print(f"  • Debug Mode: OFF")
    print("=" * 75)
    
    app.run(host=host, port=port, debug=False)
