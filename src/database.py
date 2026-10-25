"""
src/database.py

Task 5: Database Trail Management for VisionGuard.
Persists query logs with SQLite, tracking:
  - Timestamp (UTC)
  - Filename (Sanitized)
  - Predicted Label
  - Confidence Score
  - Decision Threshold
  - Model Version
  - Inference Latency (ms)
"""

import os
import sqlite3
from datetime import datetime

DEFAULT_DB_PATH = os.path.join("data", "visionguard.db")


def get_db_path():
    """Retrieves DB path from environment variable or default."""
    return os.getenv("DATABASE_PATH", DEFAULT_DB_PATH)


def init_db(db_path=None):
    """Initializes SQLite database and creates predictions table if not exists."""
    path = db_path or get_db_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            filename TEXT NOT NULL,
            predicted_label TEXT NOT NULL,
            confidence REAL NOT NULL,
            decision_threshold REAL NOT NULL,
            model_version TEXT NOT NULL,
            latency_ms REAL NOT NULL
        )
    """)
    conn.commit()
    conn.close()
    return path


def log_prediction(filename, predicted_label, confidence, decision_threshold, model_version, latency_ms, db_path=None):
    """Inserts a prediction audit record into the database."""
    path = db_path or get_db_path()
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    cursor.execute("""
        INSERT INTO predictions (timestamp, filename, predicted_label, confidence, decision_threshold, model_version, latency_ms)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (timestamp, filename, predicted_label, float(confidence), float(decision_threshold), model_version, float(latency_ms)))
    
    conn.commit()
    conn.close()


def get_prediction_history(limit=50, db_path=None):
    """Fetches recent prediction records in reverse chronological order."""
    path = db_path or get_db_path()
    if not os.path.exists(path):
        init_db(path)
        
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, timestamp, filename, predicted_label, confidence, decision_threshold, model_version, latency_ms
        FROM predictions
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def check_db_health(db_path=None):
    """Verifies SQLite database accessibility and returns record count."""
    path = db_path or get_db_path()
    try:
        if not os.path.exists(path):
            init_db(path)
        conn = sqlite3.connect(path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM predictions")
        count = cursor.fetchone()[0]
        conn.close()
        return True, count
    except Exception as e:
        return False, str(e)


if __name__ == "__main__":
    db = init_db()
    print(f"[OK] Database initialized at '{db}'")
    status, count = check_db_health()
    print(f"[OK] Database healthy. Current records: {count}")
