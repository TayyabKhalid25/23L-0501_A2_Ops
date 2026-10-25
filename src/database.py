"""
src/database.py

Task 5 & 6: Database Trail Management for VisionGuard.
Supports both SQLite (local development) and PostgreSQL (Docker Compose / Cloud), tracking:
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

DEFAULT_SQLITE_PATH = os.path.join("data", "visionguard.db")


def is_postgres():
    """Returns True if DATABASE_URL indicates PostgreSQL."""
    db_url = os.getenv("DATABASE_URL", "")
    return db_url.startswith("postgres://") or db_url.startswith("postgresql://")


def get_connection():
    """Returns a database connection based on environment configuration."""
    if is_postgres():
        import psycopg2
        import psycopg2.extras
        db_url = os.getenv("DATABASE_URL")
        conn = psycopg2.connect(db_url)
        return conn, "postgres"
    else:
        path = os.getenv("DATABASE_PATH", DEFAULT_SQLITE_PATH)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        return conn, "sqlite"


def init_db():
    """Initializes database and creates predictions table if not exists."""
    conn, db_type = get_connection()
    cursor = conn.cursor()
    
    if db_type == "postgres":
        import psycopg2
        try:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id SERIAL PRIMARY KEY,
                    timestamp VARCHAR(64) NOT NULL,
                    filename VARCHAR(255) NOT NULL,
                    predicted_label VARCHAR(32) NOT NULL,
                    confidence DOUBLE PRECISION NOT NULL,
                    decision_threshold DOUBLE PRECISION NOT NULL,
                    model_version VARCHAR(64) NOT NULL,
                    latency_ms DOUBLE PRECISION NOT NULL
                );
            """)
            conn.commit()
        except psycopg2.Error:
            conn.rollback()
    else:
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
    return "postgres" if db_type == "postgres" else os.getenv("DATABASE_PATH", DEFAULT_SQLITE_PATH)


def log_prediction(filename, predicted_label, confidence, decision_threshold, model_version, latency_ms):
    """Inserts a prediction audit record into the database."""
    conn, db_type = get_connection()
    cursor = conn.cursor()
    
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    placeholder = "%s" if db_type == "postgres" else "?"
    
    query = f"""
        INSERT INTO predictions (timestamp, filename, predicted_label, confidence, decision_threshold, model_version, latency_ms)
        VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})
    """
    params = (timestamp, filename, predicted_label, float(confidence), float(decision_threshold), model_version, float(latency_ms))
    
    cursor.execute(query, params)
    conn.commit()
    conn.close()


def get_prediction_history(limit=50):
    """Fetches recent prediction records in reverse chronological order."""
    conn, db_type = get_connection()
    cursor = conn.cursor()
    
    placeholder = "%s" if db_type == "postgres" else "?"
    query = f"""
        SELECT id, timestamp, filename, predicted_label, confidence, decision_threshold, model_version, latency_ms
        FROM predictions
        ORDER BY id DESC
        LIMIT {placeholder}
    """
    cursor.execute(query, (limit,))
    
    if db_type == "postgres":
        columns = [desc[0] for desc in cursor.description]
        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
    else:
        rows = [dict(row) for row in cursor.fetchall()]
        
    conn.close()
    return rows


def check_db_health():
    """Verifies database accessibility and returns record count and engine type."""
    try:
        conn, db_type = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM predictions")
        count = cursor.fetchone()[0]
        conn.close()
        return True, count, db_type
    except Exception as e:
        return False, str(e), "unknown"


if __name__ == "__main__":
    db_loc = init_db()
    print(f"[OK] Database initialized ({db_loc})")
    status, count, engine = check_db_health()
    print(f"[OK] Database health: status={status}, records={count}, engine={engine}")
