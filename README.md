# VisionGuard: AI vs Real Image Classification & MLOps Pipeline

[![VisionGuard CI](https://github.com/TayyabKhalid25/23L-0501_A2_Ops/actions/workflows/ci.yml/badge.svg)](https://github.com/TayyabKhalid25/23L-0501_A2_Ops/actions/workflows/ci.yml)
[![DagsHub Project](https://img.shields.io/badge/DagsHub-Project-blue)](https://dagshub.com/tayyabkhalid2004/23L-0501_A2_Ops)
[![Docker](https://img.shields.io/badge/Docker-Compose%20v2-2496ED?logo=docker&logoColor=white)](docker-compose.yml)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://python.org)

VisionGuard is an end-to-end production MLOps system that classifies uploaded images as either **AI-Generated** or **Authentic (Real)**. The system pairs fine-tuned deep learning models with automated data version control (DVC), experiment tracking (MLflow), production serving (Flask + Gunicorn), an audit trail database (PostgreSQL / SQLite), containerization (Docker Compose), and continuous integration (GitHub Actions).

---

## System Architecture

```
                    +-----------------------------+
                    |        User Browser         |
                    +--------------+--------------+
                                   | HTTP (Port 5000)
                                   v
+-----------------------------------------------------------------+
| Docker Compose Isolated Network (visionguard_internal_network)   |
|                                                                 |
|  +---------------------------+   TCP    +--------------------+  |
|  |     visionguard-web       |--------->|   visionguard-db   |  |
|  | (Gunicorn + Flask WSGI)   |  (5432)  | (PostgreSQL 16)    |  |
|  | - Non-root user: appuser  |          | - Named volume     |  |
|  | - Healthcheck: /health    |          | - Port closed to   |  |
|  | - TensorFlow-CPU runtime  |          |   outside host     |  |
|  +-------------+-------------+          +---------+----------+  |
|                |                                  |             |
+----------------|----------------------------------|-------------+
                 v                                  v
      Mounted Model Artifact               Persistent Data Volume
 (models/champion_model.keras:ro)         (visionguard_postgres_data)
```

---

## Quickstart with Docker Compose

Ensure Docker and Docker Compose are installed and running.

```bash
# 1. Clone repository
git clone https://github.com/TayyabKhalid25/23L-0501_A2_Ops.git
cd 23L-0501_A2_Ops

# 2. Configure environment (copy template)
cp .env.example .env

# 3. Launch isolated container stack
docker compose up --build -d

# 4. Check service health and logs
docker compose ps
docker compose logs -f web

# 5. Open in browser:
# Web App UI:       http://localhost:5000
# Health Status:    http://localhost:5000/health
# History Trail:    http://localhost:5000/history
```

To stop the services:
```bash
docker compose down
```

---

## Local Development (Without Docker)

```bash
# 1. Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1    # Windows
# source .venv/bin/activate     # Linux / macOS

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run test suite
pytest tests/test_app.py -v

# 4. Start local development server
python app.py
```

---

## Secrets & Workflow Security (Task 6.6)

### Where Secrets Are Stored
- **Production / CI Secrets:** Stored securely within GitHub's encrypted secret store under **Repository Settings -> Secrets and variables -> Actions**.
- **Local Secrets:** Kept strictly inside an untracked `.env` file that is excluded from version control via `.gitignore` and `.dockerignore`.

### Why Secrets Must NEVER Appear in Workflow Files
1. **Repository Visibility & Fork Vulnerabilities:** Committing API keys, database credentials, or tokens directly into `.github/workflows/ci.yml` exposes them in plaintext to anyone with repository access or through pull request forks.
2. **Immutable Git History:** Once committed into Git, tokens remain in the commit history and reflog even if deleted in a later commit, requiring manual repository history rewriting or token revocation.
3. **Auditability & Rotation:** Utilizing GitHub Secrets allows centralized key rotation without making modifying code or triggering invalid git diffs.
