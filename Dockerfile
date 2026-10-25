# Multi-stage / optimized slim Python base for VisionGuard Web App (Task 6.2)
FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered standard I/O
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5000 \
    MODEL_PATH=/app/models/champion_model.keras \
    DECISION_THRESHOLD=0.279 \
    FLASK_ENV=production

# Install minimal OS dependencies (curl for container healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create dedicated non-root user and group (Task 6.2 Principle of Least Privilege)
RUN groupadd -g 1000 appgroup && \
    useradd -u 1000 -g appgroup -s /bin/bash -m appuser

# Set application working directory
WORKDIR /app

# Leverage Docker layer caching: install requirements before copying source code.
# Any edits to app.py, HTML templates, or CSS will not invalidate this expensive layer.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code and static assets
COPY src/ ./src/
COPY templates/ ./templates/
COPY static/ ./static/
COPY app.py .

# Copy model artifacts if present in build context
COPY models/ /app/models/

# Create runtime data directory and ensure ownership belongs to non-root appuser
RUN mkdir -p /app/data && \
    chown -R appuser:appgroup /app

# Switch to non-root user (Task 6.2)
USER appuser

# Expose web service port
EXPOSE 5000

# Docker Healthcheck targeting production /health endpoint (Task 6.2)
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Production WSGI server using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "app:app"]
