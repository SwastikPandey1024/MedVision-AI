# ========================================================
# MedVision-AI Production Docker Container
# Multi-purpose: REST API (FastAPI) & Interactive UI (Streamlit)
# ========================================================

FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TF_ENABLE_ONEDNN_OPTS=0 \
    PORT=8000 \
    APP_HOME=/app

WORKDIR ${APP_HOME}

# Install minimal OS dependencies for OpenCV, DICOM parsing, and healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root system user for secure container execution
RUN groupadd -g 10001 medvision && \
    useradd -u 10001 -g medvision -s /bin/bash -m appuser

# Install Python package dependencies
COPY requirements.txt requirements-dev.txt pyproject.toml ./
RUN pip install --no-cache-dir -U pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code, API, UI, and documentation
COPY src/ ./src/
COPY api/ ./api/
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY docs/ ./docs/
COPY final_artifacts/ ./final_artifacts/
COPY test_dl.dcm ./

# Install local package in editable/standard mode
RUN pip install --no-cache-dir -e .

# Set ownership to non-root user
RUN chown -R appuser:medvision ${APP_HOME}

USER appuser

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Container healthcheck probe against FastAPI /health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command starts the REST API service
CMD ["uvicorn", "medvision.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
