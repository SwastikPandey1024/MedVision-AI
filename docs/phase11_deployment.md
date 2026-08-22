# 🐳 MedVision-AI: Phase 11 — Docker Containerization & Cloud Deployment

> **Comprehensive containerization architecture, Docker configuration, and cloud deployment guide for MedVision-AI services.**

---

## 📌 1. Overview & Container Architecture

MedVision-AI provides production-grade containerization supporting independent or unified execution of the **FastAPI REST API** and the **Streamlit Diagnostic Dashboard**.

### Security & Operational Standards
- **Base Image:** `python:3.11-slim` for minimal surface area and vulnerability reduction.
- **Non-Root Execution:** Runs under dedicated `appuser` (UID 10001) to prevent privilege escalation.
- **Container Healthcheck:** Built-in polling probe against `GET /health`.
- **Environment Isolation:** Zero credentials, keys, or sensitive configs baked into the container.
- **Inference Hardware:** Default configuration configured for CPU inference; GPU acceleration enabled when CUDA runtime drivers are passed (`--gpus all`).

---

## 🏗️ 2. Local Docker Commands

### Build Image
```bash
docker build -t medvision-ai:latest .
```

### Run REST API Container (Port 8000)
```bash
docker run -d \
  --name medvision-api \
  -p 8000:8000 \
  --restart unless-stopped \
  medvision-ai:latest
```

### Run Streamlit UI Container (Port 8501)
```bash
docker run -d \
  --name medvision-ui \
  -p 8501:8501 \
  --restart unless-stopped \
  medvision-ai:latest \
  streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

### Multi-Service Orchestration with Docker Compose
```bash
# Start both API and Streamlit UI
docker compose up -d

# Check service status
docker compose ps

# View live container logs
docker compose logs -f

# Stop services
docker compose down
```

---

## ☁️ 3. Cloud Deployment Recipes

### A. Render (Web Service)
1. Link GitHub repository.
2. Select **Docker** environment.
3. Set **Start Command:** `uvicorn medvision.api.main:app --host 0.0.0.0 --port $PORT` (or Streamlit).
4. Configure Health Check Path: `/health`.

---

### B. Hugging Face Spaces (Docker SDK)
1. Create a new Space with SDK: **Docker**.
2. Push repository with root `Dockerfile`.
3. Expose port `7860` by adding `--server.port 7860` in Streamlit CMD or setting `PORT=7860`.

---

### C. Google Cloud Run
```bash
# Authenticate & submit build
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/medvision-ai:latest

# Deploy serverless container
gcloud run deploy medvision-api \
  --image gcr.io/YOUR_PROJECT_ID/medvision-ai:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8000 \
  --memory 2Gi \
  --cpu 2
```

---

### D. AWS ECS / Fargate
1. Push image to Amazon Elastic Container Registry (ECR).
2. Create an ECS Task Definition with 2 GB Memory and 1 vCPU.
3. Configure Application Load Balancer targeting container port `8000` with health check on `/health`.

---

## ⚠️ Important Note on Model Binaries
Per best MLOps hygiene, model weights are not checked into public Git repositories. When deploying in continuous delivery pipelines, fetch checkpoints from secure object storage (AWS S3 / Google Cloud Storage) during container initialization or mount as volume.
