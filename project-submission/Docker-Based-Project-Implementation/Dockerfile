# ---------- Base image ----------
FROM python:3.10-slim

# ---------- Environment ----------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TRANSFORMERS_NO_TF=1 \
    TRANSFORMERS_NO_FLAX=1 \
    PORT=8080

# ---------- System dependencies ----------
RUN apt-get update && apt-get install -y \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ---------- Python dependencies ----------
# CRITICAL: numpy<2 for torch + transformers compatibility
RUN pip install --no-cache-dir \
    "numpy<2" \
    torch==2.1.2 \
    transformers==4.37.2 \
    accelerate \
    safetensors \
    fastapi \
    uvicorn

# ---------- App setup ----------
WORKDIR /app

COPY app.py /app/app.py
COPY serve.py /app/serve.py

EXPOSE 8080

# ---------- SageMaker entrypoint ----------
ENTRYPOINT ["python", "serve.py"]
CMD ["serve"]

