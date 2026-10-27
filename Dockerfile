# ================= Stage 1: trainer =================
FROM python:3.10-slim AS trainer
WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-train.txt .
RUN pip install --no-cache-dir -r requirements-train.txt

ARG EPOCHS=5
ARG SEED=42

COPY src/ ./src/
# Завантаження датасету, навчання, оцінка → /build/artifacts
RUN python -m src.train --classes yes no up down --epochs ${EPOCHS} --seed ${SEED} \
    --out /build/artifacts

# ================= Stage 2: runtime =================
FROM python:3.10-slim AS runtime
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/ ./src/
COPY --from=trainer /build/artifacts ./artifacts

ENV MODEL_PATH=/app/artifacts/model.pt \
    PORT=8000 \
    PYTHONUNBUFFERED=1

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
