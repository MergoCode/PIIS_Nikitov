import contextlib
import json
import logging
import os
import time

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from src.data import DEFAULT_LABELS, preprocess_audio_bytes
from src.model import AudioCNN

# Environment configuration
MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model.pt")
PORT = int(os.getenv("PORT", "8000"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger("kws-api")

# Global state
app_state = {
    "model": None,
    "classes": DEFAULT_LABELS,
    "device": torch.device("cpu"),
    "model_size_kb": 0.0,
    "accuracy": 0.0,
    "model_version": "1.0.0",
    "architecture": "AudioCNN",
}


def load_artifacts():
    model_path = MODEL_PATH
    if not os.path.exists(model_path) and os.path.exists("models/audio_model.pth"):
        model_path = "models/audio_model.pth"

    logger.info(f"Loading model from: {model_path}")
    base_dir = os.path.dirname(model_path)

    # Attempt to load labels.json
    labels_file = os.path.join(base_dir, "labels.json")
    if os.path.exists(labels_file):
        with open(labels_file, "r", encoding="utf-8") as f:
            app_state["classes"] = json.load(f)
    else:
        app_state["classes"] = DEFAULT_LABELS

    # Attempt to load metrics.json
    metrics_file = os.path.join(base_dir, "metrics.json")
    if os.path.exists(metrics_file):
        with open(metrics_file, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)
            app_state["accuracy"] = metrics_data.get("accuracy", 0.0)
            app_state["model_version"] = metrics_data.get("model_version", "1.0.0")
            app_state["architecture"] = metrics_data.get("architecture", "AudioCNN")

    device = torch.device("cpu")
    app_state["device"] = device

    num_classes = len(app_state["classes"])
    model = AudioCNN(num_classes=num_classes).to(device)

    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        app_state["model_size_kb"] = round(os.path.getsize(model_path) / 1024.0, 2)
    else:
        logger.warning(f"Warning: Model weight file not found at {model_path}! Initialized empty model.")

    model.eval()
    app_state["model"] = model
    logger.info(
        f"Model loaded successfully! Classes: {app_state['classes']}, "
        f"Size: {app_state['model_size_kb']} KB, Accuracy: {app_state['accuracy']}%"
    )


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield


app = FastAPI(
    title="Keyword Spotting Audio Classification API",
    version="1.0.0",
    lifespan=lifespan,
)


class ModelInfoResponse(BaseModel):
    architecture: str
    classes: list[str]
    model_size_kb: float
    accuracy: float
    model_version: str


class PredictResponse(BaseModel):
    label: str
    confidence: float
    probabilities: dict[str, float]
    latency_ms: float
    model_version: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    return {
        "architecture": app_state["architecture"],
        "classes": app_state["classes"],
        "model_size_kb": app_state["model_size_kb"],
        "accuracy": app_state["accuracy"],
        "model_version": app_state["model_version"],
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".wav", ".wave")):
        raise HTTPException(
            status_code=400,
            detail="Некоректний формат файлу. Очікується аудіофайл у форматі .wav",
        )

    try:
        content = await file.read()
        if not content:
            raise ValueError("Отримано порожній файл.")
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Помилка читання файлу: {e!s}",
        )

    # Measure latency for preprocessing and inference only
    t_start = time.perf_counter()
    try:
        spec = preprocess_audio_bytes(content).to(app_state["device"])
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Некоректний аудіофайл або помилка обробки wav: {e!s}",
        )

    model = app_state["model"]
    if model is None:
        raise HTTPException(status_code=500, detail="Модель не завантажена.")

    with torch.no_grad():
        logits = model(spec)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        confidence, pred_idx = torch.max(probs, dim=0)

    t_end = time.perf_counter()
    latency_ms = (t_end - t_start) * 1000.0

    labels = app_state["classes"]
    predicted_label = labels[pred_idx.item()]
    probabilities = {
        label: round(float(prob), 4) for label, prob in zip(labels, probs)
    }

    return {
        "label": predicted_label,
        "confidence": round(float(confidence), 4),
        "probabilities": probabilities,
        "latency_ms": round(latency_ms, 2),
        "model_version": app_state["model_version"],
    }
