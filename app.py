import io
import time
import torch
import soundfile as sf
import torchaudio.transforms as T
from fastapi import FastAPI, UploadFile, File, HTTPException
from src.model import AudioCNN

LABELS = ["yes", "no", "up", "down"]
TARGET_SR = 16000
FIXED_LEN = 16000

app = FastAPI(title="Speech Commands Classification Service")

device = torch.device("cpu")
model = AudioCNN(num_classes=len(LABELS)).to(device)
model.load_state_dict(torch.load("models/audio_model.pth", map_location=device))
model.eval()

mel_transform = T.MelSpectrogram(
    sample_rate=TARGET_SR,
    n_fft=1024,
    hop_length=512,
    n_mels=64
)
amp_to_db = T.AmplitudeToDB()

def preprocess_audio(file_bytes: bytes) -> torch.Tensor:
    data, sr = sf.read(io.BytesIO(file_bytes), dtype="float32")
    waveform = torch.from_numpy(data)
    
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    else:
        waveform = waveform.mean(dim=1, keepdim=True).t()

    if sr != TARGET_SR:
        waveform = T.Resample(sr, TARGET_SR)(waveform)

    max_val = torch.max(torch.abs(waveform))
    if max_val > 0:
        waveform = waveform / max_val

    if waveform.shape[1] < FIXED_LEN:
        waveform = torch.nn.functional.pad(waveform, (0, FIXED_LEN - waveform.shape[1]))
    else:
        waveform = waveform[:, :FIXED_LEN]

    spec = amp_to_db(mel_transform(waveform))
    spec = (spec - spec.mean()) / (spec.std() + 1e-6)
    
    return spec.unsqueeze(0)

@app.post("/predict")
async def predict_audio(file: UploadFile = File(...)):
    start_time = time.perf_counter()
    content = await file.read()
    
    try:
        spec = preprocess_audio(content).to(device)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Помилка обробки аудіо: {str(e)}")

    with torch.no_grad():
        logits = model(spec)
        probs = torch.softmax(logits, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)

    latency_ms = (time.perf_counter() - start_time) * 1000.0

    return {
        "predicted_label": LABELS[pred_idx.item()],
        "confidence": round(confidence.item(), 4),
        "all_probabilities": {label: round(prob.item(), 4) for label, prob in zip(LABELS, probs[0])},
        "latency_ms": round(latency_ms, 2)
    }

@app.get("/health")
def health_check():
    return {"status": "ok"}