import io

import soundfile as sf
import torch
import torchaudio
import torchaudio.transforms as T
from torch import nn
from torch.utils.data import Subset
from torchaudio.datasets import SPEECHCOMMANDS


# Monkey-patch torchaudio.load to use soundfile for portability
def custom_load(filepath, *args, **kwargs):
    data, samplerate = sf.read(filepath, dtype="float32")
    tensor = torch.from_numpy(data)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    else:
        tensor = tensor.t()
    return tensor, samplerate

torchaudio.load = custom_load

DEFAULT_LABELS = ["yes", "no", "up", "down"]
TARGET_SR = 16000
FIXED_LEN = 16000

mel_transform = T.MelSpectrogram(
    sample_rate=TARGET_SR,
    n_fft=1024,
    hop_length=512,
    n_mels=64
)
amp_to_db = T.AmplitudeToDB()


def transform_waveform_to_spec(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    if sample_rate != TARGET_SR:
        waveform = T.Resample(sample_rate, TARGET_SR)(waveform)

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    max_val = torch.max(torch.abs(waveform))
    if max_val > 0:
        waveform = waveform / max_val

    if waveform.shape[1] < FIXED_LEN:
        waveform = nn.functional.pad(waveform, (0, FIXED_LEN - waveform.shape[1]))
    else:
        waveform = waveform[:, :FIXED_LEN]

    spec = amp_to_db(mel_transform(waveform))
    spec = (spec - spec.mean()) / (spec.std() + 1e-6)
    return spec


def preprocess_audio_bytes(file_bytes: bytes) -> torch.Tensor:
    data, sr = sf.read(io.BytesIO(file_bytes), dtype="float32")
    waveform = torch.from_numpy(data)

    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    else:
        waveform = waveform.mean(dim=1, keepdim=True).t()

    spec = transform_waveform_to_spec(waveform, sr)
    return spec.unsqueeze(0)


def build_collate_fn(labels):
    def collate_fn(batch):
        tensors, targets = [], []
        for waveform, sample_rate, label, *_ in batch:
            spec = transform_waveform_to_spec(waveform, sample_rate)
            tensors.append(spec)
            targets.append(labels.index(label))
        return torch.stack(tensors), torch.tensor(targets)
    return collate_fn


def get_speechcommands_subset(data_root: str, labels: list, download: bool = True):
    dataset = SPEECHCOMMANDS(root=data_root, download=download)
    filtered_indices = []
    for i, fileid in enumerate(dataset._walker):
        parts = str(fileid).replace("\\", "/").strip("/").split("/")
        for part in parts:
            if part in labels:
                filtered_indices.append(i)
                break
    return Subset(dataset, filtered_indices)
