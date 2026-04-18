import os
from pathlib import Path

_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())
else:
    print(f"[env] WARNING: .env not found at {_env_path}")

os.environ["DISABLE_TORCHCODEC"]      = "1"
os.environ["DATASETS_AUDIO_BACKEND"]  = "soundfile"
os.environ["TOKENIZERS_PARALLELISM"]  = "false"
os.environ["OMP_NUM_THREADS"]         = "4"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

HF_TOKEN    = os.environ.get("HF_TOKEN",    "")
HF_USERNAME = os.environ.get("HF_USERNAME", "")
HF_REPO_ID  = os.environ.get("HF_REPO_ID",  f"{HF_USERNAME}/FairHindiSER")
IEMOCAP_ROOT = os.environ.get("IEMOCAP_ROOT", "data/IEMOCAP_full_release")


import numpy as np

def load_audio(path, dtype="float32"):
    """Load audio file -> (torch.Tensor [C, T], sample_rate).
    Drop-in replacement for torchaudio.load() using soundfile."""
    import soundfile as sf
    import torch
    audio, sr = sf.read(str(path), dtype=dtype, always_2d=True)
    return torch.from_numpy(audio.T), sr

def save_audio(path, waveform, sample_rate):
    """Save waveform tensor to file. Drop-in for torchaudio.save()."""
    import soundfile as sf
    data = waveform.cpu().numpy().T
    sf.write(str(path), data, sample_rate)