import env  
import torch
import torchaudio
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from transformers import Wav2Vec2FeatureExtractor
from env import load_audio


def augment_waveform(wav, sr=16000, p=0.5):
    import random
    if random.random() > p:
        return wav
    choice = random.random()
    if choice < 0.4:
        snr_db = random.uniform(15, 25)
        noise  = torch.randn_like(wav)
        s_pow  = wav.norm(p=2)
        n_pow  = noise.norm(p=2) + 1e-8
        wav    = wav + (s_pow / (n_pow * (10 ** (snr_db / 20)))) * noise
    elif choice < 0.7:
        factor = random.uniform(0.9, 1.1)
        orig_len = wav.shape[-1]
        import torchaudio.transforms as T
        wav = T.Resample(int(sr * factor), sr)(wav)
        if wav.shape[-1] > orig_len:
            wav = wav[..., :orig_len]
        elif wav.shape[-1] < orig_len:
            wav = torch.nn.functional.pad(wav, (0, orig_len - wav.shape[-1]))
    else:
        import random
        mask_len = int(wav.shape[-1] * random.uniform(0.0, 0.10))
        start    = random.randint(0, wav.shape[-1] - mask_len - 1)
        wav      = wav.clone()
        wav[..., start:start+mask_len] = 0.0
    return wav

SAMPLE_RATE  = 16000
MAX_DURATION = 6.0
MAX_SAMPLES  = int(SAMPLE_RATE * MAX_DURATION)
LABEL_MAP    = {"angry": 0, "happy": 1, "neutral": 2, "sad": 3}

_FEATURE_EXT = None

def get_feature_extractor():
    global _FEATURE_EXT
    if _FEATURE_EXT is None:
        _FEATURE_EXT = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    return _FEATURE_EXT


def _load_wav(path: str) -> np.ndarray:
    try:
        wav, sr = load_audio(path)
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        wav = wav.squeeze(0)
        if wav.shape[0] > MAX_SAMPLES:
            wav = wav[:MAX_SAMPLES]
        return wav.numpy().astype(np.float32)
    except Exception:
        return np.zeros(MAX_SAMPLES // 4, dtype=np.float32)


class SERDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        if "label_id" in df.columns:
            self.labels = torch.tensor(df["label_id"].values, dtype=torch.long)
        elif "label" in df.columns:
            self.labels = torch.tensor(df["label"].values, dtype=torch.long)
        else:
            self.labels = torch.tensor(
                df["emotion"].map(LABEL_MAP).values, dtype=torch.long
            )
        self.langs = df["language"].values if "language" in df.columns else np.array(["unknown"] * len(df))
        self.gends = df["gender"].values   if "gender"   in df.columns else np.array(["unknown"] * len(df))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        path = self.df.iloc[i]["path"]
        wav  = _load_wav(str(path))
        feat = get_feature_extractor()(
            wav,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
            max_length=MAX_SAMPLES,
            truncation=True,
        )
        return {
            "input_values": feat.input_values.squeeze(0),
            "label":        self.labels[i],
            "language":     str(self.langs[i]),
            "gender":       str(self.gends[i]),
        }


def collate_fn(batch):
    input_values = [b["input_values"] for b in batch]
    labels       = torch.stack([b["label"]    for b in batch])
    langs        = [b["language"] for b in batch]
    gends        = [b["gender"]   for b in batch]
    padded       = torch.nn.utils.rnn.pad_sequence(input_values, batch_first=True)
    attn         = (padded != 0).long()
    return padded, attn, labels, langs, gends