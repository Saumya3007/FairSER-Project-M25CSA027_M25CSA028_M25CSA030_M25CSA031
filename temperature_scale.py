
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
from inference import load_model, preprocess_audio

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RES_DIR  = BASE_DIR / "results"

def calibrate(stage="full", val_csv=None):
    val_csv = val_csv or str(DATA_DIR / "val.csv")
    df      = pd.read_csv(val_csv)
    model   = load_model(stage)

    print(f"Calibrating on {len(df)} val samples...")
    logits_all, labels_all = [], []
    for i, (_, row) in enumerate(df.iterrows()):
        if i % 100 == 0: print(f"  {i}/{len(df)}")
        try:
            x, attn = preprocess_audio(row["path"])
            with torch.no_grad():
                logits = model(x, attn)
            label_col = "label_id" if "label_id" in row.index else "label"
            logits_all.append(logits.cpu())
            labels_all.append(int(row[label_col]))
        except Exception as e:
            print(f"  skip {row['path']}: {e}")

    logits_t = torch.cat(logits_all, dim=0)
    labels_t = torch.tensor(labels_all)

    nll_before = F.cross_entropy(logits_t, labels_t).item()
    print(f"NLL before scaling: {nll_before:.4f}")

    T = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.LBFGS([T], lr=0.01, max_iter=100)

    def eval_T():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits_t / T, labels_t)
        loss.backward()
        return loss

    optimizer.step(eval_T)
    T_opt = float(T.item())
    nll_after = F.cross_entropy(logits_t / T_opt, labels_t).item()

    print(f"Optimal temperature T = {T_opt:.4f}")
    print(f"NLL after  scaling: {nll_after:.4f}")

    (RES_DIR / "temperature.txt").write_text(str(T_opt))
    print(f"Saved -> {RES_DIR}/temperature.txt")
    return T_opt

if __name__ == "__main__":
    calibrate()
