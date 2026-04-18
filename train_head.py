# train_head.py — Stage 02: Head-only fine-tuning, SSL backbone fully frozen
import env 
import json, torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from dataset     import SERDataset, collate_fn
from models      import FairSERModel
from train_utils import evaluate, DEVICE

LR       = 1e-3
EPOCHS   = 20
BATCH    = 64
PATIENCE = 5

BASE_DIR    = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
CKPT_DIR    = RESULTS_DIR / "checkpoints"
for d in [RESULTS_DIR, CKPT_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def make_loader(df, shuffle=False):
    return DataLoader(
        SERDataset(df), batch_size=BATCH, shuffle=shuffle,
        num_workers=8, pin_memory=True,
        persistent_workers=True, collate_fn=collate_fn
    )


def run_head(train_df, val_df, test_df):
    print("\n" + "=" * 60)
    print("  STAGE 02 — Head-Only Fine-Tuning (SSL backbone FROZEN)")
    print("=" * 60)

    tr_dl = make_loader(train_df, shuffle=True)
    va_dl = make_loader(val_df)
    te_dl = make_loader(test_df)

    model = FairSERModel().to(DEVICE)
    print(f"  Trainable params : {model.trainable_params():,}  (head + classifier only)")

    opt = torch.optim.AdamW(
        list(model.head.parameters()) + list(model.classifier.parameters()),
        lr=LR, weight_decay=1e-3
    )
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.amp.GradScaler("cuda")
    crit   = nn.CrossEntropyLoss()

    best_val_f1  = 0.0
    patience_cnt = 0
    best_ckpt    = str(CKPT_DIR / "head_best.pt")
    history      = []

    for ep in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for X, attn, y, *_ in tqdm(tr_dl, desc=f"  [Head] Ep {ep:02d}/{EPOCHS}", leave=False):
            X, attn, y = X.to(DEVICE), attn.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            with torch.amp.autocast("cuda"):
                loss = crit(model(X, attn), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            total_loss += loss.item()
        sched.step()

        macro, wtd, grp, lg, gg, preds, trues = evaluate(model, va_dl)
        avg = total_loss / len(tr_dl)
        history.append({"epoch": ep, "loss": round(avg, 4), "val_f1": round(macro, 4)})
        print(f"  [Head] ep {ep:02d}  loss={avg:.4f}  val_f1={macro:.4f}")

        if macro > best_val_f1:
            best_val_f1 = macro; patience_cnt = 0
            torch.save(model.state_dict(), best_ckpt)
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"  [Head] Early stop ep {ep}"); break

    model.load_state_dict(torch.load(best_ckpt, map_location=DEVICE))
    macro, wtd, grp, lg, gg, preds, trues = evaluate(model, te_dl)
    results = {
        "stage": "head_only", "macro_f1": round(macro, 4), "weighted_f1": round(wtd, 4),
        "group_f1": grp, "lang_gap": lg, "gender_gap": gg,
        "history": history, "y_pred": preds, "y_true": trues
    }
    out = str(RESULTS_DIR / "head_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  [Head] TEST  macro_f1={macro:.4f}  lang_gap={lg:.4f}  gender_gap={gg:.4f}")
    print(f"  [Head] Group F1: {grp}")
    print(f"  [Head] → {out}")
    return model, results