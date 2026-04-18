import env  
import os, json, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from dataset     import SERDataset, collate_fn
from models      import FairSERModel
from train_utils import evaluate, DEVICE
from main import load_splits


LR       = 2e-4
EPOCHS   = 30
BATCH    = 32
PATIENCE = 5

BASE_DIR    = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
CKPT_DIR    = RESULTS_DIR / "checkpoints"
for d in [RESULTS_DIR, CKPT_DIR]:
    d.mkdir(parents=True, exist_ok=True)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.15):
        super().__init__()
        self.gamma           = gamma
        self.weight          = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce = F.cross_entropy(
            logits, targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        pt    = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()


def make_loader(df, shuffle=False, augment=False):
    return DataLoader(
        SERDataset(df), batch_size=BATCH, shuffle=shuffle,
        num_workers=8, pin_memory=True,
        persistent_workers=True, collate_fn=collate_fn
    )


def run_lora(train_df, val_df, test_df):
    print("\n" + "=" * 60)
    print("  STAGE 01 — LoRA Fine-Tuning (Focal Loss + Class Weights)")
    print("=" * 60)

    tr_dl = make_loader(train_df, shuffle=True, augment=True)
    va_dl = make_loader(val_df)
    te_dl = make_loader(test_df)

    model = FairSERModel().to(DEVICE)
    print(f"  Trainable params : {model.trainable_params():,}")
    print(f"  Total params     : {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=1e-2
    )
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR,
        steps_per_epoch=len(tr_dl),
        epochs=EPOCHS, pct_start=0.05
    )
    scaler = torch.amp.GradScaler("cuda")

    class_weights = torch.tensor([1.0, 2.5, 2.0, 0.6]).to(DEVICE)
    crit = FocalLoss(gamma=2.0, weight=class_weights, label_smoothing=0.15)

    best_val_f1  = 0.0
    patience_cnt = 0
    best_ckpt    = str(CKPT_DIR / "lora_best.pt")
    history      = []

    for ep in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for X, attn, y, *_ in tqdm(tr_dl, desc=f"  [LoRA] Ep {ep:02d}/{EPOCHS}", leave=False):
            X, attn, y = X.to(DEVICE), attn.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            with torch.amp.autocast("cuda"):
                loss = crit(model(X, attn), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            total_loss += loss.item()

        macro, wtd, grp, lg, gg, preds, trues = evaluate(model, va_dl)
        avg = total_loss / len(tr_dl)
        history.append({"epoch": ep, "loss": round(avg, 4), "val_f1": round(macro, 4)})
        print(f"  [LoRA] ep {ep:02d}  loss={avg:.4f}  val_f1={macro:.4f}  lr={sched.get_last_lr()[0]:.2e}")

        if macro > best_val_f1:
            best_val_f1 = macro; patience_cnt = 0
            torch.save(model.state_dict(), best_ckpt)
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"  [LoRA] Early stop ep {ep}"); break

    model.load_state_dict(torch.load(best_ckpt, map_location=DEVICE))
    macro, wtd, grp, lg, gg, preds, trues = evaluate(model, te_dl)
    results = {
        "stage": "lora", "macro_f1": round(macro, 4), "weighted_f1": round(wtd, 4),
        "group_f1": grp, "lang_gap": lg, "gender_gap": gg,
        "history": history, "y_pred": preds, "y_true": trues
    }
    out = str(RESULTS_DIR / "lora_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  [LoRA] macro_f1={macro:.4f}  lang_gap={lg:.4f}  gender_gap={gg:.4f}")
    print(f"  [LoRA] Group F1 : {grp}\n  [LoRA] → {out}")
    return model, results


if __name__ == "__main__":
    train_df, val_df, test_df = load_splits()
    run_lora(train_df, val_df, test_df)
