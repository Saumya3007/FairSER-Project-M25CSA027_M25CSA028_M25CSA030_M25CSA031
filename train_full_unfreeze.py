import env  
import os, json, torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from dataset     import SERDataset, collate_fn
from models      import FairSERModel
from train_utils import evaluate, DEVICE
from train_lora  import make_loader

import torch.nn.functional as _F

class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.15):
        super().__init__()
        self.gamma = gamma; self.weight = weight; self.label_smoothing = label_smoothing
    def forward(self, logits, targets):
        ce = _F.cross_entropy(logits, targets, weight=self.weight,
                              label_smoothing=self.label_smoothing, reduction="none")
        return ((1 - torch.exp(-ce)) ** self.gamma * ce).mean()



EPOCHS     = 25
PATIENCE   = 6
LAMBDA_LANG = 0.03
TEMP_LANG   = 0.10

BASE_DIR    = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
CKPT_DIR    = RESULTS_DIR / "checkpoints"
for d in [RESULTS_DIR, CKPT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def _cross_lang_loss(z, y, lang_ids, temp=TEMP_LANG):
    z = _F.normalize(z, dim=1)
    sim = torch.mm(z, z.T) / temp
    B = z.size(0)
    diag = torch.eye(B, dtype=torch.bool, device=z.device)
    task_mask = (y.unsqueeze(0) == y.unsqueeze(1))
    lang_diff = (lang_ids.unsqueeze(0) != lang_ids.unsqueeze(1))
    pos_mask = task_mask & lang_diff & ~diag
    neg_mask = ~pos_mask & ~diag
    if pos_mask.sum() == 0 or neg_mask.sum() == 0:
        return torch.tensor(0.0, device=z.device)
    pos = (sim * pos_mask.float()).sum(1) / (pos_mask.float().sum(1) + 1e-8)
    neg = (sim * neg_mask.float()).sum(1) / (neg_mask.float().sum(1) + 1e-8)
    return _F.relu(neg - pos + 0.2).mean()


UNFREEZE_SCHEDULE = {
    1:  {"transformer": [11, 10]},
    3:  {"transformer": [9,  8]},
    5:  {"transformer": [7,  6, 5]},
    8:  {"transformer": [4,  3, 2]},
    11: {"transformer": [1,  0]},
    14: {"cnn": True},             
}


def run_full_unfreeze(train_df, val_df, test_df, best_params=None):
    print("\n" + "=" * 60)
    print("  STAGE 04 — Gradual Full Encoder + CNN Unfreezing")
    print("=" * 60)

    lr_head  = best_params.get("lr_head",        1e-4) if best_params else 1e-4
    lr_trans = best_params.get("lr_transformer",  5e-6) if best_params else 5e-6
    lr_cnn   = best_params.get("lr_cnn",          1e-6) if best_params else 1e-6
    wd       = best_params.get("weight_decay",    1e-2) if best_params else 1e-2
    ls       = best_params.get("label_smooth",    0.05) if best_params else 0.05

    tr_dl = make_loader(train_df, shuffle=True)
    va_dl = make_loader(val_df)
    te_dl = make_loader(test_df)

    model = FairSERModel().to(DEVICE)
    model.param_summary()

    for ckpt_name in ["clues_lora_best.pt", "lora_best.pt"]:
        ckpt = CKPT_DIR / ckpt_name
        if ckpt.exists():
            model.load_state_dict(torch.load(str(ckpt), map_location=DEVICE))
            print(f"  [Full] Warm-started from {ckpt_name}")
            break

    scaler       = torch.amp.GradScaler("cuda")
    class_weights = torch.tensor([1.0, 2.5, 2.0, 0.6]).to(DEVICE)
    crit         = FocalLoss(gamma=2.0, weight=class_weights, label_smoothing=0.15)
    best_val_f1  = 0.0
    patience_cnt = 0
    best_ckpt    = str(CKPT_DIR / "full_best.pt")
    history      = []

    for ep in range(1, EPOCHS + 1):

        if ep in UNFREEZE_SCHEDULE:
            sched_ep = UNFREEZE_SCHEDULE[ep]
            if "transformer" in sched_ep:
                model.unfreeze_transformer_layers(sched_ep["transformer"])
            if sched_ep.get("cnn"):
                model.unfreeze_feature_extractor()
            model.param_summary()

        param_groups = model.get_param_groups(
            cnn_lr=lr_cnn, transformer_lr=lr_trans, head_lr=lr_head
        )
        for g in param_groups:
            g["weight_decay"] = wd
        opt = torch.optim.AdamW(param_groups)

        model.train()
        total_loss = 0.0
        for X, attn, y, langs, *_ in tqdm(tr_dl, desc=f"  [Full] Ep {ep:02d}/{EPOCHS}", leave=False):
            X, attn, y = X.to(DEVICE), attn.to(DEVICE), y.to(DEVICE)
            lang_ids = torch.tensor([0 if l == "english" else 1 for l in langs],
                                    dtype=torch.long, device=DEVICE)
            opt.zero_grad()
            with torch.amp.autocast("cuda"):
                z = model.get_penultimate(X, attn)
                logits = model.classifier(z)
                ce = crit(logits, y)
                lf = _cross_lang_loss(z, y, lang_ids)
                loss = ce + LAMBDA_LANG * lf
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            total_loss += loss.item()

        macro, wtd, grp, lg, gg, preds, trues = evaluate(model, va_dl)
        avg = total_loss / len(tr_dl)
        history.append({"epoch": ep, "loss": round(avg, 4), "val_f1": round(macro, 4),
                        "trainable": model.trainable_params()})
        print(f"  [Full] ep {ep:02d}  loss={avg:.4f}  val_f1={macro:.4f}  "
              f"trainable={model.trainable_params():,}")

        if macro > best_val_f1:
            best_val_f1 = macro; patience_cnt = 0
            torch.save(model.state_dict(), best_ckpt)
            print(f"  [Full] ✓ New best {macro:.4f}")
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"  [Full] Early stop ep {ep}"); break

    model.load_state_dict(torch.load(best_ckpt, map_location=DEVICE))
    macro, wtd, grp, lg, gg, preds, trues = evaluate(model, te_dl)
    results = {
        "stage": "full_unfreeze", "macro_f1": round(macro, 4),
        "weighted_f1": round(wtd, 4), "group_f1": grp,
        "lang_gap": lg, "gender_gap": gg,
        "history": history, "y_pred": preds, "y_true": trues
    }
    out = str(RESULTS_DIR / "full_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  [Full] TEST  macro_f1={macro:.4f}  lang_gap={lg:.4f}  gender_gap={gg:.4f}")
    print(f"  [Full] Group F1: {grp}")
    print(f"  [Full] → {out}")
    return model, results