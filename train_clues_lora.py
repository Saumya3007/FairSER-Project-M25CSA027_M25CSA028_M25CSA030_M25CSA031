import env  
import os, json, torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.cluster import MiniBatchKMeans
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



LR          = 5e-5
EPOCHS      = 20
PATIENCE    = 6
TEMPERATURE = 0.10
LAMBDA_TASK = 0.05
LAMBDA_SUB  = 0.05
LAMBDA_ERR  = 0.02
LAMBDA_LANG = 0.05
K_CLUSTERS  = 16

BASE_DIR    = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
CKPT_DIR    = RESULTS_DIR / "checkpoints"


def _clues_loss(z, y, sub, logits, lang_ids=None, temp=TEMPERATURE):
    z   = F.normalize(z, dim=1)
    sim = torch.mm(z, z.T) / temp
    B   = z.size(0)

    def _contra(pos_mask):
        diag     = torch.eye(B, dtype=torch.bool, device=z.device)
        pos_mask = pos_mask & ~diag
        neg_mask = ~pos_mask & ~diag
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            return torch.tensor(0.0, device=z.device)
        pos = (sim * pos_mask.float()).sum(1) / (pos_mask.float().sum(1) + 1e-8)
        neg = (sim * neg_mask.float()).sum(1) / (neg_mask.float().sum(1) + 1e-8)
        return F.relu(neg - pos + 0.2).mean()

    task_mask = (y.unsqueeze(0) == y.unsqueeze(1))
    sub_mask  = (sub.unsqueeze(0) == sub.unsqueeze(1))
    correct   = (logits.argmax(1) == y)
    err_mask  = (correct.unsqueeze(0) == correct.unsqueeze(1)) & correct.unsqueeze(1)
    loss = (LAMBDA_TASK * _contra(task_mask) +
            LAMBDA_SUB  * _contra(sub_mask)  +
            LAMBDA_ERR  * _contra(err_mask))

    if lang_ids is not None:
        cross_lang_mask = task_mask & (lang_ids.unsqueeze(0) != lang_ids.unsqueeze(1))
        loss = loss + LAMBDA_LANG * _contra(cross_lang_mask)

    return loss


class SERDatasetWithSub(Dataset):
    def __init__(self, base_ds, subgroup_ids):
        self.base = base_ds
        self.sub  = torch.tensor(subgroup_ids, dtype=torch.long)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        item = self.base[i]
        item["subgroup"] = self.sub[i]
        return item


def collate_with_sub(batch):
    from dataset import collate_fn
    X, attn, y, langs, gends = collate_fn(batch)
    subs = torch.stack([b["subgroup"] for b in batch])
    return X, attn, y, langs, gends, subs


def _assign_subgroups(df, k=K_CLUSTERS):
    rng       = np.random.RandomState(42)
    lang_code = (df["language"] == "hindi").astype(int).values
    gend_code = (df["gender"] == "female").astype(int).values if "gender" in df.columns else np.zeros(len(df), int)
    feat      = np.stack([lang_code, gend_code, rng.randn(len(df)) * 0.1], axis=1)
    return MiniBatchKMeans(n_clusters=k, random_state=42, n_init=5).fit_predict(feat)


def run_clues_lora(train_df, val_df, test_df):
    print("\n" + "=" * 60)
    print("  STAGE 02 — CLUES Contrastive Debiasing on LoRA backbone")
    print("=" * 60)

    train_sub = _assign_subgroups(train_df)
    tr_ds     = SERDatasetWithSub(SERDataset(train_df), train_sub)
    tr_dl     = DataLoader(
        tr_ds, batch_size=16, shuffle=True,
        num_workers=8, pin_memory=True,
        persistent_workers=True, collate_fn=collate_with_sub
    )
    va_dl = make_loader(val_df)
    te_dl = make_loader(test_df)

    model     = FairSERModel().to(DEVICE)
    lora_ckpt = str(CKPT_DIR / "lora_best.pt")
    if os.path.exists(lora_ckpt):
        model.load_state_dict(torch.load(lora_ckpt, map_location=DEVICE))
        print("  [CLUES] Warm-started from lora_best.pt")

    opt   = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=1e-2
    )
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.amp.GradScaler("cuda")
    class_weights = torch.tensor([1.0, 2.5, 2.0, 0.6]).to(DEVICE)
    crit   = FocalLoss(gamma=2.0, weight=class_weights, label_smoothing=0.15)

    best_val_f1  = 0.0
    patience_cnt = 0
    best_ckpt    = str(CKPT_DIR / "clues_lora_best.pt")
    history      = []

    for ep in range(1, EPOCHS + 1):
        model.train()
        total_loss = total_ce = total_cl = 0.0
        for X, attn, y, langs, _g, sub in tqdm(tr_dl, desc=f"  [CLUES] Ep {ep:02d}/{EPOCHS}", leave=False):
            X, attn, y, sub = X.to(DEVICE), attn.to(DEVICE), y.to(DEVICE), sub.to(DEVICE)
            lang_ids = torch.tensor([0 if l == "english" else 1 for l in langs],
                                    dtype=torch.long, device=DEVICE)
            opt.zero_grad()
            with torch.amp.autocast("cuda"):
                z      = model.get_penultimate(X, attn)
                logits = model.classifier(z)
                ce     = crit(logits, y)
                cl     = _clues_loss(z, y, sub, logits, lang_ids=lang_ids)
                loss   = ce + cl
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            total_loss += loss.item(); total_ce += ce.item(); total_cl += cl.item()

        sched.step()
        macro, wtd, grp, lg, gg, preds, trues = evaluate(model, va_dl)
        n = len(tr_dl)
        history.append({
            "epoch": ep, "loss": round(total_loss / n, 4),
            "ce": round(total_ce / n, 4), "cl": round(total_cl / n, 4),
            "val_f1": round(macro, 4)
        })
        print(f"  [CLUES] ep {ep:02d}  loss={total_loss/n:.4f} "
              f"(ce={total_ce/n:.4f} cl={total_cl/n:.4f})  val_f1={macro:.4f}")

        if macro > best_val_f1:
            best_val_f1 = macro; patience_cnt = 0
            torch.save(model.state_dict(), best_ckpt)
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"  [CLUES] Early stop ep {ep}"); break

        if ep % 5 == 0:
            train_sub     = _assign_subgroups(train_df)
            tr_ds.sub     = torch.tensor(train_sub, dtype=torch.long)

    model.load_state_dict(torch.load(best_ckpt, map_location=DEVICE))
    macro, wtd, grp, lg, gg, preds, trues = evaluate(model, te_dl)
    results = {
        "stage": "clues_lora", "macro_f1": round(macro, 4), "weighted_f1": round(wtd, 4),
        "group_f1": grp, "lang_gap": lg, "gender_gap": gg,
        "history": history, "y_pred": preds, "y_true": trues
    }
    out = str(RESULTS_DIR / "clues_lora_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  [CLUES] macro_f1={macro:.4f}  lang_gap={lg:.4f}  gender_gap={gg:.4f}")
    print(f"  [CLUES] Group F1 : {grp}\n  [CLUES] → {out}")
    return model, results