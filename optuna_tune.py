import env  
import json, torch, optuna
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from dataset     import SERDataset, collate_fn
from models      import FairSERModel
from train_utils import evaluate, DEVICE

BASE_DIR    = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
CKPT_DIR    = RESULTS_DIR / "checkpoints"
for d in [RESULTS_DIR, CKPT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

N_TRIALS = 30


def _objective(trial, train_df, val_df):
    lr_head      = trial.suggest_float("lr_head",        1e-5, 5e-4, log=True)
    lr_trans     = trial.suggest_float("lr_transformer", 1e-6, 1e-4, log=True)
    lr_cnn       = trial.suggest_float("lr_cnn",         1e-7, 1e-5, log=True)
    batch        = trial.suggest_categorical("batch",    [16, 32])
    wd           = trial.suggest_float("weight_decay",   1e-4, 1e-1, log=True)
    ls           = trial.suggest_float("label_smooth",   0.0,  0.15)
    dropout      = trial.suggest_float("dropout",        0.1,  0.5)
    unfreeze_cnn = trial.suggest_categorical("unfreeze_cnn", [True, False])

    model = FairSERModel().to(DEVICE)

    for m in model.head.modules():
        if isinstance(m, nn.Dropout):
            m.p = dropout

    model.unfreeze_transformer_layers([8, 9, 10, 11])
    if unfreeze_cnn:
        model.unfreeze_feature_extractor()

    tr_dl = DataLoader(
        SERDataset(train_df), batch_size=batch, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=collate_fn
    )
    va_dl = DataLoader(
        SERDataset(val_df), batch_size=batch, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=collate_fn
    )

    param_groups = model.get_param_groups(
        cnn_lr=lr_cnn, transformer_lr=lr_trans, head_lr=lr_head
    )
    for g in param_groups:
        g["weight_decay"] = wd

    opt = torch.optim.AdamW(param_groups)

    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5, eta_min=1e-7)
    scaler = torch.amp.GradScaler("cuda")
    crit   = nn.CrossEntropyLoss(label_smoothing=ls)

    best_val = 0.0
    for ep in range(5):
        model.train()
        for X, attn, y, *_ in tr_dl:
            X, attn, y = X.to(DEVICE), attn.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            with torch.amp.autocast("cuda"):
                loss = crit(model(X, attn), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        sched.step()

        macro, *_ = evaluate(model, va_dl)
        best_val  = max(best_val, macro)
        trial.report(macro, ep)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_val


def run_optuna(train_df, val_df):
    print("\n" + "=" * 60)
    print("  STAGE 03b — Optuna HPO (30 trials)")
    print("=" * 60)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    )
    study.optimize(
        lambda t: _objective(t, train_df, val_df),
        n_trials=N_TRIALS,
        show_progress_bar=True
    )

    best = study.best_params
    print(f"\n  [Optuna] Best val_f1 = {study.best_value:.4f}")
    print(f"  [Optuna] Best params : {best}")

    out = str(RESULTS_DIR / "optuna_results.json")
    with open(out, "w") as f:
        json.dump({"best_val_f1": study.best_value, "best_params": best}, f, indent=2)
    print(f"  [Optuna] → {out}")
    return best