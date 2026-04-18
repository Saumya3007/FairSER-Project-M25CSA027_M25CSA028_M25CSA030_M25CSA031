import env  
import json, torch
from torch.utils.data import DataLoader
from pathlib import Path
from dataset     import SERDataset, collate_fn
from models      import FairSERModel
from train_utils import evaluate, DEVICE

BASE_DIR    = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_zero_shot(train_df, val_df, test_df):
    print("\n" + "=" * 60)
    print("  STAGE 01 — Zero-Shot Baseline")
    print("  Frozen SSL, random head — measures raw embedding signal")
    print("=" * 60)

    te_dl = DataLoader(
        SERDataset(test_df), batch_size=64, shuffle=False,
        num_workers=8, pin_memory=True, collate_fn=collate_fn
    )
    va_dl = DataLoader(
        SERDataset(val_df), batch_size=64, shuffle=False,
        num_workers=8, pin_memory=True, collate_fn=collate_fn
    )

    model = FairSERModel().to(DEVICE)

    macro_val, *_                          = evaluate(model, va_dl)
    macro, wtd, grp, lg, gg, preds, trues = evaluate(model, te_dl)

    results = {
        "stage": "zero_shot", "macro_f1": round(macro, 4),
        "weighted_f1": round(wtd, 4), "group_f1": grp,
        "lang_gap": lg, "gender_gap": gg,
        "y_pred": preds, "y_true": trues
    }
    out = str(RESULTS_DIR / "zero_shot_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  [Zero-Shot] TEST  macro_f1={macro:.4f}  lang_gap={lg:.4f}  gender_gap={gg:.4f}")
    print(f"  [Zero-Shot] Group F1: {grp}")
    print(f"  [Zero-Shot] → {out}")
    return results