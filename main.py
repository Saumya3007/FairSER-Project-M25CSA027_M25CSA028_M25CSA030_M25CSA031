import env  
import argparse
from pathlib import Path
import pandas as pd

BASE_DIR    = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def load_splits():
    DATA_DIR = BASE_DIR / "data"
    train = pd.read_csv(DATA_DIR / "train.csv")
    val   = pd.read_csv(DATA_DIR / "val.csv")
    test  = pd.read_csv(DATA_DIR / "test.csv")
    print(f"  Splits — train:{len(train)}  val:{len(val)}  test:{len(test)}")
    return train, val, test


def main():
    parser = argparse.ArgumentParser(description="FairHindiSER pipeline")
    parser.add_argument("--skip-dataset", action="store_true")
    parser.add_argument("--skip-zero",    action="store_true")
    parser.add_argument("--skip-head",    action="store_true")
    parser.add_argument("--skip-lora",    action="store_true")
    parser.add_argument("--skip-clues",   action="store_true")
    parser.add_argument("--skip-optuna",  action="store_true")
    parser.add_argument("--skip-full",    action="store_true")
    parser.add_argument("--skip-eval",    action="store_true")
    args = parser.parse_args()  

    # ── Stage 0: Dataset ────────────────────────────────────────
    if not args.skip_dataset:
        print("\n" + "="*60)
        print("  STAGE 00 — Dataset Pipeline")
        print("="*60)
        from pipeline import run_dataset_pipeline
        run_dataset_pipeline()
    else:
        print("  [Skip] Dataset pipeline")

    train_df, val_df, test_df = load_splits()

    # ── Stage 01: Zero-Shot Baseline ─────────────────────────────
    if not args.skip_zero:
        from train_zero_shot import run_zero_shot
        run_zero_shot(train_df, val_df, test_df)
    else:
        print("  [Skip] Stage 01 — Zero-Shot")

    # ── Stage 01.1: Head-Only Fine-Tuning ──────────────────────────
    if not args.skip_head:
        from train_head import run_head
        _, head_results = run_head(train_df, val_df, test_df)
    else:
        print("  [Skip] Stage 01.1 — Head-Only")

    # ── Stage 02 LoRA Fine-Tuning ──────────────────────────────
    if not args.skip_lora:
        from train_lora import run_lora
        _, lora_results = run_lora(train_df, val_df, test_df)
    else:
        print("  [Skip] Stage 02 — LoRA")

    # ── Stage 03: CLUES Debiasing ────────────────────────────────
    if not args.skip_clues:
        from train_clues_lora import run_clues_lora as run_clues
        _, clues_results = run_clues(train_df, val_df, test_df)
    else:
        print("  [Skip] Stage 03 — CLUES")

    # ── Stage 03b: Optuna HPO ────────────────────────────────────
    best_params = None
    if not args.skip_optuna:
        from optuna_tune import run_optuna
        best_params = run_optuna(train_df, val_df)
    else:
        print("  [Skip] Stage 03b — Optuna HPO")
        import json
        optuna_json = RESULTS_DIR / "optuna_results.json"
        if optuna_json.exists():
            with open(optuna_json) as f:
                best_params = json.load(f).get("best_params")
            print(f"  [Load] Optuna best_params: {best_params}")

    # ── Stage 04: Full Encoder Unfreezing ────────────────────────
    if not args.skip_full:
        from train_full_unfreeze import run_full_unfreeze
        _, full_results = run_full_unfreeze(train_df, val_df, test_df,
                                            best_params=best_params)
    else:
        print("  [Skip] Stage 04 — Full Unfreeze")

    # ── Stage 05: AudioTrust Evaluation ──────────────────────────
    if not args.skip_eval:
        from evaluate import run_audiotrust
        run_audiotrust()
    else:
        print("  [Skip] Stage 05 — AudioTrust Eval")

    print("\n" + "="*60)
    print("  FairHindiSER pipeline complete.")
    print("="*60)


if __name__ == "__main__":
    main()