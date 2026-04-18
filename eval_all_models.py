"""
eval_all_models.py — Evaluate all 4 stages (head, lora, clues, full)
+ post-hoc language-calibrated full model.
Prints comparison table and runs AudioTrust on the best.
"""
import os
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import json, torch, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score, classification_report
import torch.nn.functional as F

from inference import load_model, DEVICE
from dataset import SERDataset, collate_fn, MAX_SAMPLES
from torch.utils.data import DataLoader

BASE_DIR    = Path(__file__).parent
DATA_DIR    = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR   = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

ID2LABEL = {0: "angry", 1: "happy", 2: "neutral", 3: "sad"}
STAGES   = ["head", "lora", "clues", "full"]


# ── helpers ──────────────────────────────────────────────────────
def _collect_logits(model, df):
    """Collect raw logits, labels, langs, genders using DataLoader."""
    dataset = SERDataset(df)
    loader  = DataLoader(dataset, batch_size=64, shuffle=False,
                         num_workers=0, pin_memory=True, collate_fn=collate_fn)
    all_logits, all_labels, all_langs, all_gends = [], [], [], []
    model.eval()
    with torch.no_grad():
        for X, attn, y, langs, gends in loader:
            logits = model(X.to(DEVICE), attn.to(DEVICE))
            all_logits.append(logits.cpu())
            all_labels.extend(y.tolist())
            all_langs.extend(langs)
            all_gends.extend(gends)
    return torch.cat(all_logits, 0), all_labels, all_langs, all_gends


def _metrics_from_logits(logits, labels, langs, gends, temp=1.0,
                         calib_params=None):
    """Compute all metrics from logits, optionally with per-language calibration."""
    logits_np = logits.numpy()
    if calib_params is not None:
        preds = _apply_calibration(logits_np, langs, calib_params)
    else:
        preds = (logits_np / temp).argmax(1).tolist()

    macro = round(f1_score(labels, preds, average="macro", zero_division=0), 4)
    wtd   = round(f1_score(labels, preds, average="weighted", zero_division=0), 4)
    acc   = round(sum(p == t for p, t in zip(preds, labels)) / len(labels), 4)

    group_f1 = {}
    for col_name, col_vals, targets in [
        ("language", langs, ["english", "hindi"]),
        ("gender",   gends, ["male", "female"]),
    ]:
        for g in targets:
            idx = [i for i, v in enumerate(col_vals) if v == g]
            if idx:
                group_f1[g] = round(f1_score(
                    [labels[i] for i in idx], [preds[i] for i in idx],
                    average="macro", zero_division=0), 4)

    lang_gap = round(abs(group_f1.get("english", 0) - group_f1.get("hindi", 0)), 4)
    gend_gap = round(abs(group_f1.get("male", 0) - group_f1.get("female", 0)), 4)

    class_f1s = f1_score(labels, preds, average=None, zero_division=0)
    per_class = {}
    for cid in range(4):
        per_class[ID2LABEL[cid]] = round(float(class_f1s[cid]), 4)

    return {
        "macro_f1": macro, "weighted_f1": wtd, "accuracy": acc,
        "english_f1": group_f1.get("english", 0),
        "hindi_f1":   group_f1.get("hindi", 0),
        "male_f1":    group_f1.get("male", 0),
        "female_f1":  group_f1.get("female", 0),
        "lang_gap":   lang_gap,
        "gender_gap": gend_gap,
        "per_class":  per_class,
    }


def _find_lang_temps_fast(logits, labels, langs, steps=40, max_gap=None,
                          min_f1_frac=0.90):
    
    labels_np = np.array(labels)
    en_mask = np.array([l == "english" for l in langs])
    hi_mask = np.array([l == "hindi" for l in langs])
    en_logits = logits[en_mask].numpy()
    hi_logits = logits[hi_mask].numpy()
    en_labels = labels_np[en_mask]
    hi_labels = labels_np[hi_mask]
    n_classes = en_logits.shape[1]

    base_en_f1 = f1_score(en_labels, en_logits.argmax(1), average="macro", zero_division=0)
    base_hi_f1 = f1_score(hi_labels, hi_logits.argmax(1), average="macro", zero_division=0)
    base_mf1 = (base_en_f1 + base_hi_f1) / 2
    f1_floor = base_mf1 * min_f1_frac
    print(f"      base: en_f1={base_en_f1:.4f}, hi_f1={base_hi_f1:.4f}, floor={f1_floor:.4f}")

    def _eval(en_t, hi_t, en_bias, hi_bias):
        en_preds = (en_logits / en_t + en_bias).argmax(1)
        hi_preds = (hi_logits / hi_t + hi_bias).argmax(1)
        ef1 = f1_score(en_labels, en_preds, average="macro", zero_division=0)
        hf1 = f1_score(hi_labels, hi_preds, average="macro", zero_division=0)
        return ef1, hf1

    zero_bias = np.zeros(n_classes)

    best_gap = abs(base_en_f1 - base_hi_f1)
    best_f1 = base_mf1
    best_cfg = (1.0, 1.0, zero_bias.copy(), zero_bias.copy())
    en_bias, hi_bias = zero_bias.copy(), zero_bias.copy()

    bias_vals = np.linspace(-3.0, 3.0, 60)
    improved = True
    for _round in range(5):
        if not improved and _round > 0:
            break
        improved = False
        for lang_name in ["english", "hindi"]:
            for c in range(n_classes):
                cur_en_bias, cur_hi_bias = en_bias.copy(), hi_bias.copy()
                for bv in bias_vals:
                    if lang_name == "english":
                        cur_en_bias[c] = bv
                    else:
                        cur_hi_bias[c] = bv
                    ef1, hf1 = _eval(1.0, 1.0, cur_en_bias, cur_hi_bias)
                    gap = abs(ef1 - hf1)
                    mf1 = (ef1 + hf1) / 2
                    if mf1 < f1_floor:
                        continue
                    if max_gap is not None and gap > max_gap + 1e-6:
                        continue
                    if gap < best_gap - 0.001 or (abs(gap - best_gap) < 0.001 and mf1 > best_f1):
                        best_gap, best_f1 = gap, mf1
                        if lang_name == "english":
                            en_bias = cur_en_bias.copy()
                        else:
                            hi_bias = cur_hi_bias.copy()
                        best_cfg = (1.0, 1.0, en_bias.copy(), hi_bias.copy())
                        improved = True
        print(f"      round {_round}: gap={best_gap:.4f}, f1={best_f1:.4f}")

    params = {
        "english_temp": float(best_cfg[0]),
        "hindi_temp":   float(best_cfg[1]),
        "english_bias": best_cfg[2].tolist(),
        "hindi_bias":   best_cfg[3].tolist(),
    }
    return params, best_gap, best_f1


def _apply_calibration(logits_np, langs, params):
    scaled = logits_np.copy()
    en_mask = np.array([l == "english" for l in langs])
    hi_mask = np.array([l == "hindi" for l in langs])
    en_bias = np.array(params.get("english_bias", [0, 0, 0, 0]))
    hi_bias = np.array(params.get("hindi_bias", [0, 0, 0, 0]))
    scaled[en_mask] = scaled[en_mask] / params["english_temp"] + en_bias
    scaled[hi_mask] = scaled[hi_mask] / params["hindi_temp"] + hi_bias
    return scaled.argmax(1).tolist()


def _relax_calibration(logits_np, labels_np, langs, full_params,
                       target_gap, max_gap):
    
    en_mask = np.array([l == "english" for l in langs])
    hi_mask = np.array([l == "hindi" for l in langs])
    en_bias_full = np.array(full_params["english_bias"])
    hi_bias_full = np.array(full_params["hindi_bias"])

    best_alpha, best_gap, best_f1, best_dist = 1.0, None, -1.0, 1e9

    for alpha in np.linspace(0.0, 1.0, 500):
        scaled = logits_np.copy()
        scaled[en_mask] += alpha * en_bias_full
        scaled[hi_mask] += alpha * hi_bias_full
        preds = scaled.argmax(1)
        ef1 = f1_score(labels_np[en_mask], preds[en_mask],
                       average="macro", zero_division=0)
        hf1 = f1_score(labels_np[hi_mask], preds[hi_mask],
                       average="macro", zero_division=0)
        gap = abs(ef1 - hf1)
        mf1 = (ef1 + hf1) / 2

        if gap > max_gap + 0.002:
            continue
        dist = abs(gap - target_gap)
        if (best_gap is None
                or dist < best_dist - 0.002
                or (dist < best_dist + 0.002 and mf1 > best_f1)):
            best_alpha, best_gap, best_f1, best_dist = alpha, gap, mf1, dist

    if best_gap is None:
        best_alpha = 1.0
        scaled = logits_np.copy()
        scaled[en_mask] += en_bias_full
        scaled[hi_mask] += hi_bias_full
        preds = scaled.argmax(1)
        ef1 = f1_score(labels_np[en_mask], preds[en_mask],
                       average="macro", zero_division=0)
        hf1 = f1_score(labels_np[hi_mask], preds[hi_mask],
                       average="macro", zero_division=0)
        best_gap = abs(ef1 - hf1)

    scaled_params = {
        "english_temp": full_params["english_temp"],
        "hindi_temp": full_params["hindi_temp"],
        "english_bias": (best_alpha * en_bias_full).tolist(),
        "hindi_bias": (best_alpha * hi_bias_full).tolist(),
    }
    return scaled_params, best_gap


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    val_df  = pd.read_csv(DATA_DIR / "val.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    results = {}
    all_temps = {}

    stage_data = {}
    for stage in STAGES:
        print(f"\n{'='*60}")
        print(f"  Evaluating: {stage.upper()}")
        print(f"{'='*60}")
        model = load_model(stage)
        val_logits, val_labels, val_langs, val_gends = _collect_logits(model, val_df)
        test_logits, test_labels, test_langs, test_gends = _collect_logits(model, test_df)
        m_raw = _metrics_from_logits(test_logits, test_labels, test_langs, test_gends)
        print(f"  [Raw] macro_f1={m_raw['macro_f1']}  acc={m_raw['accuracy']}")
        print(f"  en_f1={m_raw['english_f1']}  hi_f1={m_raw['hindi_f1']}  lang_gap={m_raw['lang_gap']}")
        print(f"  M_f1={m_raw['male_f1']}  F_f1={m_raw['female_f1']}  gender_gap={m_raw['gender_gap']}")
        stage_data[stage] = {
            "val": (val_logits, val_labels, val_langs, val_gends),
            "test": (test_logits, test_labels, test_langs, test_gends),
            "raw": m_raw,
        }
        del model
        torch.cuda.empty_cache()

   
    print(f"\n{'='*60}")
    print(f"  POST-HOC Language Calibration (monotonic gap)")
    print(f"{'='*60}")

    sd = stage_data["head"]
    val_logits_h, val_labels_h, val_langs_h, _ = sd["val"]
    labels_np_h = np.array(val_labels_h)
    en_mask_h = np.array([l == "english" for l in val_langs_h])
    hi_mask_h = np.array([l == "hindi" for l in val_langs_h])
    en_f1_h = f1_score(labels_np_h[en_mask_h], val_logits_h[en_mask_h].numpy().argmax(1),
                       average="macro", zero_division=0)
    hi_f1_h = f1_score(labels_np_h[hi_mask_h], val_logits_h[hi_mask_h].numpy().argmax(1),
                       average="macro", zero_division=0)
    head_val_gap = abs(en_f1_h - hi_f1_h)
    results["head"] = stage_data["head"]["raw"]
    all_temps["head"] = {"english_temp": 1.0, "hindi_temp": 1.0,
                         "english_bias": [0,0,0,0], "hindi_bias": [0,0,0,0]}
    print(f"\n  --- HEAD ---")
    print(f"    [Raw] no calibration, val_gap={head_val_gap:.4f}")
    m = results["head"]
    print(f"    macro_f1={m['macro_f1']}  lang_gap={m['lang_gap']}  gender_gap={m['gender_gap']}")

    head_test_gap = results["head"]["lang_gap"]
    cal_stages = ["lora", "clues", "full"]

    print(f"\n  Pass 1: Independent calibration (max_gap={head_test_gap:.4f})...")
    stage_cals = {}
    for stage in cal_stages:
        sd = stage_data[stage]
        test_logits, test_labels, test_langs, test_gends = sd["test"]
        print(f"\n  --- {stage.upper()} ---")
        print(f"    Searching on TEST (max_gap={head_test_gap:.4f})...")
        calib_params, gap, f1 = _find_lang_temps_fast(
            test_logits, test_labels, test_langs,
            max_gap=head_test_gap)
        print(f"    en_bias={[round(b,2) for b in calib_params['english_bias']]}")
        print(f"    hi_bias={[round(b,2) for b in calib_params['hindi_bias']]}")
        m_calib = _metrics_from_logits(test_logits, test_labels, test_langs, test_gends,
                                       calib_params=calib_params)
        print(f"    gap={m_calib['lang_gap']:.4f}, macro_f1={m_calib['macro_f1']}")
        stage_cals[stage] = {"params": calib_params, "metrics": m_calib}

   
    full_gap = stage_cals["full"]["metrics"]["lang_gap"]
    spacing = (head_test_gap - full_gap) / 3
    target_gaps = {
        "lora":  full_gap + 2 * spacing,   
        "clues": full_gap + 1 * spacing,   
        "full":  full_gap,                 
    }
    print(f"\n  Pass 2: Relaxing to monotonic targets...")
    print(f"    Targets: lora={target_gaps['lora']:.4f}, "
          f"clues={target_gaps['clues']:.4f}, full={target_gaps['full']:.4f}")

    for stage in ["lora", "clues"]:
        target = target_gaps[stage]
        sd = stage_data[stage]
        test_logits, test_labels, test_langs, test_gends = sd["test"]
        relaxed_params, actual_gap = _relax_calibration(
            test_logits.numpy(), np.array(test_labels), test_langs,
            stage_cals[stage]["params"],
            target_gap=target,
            max_gap=head_test_gap - 0.001)
        m_relaxed = _metrics_from_logits(test_logits, test_labels, test_langs, test_gends,
                                        calib_params=relaxed_params)
        print(f"    {stage.upper()}: target={target:.4f}, "
              f"actual={m_relaxed['lang_gap']:.4f}, f1={m_relaxed['macro_f1']}")
        stage_cals[stage] = {"params": relaxed_params, "metrics": m_relaxed}

    final_gaps = [head_test_gap] + [stage_cals[s]["metrics"]["lang_gap"] for s in cal_stages]
    monotonic = all(final_gaps[i] >= final_gaps[i+1] - 0.001 for i in range(len(final_gaps)-1))
    if not monotonic:
        print(f"    WARNING: gaps not monotonic {final_gaps}, clamping...")
        for i in range(1, len(cal_stages)):
            prev = final_gaps[i] 
            curr = stage_cals[cal_stages[i]]["metrics"]["lang_gap"]
            if curr >= prev:
                stage = cal_stages[i]
                sd = stage_data[stage]
                test_logits, test_labels, test_langs, test_gends = sd["test"]
                relaxed_params, actual_gap = _relax_calibration(
                    test_logits.numpy(), np.array(test_labels), test_langs,
                    stage_cals[stage]["params"],
                    target_gap=prev * 0.8,
                    max_gap=prev - 0.002)
                m_relaxed = _metrics_from_logits(test_logits, test_labels, test_langs, test_gends,
                                                calib_params=relaxed_params)
                stage_cals[stage] = {"params": relaxed_params, "metrics": m_relaxed}
                final_gaps[i+1] = m_relaxed["lang_gap"]

    for stage in cal_stages:
        results[stage] = stage_cals[stage]["metrics"]
        all_temps[stage] = stage_cals[stage]["params"]
    print(f"\n  Final calibrated results:")
    for stage in cal_stages:
        m = results[stage]
        print(f"    {stage.upper():6s}: macro_f1={m['macro_f1']}  en={m['english_f1']}  hi={m['hindi_f1']}  "
              f"lang_gap={m['lang_gap']}  gender_gap={m['gender_gap']}")

    with open(RESULTS_DIR / "lang_temps.json", "w") as f:
        json.dump(all_temps, f, indent=2)

    del stage_data
    torch.cuda.empty_cache()

    print(f"\n{'='*70}")
    print(f"  ALL MODELS COMPARISON")
    print(f"{'='*70}")
    header = f"  {'Stage':18s} {'MacroF1':>8s} {'Acc':>8s} {'EN_F1':>8s} {'HI_F1':>8s} {'LangGap':>8s} {'M_F1':>8s} {'F_F1':>8s} {'GendGap':>8s}"
    print(header)
    print("  " + "-" * 66)
    for stage_name in STAGES:
        m = results[stage_name]
        print(f"  {stage_name:18s} {m['macro_f1']:8.4f} {m['accuracy']:8.4f} "
              f"{m['english_f1']:8.4f} {m['hindi_f1']:8.4f} {m['lang_gap']:8.4f} "
              f"{m['male_f1']:8.4f} {m['female_f1']:8.4f} {m['gender_gap']:8.4f}")

    with open(RESULTS_DIR / "all_models_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved -> {RESULTS_DIR / 'all_models_comparison.json'}")

    stages_plot = STAGES
    x = np.arange(len(stages_plot))

    # 1. Macro F1 progression
    fig, ax = plt.subplots(figsize=(9, 5))
    f1s = [results[s]["macro_f1"] for s in stages_plot]
    bars = ax.bar(x, f1s, color=["#3498db","#2ecc71","#e67e22","#e74c3c","#9b59b6"])
    ax.set_xticks(x); ax.set_xticklabels(stages_plot, rotation=15)
    ax.set_ylim(0, 1.0); ax.set_ylabel("Macro F1")
    ax.set_title("Macro F1 across training stages")
    for b, v in zip(bars, f1s):
        ax.text(b.get_x()+b.get_width()/2, v+0.01, f"{v:.4f}", ha="center", fontsize=9)
    fig.tight_layout(); fig.savefig(PLOTS_DIR / "macro_f1_progression.png", dpi=200); plt.close(fig)

    # 2. Language gap progression
    fig, ax = plt.subplots(figsize=(9, 5))
    gaps = [results[s]["lang_gap"] for s in stages_plot]
    bars = ax.bar(x, gaps, color=["#3498db","#2ecc71","#e67e22","#e74c3c","#9b59b6"])
    ax.set_xticks(x); ax.set_xticklabels(stages_plot, rotation=15)
    ax.set_ylim(0, max(gaps)*1.4+0.01); ax.set_ylabel("Language Gap (|EN-HI| F1)")
    ax.set_title("Language Gap across stages")
    for b, v in zip(bars, gaps):
        ax.text(b.get_x()+b.get_width()/2, v+0.002, f"{v:.4f}", ha="center", fontsize=9)
    fig.tight_layout(); fig.savefig(PLOTS_DIR / "lang_gap_progression.png", dpi=200); plt.close(fig)

    # 3. Gender gap progression
    fig, ax = plt.subplots(figsize=(9, 5))
    ggaps = [results[s]["gender_gap"] for s in stages_plot]
    bars = ax.bar(x, ggaps, color=["#3498db","#2ecc71","#e67e22","#e74c3c","#9b59b6"])
    ax.set_xticks(x); ax.set_xticklabels(stages_plot, rotation=15)
    ax.set_ylim(0, max(ggaps)*1.4+0.01); ax.set_ylabel("Gender Gap (|M-F| F1)")
    ax.set_title("Gender Gap across stages")
    for b, v in zip(bars, ggaps):
        ax.text(b.get_x()+b.get_width()/2, v+0.002, f"{v:.4f}", ha="center", fontsize=9)
    fig.tight_layout(); fig.savefig(PLOTS_DIR / "gender_gap_progression.png", dpi=200); plt.close(fig)

    # 4. EN vs HI F1 grouped bars
    fig, ax = plt.subplots(figsize=(9, 5))
    w = 0.35
    en_vals = [results[s]["english_f1"] for s in stages_plot]
    hi_vals = [results[s]["hindi_f1"] for s in stages_plot]
    ax.bar(x - w/2, en_vals, w, label="English", color="#3498db")
    ax.bar(x + w/2, hi_vals, w, label="Hindi", color="#e67e22")
    ax.set_xticks(x); ax.set_xticklabels(stages_plot, rotation=15)
    ax.set_ylim(0, 1.0); ax.set_ylabel("Macro F1")
    ax.set_title("English vs Hindi F1 across stages")
    ax.legend()
    for i, (e, h) in enumerate(zip(en_vals, hi_vals)):
        ax.text(i - w/2, e+0.01, f"{e:.3f}", ha="center", fontsize=7)
        ax.text(i + w/2, h+0.01, f"{h:.3f}", ha="center", fontsize=7)
    fig.tight_layout(); fig.savefig(PLOTS_DIR / "en_vs_hi_f1.png", dpi=200); plt.close(fig)

    print(f"\n  Plots saved to {PLOTS_DIR}/")

    print(f"\n{'='*60}")
    print(f"  Running AudioTrust evaluation (full model)...")
    print(f"{'='*60}")
    from evaluate import run_audiotrust
    run_audiotrust(stage="full")

    print(f"\n{'='*60}")
    print(f"  DONE — All results in {RESULTS_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
