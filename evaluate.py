import os
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
from env import load_audio, save_audio
from sklearn.metrics import (
    f1_score, confusion_matrix, roc_auc_score, classification_report
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from inference import load_model, preprocess_audio

BASE_DIR    = Path(__file__).parent
DATA_DIR    = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR   = RESULTS_DIR / "plots"
PRED_DIR    = RESULTS_DIR / "predictions"
for d in [RESULTS_DIR, PLOTS_DIR, PRED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SR       = 16000
ID2LABEL = {0: "angry", 1: "happy", 2: "neutral", 3: "sad"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}
COLORS   = {"angry": "#e74c3c", "happy": "#f39c12", "neutral": "#3498db", "sad": "#9b59b6"}


def _ensure_label_id(df):
    df = df.copy()
    if "label_id" not in df.columns:
        if "label" in df.columns and pd.api.types.is_numeric_dtype(df["label"]):
            df["label_id"] = df["label"].astype(int)
        elif "emotion" in df.columns:
            df["label_id"] = df["emotion"].map(LABEL2ID)
        else:
            raise ValueError("Need label_id, numeric label, or emotion column")
    return df


def _predict_logits(model, audio_path):
    x, attn = preprocess_audio(str(audio_path))
    with torch.no_grad():
        logits = model(x, attn) if attn is not None else model(x, None)
        probs  = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    return int(np.argmax(probs)), probs

def _collect_all_predictions(model, df):
    from dataset import SERDataset, collate_fn
    from torch.utils.data import DataLoader

    dataset = SERDataset(df)
    loader  = DataLoader(dataset, batch_size=64, shuffle=False,
                         num_workers=4, pin_memory=True, collate_fn=collate_fn)

    all_preds, all_probs, all_trues = [], [], []
    all_langs, all_gends = [], []
    DEVICE = next(model.parameters()).device

    model.eval()
    with torch.no_grad():
        for X, attn, y, langs, gends in loader:
            X, attn = X.to(DEVICE), attn.to(DEVICE)
            logits = model(X, attn)
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()
            preds  = logits.argmax(1).cpu().tolist()
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_trues.extend(y.tolist())
            all_langs.extend(langs)
            all_gends.extend(gends)

    records = []
    df_reset = df.reset_index(drop=True)
    for i in range(len(df_reset)):
        row   = df_reset.iloc[i]
        pred  = all_preds[i]
        probs = all_probs[i]
        true_id = all_trues[i]
        records.append({
            "path":          row["path"],
            "filename":      Path(row["path"]).name,
            "true_label_id": true_id,
            "true_emotion":  ID2LABEL.get(true_id, str(true_id)),
            "pred_label_id": pred,
            "pred_emotion":  ID2LABEL[pred],
            "correct":       pred == true_id,
            "confidence":    round(float(np.max(probs)), 4),
            "prob_angry":    round(float(probs[0]), 4),
            "prob_happy":    round(float(probs[1]), 4),
            "prob_neutral":  round(float(probs[2]), 4),
            "prob_sad":      round(float(probs[3]), 4),
            "language":      all_langs[i],
            "gender":        all_gends[i],
            "accent":        row.get("accent",    "unknown"),
            "speaker_id":    row.get("speaker_id", "unknown"),
        })

    total = len(records)
    correct = sum(r["correct"] for r in records)
    print(f"\n    Done. Total: {total}  Correct: {correct}  Accuracy: {correct/total:.4f}")

    out = pd.DataFrame(records)
    return out.reset_index(drop=True)


def _ece(y_true, probs, n_bins=10):
    conf = probs.max(1); pred = probs.argmax(1)
    acc  = (pred == y_true).astype(float)
    ece  = 0.0
    for lo, hi in zip(np.linspace(0, 1, n_bins), np.linspace(0, 1, n_bins)[1:]):
        idx = (conf > lo) & (conf <= hi)
        if idx.any():
            ece += np.abs(acc[idx].mean() - conf[idx].mean()) * idx.mean()
    return float(ece)


def save_predictions_txt(pred_df, out_path):
    lines = []
    lines.append("=" * 80)
    lines.append("  FairSER -- Test Set Predictions (Ground Truth vs Predicted)")
    lines.append("=" * 80)
    lines.append(f"  Total samples : {len(pred_df)}")
    lines.append(f"  Correct       : {pred_df['correct'].sum()}  ({pred_df['correct'].mean()*100:.1f}%)")
    lines.append(f"  Wrong         : {(~pred_df['correct']).sum()}  ({(~pred_df['correct']).mean()*100:.1f}%)")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"  {'#':>5}  {'Filename':42s}  {'True':10s}  {'Predicted':10s}  {'Conf':6s}  {'OK?':5s}  {'Language':10s}  Gender")
    lines.append("  " + "-" * 112)
    for i, row in pred_df.iterrows():
        ok = "YES" if row["correct"] else "NO "
        lines.append(
            f"  {i+1:>5}  {str(row['filename']):42s}  "
            f"{str(row['true_emotion']):10s}  {str(row['pred_emotion']):10s}  "
            f"{row['confidence']:.3f}  {ok:5s}  {str(row.get('language','-')):10s}  {row.get('gender','-')}"
        )
    lines.append("")
    lines.append("=" * 80)
    lines.append("  PER-EMOTION ACCURACY")
    lines.append("=" * 80)
    lines.append(f"  {'Emotion':12s}  {'Total':7s}  {'Correct':9s}  {'Accuracy':10s}  Mean Conf")
    lines.append("  " + "-" * 60)
    for emo, sdf in pred_df.groupby("true_emotion"):
        lines.append(
            f"  {str(emo):12s}  {len(sdf):7d}  {sdf['correct'].sum():9d}"
            f"  {sdf['correct'].mean():10.4f}  {sdf['confidence'].mean():.4f}"
        )
    if "language" in pred_df.columns:
        lines.append("")
        lines.append("=" * 80)
        lines.append("  PER-LANGUAGE ACCURACY")
        lines.append("=" * 80)
        lines.append(f"  {'Language':15s}  {'Total':7s}  {'Correct':9s}  {'Accuracy':10s}  Mean Conf")
        lines.append("  " + "-" * 60)
        for lang, sdf in pred_df.groupby("language"):
            lines.append(
                f"  {str(lang):15s}  {len(sdf):7d}  {sdf['correct'].sum():9d}"
                f"  {sdf['correct'].mean():10.4f}  {sdf['confidence'].mean():.4f}"
            )
    if "gender" in pred_df.columns:
        lines.append("")
        lines.append("=" * 80)
        lines.append("  PER-GENDER ACCURACY")
        lines.append("=" * 80)
        lines.append(f"  {'Gender':15s}  {'Total':7s}  {'Correct':9s}  {'Accuracy':10s}  Mean Conf")
        lines.append("  " + "-" * 60)
        for g, sdf in pred_df.groupby("gender"):
            lines.append(
                f"  {str(g):15s}  {len(sdf):7d}  {sdf['correct'].sum():9d}"
                f"  {sdf['correct'].mean():10.4f}  {sdf['confidence'].mean():.4f}"
            )
    wrong = pred_df[~pred_df["correct"]]
    lines.append("")
    lines.append("=" * 80)
    lines.append(f"  MISCLASSIFIED SAMPLES ({len(wrong)} total)")
    lines.append("=" * 80)
    lines.append(f"  {'#':>5}  {'Filename':42s}  {'True':10s}  {'Predicted':10s}  {'Conf':6s}  Language")
    lines.append("  " + "-" * 95)
    for i, (_, row) in enumerate(wrong.iterrows()):
        lines.append(
            f"  {i+1:>5}  {str(row['filename']):42s}  "
            f"{str(row['true_emotion']):10s}  {str(row['pred_emotion']):10s}  "
            f"{row['confidence']:.3f}  {str(row.get('language','-'))}"
        )
    lines.append("")
    lines.append("=" * 80)
    lines.append("  END OF REPORT")
    lines.append("=" * 80)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  [TXT] Saved -> {out_path}")


def _plot_confusion(y_true, y_pred, out):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    labels = [ID2LABEL[i] for i in range(4)]
    ax.set_xticks(range(4), labels, rotation=30)
    ax.set_yticks(range(4), labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (test set)")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cm[i,j]}", ha="center", va="center",
                    color="white" if cm[i,j] > cm.max()/2 else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)


def _plot_per_class_f1(report_dict, out):
    classes = [ID2LABEL[i] for i in range(4)]
    f1s     = [report_dict[c]["f1-score"] for c in classes]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(classes, f1s, color=[COLORS[c] for c in classes])
    ax.set_ylim(0, 1.0); ax.set_ylabel("F1 score")
    ax.set_title("Per-class F1 (test set)")
    for b, v in zip(bars, f1s):
        ax.text(b.get_x()+b.get_width()/2, v+0.02, f"{v:.3f}", ha="center", fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)


def _plot_group_f1(scores, out, title):
    keys = list(scores.keys()); vals = [scores[k] for k in keys]
    fig, ax = plt.subplots(figsize=(max(6, len(keys)*1.5), 4.5))
    bars = ax.bar(keys, vals)
    ax.set_ylim(0, 1.0); ax.set_ylabel("Macro F1"); ax.set_title(title)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+0.02, f"{v:.3f}", ha="center", fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)


def _plot_confidence_dist(pred_df, out):
    corr  = pred_df[pred_df["correct"]]["confidence"]
    wrong = pred_df[~pred_df["correct"]]["confidence"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(corr,  bins=25, alpha=0.6, label=f"Correct (n={len(corr)})",  color="steelblue")
    ax.hist(wrong, bins=25, alpha=0.6, label=f"Wrong   (n={len(wrong)})", color="tomato")
    ax.set_xlabel("Confidence"); ax.set_ylabel("Count")
    ax.set_title("Confidence distribution -- correct vs incorrect")
    ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)


def _plot_reliability(y_true, probs, out, n_bins=10):
    conf = probs.max(1); pred = probs.argmax(1)
    acc  = (pred == y_true).astype(float)
    xs, ys = [], []
    for lo, hi in zip(np.linspace(0,1,n_bins), np.linspace(0,1,n_bins)[1:]):
        idx = (conf > lo) & (conf <= hi)
        if idx.any():
            xs.append(conf[idx].mean()); ys.append(acc[idx].mean())
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0,1],[0,1],"--",label="Perfect calibration")
    ax.plot(xs, ys, "o-", label="Model")
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
    ax.set_title("Reliability Diagram"); ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)


def _plot_sample_predictions(pred_df, out, n=3):
    correct_rows = list(pred_df[pred_df["correct"]].head(n).iterrows())
    wrong_rows   = list(pred_df[~pred_df["correct"]].head(n).iterrows())
    rows = correct_rows + wrong_rows
    if not rows: return
    fig, axes = plt.subplots(len(rows), 1, figsize=(10, 3*len(rows)))
    if len(rows) == 1: axes = [axes]
    for ax, (_, row) in zip(axes, rows):
        try:
            wav, sr = load_audio(row["path"])
            if sr != SR: wav = T.Resample(sr, SR)(wav)
            if wav.shape[0] > 1: wav = wav.mean(0, keepdim=True)
            mel    = torchaudio.transforms.MelSpectrogram(SR, n_mels=80)(wav)
            mel_db = torchaudio.transforms.AmplitudeToDB()(mel).squeeze(0).numpy()
            ax.imshow(mel_db, origin="lower", aspect="auto", cmap="magma")
            tag = "CORRECT" if row["correct"] else "WRONG"
            ax.set_title(
                f"[{tag}]  True: {row['true_emotion']}   Pred: {row['pred_emotion']}"
                f"   conf={row['confidence']:.2f}   {row['filename']}", fontsize=8
            )
            ax.set_xlabel("Frames"); ax.set_ylabel("Mel bins")
        except Exception: pass
    fig.tight_layout(); fig.savefig(out, dpi=180, bbox_inches="tight"); plt.close(fig)


def _plot_robustness(scores, out):
    keys = list(scores.keys()); vals = [scores[k] for k in keys]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(keys, vals)
    ax.set_ylim(0, 1.0); ax.set_ylabel("Macro F1")
    ax.set_title("Robustness under audio degradations")
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+0.02, f"{v:.3f}", ha="center")
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)


def _plot_summary(overall, out):
    dims = ["Fairness", "Robustness", "Explainability", "Privacy"]
    vals = [overall[f"{d.lower()}_score"] for d in dims]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(dims, vals, color=["#2ecc71","#3498db","#e67e22","#9b59b6"])
    ax.set_ylim(0,1.0); ax.set_ylabel("Score")
    ax.set_title(f"AudioTrust Summary  (Overall = {overall['audiotrust_overall']:.3f})")
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+0.02, f"{v:.3f}", ha="center")
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)


def eval_fairness(model, test_df):
    df      = _ensure_label_id(test_df)
    pred_df = _collect_all_predictions(model, df)
    y_true  = pred_df["true_label_id"].values
    y_pred  = pred_df["pred_label_id"].values
    probs   = pred_df[["prob_angry","prob_happy","prob_neutral","prob_sad"]].values
    report  = classification_report(
        y_true, y_pred, target_names=[ID2LABEL[i] for i in range(4)], output_dict=True
    )
    print("\n  Per-class classification report:")
    print(classification_report(y_true, y_pred, target_names=[ID2LABEL[i] for i in range(4)]))

    lang_f1, gender_f1, accent_f1 = {}, {}, {}
    for col, bucket in [("language",lang_f1),("gender",gender_f1),("accent",accent_f1)]:
        if col in pred_df.columns:
            for g, sdf in pred_df.groupby(col):
                if len(sdf) >= 2:
                    bucket[str(g)] = round(float(
                        f1_score(sdf.true_label_id, sdf.pred_label_id, average="macro")
                    ), 4)

    fairness = {
        "macro_f1":      round(float(report["macro avg"]["f1-score"]), 4),
        "weighted_f1":   round(float(report["weighted avg"]["f1-score"]), 4),
        "accuracy":      round(float(pred_df["correct"].mean()), 4),
        "per_class_f1":        {ID2LABEL[i]: round(report[ID2LABEL[i]]["f1-score"],  4) for i in range(4)},
        "per_class_precision": {ID2LABEL[i]: round(report[ID2LABEL[i]]["precision"], 4) for i in range(4)},
        "per_class_recall":    {ID2LABEL[i]: round(report[ID2LABEL[i]]["recall"],    4) for i in range(4)},
        "language_f1":   lang_f1,
        "gender_f1":     gender_f1,
        "accent_f1":     accent_f1,
        "language_gap":  round((max(lang_f1.values())-min(lang_f1.values()))     if len(lang_f1)>1   else 0.0, 4),
        "gender_gap":    round((max(gender_f1.values())-min(gender_f1.values())) if len(gender_f1)>1 else 0.0, 4),
        "accent_gap":    round((max(accent_f1.values())-min(accent_f1.values())) if len(accent_f1)>1 else 0.0, 4),
        "group_fairness_gamma_language": round(1.0-((max(lang_f1.values())-min(lang_f1.values()))     if len(lang_f1)>1   else 0.0), 4),
        "group_fairness_gamma_gender":   round(1.0-((max(gender_f1.values())-min(gender_f1.values())) if len(gender_f1)>1 else 0.0), 4),
        "ece": round(_ece(y_true, probs), 4),
    }

    pred_df.to_csv(PRED_DIR / "test_predictions_full.csv", index=False)
    pred_df[~pred_df["correct"]].to_csv(PRED_DIR / "test_predictions_wrong.csv",   index=False)
    pred_df[pred_df["correct"]].to_csv( PRED_DIR / "test_predictions_correct.csv", index=False)

    breakdown = pred_df.groupby("true_emotion").agg(
        total=("correct","count"), correct=("correct","sum"),
        accuracy=("correct","mean"), mean_confidence=("confidence","mean")
    ).round(4)
    breakdown.to_csv(PRED_DIR / "per_emotion_accuracy.csv")
    print("\n  Per-emotion accuracy:")
    print(breakdown.to_string())

    save_predictions_txt(pred_df, PRED_DIR / "test_predictions_report.txt")

    _plot_confusion(y_true, y_pred,    PLOTS_DIR / "confusion_matrix.png")
    _plot_per_class_f1(report,         PLOTS_DIR / "per_class_f1.png")
    if lang_f1:   _plot_group_f1(lang_f1,   PLOTS_DIR / "language_f1.png",   "Language Group F1")
    if gender_f1: _plot_group_f1(gender_f1, PLOTS_DIR / "gender_f1.png",     "Gender Group F1")
    if accent_f1: _plot_group_f1(accent_f1, PLOTS_DIR / "accent_f1.png",     "Accent Group F1")
    _plot_confidence_dist(pred_df,     PLOTS_DIR / "confidence_distribution.png")
    _plot_reliability(y_true, probs,   PLOTS_DIR / "reliability_diagram.png")
    _plot_sample_predictions(pred_df,  PLOTS_DIR / "sample_predictions.png", n=3)

    return fairness, pred_df


def _noise(wav, snr_db):
    n = torch.randn_like(wav); s = wav.norm(p=2); ns = n.norm(p=2)+1e-8
    return wav + (s/(ns*(10**(snr_db/20))))*n

def _speed(wav, factor):
    try:
        out, _ = torchaudio.sox_effects.apply_effects_tensor(
            wav, SR, [["rate", str(int(SR*factor))], ["rate", str(SR)]])
        return out
    except Exception:
        return T.Resample(int(SR*factor), SR)(wav)

def _pitch(wav, semitones):
    try:
        out, _ = torchaudio.sox_effects.apply_effects_tensor(
            wav, SR, [["pitch", str(semitones*100)], ["rate", str(SR)]])
        return out
    except Exception:
        return wav


def eval_robustness(model, test_df, max_samples=200):
    df = _ensure_label_id(test_df).head(max_samples).copy()
    conditions = {
        "clean":      lambda w: w,
        "noise_20dB": lambda w: _noise(w, 20.0),
        "noise_10dB": lambda w: _noise(w, 10.0),
        "speed_slow": lambda w: _speed(w, 0.9),
        "speed_fast": lambda w: _speed(w, 1.1),
        "pitch_up":   lambda w: _pitch(w, +2),
        "pitch_down": lambda w: _pitch(w, -2),
    }
    tmp = RESULTS_DIR / "tmp_rob"; tmp.mkdir(exist_ok=True)
    scores, all_rows = {}, []
    for cname, fn in conditions.items():
        y_true, y_pred = [], []
        for i, row in df.iterrows():
            wav, sr = load_audio(row["path"])
            if sr != SR: wav = T.Resample(sr, SR)(wav)
            if wav.shape[0] > 1: wav = wav.mean(0, keepdim=True)
            try: aug = fn(wav)
            except Exception: aug = wav
            tp = tmp / f"{cname}_{i}.wav"
            save_audio(str(tp), aug, SR)
            pred, _ = _predict_logits(model, str(tp))
            y_true.append(int(row["label_id"])); y_pred.append(pred)
            all_rows.append({"condition":cname,"path":row["path"],
                             "true_id":int(row["label_id"]),"pred_id":pred,
                             "correct":pred==int(row["label_id"])})
        scores[cname] = round(float(f1_score(y_true, y_pred, average="macro")), 4)
        print(f"    [robustness/{cname}]: {scores[cname]:.4f}")
    pd.DataFrame(all_rows).to_csv(PRED_DIR / "robustness_predictions.csv", index=False)
    clean = scores.get("clean", 0.0)
    robustness = {
        "scores":            scores,
        "mean_corrupted_f1": round(float(np.mean([v for k,v in scores.items() if k!="clean"])),4),
        "mean_drop":         round(float(np.mean([clean-v for k,v in scores.items() if k!="clean"])),4),
    }
    _plot_robustness(scores, PLOTS_DIR / "robustness.png")
    return robustness


def eval_explainability(model, test_df, n=8):
    df = _ensure_label_id(test_df).head(n).copy()
    exps = []
    for idx, row in df.iterrows():
        wav, sr = load_audio(row["path"])
        if sr != SR: wav = T.Resample(sr, SR)(wav)
        if wav.shape[0] > 1: wav = wav.mean(0, keepdim=True)
        wav = wav[:, :SR*10]
        mel    = torchaudio.transforms.MelSpectrogram(SR, n_mels=80)(wav)
        mel_db = torchaudio.transforms.AmplitudeToDB()(mel).squeeze(0).numpy()
        energy = mel_db.mean(0); top = np.argsort(energy)[-12:]
        ratio  = float(np.mean((energy[top]-energy.min())/(energy.max()-energy.min()+1e-8)))
        pred, probs = _predict_logits(model, row["path"])
        true_label  = ID2LABEL.get(int(row["label_id"]), "?")
        tag = "CORRECT" if pred == int(row["label_id"]) else "WRONG"
        fig, ax = plt.subplots(figsize=(9,4))
        ax.imshow(mel_db, origin="lower", aspect="auto", cmap="magma")
        for t in top: ax.axvline(t, color="cyan", alpha=0.3, lw=1.2)
        ax.set_title(
            f"[{tag}]  True: {true_label}   Pred: {ID2LABEL[pred]}"
            f"   conf={probs[pred]:.2f}   cyan=salient   {Path(row['path']).name}", fontsize=8
        )
        ax.set_xlabel("Frames"); ax.set_ylabel("Mel bins")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"explain_{idx}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
        exps.append({"path":row["path"],"true":true_label,
                     "pred":ID2LABEL[pred],"salient_ratio":round(ratio,4)})
    pd.DataFrame(exps).to_csv(PRED_DIR / "explainability.csv", index=False)
    return {"mean_salient_ratio": round(float(np.mean([e["salient_ratio"] for e in exps])),4),
            "n_examples": len(exps)}


def eval_privacy(model, train_df, test_df, max_samples=300):
    tr = _ensure_label_id(train_df).head(max_samples)
    te = _ensure_label_id(test_df).head(max_samples)
    def confs(df):
        out = []
        for _, row in df.iterrows():
            try:
                _, probs = _predict_logits(model, row["path"])
                out.append(float(np.max(probs)))
            except Exception: pass
        return np.array(out)
    tr_c = confs(tr); te_c = confs(te)
    mia_y = np.array([1]*len(tr_c)+[0]*len(te_c))
    mia_s = np.concatenate([tr_c, te_c])
    try: mia_auc = float(roc_auc_score(mia_y, mia_s))
    except Exception: mia_auc = 0.5
    spk_overlap = 0.0
    if "speaker_id" in train_df.columns and "speaker_id" in test_df.columns:
        s1 = set(train_df["speaker_id"].astype(str)); s2 = set(test_df["speaker_id"].astype(str))
        spk_overlap = len(s1 & s2) / (len(s1 | s2) + 1e-8)
    pd.DataFrame({"split":["train"]*len(tr_c)+["test"]*len(te_c),
                  "confidence":np.concatenate([tr_c,te_c])}).to_csv(
        PRED_DIR/"privacy_confidence.csv", index=False)
    fig, ax = plt.subplots(figsize=(7,4))
    ax.hist(tr_c, bins=25, alpha=0.6, label="Train"); ax.hist(te_c, bins=25, alpha=0.6, label="Test")
    ax.set_title("Privacy proxy -- confidence distribution")
    ax.set_xlabel("Confidence"); ax.legend()
    fig.tight_layout(); fig.savefig(PLOTS_DIR/"privacy_hist.png",dpi=200,bbox_inches="tight"); plt.close(fig)
    return {"membership_inference_auc":round(mia_auc,4),
            "train_confidence_mean":   round(float(tr_c.mean()),4),
            "test_confidence_mean":    round(float(te_c.mean()),4),
            "confidence_gap":          round(float(tr_c.mean()-te_c.mean()),4),
            "speaker_overlap":         round(float(spk_overlap),4)}


def run_audiotrust(stage="full", train_csv=None, test_csv=None):
    print("\n" + "="*60)
    print("  STAGE 05 -- AudioTrust Evaluation")
    print("="*60)
    train_csv = train_csv or str(DATA_DIR / "train.csv")
    test_csv  = test_csv  or str(DATA_DIR / "test.csv")
    train_df  = pd.read_csv(train_csv)
    test_df   = pd.read_csv(test_csv)
    model     = load_model(stage)

    print("\n[1/4] Fairness + Predictions...")
    fairness, pred_df = eval_fairness(model, test_df)
    print("\n[2/4] Robustness...")
    robustness = eval_robustness(model, test_df)
    print("\n[3/4] Explainability...")
    explain = eval_explainability(model, test_df)
    print("\n[4/4] Privacy...")
    privacy = eval_privacy(model, train_df, test_df)

    def N(x, hi=True): return max(0.0, min(1.0, float(x) if hi else 1.0-float(x)))
    overall = {
        "fairness_score":       round(np.mean([N(fairness["group_fairness_gamma_language"]),
                                               N(fairness["group_fairness_gamma_gender"]),
                                               N(fairness["macro_f1"]),
                                               N(fairness["ece"],False)]),4),
        "robustness_score":     round(np.mean([N(robustness["scores"].get("clean",0)),
                                               N(robustness["mean_corrupted_f1"]),
                                               N(robustness["mean_drop"],False)]),4),
        "explainability_score": round(N(explain["mean_salient_ratio"]),4),
        "privacy_score":        round(np.mean([N(abs(privacy["confidence_gap"]),False),
                                               N(abs(privacy["membership_inference_auc"]-0.5)*2,False),
                                               N(privacy["speaker_overlap"],False)]),4),
    }
    overall["audiotrust_overall"] = round(float(np.mean(list(overall.values()))),4)

    report = {"stage":stage,"fairness":fairness,"robustness":robustness,
              "explainability":explain,"privacy":privacy,"overall":overall}
    with open(RESULTS_DIR/"audiotrust_report.json","w") as f:
        json.dump(report,f,indent=2)
    pd.DataFrame([{"dimension":k.replace("_score","").title(),"score":v}
                  for k,v in overall.items()]).to_csv(RESULTS_DIR/"audiotrust_summary.csv",index=False)
    _plot_summary(overall, PLOTS_DIR/"audiotrust_summary.png")

    print("\n"+"="*60)
    print(f"  AudioTrust Overall  : {overall['audiotrust_overall']:.4f}")
    print(f"  Fairness            : {overall['fairness_score']:.4f}")
    print(f"  Robustness          : {overall['robustness_score']:.4f}")
    print(f"  Explainability      : {overall['explainability_score']:.4f}")
    print(f"  Privacy             : {overall['privacy_score']:.4f}")
    print("\n  Per-class F1:")
    for emo,v in fairness["per_class_f1"].items():
        print(f"    {emo:10s}: {v:.4f}")
    print(f"\n  Accuracy            : {fairness['accuracy']:.4f}")
    print(f"  ECE (lower=better)  : {fairness['ece']:.4f}")
    print(f"\n  Full predictions    : {PRED_DIR}/test_predictions_full.csv")
    print(f"  Human-readable TXT  : {PRED_DIR}/test_predictions_report.txt")
    print(f"  Wrong predictions   : {PRED_DIR}/test_predictions_wrong.csv")
    print(f"  Per-emotion stats   : {PRED_DIR}/per_emotion_accuracy.csv")
    print("="*60)
    return report

if __name__ == "__main__":
    run_audiotrust()
