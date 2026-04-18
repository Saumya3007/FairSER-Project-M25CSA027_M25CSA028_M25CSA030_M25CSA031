import os, sys, argparse, json
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
from pathlib import Path
from transformers import Wav2Vec2Processor
from models import FairSERModel
from env import load_audio

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SR       = 16000
EMOTIONS = {0: "angry", 1: "happy", 2: "neutral", 3: "sad"}
CKPT_DIR = Path(__file__).parent / "results" / "checkpoints"


def load_model(stage: str = "full") -> FairSERModel:
    ckpt_map = {
        "full":  CKPT_DIR / "full_best.pt",
        "clues": CKPT_DIR / "clues_lora_best.pt",
        "lora":  CKPT_DIR / "lora_best.pt",
        "head":  CKPT_DIR / "head_best.pt",
    }
    ckpt = ckpt_map[stage]
    assert ckpt.exists(), f"Checkpoint not found: {ckpt}"
    model = FairSERModel().to(DEVICE)
    state = torch.load(ckpt, map_location=DEVICE, weights_only=True)
    sd    = state.get("model_state_dict", state)
    model.load_state_dict(sd)
    model.eval()
    print(f"  [Inference] Loaded {stage} checkpoint: {ckpt.name}")
    return model

def preprocess_audio(audio_path: str):
   
    from dataset import get_feature_extractor, SAMPLE_RATE, MAX_SAMPLES

    extractor = get_feature_extractor()
    wav, sr = load_audio(audio_path)
    if sr != SAMPLE_RATE:
        wav = T.Resample(sr, SAMPLE_RATE)(wav)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    wav = wav.squeeze(0)
    if wav.shape[0] > MAX_SAMPLES:
        wav = wav[:MAX_SAMPLES]
    wav_np = wav.numpy().astype(np.float32)

    feat = extractor(
        wav_np,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
        max_length=MAX_SAMPLES,
        truncation=True,
    )
    x = feat.input_values.to(DEVICE)  
    attn = (x != 0).long()
    return x, attn


def predict_single(model, audio_path: str) -> dict:
    x, attn = preprocess_audio(audio_path)
    with torch.no_grad():
        logits = model(x, attn)
        _T_file = Path(__file__).parent / 'results' / 'temperature.txt'
        if _T_file.exists():
            _T = float(_T_file.read_text().strip())
            logits = logits / _T
        probs  = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    pred = int(np.argmax(probs))
    return {
        "file":              Path(audio_path).name,
        "predicted_emotion": EMOTIONS[pred],
        "confidence":        round(float(probs[pred]), 4),
        "all_probabilities": {e: round(float(probs[i]), 4) for i, e in EMOTIONS.items()},
    }


def predict_batch(model, folder_or_csv: str) -> list:
    p = Path(folder_or_csv)
    if p.suffix == ".csv":
        import pandas as pd
        paths = pd.read_csv(p)["path"].tolist()
    else:
        exts  = ["*.wav","*.flac","*.mp3","*.ogg","*.m4a","*.opus"]
        paths = [str(f) for ext in exts for f in p.glob(ext)]
    results = []
    for ap in sorted(paths):
        try:
            r = predict_single(model, str(ap))
            results.append(r)
            print(f"  {r['file']:45s}  ->  {r['predicted_emotion']:8s}  ({r['confidence']:.2%})")
        except Exception as e:
            print(f"  SKIP {ap}: {e}")
    return results


def predict_with_groundtruth(model, csv_path: str) -> None:
    
    import pandas as pd
    from sklearn.metrics import f1_score, classification_report

    df        = pd.read_csv(csv_path)
    label_col = "label_id" if "label_id" in df.columns else "label"
    emo_map   = {0:"angry",1:"happy",2:"neutral",3:"sad"}
    rev_map   = {v:k for k,v in emo_map.items()}

    rows = []
    print(f"\n  {'#':>5}  {'Filename':42s}  {'True':10s}  {'Predicted':10s}  {'Conf':6s}  OK?")
    print("  " + "-" * 82)
    for idx, (_, row) in enumerate(df.iterrows()):
        try:
            r        = predict_single(model, row["path"])
            true_id  = int(row[label_col])
            true_emo = emo_map.get(true_id, str(true_id))
            ok       = "YES" if r["predicted_emotion"] == true_emo else "NO "
            print(
                f"  {idx+1:>5}  {Path(row['path']).name:42s}  "
                f"{true_emo:10s}  {r['predicted_emotion']:10s}  "
                f"{r['confidence']:.3f}  {ok}"
            )
            rows.append({**r, "true_emotion":true_emo, "true_label_id":true_id,
                         "correct":r["predicted_emotion"]==true_emo,
                         "language":row.get("language","-"), "gender":row.get("gender","-")})
        except Exception as e:
            print(f"  SKIP {row['path']}: {e}")

    if not rows:
        return

    out    = pd.DataFrame(rows)
    acc    = out["correct"].mean()
    preds  = [rev_map.get(r["predicted_emotion"],-1) for r in rows]
    truths = [r["true_label_id"] for r in rows]
    mf1    = f1_score(truths, preds, average="macro")

    print("\n" + "="*60)
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Macro F1  : {mf1:.4f}")
    print("\n" + classification_report(truths, preds,
          target_names=["angry","happy","neutral","sad"]))

    out_dir = Path(__file__).parent / "results" / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "manual_eval.csv", index=False)
    print(f"  CSV saved -> {out_dir / 'manual_eval.csv'}")

    txt_path = out_dir / "manual_eval_report.txt"
    lines = []
    lines.append("=" * 80)
    lines.append("  FairSER Manual Evaluation -- Ground Truth vs Predicted")
    lines.append("=" * 80)
    lines.append(f"  CSV input     : {csv_path}")
    lines.append(f"  Total samples : {len(out)}")
    lines.append(f"  Correct       : {out['correct'].sum()}  ({acc*100:.1f}%)")
    lines.append(f"  Wrong         : {(~out['correct']).sum()}  ({(1-acc)*100:.1f}%)")
    lines.append(f"  Macro F1      : {mf1:.4f}")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"  {'#':>5}  {'Filename':42s}  {'True':10s}  {'Predicted':10s}  {'Conf':6s}  {'OK?':5s}  {'Language':10s}  Gender")
    lines.append("  " + "-" * 112)
    for i, r in enumerate(rows):
        ok = "YES" if r["correct"] else "NO "
        lines.append(
            f"  {i+1:>5}  {str(r['file']):42s}  "
            f"{str(r['true_emotion']):10s}  {str(r['predicted_emotion']):10s}  "
            f"{r['confidence']:.3f}  {ok:5s}  {str(r.get('language','-')):10s}  {r.get('gender','-')}"
        )
    lines.append("")
    lines.append("=" * 80)
    lines.append("  PER-EMOTION ACCURACY")
    lines.append("=" * 80)
    lines.append(f"  {'Emotion':12s}  {'Total':7s}  {'Correct':9s}  {'Accuracy':10s}  Mean Conf")
    lines.append("  " + "-" * 60)
    for emo, sdf in out.groupby("true_emotion"):
        lines.append(
            f"  {str(emo):12s}  {len(sdf):7d}  {sdf['correct'].sum():9d}"
            f"  {sdf['correct'].mean():10.4f}  {sdf['confidence'].mean():.4f}"
        )
    wrong = out[~out["correct"]]
    lines.append("")
    lines.append("=" * 80)
    lines.append(f"  MISCLASSIFIED SAMPLES ({len(wrong)} total)")
    lines.append("=" * 80)
    lines.append(f"  {'#':>5}  {'Filename':42s}  {'True':10s}  {'Predicted':10s}  {'Conf':6s}  Language")
    lines.append("  " + "-" * 95)
    for i, (_, r) in enumerate(wrong.iterrows()):
        lines.append(
            f"  {i+1:>5}  {str(r['file']):42s}  "
            f"{str(r['true_emotion']):10s}  {str(r['predicted_emotion']):10s}  "
            f"{r['confidence']:.3f}  {str(r.get('language','-'))}"
        )
    lines.append("")
    lines.append("=" * 80)
    lines.append("  END OF REPORT")
    lines.append("=" * 80)
    with open(txt_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  TXT saved -> {txt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FairSER Inference")
    parser.add_argument("--input",  required=True,
        help="Single audio file | folder of audio files | CSV with path column")
    parser.add_argument("--stage",  default="full", choices=["full","clues","lora"])
    parser.add_argument("--eval",   action="store_true",
        help="CSV with ground-truth labels: print true vs predicted table")
    parser.add_argument("--output", default=None, help="Save JSON results here")
    args = parser.parse_args()

    model = load_model(args.stage)

    if args.eval:
        predict_with_groundtruth(model, args.input)
    elif Path(args.input).is_dir() or args.input.endswith(".csv"):
        results = predict_batch(model, args.input)
        if args.output:
            with open(args.output,"w") as f: json.dump(results,f,indent=2)
            print(f"  JSON saved -> {args.output}")
    else:
        result = predict_single(model, args.input)
        print("\n  === Prediction ===")
        print(f"  File              : {result['file']}")
        print(f"  Predicted emotion : {result['predicted_emotion']}")
        print(f"  Confidence        : {result['confidence']:.4f}")
        print("  All probabilities :")
        for emo, prob in result["all_probabilities"].items():
            bar = "#" * int(prob * 30)
            print(f"    {emo:10s}: {prob:.4f}  {bar}")
        if args.output:
            with open(args.output,"w") as f: json.dump(result,f,indent=2)
            print(f"  JSON saved -> {args.output}")
