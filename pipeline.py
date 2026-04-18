import os, re, random, warnings
os.environ["DATASETS_AUDIO_BACKEND"] = "soundfile"
os.environ["DISABLE_TORCHCODEC"]     = "1"

import numpy  as np
import pandas as pd
import soundfile as sf
import librosa
from pathlib  import Path
from tqdm     import tqdm
from sklearn.model_selection import train_test_split
import env  
from env import IEMOCAP_ROOT

warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR         = Path(__file__).parent
DATA_DIR         = BASE_DIR / "data"
HINDI_DIR        = DATA_DIR / "hindi"
IEMOCAP_ROOT     = DATA_DIR / "IEMOCAP_full_release"

SAMPLE_RATE      = 16000
SAMPLES_PER_LANG = 3200
RANDOM_SEED      = 42
MIN_DURATION     = 0.3

LABEL2ID = {"angry": 0, "happy": 1, "neutral": 2, "sad": 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

FOLDER_NUM_TO_4 = {
    "1": "angry",   "2": "angry",   "3": "sad",   "4": "happy",
    "5": "neutral", "6": "sad",     "7": "angry", "8": "happy",
}

EMO_STR = {
    "angry":   "angry",  "anger":     "angry",  "ang":  "angry",
    "disgust": "angry",  "dis":       "angry",  "sarcastic": "angry", "sar": "angry",
    "fear":    "sad",    "fea":       "sad",
    "sad":     "sad",    "sadness":   "sad",
    "happy":   "happy",  "happiness": "happy",  "hap":  "happy",
    "excited": "happy",  "exc":       "happy",  "surprise": "happy",  "sur": "happy",
    "neutral": "neutral","neu":       "neutral",
}

IEMOCAP_EMO = {
    "ang": "angry",  "angry":     "angry",
    "hap": "happy",  "happiness": "happy",
    "exc": "happy",  "excited":   "happy",
    "neu": "neutral","neutral":   "neutral",
    "sad": "sad",    "sadness":   "sad",
}

def _norm(raw):
    if not raw:
        return None
    c = str(raw).strip().lower().rstrip(".")
    if c in EMO_STR:
        return EMO_STR[c]
    first = re.split(r"[_\-\s/]", c)[0]
    if first in EMO_STR:
        return EMO_STR[first]
    if len(c) >= 3 and c[:3] in EMO_STR:
        return EMO_STR[c[:3]]
    return None

def _gender(stem):
    s = stem.upper()
    if re.match(r"^F\d", s) or "_F_" in s or "FEMALE" in s:
        return "female"
    if re.match(r"^M\d", s) or "_M_" in s or "MALE"   in s:
        return "male"
    return "unknown"

def _save_wav(audio, orig_sr, out_dir, fname):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if audio.ndim > 1:
        audio = audio[:, 0]

    if orig_sr != SAMPLE_RATE:
        audio = librosa.resample(
            audio.astype(np.float32),
            orig_sr=orig_sr,
            target_sr=SAMPLE_RATE
        )

    mx = np.abs(audio).max()
    if mx > 1e-6:
        audio = audio / mx * 0.95

    p = str(out_dir / fname)
    sf.write(p, audio.astype(np.float32), SAMPLE_RATE)
    return p

def _strat_sample(df, n, seed=RANDOM_SEED):
    
    per_class = n // len(LABEL2ID)
    parts = []

    for emo in LABEL2ID.keys():
        grp = df[df["emotion"] == emo]
        if len(grp) == 0:
            continue
        parts.append(grp.sample(min(len(grp), per_class), random_state=seed))

    sampled = pd.concat(parts, ignore_index=True)

    need = n - len(sampled)
    if need > 0:
        used_paths = set(sampled["path"].tolist())
        rest = df[~df["path"].isin(used_paths)]
        if len(rest) > 0:
            extra = rest.sample(min(need, len(rest)), random_state=seed)
            sampled = pd.concat([sampled, extra], ignore_index=True)

    return sampled.reset_index(drop=True)

def diagnose_hindi():
    base = HINDI_DIR
    wavs = list(base.rglob("*.wav")) + list(base.rglob("*.WAV"))
    print(f"\n[Diagnose] {len(wavs)} WAVs under {base}\nFirst 30:")
    for w in sorted(wavs)[:30]:
        emo = None
        for part in reversed(list(w.parts[:-1])):
            p = part.strip()
            if p in FOLDER_NUM_TO_4:
                emo = FOLDER_NUM_TO_4[p]
                break
            e = _norm(p)
            if e:
                emo = e
                break
        print(f"  {w.relative_to(base)}  ->  {emo}")

def load_hindi_iitkgp(max_clips=SAMPLES_PER_LANG):
    base = HINDI_DIR
    if not base.exists():
        raise FileNotFoundError(f"[Hindi] Not found: {base}")

    root = base
    for sub in sorted(base.rglob("*")):
        if sub.is_dir() and sub.name.strip() == "1":
            cand = sub.parent
            n_num = sum(
                1 for d in cand.iterdir()
                if d.is_dir() and d.name.strip().isdigit()
            )
            if n_num >= 5:
                root = cand
                print(f"[Hindi] Numbered layout root: {root}")
                break

    wavs = list(root.rglob("*.wav")) + list(root.rglob("*.WAV"))
    print(f"[Hindi] {len(wavs)} WAV files found")

    save_dir = DATA_DIR / "hindi_processed"
    rows = []
    skip_emo = 0
    skip_err = 0
    skip_short = 0

    for wav in tqdm(wavs, desc="IITKGP Hindi"):
        emo = None
        for part in reversed(list(wav.parts[:-1])):
            p = part.strip()
            if p in FOLDER_NUM_TO_4:
                emo = FOLDER_NUM_TO_4[p]
                break
            e = _norm(p)
            if e:
                emo = e
                break

        if emo is None:
            skip_emo += 1
            continue

        try:
            audio, sr = sf.read(str(wav), dtype="float32", always_2d=False)
        except Exception:
            skip_err += 1
            continue

        if audio.ndim > 1:
            audio = audio[:, 0]

        dur = len(audio) / sr
        if dur < MIN_DURATION:
            skip_short += 1
            continue

        stem = wav.stem
        rows.append({
            "path":       _save_wav(audio, sr, save_dir, f"hi_{emo}_{stem}.wav"),
            "emotion":    emo,
            "language":   "hindi",
            "gender":     _gender(stem),
            "accent":     "hindi_iitkgp",
            "speaker_id": f"iitkgp_{wav.parent.name.strip()}",
            "duration":   round(dur, 2),
        })

    print(f"[Hindi] Skipped: no_emo={skip_emo} err={skip_err} short={skip_short}")
    print(f"[Hindi] Loaded: {len(rows)} clips")

    if not rows:
        raise RuntimeError("[Hindi] 0 clips loaded! Run: python dataset.py --diagnose")

    df = pd.DataFrame(rows)
    print(f"[Hindi] Columns: {list(df.columns)}")
    print(f"[Hindi] Emotion counts:\n{df['emotion'].value_counts().to_string()}\n")

    if len(df) > max_clips:
        df = _strat_sample(df, max_clips)
    else:
        df = df.reset_index(drop=True)

    print(f"[Hindi] Final clips: {len(df)}")
    print(f"[Hindi] Columns after sample: {list(df.columns)}")
    return df

def load_english_iemocap(max_clips=SAMPLES_PER_LANG):
    root = IEMOCAP_ROOT
    if not root.exists():
        raise FileNotFoundError(f"[IEMOCAP] Not found: {root}")
    print(f"[IEMOCAP] Loading from: {root}")

    save_dir = DATA_DIR / "english_processed"
    rows = []

    for sess_n in range(1, 6):
        sess_dir = root / f"Session{sess_n}"
        if not sess_dir.exists():
            print(f"  [IEMOCAP] WARNING: Session{sess_n} missing")
            continue

        emo_map = {}
        for txt in (sess_dir / "dialog").rglob("*.txt"):
            for line in txt.read_text(errors="replace").splitlines():
                if not line.startswith("["):
                    continue
                parts = line.split("\t")
                if len(parts) < 3:
                    continue
                utt_id = parts[1].strip()
                raw_e = parts[2].strip().lower()
                emo = IEMOCAP_EMO.get(raw_e[:3], IEMOCAP_EMO.get(raw_e))
                if emo:
                    emo_map[utt_id] = emo

        sess_count = 0
        for wav in (sess_dir / "sentences" / "wav").rglob("*.wav"):
            utt_id = wav.stem
            emotion = emo_map.get(utt_id)
            if emotion is None:
                continue

            try:
                audio, sr = sf.read(str(wav), dtype="float32", always_2d=False)
            except Exception:
                continue

            if audio.ndim > 1:
                audio = audio[:, 0]

            if len(audio) / sr < MIN_DURATION:
                continue

            m = re.search(r"Ses\d+(F|M)_", utt_id)
            gender = "female" if (m and m.group(1) == "F") else "male"

            rows.append({
                "path":       _save_wav(audio, sr, save_dir, f"en_s{sess_n}_{utt_id}.wav"),
                "emotion":    emotion,
                "language":   "english",
                "gender":     gender,
                "accent":     "american",
                "speaker_id": f"iemocap_S{sess_n}_{gender}",
                "duration":   round(len(audio) / sr, 2),
            })
            sess_count += 1

        print(f"  [IEMOCAP] Session{sess_n}: {sess_count} clips")

    print(f"[IEMOCAP] Total 4-class clips: {len(rows)}")
    if not rows:
        raise RuntimeError("[IEMOCAP] 0 clips loaded!")

    df = pd.DataFrame(rows)
    print(f"[IEMOCAP] Columns: {list(df.columns)}")
    print(f"[IEMOCAP] Emotion counts:\n{df['emotion'].value_counts().to_string()}\n")

    if len(df) > max_clips:
        df = _strat_sample(df, max_clips)
    else:
        df = df.reset_index(drop=True)

    print(f"[IEMOCAP] Final clips: {len(df)}")
    print(f"[IEMOCAP] Columns after sample: {list(df.columns)}")
    return df

def run_dataset_pipeline():
    print("\n" + "=" * 60)
    print("  FairHindiSER Dataset Pipeline")
    print("  3200 Hindi + 3200 English = 6400 total")
    print("=" * 60 + "\n")

    hi_df = load_hindi_iitkgp(max_clips=SAMPLES_PER_LANG)
    en_df = load_english_iemocap(max_clips=SAMPLES_PER_LANG)

    print(f"\n[Pipeline] hi_df type={type(hi_df).__name__}  shape={hi_df.shape}  cols={list(hi_df.columns)}")
    print(f"[Pipeline] en_df type={type(en_df).__name__}  shape={en_df.shape}  cols={list(en_df.columns)}")

    for label, obj in [("hi_df", hi_df), ("en_df", en_df)]:
        if not isinstance(obj, pd.DataFrame):
            raise TypeError(f"{label} is {type(obj).__name__}, not DataFrame")
        for col in ["path", "emotion", "language", "gender"]:
            if col not in obj.columns:
                raise RuntimeError(f"{label} missing column '{col}'. Has: {list(obj.columns)}")

    n = min(len(hi_df), len(en_df), SAMPLES_PER_LANG)
    hi_df = hi_df.sample(n, random_state=RANDOM_SEED).reset_index(drop=True)
    en_df = en_df.sample(n, random_state=RANDOM_SEED).reset_index(drop=True)
    df = pd.concat([hi_df, en_df], ignore_index=True)

    bad = df[~df["emotion"].isin(LABEL2ID)]
    if len(bad) > 0:
        print(f"[Pipeline] Dropping {len(bad)} unknown emotions: {bad['emotion'].unique()}")
        df = df[df["emotion"].isin(LABEL2ID)].reset_index(drop=True)

    df["label"] = df["emotion"].map(LABEL2ID).astype(int)
    df["label_id"] = df["label"]

    print("\n[Pipeline] Final distribution:")
    print(df.groupby(["language", "emotion"]).size().unstack(fill_value=0).to_string())
    print(
        f"\n[Pipeline] Total={len(df)}  "
        f"Hindi={len(df[df['language']=='hindi'])}  "
        f"English={len(df[df['language']=='english'])}"
    )

    df["strat"] = df["emotion"].astype(str) + "_" + df["language"].astype(str)

    tr, tmp = train_test_split(
        df,
        test_size=0.30,
        stratify=df["strat"],
        random_state=RANDOM_SEED
    )
    va, te = train_test_split(
        tmp,
        test_size=0.50,
        stratify=tmp["strat"],
        random_state=RANDOM_SEED
    )

    for d in [tr, va, te]:
        d.drop(columns=["strat"], inplace=True, errors="ignore")

    print(f"[Pipeline] Split: train={len(tr)}  val={len(va)}  test={len(te)}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tr.to_csv(DATA_DIR / "train.csv", index=False)
    va.to_csv(DATA_DIR / "val.csv", index=False)
    te.to_csv(DATA_DIR / "test.csv", index=False)
    print("[Pipeline] Saved train.csv  val.csv  test.csv\n")

    return tr, va, te

if __name__ == "__main__":
    import sys
    if "--diagnose" in sys.argv:
        diagnose_hindi()
    else:
        tr, va, te = run_dataset_pipeline()
        print(f"Done — train={len(tr)}  val={len(va)}  test={len(te)}")