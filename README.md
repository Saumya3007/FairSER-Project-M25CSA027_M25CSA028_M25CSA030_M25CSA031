# FairSER: Speech Emotion Recognition Project

## Hugging Face Model: https://huggingface.co/Saumya3007/spee_project_fairhindiser-clues
## Overview
FairSER is a Speech Emotion Recognition (SER) project designed to identify emotions from speech data. The project includes various stages of training, evaluation, and inference, with a focus on fairness and robustness.


## Features
- Multi-stage training pipeline with support for:
  - Zero-shot learning
  - Head-only fine-tuning
  - LoRA fine-tuning
  - CLUES debiasing
  - Full encoder unfreezing
- Hyperparameter optimization using Optuna
- Evaluation with AudioTrust metrics
- Direct inference for single audio files or batches


## Dataset Information

The FairSER project results are done on the following datasets:

1. **IMEOCAP Dataset**:
   - Download the dataset online.
   - Place the dataset files in the `data/` folder.

2. **IITKGP Hindi Dataset**:
   - Download the dataset from the [AI4Bharat Kosh website](https://ai4bharat.org/kosh).
   - Place the dataset files in the `data/` folder.

Ensure that the datasets are properly organized in the `data/` directory before running the pipeline.

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd FairSER(Speech Emotion Recognition Project)
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Entrypoint: `main.py`
The main pipeline can be executed using `main.py`. It supports various stages of training and evaluation. Use the following command:
```bash
python main.py [options]
```

#### Options:
- `--skip-dataset`: Skip dataset preparation
- `--skip-zero`: Skip zero-shot learning stage
- `--skip-head`: Skip head-only fine-tuning
- `--skip-lora`: Skip LoRA fine-tuning
- `--skip-clues`: Skip CLUES debiasing
- `--skip-optuna`: Skip hyperparameter optimization
- `--skip-full`: Skip full encoder unfreezing
- `--skip-eval`: Skip evaluation

### Direct Inference: `inference.py`
For direct inference, use `inference.py`. It supports single audio files, batches, and evaluation with ground truth.

#### Examples:
1. Predict emotion for a single audio file:
   ```bash
   python inference.py --file <path-to-audio>
   ```
2. Predict emotions for a batch of audio files:
   ```bash
   python inference.py --folder <path-to-folder>
   ```



## Project Summary

FairHindiSER is a speech emotion recognition pipeline built on **wav2vec2‑base** with staged adaptation (head‑only, LoRA, CLUES debiasing, gradual full unfreezing) and **AudioTrust** evaluation of fairness, robustness, explainability and privacy.

The model is trained on a **balanced 50/50 Hindi–English corpus** (IITKGP Hindi + IEMOCAP English, 4 emotions), and evaluated with group metrics across language, gender and accent.[file:50][file:58]

---

## 1. Features

- SSL backbone: `facebook/wav2vec2-base` with masked‑mean pooling and a FairSER MLP head.[file:52]
- Training stages:
  - Stage 00 – Dataset pipeline (builds `train.csv`, `val.csv`, `test.csv`).[file:50]
  - Stage 01 – Zero‑shot baseline (frozen SSL, random head).[file:43]
  - Stage 02 – Head‑only fine‑tuning (SSL frozen).[file:44]
  - Stage 02b – LoRA fine‑tuning (Q/V adapters, focal loss).[file:46][file:52]
  - Stage 03 – CLUES contrastive debiasing on LoRA backbone.[file:47]
  - Stage 03b – Optuna hyperparameter search.[file:51]
  - Stage 04 – Gradual full‑encoder + CNN unfreezing.[file:45]
  - Stage 05 – AudioTrust evaluation (fairness, robustness, explainability, privacy).[file:58][file:41]
- Evaluation artefacts:
  - Per‑class F1 and confusion matrix.
  - Group F1 by language, gender, accent.
  - Robustness curves under noise/speed/pitch perturbations.
  - Calibration, confidence histograms, basic privacy proxies.[file:58][file:41]

---

    
##  Project Structure

```text
FairSer/
├── main.py                 # Orchestrates the full pipeline (Stages 00–05)
├── env.py                  # Environment variables, paths (e.g., data roots)
├── requirements.txt        # Python dependencies
│
├── pipeline.py             # Stage 00: dataset building (Hindi + IEMOCAP)
├── dataset.py              # SERDataset + collate_fn for training/eval
├── models.py               # FairSERModel, LoRA layers, param groups
├── losses.py               # (If used) extra loss utilities
├── train_utils.py          # evaluate(), DEVICE, common helpers
│
├── train_zero_shot.py      # Stage 01: zero-shot baseline (frozen SSL, random head)
├── train_head.py           # Stage 02: head-only fine-tuning (backbone frozen)
├── train_lora.py           # Stage 02b: LoRA fine-tuning (q_proj/v_proj, layers 8–11)
├── train_clues_lora.py     # Stage 03: CLUES contrastive debiasing on LoRA backbone
├── optuna_tune.py          # Stage 03b: Optuna HPO for full-unfreeze stage
├── train_full_unfreeze.py  # Stage 04: gradual encoder + CNN unfreezing
│
├── inference.py            # Inference helpers, model loader, single-file prediction
├── evaluate.py             # Stage 05: AudioTrust evaluation + plots/reports
│
├── data/                   # Built/processed data & CSV splits
│   ├── hindi/              # Raw Hindi IITKGP corpus (input, not tracked)
│   ├── IEMOCAP_full_release/   # Raw IEMOCAP (input, not tracked)
│   ├── hindi_processed/    # 16kHz normalized Hindi WAVs (auto-generated)
│   ├── english_processed/  # 16kHz normalized English WAVs (auto-generated)
│   ├── train.csv           # Final train split (balanced 50/50 Hindi–English)
│   ├── val.csv             # Final validation split
│   └── test.csv            # Final test split
│
├── results/
│   ├── zero_shot_results.json      # Stage 01 metrics (macro F1, gaps, history)
│   ├── head_results.json           # Stage 02 metrics
│   ├── lora_results.json           # Stage 02b metrics
│   ├── clues_lora_results.json     # Stage 03 metrics
│   ├── optuna_results.json         # Stage 03b best hyperparameters
│   ├── full_results.json           # Stage 04 metrics (final SER model)
│   │
│   ├── checkpoints/
│   │   ├── zero_shot_best.pt       # (optional) zero-shot snapshot
│   │   ├── head_best.pt            # Best head-only model (backbone frozen)
│   │   ├── lora_best.pt            # Best LoRA model
│   │   ├── clues_lora_best.pt      # Best CLUES-debiased LoRA model
│   │   └── full_best.pt            # Best fully-unfrozen FairHindiSER model
│   │
│   ├── plots/
│   │   ├── confusion_matrix.png           # Test confusion matrix
│   │   ├── per_class_f1.png               # Per-class F1 (angry, happy, neutral, sad)
│   │   ├── language_f1.png                # Macro F1 by language group
│   │   ├── gender_f1.png                  # Macro F1 by gender group
│   │   ├── accent_f1.png                  # Macro F1 by accent group
│   │   ├── robustness.png                 # Macro F1 under noise/speed/pitch
│   │   ├── confidence_distribution.png    # Correct vs incorrect confidence hist
│   │   ├── reliability_diagram.png        # Calibration curve (ECE)
│   │   ├── privacy_hist.png               # Train vs test confidence hist
│   │   ├── sample_predictions.png         # Example spectrograms, correct/wrong
│   │   ├── explain_0.png ... explain_7.png# Saliency-like spectrogram overlays
│   │   └── audiotrust_summary.png         # AudioTrust bar-chart summary
│   │
│   └── predictions/
│       ├── test_predictions_full.csv      # All test predictions + metadata
│       ├── test_predictions_wrong.csv     # Misclassified subset
│       ├── test_predictions_correct.csv   # Correct subset
│       ├── per_emotion_accuracy.csv       # Per-class accuracy + mean confidence
│       ├── robustness_predictions.csv     # Predictions per robustness condition
│       ├── explainability.csv             # CLUES-style saliency stats
│       ├── privacy_confidence.csv         # Train vs test confidence scores
│       ├── inference_errors.csv           # (if any) errors during eval
│       └── test_predictions_report.txt    # Human-readable prediction report
│
└── fairser_cvpr_report.pdf   # (Optional) PDF report built on this pipeline


## Results
Results are stored in the `results/` directory:
- `all_models_comparison.json`: Comparison of all models
- `predictions/`: Contains prediction results and evaluation metrics
- `plots/`: Visualization of results
- `checkpoints/`:checkpoints
