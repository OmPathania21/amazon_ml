# Amazon ML Challenge 2026 — Business Entity Resolution

Pipeline: **clean → block → features → LightGBM (5-fold) → decision rule tuned on macro F0.5**.

## Setup (once)

Use Python 3.10–3.12.

```bash
git clone <repo-url>
cd amazon_ml
python -m venv .venv
.venv\Scripts\activate           # Windows
# source .venv/bin/activate      # Mac / Linux
pip install -r requirements.txt
```

The dataset is not in git (too big). Download and arrange it with:

```bash
python src/00_download_data.py
```

It downloads the challenge zip from Google Drive, unzips it and produces:

```
amazon_ml/data/train/train_source1.tsv ... train_ground_truth.tsv
amazon_ml/data/test/test_source1.tsv ...
```

If the download fails (Drive quota), download the zip in a browser from
https://drive.google.com/file/d/10kOaB9eVp0096S069W8a8mT9-IEm3m8f/view and run
`python src/00_download_data.py --zip path/to/the.zip`.
(`student_resource/dataset/` is also picked up if it exists instead of `data/`.)

## Run

**1. Quick trial on 5% of train (~5–10 min).** Checks everything works:

```bash
python src/01_clean.py --sample 0.05
python src/02_labels_folds.py --sample 0.05
python src/03_block.py --sample 0.05
python src/04_features.py --sample 0.05
python src/05_train.py --sample 0.05
python src/06_tune.py --sample 0.05
```

**2. Full training run.** Same commands without `--sample`:

```bash
python src/01_clean.py
python src/02_labels_folds.py
python src/03_block.py
python src/04_features.py
python src/05_train.py
python src/06_tune.py
```

Outputs go to `work/sample0.05/` or `work/full/`. Every step skips work that already
exists; add `--force` to redo a step. Set `ER_JOBS=8` to limit CPU threads.

## What each step prints (send these numbers back)

| Step | Checkpoint |
|---|---|
| 01_clean | record counts + example cleaned rows |
| 02_labels_folds | S1 count, singleton rate, S1 per fold |
| 03_block | **BLOCKING RECALL** top-1..top-5, per country |
| 05_train | per fold: AUC, logloss, best iteration; top features |
| 06_tune | **macro F0.5** (overall / per country / singletons), chosen rule |

## Files

| File | Role |
|---|---|
| `src/config.py` | paths, settings (TOP_K, MAX_DF, LightGBM params, sample mode) |
| `src/normalize.py` | all cleaning rules for names and addresses (`python src/normalize.py` shows examples) |
| `src/features.py` | pair features (rapidfuzz fuzzy scores, exact matches, relative/rank features) |
| `src/metrics.py` | macro F0.5, one-owner rule, threshold / expected-F0.5 decision rules |
| `src/01_clean.py` | raw TSV → cleaned parquet per source |
| `src/02_labels_folds.py` | ground truth lookup + 5 folds grouped by Source-1 entity |
| `src/03_block.py` | TF-IDF token blocking: top-K S1 per S2/S3 record, per country |
| `src/04_features.py` | features for every candidate pair |
| `src/05_train.py` | LightGBM 5-fold, out-of-fold predictions, models |
| `src/06_tune.py` | decision rule search + validation macro F0.5 |
