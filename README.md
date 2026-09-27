# Amazon ML Challenge 2026 — Business Entity Resolution

Pipeline: **clean → block → features → LightGBM (5-fold) → decision rule tuned on macro F0.5**.

## Setup (once)

Use Python 3.13 (results were produced with Python 3.13.1 on Windows 11; `requirements.txt` pins the exact package versions).

Run every command from this folder (the one containing `src/`, `README.md` and
`requirements.txt`; in the submission zip that is `code/business_entity_resolution/`).

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows
# source .venv/bin/activate      # Mac / Linux
pip install -r requirements.txt
```

**Data.** Put the challenge dataset next to `src/` as:

```
data/train/train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
data/test/test_source1.tsv    test_source2.tsv   test_source3.tsv
```

Either copy the organisers' `dataset/train` and `dataset/test` folders into `data/`, or
let the script download the challenge zip from Google Drive and arrange it:

```bash
python src/00_download_data.py                      # download + unzip + arrange
python src/00_download_data.py --zip path/to/the.zip  # zip already downloaded
```

(A `student_resource/dataset/` folder next to `src/` is also picked up instead of `data/`.)
All outputs are written inside this folder: `work/` (intermediate files) and `output/`
(`matching_results.tsv`, `candidate_pairs.tsv`).

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

**3. Test set → submission.** After the full training run:

```bash
python src/01_clean.py --split test
python src/03_block.py --split test
python src/04_features.py --split test
python src/07_predict.py
python src/08_write_submission.py
```

This writes `output/matching_results.tsv` (upload this) and `output/candidate_pairs.tsv`,
then runs the organisers' validator (must print `PASS`).
Quick first submission before the full training finishes: run the three `--split test`
commands, then `python src/07_predict.py --models sample0.05` and `python src/08_write_submission.py`.

**4. Error analysis and final package.**

```bash
python src/09_error_analysis.py                  # TP/FP/FN, causes, examples -> work/full/error_analysis.md
python src/10_make_zip.py --team "<team name>"   # -> <team>_submission.zip
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
| `src/00_download_data.py` | download the challenge zip from Google Drive → `data/train`, `data/test` |
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
| `src/07_predict.py` | score test pairs with the 5 fold models (averaged) |
| `src/08_write_submission.py` | one-owner rule + decision rule → both TSVs → validator |
| `src/09_error_analysis.py` | pair-level TP/FP/FN, entity outcomes, blocking quality, error causes, examples |
| `src/10_make_zip.py` | builds `<team>_submission.zip` in the required layout |
