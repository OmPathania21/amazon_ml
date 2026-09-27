"""Shared paths, settings and helpers for every pipeline step."""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _default_data_dir():
    for cand in (ROOT / "data", ROOT / "student_resource" / "dataset"):
        if (cand / "train").is_dir():
            return cand
    return ROOT / "data"


DATA_DIR = Path(os.environ.get("ER_DATA", _default_data_dir()))
WORK_ROOT = Path(os.environ.get("ER_WORK", ROOT / "work"))

N_JOBS = int(os.environ.get("ER_JOBS", os.cpu_count() or 4))
SEED = 42
N_FOLDS = 5

# ---- blocking (v2: three passes) ----
TOP_K = 5                 # (used only in reports)
PASS_K = {"combo": 3, "name": 3, "addr": 3}   # S1 candidates per S2/S3 record, per pass
PRUNE_RATIO = 0.75        # keep a non-top candidate only if score >= ratio * best of that pass
MAX_DF = 400              # minimum document-frequency cap for blocking tokens
MAX_DF_FRAC = 0.003       # cap grows with the country's S1 size (0.3% of its S1 records)
BLOCK_CHUNK = 50_000      # S2/S3 records per sparse matmul chunk

# ---- features ----
FEAT_CHUNK = 1_000_000    # candidate pairs per feature part file

# ---- training ----
MAX_TRAIN_ROWS = 6_000_000   # rows sampled per fold for training
MAX_VALID_ROWS = 1_000_000   # rows of the held-out fold used for early stopping
LGB_PARAMS = dict(
    objective="binary",
    learning_rate=0.1,
    num_leaves=127,
    min_data_in_leaf=200,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    max_bin=255,
    metric=["binary_logloss", "auc"],
    verbose=-1,
    seed=SEED,
)
NUM_BOOST_ROUND = 2000
EARLY_STOP = 50

SOURCES = (1, 2, 3)
META_COLS = ["s1", "cand", "label", "fold", "country"]


def parse_args(desc, split=False):
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--sample", type=float, default=0.0,
                   help="fraction of TRAIN Source-1 entities to use (0 = full data)")
    p.add_argument("--force", action="store_true", help="recompute even if outputs exist")
    if split:
        p.add_argument("--split", choices=["train", "test"], default="train")
    return p.parse_args()


def work_dir(args):
    name = "full" if not args.sample else f"sample{args.sample:g}"
    d = WORK_ROOT / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def raw_path(split, src):
    return DATA_DIR / split / f"{split}_source{src}.tsv"


GT_PATH = DATA_DIR / "train" / "train_ground_truth.tsv"


def read_tsv(path, **kw):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                       quoting=3, engine="c", **kw)


def id_to_key(ids):
    """'S2-12345' -> 2*10**10 + 12345 (int64). Compact join key."""
    s = pd.Series(ids, copy=False).astype(str)
    src = s.str.slice(1, 2).astype(np.int64)
    num = s.str.slice(3).astype(np.int64)
    return (src * 10**10 + num).to_numpy(np.int64)


def hash01(keys, salt):
    """Deterministic pseudo-random value in [0, 1) per key (same on every machine)."""
    x = np.asarray(keys, dtype=np.uint64) + np.uint64(salt)
    x = (x ^ (x >> np.uint64(33))) * np.uint64(0xFF51AFD7ED558CCD)
    x = (x ^ (x >> np.uint64(33))) * np.uint64(0xC4CEB9FE1A85EC53)
    x = x ^ (x >> np.uint64(33))
    return (x >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def fold_of(keys):
    return (hash01(keys, 7) * N_FOLDS).astype(np.int8)


class Timer:
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self.t = time.time()
        print(f"[..] {self.label}", flush=True)
        return self

    def __exit__(self, *a):
        print(f"[ok] {self.label}  ({time.time() - self.t:.1f}s)", flush=True)
