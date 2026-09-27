"""Step 8: apply the one-owner rule + tuned decision rule and write the submission.

Outputs (tab separated, one row per test Source-1 entity):
  output/matching_results.tsv   source1_entity_id, matched_entity_ids   <- upload this
  output/candidate_pairs.tsv    source1_entity_id, candidate_entity_ids (pairs fed to the model)
Then runs the organisers' validator.
"""
import argparse
import json
import subprocess
import sys

import numpy as np
import pandas as pd

import config as C
import metrics as M

OUT_DIR = C.ROOT / "output"


def id_lookup(d):
    parts = [pd.read_parquet(d / f"clean_test_s{s}.parquet", columns=["key", "entity_id"]) for s in (2, 3)]
    ids = pd.concat(parts, ignore_index=True)
    return pd.Index(ids.key.to_numpy()), ids.entity_id.to_numpy(dtype=object)


def write_lists(path, header, s1, pairs, cidx, cids):
    """One row per S1 entity (in test_source1 order), comma-joined candidate ids."""
    pairs = pairs.sort_values(["s1", "prob"], ascending=[True, False], kind="stable")
    keys, cand = pairs.s1.to_numpy(), pairs.cand.to_numpy()
    starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]]) if len(keys) else np.array([], int)
    ends = np.r_[starts[1:], len(keys)]
    joined = {}
    block = 100_000
    for g in range(0, len(starts), block):
        lo, hi = starts[g], ends[min(g + block, len(starts)) - 1]
        names = cids[cidx.get_indexer(cand[lo:hi])]
        for a, b in zip(starts[g:g + block], ends[g:g + block]):
            joined[keys[a]] = ",".join(names[a - lo:b - lo])
    lists = pd.Series([joined.get(k, "") for k in s1.key.to_numpy()])
    out = pd.DataFrame({header[0]: s1.entity_id.to_numpy(), header[1]: lists.to_numpy()})
    out.to_csv(path, sep="\t", index=False, lineterminator="\n")
    nonempty = (lists != "").sum()
    print(f"   {path.name}: {len(out):,} rows, {nonempty:,} non-empty, {len(pairs):,} ids")


def find_validator():
    for p in (C.DATA_DIR / "validate_submission.py", C.ROOT / "data" / "validate_submission.py",
              C.ROOT / "student_resource" / "utils" / "validate_submission.py"):
        if p.is_file():
            return p
    return None


def main():
    p = argparse.ArgumentParser(description="Step 8: write submission files")
    p.add_argument("--decision", default=None,
                   help="run folder whose decision.json to use (default: same as the models)")
    args = p.parse_args()

    d = C.WORK_ROOT / "full"
    models_run = (d / "test_pred_models.txt").read_text().strip()
    dec = json.loads((C.WORK_ROOT / (args.decision or models_run) / "decision.json").read_text())
    print(f"decision rule: {dec['method']} param={dec['param']} (validation F0.5 {dec['val_f05']:.5f})")
    if dec.get("per_country"):
        print(f"per-country rules: {dec['per_country']}  (other countries use the global rule)")

    pred = pd.read_parquet(d / "test_pred.parquet")
    s1 = pd.read_parquet(d / "clean_test_s1.parquet", columns=["key", "entity_id"])
    cidx, cids = id_lookup(d)

    best = M.assign_best(pred)
    chosen = M.apply_decision(best, dec)
    print(f"pairs: candidates {len(pred):,}  after one-owner {len(best):,}  matched {len(chosen):,}")
    per = chosen.groupby("s1").size().reindex(s1.key.to_numpy(), fill_value=0)
    print(f"S1 with no match: {(per == 0).mean():.4f}   mean matches: {per.mean():.3f}")
    print("matches per S1 by country:",
          chosen.groupby("country").size().div(pred.groupby("country").s1.nunique()).round(3).to_dict())

    OUT_DIR.mkdir(exist_ok=True)
    write_lists(OUT_DIR / "matching_results.tsv", ["source1_entity_id", "matched_entity_ids"],
                s1, chosen, cidx, cids)
    write_lists(OUT_DIR / "candidate_pairs.tsv", ["source1_entity_id", "candidate_entity_ids"],
                s1, pred, cidx, cids)

    v = find_validator()
    if v is None:
        print("validator not found - run it manually")
        return
    test_dir = C.DATA_DIR / "test"
    print(f"\nrunning {v}")
    r = subprocess.run([sys.executable, str(v), "--matching", str(OUT_DIR / "matching_results.tsv"),
                        "--candidate", str(OUT_DIR / "candidate_pairs.tsv"), "--test-dir", str(test_dir)])
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
