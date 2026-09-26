"""Step 7: score every TEST candidate pair with the 5 fold models (averaged).

Needs: 01_clean / 03_block / 04_features run with --split test, and trained models.
By default the models of the full run are used; --models sample0.05 uses the models
from the 5% trial instead (quick first submission).

Output: work/full/test_pred.parquet  (s1, cand, country, prob)
"""
import argparse
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

import config as C


def main():
    p = argparse.ArgumentParser(description="Step 7: predict test pairs")
    p.add_argument("--models", default="full", help="run folder whose models to use (full / sample0.05)")
    args = p.parse_args()

    d = C.WORK_ROOT / "full"
    mdir = C.WORK_ROOT / args.models / "models"
    feats = json.loads((mdir / "features.json").read_text())
    boosters = [lgb.Booster(model_file=str(f)) for f in sorted(mdir.glob("lgb_fold*.txt"))]
    assert boosters, f"no models in {mdir}"
    files = sorted((d / "feat_test").glob("*.parquet"))
    assert files, "run 01_clean, 03_block, 04_features with --split test first"
    print(f"{len(boosters)} models from {mdir}, {len(files)} test feature files")

    out = []
    with C.Timer("predict test pairs"):
        for i, f in enumerate(files):
            df = pd.read_parquet(f, columns=feats + ["s1", "cand", "country"])
            X = df[feats].to_numpy(np.float32)
            prob = np.mean([b.predict(X, num_threads=C.N_JOBS) for b in boosters], axis=0)
            out.append(pd.DataFrame({"s1": df.s1.to_numpy(), "cand": df.cand.to_numpy(),
                                     "country": df.country.to_numpy(), "prob": prob.astype(np.float32)}))
            print(f"     {i + 1}/{len(files)} files", flush=True)
    pred = pd.concat(out, ignore_index=True)
    pred.to_parquet(d / "test_pred.parquet", index=False)
    (d / "test_pred_models.txt").write_text(args.models)
    print(f"{len(pred):,} test pairs scored  ->  {d / 'test_pred.parquet'}")
    print("probability > 0.5 share:", round(float((pred.prob > 0.5).mean()), 4))


if __name__ == "__main__":
    main()
