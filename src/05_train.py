"""Step 5: train LightGBM with 5-fold cross-validation grouped by Source-1 entity.

For each fold k: train on pairs whose S1 entity is not in fold k (uniformly sampled
to at most MAX_TRAIN_ROWS), early-stop on fold k, then predict every pair of fold k.
Result: an out-of-fold probability for every training pair (used by step 6) and
5 models (averaged later on the test set).

Outputs: work/<run>/models/lgb_fold<k>.txt, oof.parquet, feature_importance.csv
"""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import config as C
from features import feature_columns


def read_rows(files, cols, keep_fn, rng=None, rate=1.0):
    parts = []
    for f in files:
        df = pd.read_parquet(f, columns=cols)
        m = keep_fn(df)
        if rate < 1.0:
            m &= rng.random(len(df)) < rate
        parts.append(df[m])
    return pd.concat(parts, ignore_index=True)


def main():
    args = C.parse_args("Step 5: train LightGBM")
    d = C.work_dir(args)
    files = sorted((d / "feat_train").glob("*.parquet"))
    assert files, "run 04_features.py first"
    feats = feature_columns(pd.read_parquet(files[0]).head(1))
    (d / "models").mkdir(exist_ok=True)
    (d / "models" / "features.json").write_text(json.dumps(feats))

    folds = np.concatenate([pq.read_table(f, columns=["fold"])["fold"].to_numpy() for f in files])
    per_fold = np.bincount(folds, minlength=C.N_FOLDS)
    print(f"{len(files)} feature files, {len(folds):,} pairs, {len(feats)} features")
    print("pairs per fold:", per_fold.tolist())

    rng = np.random.default_rng(C.SEED)
    oof, importance = [], []
    for k in range(C.N_FOLDS):
        model_path = d / "models" / f"lgb_fold{k}.txt"
        if model_path.exists() and not args.force:
            print(f"[skip] fold {k} model exists")
            booster = lgb.Booster(model_file=str(model_path))
        else:
            with C.Timer(f"fold {k}: train"):
                n_tr = len(folds) - per_fold[k]
                tr = read_rows(files, feats + ["label", "fold"], lambda x: x.fold != k,
                               rng, min(1.0, C.MAX_TRAIN_ROWS / max(n_tr, 1)))
                va = read_rows(files, feats + ["label", "fold"], lambda x: x.fold == k,
                               rng, min(1.0, C.MAX_VALID_ROWS / max(per_fold[k], 1)))
                print(f"     train {len(tr):,} rows ({tr.label.mean():.3f} positive), valid {len(va):,}")
                dtr = lgb.Dataset(tr[feats].to_numpy(np.float32), tr.label.to_numpy(), feature_name=feats)
                dva = lgb.Dataset(va[feats].to_numpy(np.float32), va.label.to_numpy(), reference=dtr)
                del tr, va
                params = dict(C.LGB_PARAMS, num_threads=C.N_JOBS)
                booster = lgb.train(params, dtr, C.NUM_BOOST_ROUND, valid_sets=[dva], valid_names=["valid"],
                                    callbacks=[lgb.early_stopping(C.EARLY_STOP, verbose=False),
                                               lgb.log_evaluation(200)])
                booster.save_model(str(model_path), num_iteration=booster.best_iteration)
                best = booster.best_score["valid"]
                print(f"     best iter {booster.best_iteration}  auc {best['auc']:.5f}  "
                      f"logloss {best['binary_logloss']:.5f}")
                del dtr, dva

        with C.Timer(f"fold {k}: predict held-out pairs"):
            ho = read_rows(files, feats + ["s1", "cand", "label", "fold", "country"], lambda x: x.fold == k)
            ho["prob"] = booster.predict(ho[feats].to_numpy(np.float32), num_threads=C.N_JOBS).astype(np.float32)
            oof.append(ho[["s1", "cand", "label", "fold", "country", "prob"]])
        importance.append(pd.Series(booster.feature_importance("gain"), index=booster.feature_name()))

    oof = pd.concat(oof, ignore_index=True)
    oof.to_parquet(d / "oof.parquet", index=False)
    imp = pd.concat(importance, axis=1).mean(axis=1).sort_values(ascending=False)
    imp.rename("gain").to_csv(d / "feature_importance.csv")
    print("\nTop 20 features by gain:")
    print((imp / imp.sum()).head(20).round(4).to_string())


if __name__ == "__main__":
    main()
