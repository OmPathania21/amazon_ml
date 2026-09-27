"""Step 6c (optional): second-stage model on top of the first model's probabilities.

For every pair it looks at the probability in context - its rank and gap among the other
candidates of the same S1 entity and of the same S2/S3 record, how many strong candidates
the entity has, their sum, etc. - and re-scores the pair. It is trained with the same
5 folds on the out-of-fold probabilities (oof.parquet), so its own validation is honest.

Nothing is recomputed upstream. The stacked probabilities replace test_pred.parquet and
decision.json ONLY if the validation macro F0.5 improves; the originals are kept as
*_base files. Afterwards run 08_write_submission.py as usual.
"""
import json
import shutil

import lightgbm as lgb
import numpy as np
import pandas as pd

import config as C
import metrics as M

PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=63, min_data_in_leaf=500,
              feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, metric="binary_logloss",
              verbose=-1, seed=C.SEED)
ROWS_PER_FOLD = 4_000_000


def stack_features(df):
    p = df.prob.astype(np.float32)
    f = pd.DataFrame({"prob": p})
    f["logit"] = np.log(p.clip(1e-6, 1 - 1e-6) / (1 - p.clip(1e-6, 1 - 1e-6))).astype(np.float32)
    for key, tag in (("s1", "s"), ("cand", "c")):
        g = p.groupby(df[key])
        mx = g.transform("max")
        f[f"{tag}_rank"] = g.rank(ascending=False, method="first").astype(np.float32)
        f[f"{tag}_n"] = g.transform("size").astype(np.float32)
        f[f"{tag}_max"] = mx.astype(np.float32)
        f[f"{tag}_gap"] = (mx - p).astype(np.float32)
        f[f"{tag}_sum"] = g.transform("sum").astype(np.float32)
        f[f"{tag}_n50"] = (p > 0.5).groupby(df[key]).transform("sum").astype(np.float32)
    # second best of the record (how contested this S2/S3 record is)
    f["c_second"] = p.where(f.c_rank == 2).groupby(df.cand).transform("max").fillna(0).astype(np.float32)
    f["share_s"] = (p / f.s_sum.clip(lower=1e-6)).astype(np.float32)
    return f


def main():
    d = C.WORK_ROOT / "full"
    dec = json.loads((d / "decision.json").read_text())
    base_f = dec["val_f05"]
    print(f"current validation macro F0.5: {base_f:.5f}")

    with C.Timer("stack features (train)"):
        oof = pd.read_parquet(d / "oof.parquet", columns=["s1", "cand", "label", "fold", "country", "prob"])
        X = stack_features(oof)
    feats = list(X.columns)
    y, folds = oof.label.to_numpy(), oof.fold.to_numpy()

    rng = np.random.default_rng(C.SEED)
    new_prob = np.zeros(len(oof), np.float32)
    boosters = []
    for k in range(C.N_FOLDS):
        with C.Timer(f"stack fold {k}"):
            tr = np.flatnonzero(folds != k)
            if len(tr) > ROWS_PER_FOLD:
                tr = rng.choice(tr, ROWS_PER_FOLD, replace=False)
            va = np.flatnonzero(folds == k)
            vs = rng.choice(va, min(len(va), 1_000_000), replace=False)
            dtr = lgb.Dataset(X.iloc[tr].to_numpy(np.float32), y[tr], feature_name=feats)
            dva = lgb.Dataset(X.iloc[vs].to_numpy(np.float32), y[vs], reference=dtr)
            b = lgb.train(dict(PARAMS, num_threads=C.N_JOBS), dtr, 600, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(30, verbose=False)])
            new_prob[va] = b.predict(X.iloc[va].to_numpy(np.float32), num_iteration=b.best_iteration,
                                     num_threads=C.N_JOBS)
            boosters.append(b)
            print(f"     best iter {b.best_iteration}")

    info = pd.read_parquet(d / "s1_info.parquet")
    s1_index, n_true = pd.Index(info.s1.to_numpy()), info.n_true.to_numpy()
    oof["prob"] = new_prob
    with C.Timer("search decision rule on stacked probabilities"):
        best = M.assign_best(oof)
        res, top = M.search(best, s1_index, n_true)
    new_f = float(top.f05)
    print(f"\nSTACKED validation macro F0.5: {new_f:.5f}  ({top.method} param={top.param:.3f})"
          f"   vs current {base_f:.5f}")
    if new_f <= base_f:
        print("-> no improvement: nothing changed, keep the current submission")
        return

    with C.Timer("apply stacking to test pairs"):
        tp = pd.read_parquet(d / "test_pred.parquet")
        Xt = stack_features(tp).to_numpy(np.float32)
        tp["prob"] = np.mean([b.predict(Xt, num_iteration=b.best_iteration, num_threads=C.N_JOBS)
                              for b in boosters], axis=0).astype(np.float32)
    for name in ("test_pred.parquet", "decision.json"):
        base = d / name.replace(".", "_base.", 1)
        if not base.exists():
            shutil.copy2(d / name, base)
    tp.to_parquet(d / "test_pred.parquet", index=False)
    (d / "decision.json").write_text(json.dumps(
        {"method": top.method, "param": float(top.param), "val_f05": new_f, "stacked": True}, indent=2))
    print(f"-> improved by +{new_f - base_f:.5f}: test_pred.parquet and decision.json updated "
          f"(originals saved as *_base). Now run 08_write_submission.py")


if __name__ == "__main__":
    main()
