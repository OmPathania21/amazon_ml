"""Step 6: turn out-of-fold probabilities into matches and measure macro F0.5.

1. each S2/S3 record keeps only its best S1 entity (one-owner rule)
2. search the decision rule (global threshold, or per-entity expected-F0.5 top-k)
3. report validation macro F0.5 overall / per country / singletons vs matched

Output: work/<run>/decision.json, tune_results.csv
"""
import json

import numpy as np
import pandas as pd

import config as C
import metrics as M


def report(name, f, info):
    print(f"\n{name}: macro F0.5 = {f.mean():.5f}")
    df = info.assign(f=f)
    print("   by country :", df.groupby("country").f.mean().round(5).to_dict())
    print(f"   singletons : {df[df.n_true == 0].f.mean():.5f}  ({(df.n_true == 0).mean():.3%} of S1)")
    print(f"   has matches: {df[df.n_true > 0].f.mean():.5f}")


def main():
    args = C.parse_args("Step 6: decision rule + validation score")
    d = C.work_dir(args)
    oof = pd.read_parquet(d / "oof.parquet")
    info = pd.read_parquet(d / "s1_info.parquet")
    s1_index = pd.Index(info.s1.to_numpy())
    n_true = info.n_true.to_numpy()

    print(f"pairs {len(oof):,}   S1 entities {len(info):,}")
    report("CEILING (perfect model on our candidates)", M.macro_f05(oof[oof.label == 1], s1_index, n_true), info)

    best = M.assign_best(oof)
    print(f"\nafter one-owner rule: {len(best):,} pairs  (positives kept: "
          f"{best.label.sum():,} of {oof.label.sum():,})")

    with C.Timer("search decision rule"):
        res, top = M.search(best, s1_index, n_true)
    res.to_csv(d / "tune_results.csv", index=False)
    for method, g in res.groupby("method"):
        r = g.loc[g.f05.idxmax()]
        print(f"   best {method:10s}: param={r.param:.3f}  F0.5={r.f05:.5f}")

    chosen = M.apply_rule(best, top.method, float(top.param))
    f = M.macro_f05(chosen, s1_index, n_true)
    report(f"CHOSEN {top.method} (param={top.param:.3f})", f, info)

    tp = np.bincount(s1_index.get_indexer(chosen.s1), weights=chosen.label, minlength=len(info))
    npred = np.bincount(s1_index.get_indexer(chosen.s1), minlength=len(info))
    print(f"   pair precision {tp.sum() / max(npred.sum(), 1):.4f}   pair recall {tp.sum() / n_true.sum():.4f}")
    print(f"   singletons predicted empty: {(npred[n_true == 0] == 0).mean():.4f}")

    dec = {"method": top.method, "param": float(top.param), "val_f05": float(f.mean())}

    # per-country rules (countries seen in training); unseen countries keep the global rule
    with C.Timer("search per-country decision rules"):
        per = {}
        for country, g in best.groupby("country", sort=False):
            m = info.country.to_numpy() == country
            _, t = M.search(g, pd.Index(info.s1.to_numpy()[m]), n_true[m])
            per[str(country)] = {"method": t.method, "param": float(t.param)}
            print(f"   {country}: {t.method} param={t.param:.3f}  F0.5={t.f05:.5f}")
    dec_pc = dict(dec, per_country=per)
    f_pc = M.macro_f05(M.apply_decision(best, dec_pc), s1_index, n_true)
    report("CHOSEN per-country rules", f_pc, info)
    if f_pc.mean() > f.mean():
        dec = dict(dec_pc, val_f05=float(f_pc.mean()))
        print(f"   -> using per-country rules (+{f_pc.mean() - f.mean():.5f})")
    else:
        print("   -> per-country rules do not help; keeping the global rule")

    (d / "decision.json").write_text(json.dumps(dec, indent=2))
    print(f"\nsaved {d / 'decision.json'}")


if __name__ == "__main__":
    main()
