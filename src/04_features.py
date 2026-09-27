"""Step 4: compute pair features for every candidate pair.

Output: work/<run>/feat_<split>/<country>_<part>.parquet
Columns: s1, cand, label (-1 on test), fold (-1 on test), country, features...
"""
import re

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

import config as C
from features import (REC_COLS, group_features, partner_features, partner_keys, query_relative,
                      string_features)

def safe_name(name):
    return re.sub(r"[^A-Za-z0-9]+", "_", str(name)) or "none"


def load_records(d, split, srcs, country):
    tables = []
    for src in srcs:
        t = pq.read_table(d / f"clean_{split}_s{src}.parquet", columns=["key", "country"] + REC_COLS)
        tables.append(t.filter(pc.equal(t["country"], country)))
    t = pa.concat_tables(tables)
    return t, pd.Index(t["key"].to_numpy())


def take(table, idx):
    assert (idx >= 0).all(), "candidate key missing from cleaned records"
    sub = table.take(pa.array(idx))
    return {c: sub[c].to_pylist() for c in REC_COLS}


def chunk_bounds(cand, size):
    starts = np.flatnonzero(np.r_[True, cand[1:] != cand[:-1]])
    cuts = [0]
    while cuts[-1] + size < len(cand):
        j = np.searchsorted(starts, cuts[-1] + size)
        if j >= len(starts):
            break
        cuts.append(int(starts[j]))
    cuts.append(len(cand))
    return list(zip(cuts[:-1], cuts[1:]))


def main():
    args = C.parse_args("Step 4: pair features", split=True)
    d = C.work_dir(args)
    out_dir = d / f"feat_{args.split}"
    out_dir.mkdir(exist_ok=True)

    if args.split == "train":
        truth = pd.read_parquet(d / "truth.parquet")
        tidx, tval = pd.Index(truth.cand.to_numpy()), truth.s1.to_numpy()
        info = pd.read_parquet(d / "s1_info.parquet")
        fidx, fval = pd.Index(info.s1.to_numpy()), info.fold.to_numpy()

    for cfile in sorted(d.glob(f"cand_{args.split}_*.parquet")):
        tag = cfile.stem[len(f"cand_{args.split}_"):]
        done = out_dir / f"{tag}.done"
        if done.exists() and not args.force:
            print(f"[skip] features {tag} exist")
            continue
        for old in out_dir.glob(f"{tag}_*.parquet"):
            old.unlink()

        c = pd.read_parquet(cfile)
        if c.empty:
            done.touch()
            continue
        c = c.sort_values(["cand", "rank"], kind="stable").reset_index(drop=True)
        with C.Timer(f"features {args.split} {tag}: {len(c):,} pairs"):
            c = group_features(c)
            c["partner"], c["partner_sc"] = partner_keys(c)
            if args.split == "train":
                pos = tidx.get_indexer(c.cand.to_numpy())
                true_s1 = np.where(pos >= 0, tval[np.maximum(pos, 0)], -1)
                c["label"] = (true_s1 == c.s1.to_numpy()).astype(np.int8)
                c["fold"] = fval[fidx.get_indexer(c.s1.to_numpy())].astype(np.int8)
            else:
                c["label"] = np.int8(-1)
                c["fold"] = np.int8(-1)

            country = pd.read_parquet(d / f"clean_{args.split}_s1.parquet", columns=["country"]).country
            country = next(x for x in country.unique() if safe_name(x) == tag)
            s1t, s1i = load_records(d, args.split, [1], country)
            qt, qi = load_records(d, args.split, [2, 3], country)

            bounds = chunk_bounds(c.cand.to_numpy(), C.FEAT_CHUNK)
            for i, (lo, hi) in enumerate(bounds):
                ch = c.iloc[lo:hi].reset_index(drop=True)
                a = take(s1t, s1i.get_indexer(ch.s1.to_numpy()))
                b = take(qt, qi.get_indexer(ch.cand.to_numpy()))
                pidx = qi.get_indexer(ch.partner.to_numpy())
                has_p = pidx >= 0
                p = take(qt, np.where(has_p, pidx, 0))
                pf = partner_features(b, p, has_p, ch.partner_sc.to_numpy())
                ch = ch.drop(columns=["partner", "partner_sc"])
                ch = pd.concat([ch, pd.DataFrame(string_features(a, b)), pd.DataFrame(pf)], axis=1)
                ch = query_relative(ch)
                ch["country"] = country
                ch.to_parquet(out_dir / f"{tag}_{i:04d}.parquet", index=False)
                print(f"     part {i + 1}/{len(bounds)}", flush=True)
            done.touch()


if __name__ == "__main__":
    main()
