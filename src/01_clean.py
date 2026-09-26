"""Step 1: clean names and addresses of all three sources.

Output: work/<run>/clean_<split>_s<src>.parquet  (one row per record)
Sample mode keeps a fraction of TRAIN Source-1 entities, all their matches, and the
same fraction of unmatched S2/S3 records (so the match/decoy mix stays realistic).
"""
from multiprocessing import Pool

import numpy as np
import pandas as pd

import config as C
from normalize import FIELDS, clean_record

CHUNK = 20_000


def _clean_chunk(args):
    names, addrs = args
    rows = [clean_record(n, a) for n, a in zip(names, addrs)]
    return {f: [r[f] for r in rows] for f in FIELDS}


def clean_frame(df, pool):
    names, addrs = df.business_name.tolist(), df.business_address.tolist()
    jobs = [(names[i:i + CHUNK], addrs[i:i + CHUNK]) for i in range(0, len(df), CHUNK)]
    parts = {f: [] for f in FIELDS}
    for i, res in enumerate(pool.imap(_clean_chunk, jobs)):
        for f in FIELDS:
            parts[f].extend(res[f])
        if (i + 1) % 25 == 0:
            print(f"     {min((i + 1) * CHUNK, len(df)):,}/{len(df):,}", flush=True)
    out = pd.DataFrame({
        "entity_id": df.entity_id.to_numpy(),
        "key": C.id_to_key(df.entity_id),
        "country": df.country.to_numpy(),
    })
    for f in FIELDS:
        out[f] = parts[f]
    out["is_domain"] = out.is_domain.astype(np.int8)
    out["addr_missing"] = out.addr_missing.astype(np.int8)
    assert out.key.is_unique, "entity id -> key collision"
    return out


def sample_filter(args, src, df, gt_cache):
    """Return the rows of df kept in sample mode (train only)."""
    keys = C.id_to_key(df.entity_id)
    if src == 1:
        return df[C.hash01(keys, 1) < args.sample]
    if "kept" not in gt_cache:
        gt = C.read_tsv(C.GT_PATH)
        s1k = C.id_to_key(gt.source1_entity_id)
        m = gt.matched_entity_ids.str.split(",").explode()
        m = m[m.fillna("") != ""]
        mk = C.id_to_key(m.to_numpy())
        owner = s1k[m.index.to_numpy()]
        gt_cache["all"] = pd.Index(mk)
        gt_cache["kept"] = pd.Index(mk[C.hash01(owner, 1) < args.sample])
    matched_kept = gt_cache["kept"].get_indexer(keys) >= 0
    unmatched = gt_cache["all"].get_indexer(keys) < 0
    return df[matched_kept | (unmatched & (C.hash01(keys, 1) < args.sample))]


def main():
    args = C.parse_args("Step 1: clean records", split=True)
    if args.sample and args.split == "test":
        raise SystemExit("--sample only applies to the train split")
    out_dir = C.work_dir(args)
    gt_cache = {}
    with Pool(C.N_JOBS) as pool:
        for src in C.SOURCES:
            out = out_dir / f"clean_{args.split}_s{src}.parquet"
            if out.exists() and not args.force:
                print(f"[skip] {out.name} exists")
                continue
            with C.Timer(f"clean {args.split} source{src}"):
                df = C.read_tsv(C.raw_path(args.split, src))
                if args.sample:
                    df = sample_filter(args, src, df, gt_cache).reset_index(drop=True)
                print(f"     {len(df):,} records", flush=True)
                clean = clean_frame(df, pool)
                clean.to_parquet(out, index=False)
                del df, clean

    if args.split == "train":
        s1 = pd.read_parquet(out_dir / "clean_train_s1.parquet")
        print("\nExample cleaned Source-1 records:")
        cols = ["entity_id", "name_core", "name_legal", "addr_clean", "house_no", "postcode", "state"]
        print(s1.sample(8, random_state=C.SEED)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
