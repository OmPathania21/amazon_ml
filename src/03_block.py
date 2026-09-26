"""Step 3: blocking - for every S2/S3 record find its TOP_K most similar S1 records.

Each record is a bag of prefixed tokens (name words, name skeletons, joined name,
address words, house number, postcode, numbers). Tokens are weighted by IDF over
Source 1, tokens that are too common (df > MAX_DF) are dropped, and the TOP_K S1
records by TF-IDF cosine are kept, per country, with a multithreaded sparse top-k
matrix product.

Output: work/<run>/cand_<split>_<country>.parquet  (s1, cand, score, rank)
"""
import re

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sparse_dot_topn import sp_matmul_topn

import config as C


def safe(name):
    return re.sub(r"[^A-Za-z0-9]+", "_", str(name)) or "none"


def load(d, split, src):
    return pd.read_parquet(d / f"clean_{split}_s{src}.parquet", columns=["key", "country", "btok"])


def build_matrix(btok, vocab, idf):
    toks = pd.Series(btok, copy=False).str.split(" ")
    lens = toks.str.len().to_numpy()
    col = vocab.get_indexer(np.concatenate(toks.to_numpy()) if len(toks) else np.array([], object))
    row = np.repeat(np.arange(len(toks)), lens)
    keep = col >= 0
    m = sp.csr_matrix((idf[col[keep]], (row[keep], col[keep])),
                      shape=(len(toks), len(vocab)), dtype=np.float32)
    norm = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel())
    norm[norm == 0] = 1.0
    return sp.diags((1.0 / norm).astype(np.float32)) @ m


def block_country(s1c, qc):
    toks = s1c.btok.str.split(" ").explode()
    df = toks[toks.fillna("") != ""].value_counts()
    df = df[df <= C.MAX_DF]
    vocab = pd.Index(df.index)
    idf = (np.log((len(s1c) + 1) / df.to_numpy()) + 1.0).astype(np.float32)
    B = build_matrix(s1c.btok.to_numpy(), vocab, idf).T.tocsr()
    s1_keys = s1c.key.to_numpy()

    out = []
    for start in range(0, len(qc), C.BLOCK_CHUNK):
        q = qc.iloc[start:start + C.BLOCK_CHUNK]
        A = build_matrix(q.btok.to_numpy(), vocab, idf)
        R = sp_matmul_topn(A, B, top_n=C.TOP_K, sort=True, n_threads=C.N_JOBS)
        counts = np.diff(R.indptr)
        rows = np.repeat(np.arange(len(q)), counts)
        rank = np.arange(R.nnz) - np.repeat(R.indptr[:-1], counts)
        out.append(pd.DataFrame({
            "s1": s1_keys[R.indices],
            "cand": q.key.to_numpy()[rows],
            "score": R.data.astype(np.float32),
            "rank": rank.astype(np.int8),
        }))
        done = start + len(q)
        if len(out) % 20 == 0 or done == len(qc):
            print(f"     {done:,}/{len(qc):,} queries", flush=True)
    cols = {"s1": np.int64, "cand": np.int64, "score": np.float32, "rank": np.int8}
    if not out:
        return pd.DataFrame({c: pd.Series(dtype=t) for c, t in cols.items()}), len(vocab)
    return pd.concat(out, ignore_index=True), len(vocab)


def report_recall(d, cands, split):
    if split != "train":
        return
    truth = pd.read_parquet(d / "truth.parquet")
    hit = cands.merge(truth, on=["cand", "s1"], how="inner")
    total = len(truth)
    print(f"\nBLOCKING RECALL (true pairs found / all true pairs = {total:,})")
    for k in range(1, C.TOP_K + 1):
        print(f"   top-{k}: {(hit['rank'] < k).sum() / total:.4f}")
    info = pd.read_parquet(d / "s1_info.parquet")
    hit = hit.merge(info[["s1", "country"]], on="s1")
    t = truth.merge(info[["s1", "country"]], on="s1")
    print("   by country:", (hit.groupby("country").size() / t.groupby("country").size()).round(4).to_dict())
    print(f"   candidate pairs: {len(cands):,}   positives among them: {len(hit) / len(cands):.3f}")


def main():
    args = C.parse_args("Step 3: blocking", split=True)
    d = C.work_dir(args)
    s1 = load(d, args.split, 1)
    q = pd.concat([load(d, args.split, 2), load(d, args.split, 3)], ignore_index=True)

    frames = []
    for country in sorted(s1.country.unique()):
        out = d / f"cand_{args.split}_{safe(country)}.parquet"
        if out.exists() and not args.force:
            print(f"[skip] {out.name} exists")
            frames.append(pd.read_parquet(out))
            continue
        s1c = s1[s1.country == country].reset_index(drop=True)
        qc = q[q.country == country].reset_index(drop=True)
        with C.Timer(f"block {args.split} {country}: {len(s1c):,} S1 x {len(qc):,} S2/S3"):
            cands, nvocab = block_country(s1c, qc)
            print(f"     vocab {nvocab:,} tokens, {len(cands):,} candidate pairs")
            cands.to_parquet(out, index=False)
        frames.append(cands)

    report_recall(d, pd.concat(frames, ignore_index=True), args.split)


if __name__ == "__main__":
    main()
