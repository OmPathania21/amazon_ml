"""Step 3: blocking v2 - three TF-IDF passes, union, adaptive pruning.

For every S2/S3 record (query) we search its country's Source-1 records three times:
  combo : rare combinations (house number x name skeleton, postcode x name, street word x
          name, name-skeleton pairs) - pins down the right business even at full scale
  name  : name words / skeletons / joined name - works when the address is empty
  addr  : address words, numbers, number x street-word pairs - works when the name differs
Each pass keeps its top-K S1 records by TF-IDF cosine (multithreaded sparse top-k product).
Tokens found in too many S1 records are dropped (the cap grows with the country size).
The three candidate lists are merged; a candidate is kept if it is the best of some
pass or within PRUNE_RATIO of that pass's best score. Keeping the lists short keeps the
candidate set per Source-1 entity small.

Output: work/<run>/cand_<split>_<country>.parquet
        s1, cand, score (sum of pass scores), rank, sc_combo, sc_name, sc_addr, n_pass
"""
import os
import re
from multiprocessing import Pool

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sparse_dot_topn import sp_matmul_topn

import config as C
from normalize import blocking_tokens

PASSES = ["combo", "name", "addr"]
TOK_COLS = ["name_core", "name_alt", "name_nospace", "addr_clean", "house_no", "postcode", "numbers"]
SUB = 5_000
MAX_WORKERS = 8          # tokenizer processes (each holds a copy of Python + libraries)


def safe(name):
    return re.sub(r"[^A-Za-z0-9]+", "_", str(name)) or "none"


def _tok_worker(cols):
    flats = {p: [] for p in PASSES}
    lens = {p: [] for p in PASSES}
    for row in zip(*cols):
        for p, h in zip(PASSES, blocking_tokens(*row)):
            flats[p].extend(h)
            lens[p].append(len(h))
    return {p: (np.array(flats[p], np.int64), np.array(lens[p], np.int32)) for p in PASSES}


def tokenize(df, pool):
    cols = [df[c].tolist() for c in TOK_COLS]
    jobs = [[c[i:i + SUB] for c in cols] for i in range(0, len(df), SUB)]
    parts = list(pool.imap(_tok_worker, jobs))
    return {p: (np.concatenate([x[p][0] for x in parts]), np.concatenate([x[p][1] for x in parts]))
            for p in PASSES}


def build(flat, lens, vocab, idf):
    n = len(lens)
    if len(vocab) == 0:
        return sp.csr_matrix((n, 1), dtype=np.float32)
    col = np.searchsorted(vocab, flat)
    col[col >= len(vocab)] = 0
    ok = vocab[col] == flat
    row = np.repeat(np.arange(n), lens)
    m = sp.csr_matrix((idf[col[ok]], (row[ok], col[ok])), shape=(n, len(vocab)), dtype=np.float32)
    norm = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel())
    norm[norm == 0] = 1.0
    return sp.diags((1.0 / norm).astype(np.float32)) @ m


def block_country(s1c, qc, pool):
    n1 = len(s1c)
    cap = max(C.MAX_DF, int(C.MAX_DF_FRAC * n1))
    toks = tokenize(s1c, pool)
    index = {}
    for p in PASSES:
        vocab, df = np.unique(toks[p][0], return_counts=True)
        keep = df <= cap
        vocab, df = vocab[keep], df[keep]
        idf = (np.log((n1 + 1) / df) + 1.0).astype(np.float32)
        index[p] = (vocab, idf, build(*toks[p], vocab, idf).T.tocsr())
        print(f"     {p:5s} vocab {len(vocab):,} tokens (df cap {cap})", flush=True)
    del toks
    s1_keys = s1c.key.to_numpy()

    out = []
    for start in range(0, len(qc), C.BLOCK_CHUNK):
        q = qc.iloc[start:start + C.BLOCK_CHUNK]
        qt = tokenize(q, pool)
        frames = []
        for p in PASSES:
            vocab, idf, B = index[p]
            A = build(*qt[p], vocab, idf)
            R = sp_matmul_topn(A, B, top_n=C.PASS_K[p], sort=True, n_threads=C.N_JOBS)
            cnt = np.diff(R.indptr)
            rows = np.repeat(np.arange(len(q)), cnt)
            best = np.repeat(R.data[R.indptr[:-1][cnt > 0]], cnt[cnt > 0])   # rows are sorted: first = best
            rank = np.arange(R.nnz) - np.repeat(R.indptr[:-1], cnt)
            keep = (rank == 0) | (R.data >= C.PRUNE_RATIO * best)
            frames.append(pd.DataFrame({"q": rows[keep], "s": R.indices[keep], p: R.data[keep]}))
        u = pd.concat(frames, ignore_index=True).groupby(["q", "s"], sort=False).max().reset_index()
        for p in PASSES:
            u[p] = u[p].fillna(0).astype(np.float32)
        u["score"] = (u.combo + u.name + u.addr).astype(np.float32)
        u["n_pass"] = ((u.combo > 0).astype(np.int8) + (u.name > 0) + (u.addr > 0)).astype(np.int8)
        u = u.sort_values(["q", "score"], ascending=[True, False], kind="stable")
        u["rank"] = u.groupby("q", sort=False).cumcount().astype(np.int8)
        out.append(pd.DataFrame({
            "s1": s1_keys[u.s.to_numpy()], "cand": q.key.to_numpy()[u.q.to_numpy()],
            "score": u.score.to_numpy(), "rank": u["rank"].to_numpy(),
            "sc_combo": u.combo.to_numpy(), "sc_name": u.name.to_numpy(), "sc_addr": u.addr.to_numpy(),
            "n_pass": u.n_pass.to_numpy()}))
        done = start + len(q)
        if len(out) % 20 == 0 or done == len(qc):
            print(f"     {done:,}/{len(qc):,} queries", flush=True)
    return pd.concat(out, ignore_index=True) if out else None


def report(d, cands, split, n_s1):
    per = cands.groupby("s1").size()
    print(f"\ncandidate pairs: {len(cands):,}   per S1 entity: mean {len(cands) / max(n_s1, 1):.1f}, "
          f"median {per.median():.0f}, p95 {per.quantile(.95):.0f}")
    if split != "train":
        return
    truth = pd.read_parquet(d / "truth.parquet")
    hit = cands.merge(truth, on=["cand", "s1"], how="inner")
    total = len(truth)
    print(f"\nBLOCKING RECALL (true pairs found / all true pairs = {total:,}):  {len(hit) / total:.4f}")
    for p in PASSES:
        print(f"   found by {p:5s} pass: {(hit['sc_' + p] > 0).sum() / total:.4f}")
    for k in (1, 2, 3, 5):
        print(f"   within rank {k}: {(hit['rank'] < k).sum() / total:.4f}")
    info = pd.read_parquet(d / "s1_info.parquet")
    h = hit.merge(info[["s1", "country"]], on="s1").groupby("country").size()
    t = truth.merge(info[["s1", "country"]], on="s1").groupby("country").size()
    print("   by country:", (h / t).round(4).to_dict())
    print(f"   positives among candidates: {len(hit) / len(cands):.3f}")


def load_country(d, split, srcs, country, cols):
    return pd.concat([pd.read_parquet(d / f"clean_{split}_s{s}.parquet", columns=cols,
                                      filters=[("country", "==", country)]) for s in srcs],
                     ignore_index=True)


def read_ok(path):
    """Read a finished candidate file; None if it is missing or damaged (interrupted save)."""
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        print(f"[redo] {path.name} is damaged (interrupted save) - recomputing")
        path.unlink()
        return None


def main():
    args = C.parse_args("Step 3: blocking v2", split=True)
    d = C.work_dir(args)
    cols = ["key", "country"] + TOK_COLS
    countries = sorted(pd.read_parquet(d / f"clean_{args.split}_s1.parquet", columns=["country"]).country.unique())
    n_s1 = 0
    frames = []
    with Pool(min(C.N_JOBS, MAX_WORKERS)) as pool:
        for country in countries:
            out = d / f"cand_{args.split}_{safe(country)}.parquet"
            done = None if args.force else read_ok(out)
            s1c = load_country(d, args.split, [1], country, cols)
            n_s1 += len(s1c)
            if done is not None:
                print(f"[skip] {out.name} exists")
                frames.append(done[["s1", "cand", "rank", "sc_combo", "sc_name", "sc_addr"]])
                continue
            qc = load_country(d, args.split, [2, 3], country, cols)
            with C.Timer(f"block {args.split} {country}: {len(s1c):,} S1 x {len(qc):,} S2/S3"):
                cands = block_country(s1c, qc, pool)
                del qc
                if cands is None:
                    continue
                print(f"     {len(cands):,} candidate pairs ({len(cands) / max(len(s1c), 1):.1f} per S1)")
                tmp = out.with_name(out.name + ".tmp")
                cands.to_parquet(tmp, index=False)
                os.replace(tmp, out)
            frames.append(cands[["s1", "cand", "rank", "sc_combo", "sc_name", "sc_addr"]])
            del cands, s1c
    report(d, pd.concat(frames, ignore_index=True), args.split, n_s1)


if __name__ == "__main__":
    main()
