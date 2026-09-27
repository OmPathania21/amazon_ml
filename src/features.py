"""Pair features. `a` = Source-1 side, `b` = S2/S3 candidate side.

Each side is a dict {column name: list of str/int}, aligned row by row.
Fuzzy scores are 0-100 floats from rapidfuzz (compiled, multithreaded).
"""
import re

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist

import config as C

STR_COLS = ["name_clean", "name_core", "name_key", "name_alt", "name_skel", "name_nospace",
            "name_legal", "addr_clean", "house_no", "postcode", "state", "numbers", "landmark"]
INT_COLS = ["is_domain", "addr_missing", "name_nonascii"]
REC_COLS = STR_COLS + INT_COLS


def _cd(x, y, scorer):
    return cpdist(x, y, scorer=scorer, workers=C.N_JOBS, dtype=np.float32)


def _eq_conf(x, y):
    x, y = np.asarray(x, dtype=object), np.asarray(y, dtype=object)
    both = (x != "") & (y != "")
    eq = both & (x == y)
    return eq.astype(np.int8), (both & ~eq).astype(np.int8), (~both).astype(np.int8)


def string_features(a, b):
    f = {}
    ca, cb = a["name_core"], b["name_core"]
    f["n_set"] = _cd(ca, cb, fuzz.token_set_ratio)
    f["n_sort"] = _cd(ca, cb, fuzz.token_sort_ratio)
    f["n_ratio"] = _cd(ca, cb, fuzz.ratio)
    f["n_partial"] = _cd(ca, cb, fuzz.partial_ratio)
    f["n_jw"] = _cd(ca, cb, JaroWinkler.normalized_similarity) * 100
    f["n_full_set"] = _cd(a["name_clean"], b["name_clean"], fuzz.token_set_ratio)
    f["key_set"] = _cd(a["name_key"], b["name_key"], fuzz.token_set_ratio)
    f["key_sort"] = _cd(a["name_key"], b["name_key"], fuzz.token_sort_ratio)
    f["skel_set"] = _cd(a["name_skel"], b["name_skel"], fuzz.token_set_ratio)
    f["skel_ratio"] = _cd(a["name_skel"], b["name_skel"], fuzz.ratio)
    f["nosp_ratio"] = _cd(a["name_nospace"], b["name_nospace"], fuzz.ratio)
    f["nosp_partial"] = _cd(a["name_nospace"], b["name_nospace"], fuzz.partial_ratio)

    alt_a = [x or y for x, y in zip(a["name_alt"], ca)]
    alt_b = [x or y for x, y in zip(b["name_alt"], cb)]
    f["alt_best"] = np.maximum.reduce([f["n_set"], _cd(alt_a, cb, fuzz.token_set_ratio),
                                       _cd(ca, alt_b, fuzz.token_set_ratio),
                                       _cd(alt_a, alt_b, fuzz.token_set_ratio)])
    f["has_alt"] = np.array([(x != "") or (y != "") for x, y in zip(a["name_alt"], b["name_alt"])], np.int8)

    aa, ab = a["addr_clean"], b["addr_clean"]
    f["a_set"] = _cd(aa, ab, fuzz.token_set_ratio)
    f["a_sort"] = _cd(aa, ab, fuzz.token_sort_ratio)
    f["a_ratio"] = _cd(aa, ab, fuzz.ratio)
    f["a_partial"] = _cd(aa, ab, fuzz.partial_ratio)

    f["state_eq"], f["state_conf"], f["state_miss"] = _eq_conf(a["state"], b["state"])
    f["post_eq"], f["post_conf"], f["post_miss"] = _eq_conf(a["postcode"], b["postcode"])
    f["legal_eq"], f["legal_conf"], f["legal_miss"] = _eq_conf(a["name_legal"], b["name_legal"])
    f["house_eq"], f["house_conf"], f["house_miss"] = _eq_conf(a["house_no"], b["house_no"])
    f["house_part"] = np.array(
        [bool(x) and bool(y) and x != y and (x.endswith(y) or y.endswith(x) or x.startswith(y) or y.startswith(x))
         for x, y in zip(a["house_no"], b["house_no"])], np.int8)

    ha, hb = _house_int(a["house_no"]), _house_int(b["house_no"])
    both = (ha >= 0) & (hb >= 0)
    f["house_numdiff"] = np.where(both, np.abs(ha - hb), -1).astype(np.float32)
    f["house_numreldiff"] = np.where(both, np.abs(ha - hb) / np.maximum(np.maximum(ha, hb), 1), -1).astype(np.float32)
    f["house_numeq"] = (both & (ha == hb)).astype(np.int8)
    f["nonascii_a"] = np.asarray(a["name_nonascii"], np.int8)
    f["nonascii_b"] = np.asarray(b["name_nonascii"], np.int8)

    jac, common = [], []
    for x, y in zip(a["numbers"], b["numbers"]):
        sx, sy = set(x.split()), set(y.split())
        u = len(sx | sy)
        i = len(sx & sy)
        jac.append(i / u if u else -1.0)
        common.append(i)
    f["num_jacc"] = np.array(jac, np.float32)
    f["num_common"] = np.array(common, np.int16)

    f["key_first_eq"] = np.array(
        [bool(x) and x.split()[0] == (y.split() or [""])[0] for x, y in zip(a["name_key"], b["name_key"])], np.int8)
    f["dom_a"] = np.asarray(a["is_domain"], np.int8)
    f["dom_b"] = np.asarray(b["is_domain"], np.int8)
    f["amiss_a"] = np.asarray(a["addr_missing"], np.int8)
    f["amiss_b"] = np.asarray(b["addr_missing"], np.int8)
    f["lm_any"] = np.array([(x != "") or (y != "") for x, y in zip(a["landmark"], b["landmark"])], np.int8)
    f["len_a"] = np.array([len(x) for x in ca], np.int16)
    f["len_b"] = np.array([len(x) for x in cb], np.int16)
    f["ntok_a"] = np.array([x.count(" ") + 1 if x else 0 for x in ca], np.int8)
    f["ntok_b"] = np.array([x.count(" ") + 1 if x else 0 for x in cb], np.int8)
    return f


_LEAD_INT = re.compile(r"(\d+)")


def _house_int(values):
    out = np.full(len(values), -1, np.float64)
    for i, v in enumerate(values):
        m = _LEAD_INT.search(v) if v else None
        if m and len(m.group(1)) <= 9:
            out[i] = int(m.group(1))
    return out


def partner_keys(c):
    """For each pair, the S1 entity's strongest OTHER candidate (by blocking score);
    -1 when the entity has no other candidate."""
    o = c[["s1", "cand", "score"]].sort_values(["s1", "score"], ascending=[True, False], kind="stable")
    first = o.drop_duplicates("s1").set_index("s1")
    second = o[o.duplicated("s1")].drop_duplicates("s1").set_index("s1")
    b1 = first.cand.reindex(c.s1).to_numpy()
    b2 = second.cand.reindex(c.s1).fillna(-1).to_numpy().astype(np.int64)
    s1 = first.score.reindex(c.s1).to_numpy()
    s2 = second.score.reindex(c.s1).fillna(0).to_numpy()
    is_best = b1 == c.cand.to_numpy()
    return np.where(is_best, b2, b1), np.where(is_best, s2, s1).astype(np.float32)


def partner_features(b, p, has_p, p_score):
    """Similarity of the candidate to the S1 entity's strongest other candidate:
    records of the same business in S2 and S3 agree with each other."""
    f = {}
    f["partner_name_sim"] = np.where(has_p, _cd(b["name_core"], p["name_core"], fuzz.token_set_ratio), -1)
    f["partner_addr_sim"] = np.where(has_p, _cd(b["addr_clean"], p["addr_clean"], fuzz.token_set_ratio), -1)
    f["partner_score"] = np.where(has_p, p_score, 0).astype(np.float32)
    return {k: np.asarray(v, np.float32) for k, v in f.items()}


def group_features(c):
    """Blocking-score features relative to the other candidates of the same
    S2/S3 record (query side) and of the same S1 record. `c` sorted by cand, rank."""
    g = c.groupby("cand", sort=False).score
    c["q_best"] = g.transform("max").astype(np.float32)
    c["q_n"] = g.transform("size").astype(np.int8)
    nxt = g.shift(-1).fillna(0).astype(np.float32)
    c["gap_next"] = (c.score - nxt).astype(np.float32)
    c["gap_best"] = (c.q_best - c.score).astype(np.float32)
    c["ratio_best"] = (c.score / c.q_best.clip(lower=1e-6)).astype(np.float32)

    g1 = c.groupby("s1", sort=False).score
    c["s1_n"] = g1.transform("size").astype(np.int16)
    c["s1_rank"] = g1.rank(ascending=False, method="first").astype(np.int16)
    c["s1_ratio"] = (c.score / g1.transform("max").clip(lower=1e-6)).astype(np.float32)
    c["s1_top1_n"] = (c["rank"] == 0).groupby(c.s1).transform("sum").astype(np.int16)
    c["is_s3"] = (c.cand // 10**10 == 3).astype(np.int8)
    return c


def query_relative(df):
    """String-score features relative to the other S1 candidates of the same S2/S3
    record (all rows of a record are always in the same chunk)."""
    df["comb"] = (0.6 * df.n_set + 0.4 * df.a_set).astype(np.float32)
    for col in ["n_set", "a_set", "comb", "nosp_partial"]:
        best = df.groupby("cand", sort=False)[col].transform("max")
        df[col + "_rel"] = (df[col] - best).astype(np.float32)
    df["comb_rank"] = df.groupby("cand", sort=False).comb.rank(ascending=False, method="min").astype(np.int8)
    return df


def feature_columns(df):
    return [c for c in df.columns if c not in C.META_COLS]


__all__ = ["REC_COLS", "string_features", "group_features", "query_relative", "feature_columns",
           "partner_keys", "partner_features", "pd"]
