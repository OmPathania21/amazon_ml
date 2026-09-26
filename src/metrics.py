"""Competition metric and the decision rules that turn pair probabilities into matches.

Per Source-1 entity: F0.5 = 1.25*tp / (0.25*n_true + n_pred), and 1.0 when both
n_true and n_pred are 0 (a correctly predicted singleton). Score = mean over ALL
Source-1 entities.
"""
import numpy as np
import pandas as pd


def per_entity_f05(tp, n_pred, n_true):
    denom = 0.25 * n_true + n_pred
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(denom == 0, 1.0, 1.25 * tp / np.where(denom == 0, 1, denom))
    return f


def macro_f05(pred, s1_index, n_true):
    """pred: DataFrame of chosen pairs with columns s1, label.
    s1_index: pd.Index of all evaluated S1 keys; n_true: aligned true-match counts."""
    pos = s1_index.get_indexer(pred.s1.to_numpy())
    ok = pos >= 0
    n = len(s1_index)
    tp = np.bincount(pos[ok], weights=pred.label.to_numpy()[ok], minlength=n)
    npred = np.bincount(pos[ok], minlength=n)
    return per_entity_f05(tp, npred, n_true)


def assign_best(df):
    """Each S2/S3 record may belong to at most one S1 entity: keep its best pair."""
    return df.sort_values("prob", ascending=False, kind="stable").drop_duplicates("cand")


def select_threshold(best, t):
    return best[best.prob >= t]


def select_expected_f(best, floor=0.0):
    """Per S1 entity pick the top-k candidates that maximise the expected F0.5
    (plug-in approximation), comparing against predicting nothing, whose expected
    score is P(no true match) = prod(1 - p)."""
    b = best[best.prob >= floor].sort_values(["s1", "prob"], ascending=[True, False], kind="stable")
    if b.empty:
        return b
    p = b.prob.to_numpy().clip(1e-6, 1 - 1e-6)
    g = b.s1.to_numpy()
    starts = np.r_[True, g[1:] != g[:-1]]
    gid = np.cumsum(starts) - 1
    first = np.flatnonzero(starts)
    k = np.arange(len(p)) - first[gid] + 1
    cum = np.cumsum(p)
    cum = cum - np.r_[0, cum[first[1:] - 1]][gid]
    total = np.bincount(gid, weights=p)[gid]
    ef = 1.25 * cum / (0.25 * total + k)
    ef0 = np.exp(np.bincount(gid, weights=np.log1p(-p)))[gid]
    gmax = pd.Series(ef).groupby(gid).transform("max").to_numpy()
    kstar = pd.Series(np.where(ef >= gmax, k, 10**6)).groupby(gid).transform("min").to_numpy()
    keep = (k <= kstar) & (gmax > ef0)
    return b[keep]


def search(best, s1_index, n_true):
    """Try both decision rules over a grid; return (results table, best row)."""
    rows = []
    for t in np.round(np.arange(0.05, 0.96, 0.025), 3):
        f = macro_f05(select_threshold(best, t), s1_index, n_true).mean()
        rows.append(("threshold", t, f))
    for t in np.round(np.arange(0.0, 0.951, 0.05), 3):
        f = macro_f05(select_expected_f(best, t), s1_index, n_true).mean()
        rows.append(("expected_f", t, f))
    res = pd.DataFrame(rows, columns=["method", "param", "f05"])
    return res, res.loc[res.f05.idxmax()]


def apply_rule(best, method, param):
    if method == "threshold":
        return select_threshold(best, param)
    return select_expected_f(best, param)
