"""Step 9: error analysis on the out-of-fold (validation) predictions.

Uses the same decision rule as the submission and reports:
  1. pair-level TP / FP / FN, with FN split into "lost at blocking",
     "taken by another S1 (one-owner rule)" and "below the decision threshold"
  2. entity-level outcomes: singletons (TN / FP) and matched entities (perfect / partial / zero)
  3. blocking quality: candidates per S1, reduction ratio, recall ceiling
  4. error rates by cause (missing address, domain name, state / house conflict,
     dissimilar names, country, source)
  5. real examples of false positives and false negatives (original raw records)

Output: printed, and work/<run>/error_analysis.md
"""
import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

import config as C
import metrics as M

REC_COLS = ["key", "country", "name_core", "addr_clean", "house_no", "state", "is_domain", "addr_missing"]


class Report:
    def __init__(self):
        self.lines = []

    def __call__(self, text=""):
        print(text)
        self.lines.append(text)

    def table(self, df, floatfmt="{:.4f}"):
        df = df.copy()
        for c in df.columns:
            if df[c].dtype.kind == "f":
                df[c] = df[c].map(lambda v: floatfmt.format(v))
        cols = [str(c) for c in df.columns]
        self("| " + " | ".join(cols) + " |")
        self("|" + "---|" * len(cols))
        for row in df.itertuples(index=False):
            self("| " + " | ".join(str(v) for v in row) + " |")
        self()


def load_records(d):
    tables = [pq.read_table(d / f"clean_train_s{s}.parquet", columns=REC_COLS) for s in C.SOURCES]
    t = pa.concat_tables(tables)
    return t, pd.Index(t["key"].to_numpy())


def attach(pairs, t, idx):
    """Add S1-side (a_) and candidate-side (b_) record fields to a pair frame."""
    for side, col in (("a_", "s1"), ("b_", "cand")):
        pos = idx.get_indexer(pairs[col].to_numpy())
        sub = t.take(pa.array(pos))
        for c in REC_COLS[1:]:
            pairs[side + c] = sub[c].to_numpy(zero_copy_only=False)
    pairs["name_sim"] = cpdist(pairs.a_name_core.tolist(), pairs.b_name_core.tolist(),
                               scorer=fuzz.token_set_ratio, workers=C.N_JOBS)
    return pairs


def causes(p):
    both = lambda a, b: (p[a] != "") & (p[b] != "")
    return {
        "candidate address missing": p.b_addr_missing == 1,
        "candidate name is a domain": p.b_is_domain == 1,
        "state conflict": both("a_state", "b_state") & (p.a_state != p.b_state),
        "house number conflict": both("a_house_no", "b_house_no") & (p.a_house_no != p.b_house_no),
        "names very different (<60)": p.name_sim < 60,
        "names similar (>=90)": p.name_sim >= 90,
        "country = India": p.a_country == "India",
        "country = US": p.a_country == "US",
        "candidate from S2": p.cand // 10**10 == 2,
        "candidate from S3": p.cand // 10**10 == 3,
    }


def raw_lookup(keys):
    keys = set(int(k) for k in keys)
    out = {}
    for s in C.SOURCES:
        df = C.read_tsv(C.raw_path("train", s))
        k = C.id_to_key(df.entity_id)
        m = np.isin(k, list(keys))
        for kk, n, a in zip(k[m], df.business_name[m], df.business_address[m]):
            out[int(kk)] = (n, a)
    return out


def main():
    args = C.parse_args("Step 9: error analysis")
    d = C.work_dir(args)
    R = Report()
    dec = json.loads((d / "decision.json").read_text())
    oof = pd.read_parquet(d / "oof.parquet", columns=["s1", "cand", "label", "prob"])
    truth = pd.read_parquet(d / "truth.parquet")
    info = pd.read_parquet(d / "s1_info.parquet")

    with C.Timer("apply decision rule"):
        best = M.assign_best(oof)
        chosen = M.apply_rule(best, dec["method"], dec["param"])

    # ---------------- 1. pair level ----------------
    tp = int(chosen.label.sum())
    fp = int(len(chosen) - tp)
    n_true = len(truth)
    cand_pos = oof[oof.label == 1][["s1", "cand"]]
    fn_block = n_true - len(cand_pos)
    kept_pos = best[best.label == 1]
    fn_owner = len(cand_pos) - len(kept_pos)
    fn_thresh = len(kept_pos) - tp

    R(f"# Error analysis ({'full' if not args.sample else f'sample {args.sample:g}'} run, out-of-fold predictions)")
    R()
    R(f"Decision rule: `{dec['method']}` param={dec['param']}  |  validation macro F0.5 = {dec['val_f05']:.5f}")
    R()
    R("## 1. Pair-level confusion")
    R()
    R.table(pd.DataFrame([
        ("TP  true match, predicted", tp, tp / n_true),
        ("FP  wrong merge, predicted", fp, fp / max(len(chosen), 1)),
        ("FN  lost at blocking (never a candidate)", fn_block, fn_block / n_true),
        ("FN  one-owner rule gave it to another S1", fn_owner, fn_owner / n_true),
        ("FN  below the decision threshold", fn_thresh, fn_thresh / n_true),
    ], columns=["outcome", "pairs", "share"]))
    R(f"FP share is relative to predicted pairs ({len(chosen):,}); FN/TP shares relative to true pairs ({n_true:,}).")
    R(f"Pair precision {tp / max(len(chosen), 1):.4f}, pair recall {tp / n_true:.4f}. "
      "TN is not meaningful per pair (trillions of non-pairs), so it is reported per entity below.")
    R()

    # ---------------- 2. entity level ----------------
    s1_index = pd.Index(info.s1.to_numpy())
    pos = s1_index.get_indexer(chosen.s1.to_numpy())
    npred = np.bincount(pos, minlength=len(info))
    ntp = np.bincount(pos, weights=chosen.label.to_numpy(), minlength=len(info))
    nt = info.n_true.to_numpy()
    f = M.per_entity_f05(ntp, npred, nt)
    single = nt == 0
    R("## 2. Entity-level outcomes")
    R()
    R.table(pd.DataFrame([
        ("singleton, predicted empty (TN)", int((single & (npred == 0)).sum())),
        ("singleton, given a wrong match (FP)", int((single & (npred > 0)).sum())),
        ("has matches, all found, no wrong ones", int((~single & (ntp == nt) & (npred == ntp)).sum())),
        ("has matches, partly found, no wrong ones", int((~single & (ntp > 0) & (ntp < nt) & (npred == ntp)).sum())),
        ("has matches, at least one wrong merge", int((~single & (npred > ntp)).sum())),
        ("has matches, predicted empty", int((~single & (npred == 0)).sum())),
    ], columns=["entity outcome", "S1 entities"]))
    by = info.assign(f=f).groupby("country").f.mean()
    R(f"Macro F0.5 by country: {by.round(5).to_dict()}")
    R()

    # ---------------- 3. blocking ----------------
    cand_per_s1 = oof.groupby("s1").size().reindex(info.s1, fill_value=0)
    n_s1 = info.country.value_counts()
    n_q = pd.concat([pd.read_parquet(d / f"clean_train_s{s}.parquet", columns=["country"]).country
                     for s in (2, 3)]).value_counts()
    all_pairs = int((n_s1 * n_q.reindex(n_s1.index, fill_value=0)).sum())
    R("## 3. Blocking quality")
    R()
    R.table(pd.DataFrame([
        ("candidate pairs", f"{len(oof):,}"),
        ("candidates per S1 entity (mean / median / p95)",
         f"{cand_per_s1.mean():.1f} / {cand_per_s1.median():.0f} / {cand_per_s1.quantile(.95):.0f}"),
        ("S1 entities with no candidate", f"{(cand_per_s1 == 0).mean():.4f}"),
        ("all same-country S1 x S2/S3 pairs", f"{all_pairs:,}"),
        ("reduction ratio", f"{1 - len(oof) / all_pairs:.7f}"),
        ("blocking recall (true pairs among candidates)", f"{len(cand_pos) / n_true:.4f}"),
        ("F0.5 ceiling (perfect model on candidates)",
         f"{M.macro_f05(oof[oof.label == 1], s1_index, nt).mean():.5f}"),
    ], columns=["measure", "value"]))
    R(f"({int(n_q.sum()):,} S2/S3 records queried; each keeps its top-{C.TOP_K} S1 candidates.)")
    R()

    # ---------------- 4. causes ----------------
    with C.Timer("error causes"):
        t, idx = load_records(d)
        pred_pairs = attach(chosen[["s1", "cand", "label"]].reset_index(drop=True), t, idx)
        true_pairs = truth[["s1", "cand"]].copy()
        in_cand = true_pairs.merge(cand_pos.assign(c=1), on=["s1", "cand"], how="left").c.notna().to_numpy()
        found = true_pairs.merge(chosen[chosen.label == 1][["s1", "cand"]].assign(c=1),
                                 on=["s1", "cand"], how="left").c.notna().to_numpy()
        true_pairs = attach(true_pairs, t, idx)
        rows = []
        pc, tc = causes(pred_pairs), causes(true_pairs)
        for name in pc:
            mp, mt = pc[name].to_numpy(), tc[name].to_numpy()
            rows.append((name,
                         f"{mp.mean():.3f}", f"{1 - pred_pairs.label.to_numpy()[mp].mean():.4f}" if mp.any() else "-",
                         f"{mt.mean():.3f}", f"{1 - in_cand[mt].mean():.4f}" if mt.any() else "-",
                         f"{1 - found[mt].mean():.4f}" if mt.any() else "-"))
    R("## 4. Error rates by cause")
    R()
    R("FP rate = share of predicted pairs with this property that are wrong merges. "
      "Miss rates = share of true pairs with this property lost at blocking / not found at all.")
    R()
    R.table(pd.DataFrame(rows, columns=["property", "share of predicted", "FP rate",
                                        "share of true pairs", "lost at blocking", "missed overall"]))
    sim_bins = pd.cut(true_pairs.name_sim.to_numpy(), [-1, 40, 60, 80, 90, 100],
                      labels=["0-40", "40-60", "60-80", "80-90", "90-100"])
    g = pd.DataFrame({"bin": sim_bins, "found": found})
    R("Recall by name similarity of the true pair (token set ratio):")
    R()
    R.table(pd.DataFrame({"name similarity": g.groupby("bin", observed=False).size().index.astype(str),
                          "share of true pairs": (g.groupby("bin", observed=False).size() / len(g)).to_numpy(),
                          "recall": g.groupby("bin", observed=False).found.mean().to_numpy()}))

    # ---------------- 5. examples ----------------
    rng = np.random.default_rng(C.SEED)
    fp_ex = pred_pairs[pred_pairs.label == 0]
    fp_ex = fp_ex.iloc[rng.choice(len(fp_ex), min(10, len(fp_ex)), replace=False)] if len(fp_ex) else fp_ex
    fn_b = true_pairs[~in_cand]
    fn_b = fn_b.iloc[rng.choice(len(fn_b), min(8, len(fn_b)), replace=False)] if len(fn_b) else fn_b
    fn_m = true_pairs[in_cand & ~found]
    fn_m = fn_m.iloc[rng.choice(len(fn_m), min(8, len(fn_m)), replace=False)] if len(fn_m) else fn_m
    with C.Timer("read raw records for examples"):
        raw = raw_lookup(np.concatenate([x[c].to_numpy() for x in (fp_ex, fn_b, fn_m) for c in ("s1", "cand")]))
    probs = oof.set_index(["s1", "cand"]).prob
    R("## 5. Examples (original raw records)")
    for title, ex in (("False positives (wrong merges)", fp_ex), ("False negatives lost at blocking", fn_b),
                      ("False negatives rejected by the model", fn_m)):
        R()
        R(f"### {title}")
        R()
        for s1, c in zip(ex.s1, ex.cand):
            p = probs.get((s1, c), np.nan)
            a, b = raw.get(int(s1), ("?", "?")), raw.get(int(c), ("?", "?"))
            R(f"- prob {p:.3f}" if p == p else "- (not a candidate)")
            R(f"  - S1 : {a[0]} | {a[1]}")
            R(f"  - {'S2' if c // 10**10 == 2 else 'S3'} : {b[0]} | {b[1]}")

    C_out = d / "error_analysis.md"
    C_out.write_text("\n".join(R.lines), encoding="utf-8")
    print(f"\nsaved {C_out}")


if __name__ == "__main__":
    main()
