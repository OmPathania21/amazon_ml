"""Step 2: ground-truth lookup and cross-validation folds (train only).

Outputs:
  work/<run>/truth.parquet    cand (S2/S3 key) -> s1 (its true Source-1 key)
  work/<run>/s1_info.parquet  s1 key, country, n_true (number of true matches), fold
Folds are grouped by Source-1 entity and derived from a hash of its id, so they are
identical on every machine.
"""
import numpy as np
import pandas as pd

import config as C


def main():
    args = C.parse_args("Step 2: labels and folds")
    d = C.work_dir(args)
    s1 = pd.read_parquet(d / "clean_train_s1.parquet", columns=["key", "country"])

    with C.Timer("read ground truth"):
        gt = C.read_tsv(C.GT_PATH)
        gt["s1"] = C.id_to_key(gt.source1_entity_id)
        gt = gt[gt.s1.isin(s1.key)].reset_index(drop=True)
        m = gt.matched_entity_ids.str.split(",").explode()
        m = m[m.fillna("") != ""]
        truth = pd.DataFrame({"cand": C.id_to_key(m.to_numpy()),
                              "s1": gt.s1.to_numpy()[m.index.to_numpy()]})
        assert truth.cand.is_unique, "an S2/S3 record is matched to two S1 records"

    n_true = truth.groupby("s1").size()
    info = pd.DataFrame({"s1": s1.key.to_numpy(), "country": s1.country.to_numpy()})
    info["n_true"] = info.s1.map(n_true).fillna(0).astype(np.int16).to_numpy()
    info["fold"] = C.fold_of(info.s1.to_numpy())

    truth.to_parquet(d / "truth.parquet", index=False)
    info.to_parquet(d / "s1_info.parquet", index=False)

    print(f"S1 entities: {len(info):,}   true pairs: {len(truth):,}")
    print(f"no-match (singleton) rate: {(info.n_true == 0).mean():.4f}")
    print("S1 per fold:", info.fold.value_counts().sort_index().to_dict())
    print(info.groupby("country").n_true.agg(["size", "mean"]).to_string())


if __name__ == "__main__":
    main()
