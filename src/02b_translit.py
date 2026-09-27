"""Step 2b: learn a transliteration dictionary from the training pairs and apply it.

Many Indian records write the business name in a native script (Devanagari, Telugu,
Kannada, ...). After unidecode these become e.g. "praaim inphraasttrkcr" for "Prime
Infrastructure", which shares almost no tokens with the English name.

The training ground truth pairs English Source-1 names with their native-script S2/S3
versions. Aligning the words of those pairs position by position (same number of core
words) gives counts like  praaim -> prime, yuunik -> unique. A mapping is kept when it
is frequent and dominant. Only the provided training data is used.

The dictionary is then applied to every record whose name was in a non-Latin script
(train and test): name_core / name_alt / name_clean words are replaced and the derived
fields (name_key, name_skel, name_nospace) recomputed.

Usage:  python src/02b_translit.py              (train: learn + apply)
        python src/02b_translit.py --split test (test: apply the dictionary learned on train)
Output: work/<run>/translit.json, cleaned parquet files updated in place.
"""
import json
import os

import numpy as np
import pandas as pd

import config as C
from normalize import GENERIC, skel

MIN_COUNT = 3        # pair seen at least this often
MIN_SHARE = 0.5      # and it is the translation of this token at least half of the time
NAME_COLS = ["name_clean", "name_core", "name_alt"]


def learn(d):
    truth = pd.read_parquet(d / "truth.parquet")
    s1 = pd.read_parquet(d / "clean_train_s1.parquet", columns=["key", "name_core", "name_nonascii"])
    q = pd.concat([pd.read_parquet(d / f"clean_train_s{s}.parquet", columns=["key", "name_core", "name_nonascii"])
                   for s in (2, 3)], ignore_index=True)
    q = q[q.name_nonascii == 1]
    s1 = s1[s1.name_nonascii == 0]
    pairs = (truth.merge(q.rename(columns={"key": "cand", "name_core": "b"}), on="cand")
                  .merge(s1.rename(columns={"key": "s1", "name_core": "a"}), on="s1"))
    a, b = pairs.a.str.split(), pairs.b.str.split()
    same = (a.str.len() == b.str.len()) & (a.str.len() > 0)
    a, b = a[same], b[same]
    al = pd.DataFrame({"src": np.concatenate(b.to_numpy()), "dst": np.concatenate(a.to_numpy())})
    cnt = al.groupby(["src", "dst"]).size().rename("n").reset_index()
    tot = cnt.groupby("src").n.transform("sum")
    cnt = cnt[(cnt.n >= MIN_COUNT) & (cnt.n / tot >= MIN_SHARE) & (cnt.src != cnt.dst)
              & (cnt.src.str.len() >= 2) & ~cnt.src.str.isdigit()]
    mapping = dict(zip(cnt.src, cnt.dst))
    print(f"aligned native-script training pairs: {int(same.sum()):,} of {len(pairs):,}")
    print(f"dictionary entries: {len(mapping):,}")
    for k in list(mapping)[:15]:
        print(f"   {k} -> {mapping[k]}")
    return mapping


def remap(text, mapping):
    return " ".join(mapping.get(w, w) for w in text.split())


def apply(path, mapping):
    df = pd.read_parquet(path)
    m = (df.name_nonascii == 1).to_numpy()
    if not m.any():
        return 0
    sub = df.loc[m, NAME_COLS].copy()
    for c in NAME_COLS:
        sub[c] = [remap(x, mapping) for x in sub[c]]
    core = sub.name_core.str.split()
    sub["name_key"] = [" ".join([w for w in ws if w not in GENERIC] or ws) for ws in core]
    sub["name_skel"] = [" ".join(skel(w) for w in ws) for ws in core]
    sub["name_nospace"] = ["".join(ws) for ws in core]
    changed = int((sub.name_core != df.loc[m, "name_core"]).sum())
    for c in sub.columns:
        df.loc[m, c] = sub[c].to_numpy()
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    return changed


def main():
    args = C.parse_args("Step 2b: transliteration dictionary", split=True)
    d = C.work_dir(args)
    flag = d / f"translit_applied_{args.split}.flag"
    if flag.exists() and not args.force:
        print(f"[skip] dictionary already applied to {args.split} (use --force after re-running 01_clean)")
        return
    if args.split == "train":
        with C.Timer("learn transliteration dictionary"):
            mapping = learn(d)
        (d / "translit.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=0), encoding="utf-8")
    else:
        src = C.WORK_ROOT / "full" / "translit.json"
        assert src.exists(), "run 02b_translit.py on the train split first"
        mapping = json.loads(src.read_text(encoding="utf-8"))
    for s in C.SOURCES:
        path = d / f"clean_{args.split}_s{s}.parquet"
        with C.Timer(f"apply to {path.name}"):
            print(f"     names changed: {apply(path, mapping):,}")
    flag.touch()


if __name__ == "__main__":
    main()
