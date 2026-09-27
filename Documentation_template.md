# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Overclocked
**Team Members:** Garvit Singh Rathore (Team Leader), Om Pathania, Sanskriti Rastogi, Shaurya Kesarwani
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

We built a CPU-only pipeline that scales to the full dataset. The steps are:

- **Normalization:** rule-based cleaning of names and addresses that works for any country.
- **Blocking:** IDF-weighted token blocking run from the Source-2/3 side (TF-IDF cosine, sparse top-k), which cuts 11.8 trillion possible pairs to about 49M candidates.
- **Matching:** a LightGBM classifier on 61 pair features, trained with 5-fold cross-validation grouped by Source-1 entity.
- **Decision rule:** each S2/S3 record goes to at most one S1 entity (the one-owner rule), and matches are then selected per entity by maximizing expected F0.5.

On held-out training entities the pipeline reaches **macro F0.5 = 0.919**, with **pair precision 0.994** and pair recall 0.831. The error analysis shows that the remaining loss comes mainly from blocking recall (86%), not from the matching model.

---

## 2. Methodology

### 2.1 Problem Analysis

Key facts found during exploratory analysis (training set):

| Fact | Value | Consequence for the design |
|---|---|---|
| Records | S1 2.21M, S2 5.03M, S3 5.29M (test: 1.73M / 4.89M / 5.08M) | All-pairs comparison is impossible, so blocking is required |
| Matches per S1 entity | mean 3.46; **5.6% singletons** | Recall matters, but a false match on a singleton scores 0 |
| **Each S2/S3 record matches at most one S1 entity** | 7,638,365 matched IDs, all unique | Enables the **one-owner rule** and blocking from the S2/S3 side |
| Unmatched S2/S3 records ("decoys") | ~26% | Blocking pulls in look-alikes, and the model must reject them |
| Matches across countries | 0 | Blocking and features are computed within each country label |
| Countries | train US + India; **test adds France (18% of test S1)** | No country-specific model inputs; country is only used to group records |
| Name noise | abbreviations, legal suffixes, "doing business as" names, typos with digits (`C0mpany`, `8usiness`), word order, junk prefixes (`--`, `<<`), bracketed words, domain names (`ipower.com`), generic words appended (`Services`, `Partners`, `(India)`), **names in Indian scripts** (Devanagari, Telugu, Kannada, Tamil, Bengali, Malayalam, Gujarati, Gurmukhi, Odia) | Normalization, consonant-skeleton tokens, multiple fuzzy views |
| Address noise | abbreviations, reordered components, zero-padded house numbers (`00709`), `H.No/Door No/Plot No` labels, landmarks (`Near SBI ATM`), states written as names, codes or native script (`महाराष्ट्र`, `TG`), `CDP` suffixes, broken characters (`Â\x80\x93`), **~3.4% empty addresses** | Address parsed into components (house number, postcode, state, landmark, tokens) |
| Phone numbers | **not present** in the data (only name, address and country) | No phone features |

### 2.2 Solution Strategy

**Approach Type:** Blocking + classifier + constrained decision rule (hybrid rule/ML pipeline)

**Core Innovation:** Three ideas carry most of the design:

1. **Blocking from the S2/S3 side.** Because each S2/S3 record belongs to at most one S1 entity, every S2/S3 record retrieves only its top-5 S1 candidates.
2. **Ranking features.** Features describe each candidate relative to the other candidates of the same S2/S3 record: its rank, and its score gap to the best candidate.
3. **One-owner assignment plus expected-F0.5 selection.** After scoring, each S2/S3 record keeps only its best S1 entity. For each S1 entity we then choose the number of top candidates that maximizes expected F0.5, and predicting nothing is one of the options.

Pipeline:

```
raw TSV ──► 01 clean ──► 03 block (top-5 per S2/S3, per country) ──► 04 features (61)
        ──► 05 LightGBM (5-fold, grouped by S1) ──► 06 tune decision rule on out-of-fold F0.5
        ──► 07 score test pairs (average of 5 models) ──► 08 one-owner + decision rule ──► TSVs
```

---

## 3. Candidate Generation (Blocking)

**Normalization first (`normalize.py`, identical for train and test and for every country):**

- **Names:**
  - convert to ASCII (`unidecode`; Indian scripts and accents become Latin letters) and remove broken characters;
  - collapse dotted acronyms (`L.L.C.` → `llc`);
  - `&`/`+` → `and`;
  - fix digit-for-letter typos inside words;
  - strip `M/s`;
  - split "doing business as" names (`dba`, `d/b/a`, `trading as`, …) into a main name and an alternative name;
  - detect domain names;
  - map words to canonical forms (`limited→ltd`, `private→pvt`, `corporation→corp`, `centre→center`, …);
  - pull legal suffixes into a separate field (incl. French `SARL/SAS/EURL`), also when written in Indian scripts, via their consonant skeleton;
  - produce: core name, key name (core without generic words), consonant skeleton (`raam maarketting` → `rm mrktng` = `ram marketing`), and the name with spaces removed.
- **Addresses:**
  - convert to ASCII;
  - rejoin 6-digit PINs split by a space (`400 021`);
  - strip `cdp`;
  - split on commas, and map state/region names and codes to one code (US states, Indian states incl. native-script spellings, French regions and departments), then remove them from the address text;
  - move landmark phrases into their own field;
  - remove number labels, rejoin compound house numbers (`5-513/4`), and strip leading zeros;
  - canonicalize street words (`street/str/saint→st`, `road→rd`, `boulevard/bd→blvd`, `rue`, `allee`, …);
  - extract house number, postcode (5–6 digits, not the leading house number), and all numbers.

**Blocking keys used:** For each record, a set of prefixed tokens:
- core and alternative name words;
- name consonant skeletons;
- the name with spaces removed;
- address words (≥3 characters);
- house number;
- postcode;
- all numbers of ≥2 digits.

**How candidates are retrieved:**
- Tokens are weighted by IDF over Source 1, per country.
- Tokens present in more than `MAX_DF = 400` S1 records are dropped.
- Every S2/S3 record retrieves its **top-5 S1 records by TF-IDF cosine**, using a multithreaded sparse top-k matrix product (`sparse_dot_topn`) in chunks of 50k queries.

**Candidate pairs generated:**

| | Train | Test |
|---|---|---|
| Candidate pairs | 48,923,447 | 48,444,503 |
| All same-country S1 × S2/S3 pairs | 11.84 × 10^12 | ≈ 6.7 × 10^12 |
| Reduction ratio | 0.9999959 | ≈ 0.9999928 |
| Candidates per S1 entity (mean / median / p95) | 22.2 / 8 / 80 | ~28 mean |
| Blocking time on a 16 GB laptop | ~4 min | ~4 min |

**How we ensured true matches were not lost:**
- We retrieve from the S2/S3 side, which uses the one-owner property: each S2/S3 record only needs its single true owner among its candidates.
- Several complementary token types are used: exact words, consonant skeletons for transliteration and typos, the joined name for domains, and numbers and postcodes.
- Recall was measured on training data:

| Blocking recall | top-1 | top-3 | top-5 |
|---|---|---|---|
| Full training data | 0.792 | 0.842 | **0.860** |
| 5% trial (lower density) | 0.971 | 0.983 | 0.987 |

**Limitation (quantified in Section 5):** the `MAX_DF` cap was tuned on the 5% trial. At full scale, many informative tokens (street names, house numbers, common name words) exceed 400 S1 records and are dropped. As a result, blocking recall is 86% and the F0.5 ceiling is 0.936.

---

## 4. Matching Model

**Features used (61, all computed with `rapidfuzz`, vectorized and multithreaded):**

- **Name features:**
  - token-set, token-sort, plain and partial ratios, and Jaro-Winkler on the core name;
  - token-set ratio on the full name and on the key name (generic words removed);
  - token-set and plain ratio on consonant skeletons;
  - ratio and partial ratio on the name with spaces removed (for domain names);
  - best score over the "doing business as" alternatives;
  - legal suffix equal / conflicting / missing;
  - first key word equal;
  - domain flags;
  - name lengths and token counts.
- **Address features:**
  - token-set, token-sort, plain and partial ratios;
  - state equal / conflicting / missing;
  - postcode equal / conflicting / missing;
  - house number equal / conflicting / missing / partial (one number contains the other);
  - Jaccard similarity and count of shared numbers;
  - empty-address flags for each side;
  - landmark flag.
- **Blocking and ranking features:**
  - TF-IDF cosine score;
  - rank among the S2/S3 record's candidates;
  - gap to the next and to the best candidate;
  - ratio to the best score;
  - number of candidates;
  - on the S1 side: number of candidates, rank, and ratio to the best;
  - number of S2/S3 records for which this S1 is the top candidate;
  - source (S2 or S3);
  - relative string features: name, address and combined similarity minus the best value among the same record's candidates, plus the combined-similarity rank.

The country label is **never** used as a feature.

Most important features by gain:

| Feature | Gain share |
|---|---|
| combined similarity relative to best candidate | 0.47 |
| address token-set ratio | 0.13 |
| combined name+address similarity | 0.11 |
| rank of combined similarity | 0.06 |
| shared-number Jaccard | 0.04 |

**Model type:** LightGBM binary classifier (MIT licence, no pretrained models, far below the 8B-parameter limit).

| Parameter | Value |
|---|---|
| num_leaves | 127 |
| learning_rate | 0.05 |
| min_data_in_leaf | 200 |
| feature_fraction | 0.8 |
| bagging_fraction | 0.8 |
| lambda_l2 | 1.0 |
| Early stopping | 100 rounds, up to 3000 rounds |

- **5-fold cross-validation grouped by Source-1 entity.** Folds come from a deterministic hash of the entity ID, so they are identical on every machine.
- Each fold trains on 6M uniformly sampled pairs, which keeps the probabilities calibrated.
- Best iterations were 1250–1891, with validation AUC 0.9998.
- The test set is scored by averaging the 5 fold models.

**Threshold selection method:** Directly optimize macro F0.5 on the **out-of-fold** predictions of all 2.2M training entities:

1. **One-owner rule:** each S2/S3 record keeps only its highest-probability S1 entity.
2. We compared two decision rules on the out-of-fold predictions:
   - **(a) a global probability threshold** (best: 0.65, F0.5 0.91901);
   - **(b) per-entity expected-F0.5 selection** (chosen: F0.5 0.91921). For each S1 entity, sort its candidates by probability p and choose k to maximize `1.25·Σ_{i≤k} p_i / (0.25·Σ p + k)`. Compare this with the expected score of predicting nothing, `Π(1 − p_i)`, which protects singletons. A probability floor of 0.6 is applied first.

---

## 5. Results & Error Analysis

**Validation (out-of-fold, all 2,206,821 training S1 entities):**

| Metric | Value |
|---|---|
| **Macro F0.5** | **0.91921** (India 0.9209, US 0.9181) |
| Pair precision | 0.9938 |
| Pair recall | 0.8310 |
| Singletons correctly predicted empty | 96.8% |
| F0.5 ceiling given our candidates | 0.93647 |
| Leaderboard (public), full models | [fill in] |
| Leaderboard (public), 5% trial models | [fill in] |

**Test-set sanity check (full models):**
- 3.03 matches per S1 entity, with **France 3.03**, India 2.98 and US 3.13;
- 7.4% of entities predicted empty.

France, which never appears in training, behaves like the training countries.

**Pair-level confusion (out-of-fold):**

| Outcome | Pairs | Share |
|---|---|---|
| TP (true match, predicted) | 6,347,409 | 83.1% of true pairs |
| FP (wrong merge) | 39,669 | 0.62% of predicted pairs |
| FN: lost at blocking | 1,066,988 | **14.0%** of true pairs |
| FN: one-owner rule gave the record to another S1 | 39,158 | 0.5% |
| FN: below the decision threshold | 184,810 | 2.4% |

**Entity-level outcomes:**

| Outcome | S1 entities |
|---|---|
| Singleton, predicted empty (TN) | 119,296 |
| Singleton, given a wrong match (FP) | 3,951 |
| All matches found, no wrong ones | 1,254,702 |
| Partly found, no wrong ones | 721,174 |
| At least one wrong merge | 34,719 |
| Has matches, predicted empty | 72,979 |

**Error rates by cause:**

| Property | FP rate among predictions | True pairs lost at blocking | True pairs missed overall |
|---|---|---|---|
| Candidate address missing (4.4% of true pairs) | **4.7%** | **38.0%** | **60.8%** |
| Names very different, token-set < 60 (9.6%) | 1.1% | 28.7% | 32.3% |
| House-number conflict (22.0%) | 1.5% | 21.2% | 27.7% |
| Candidate name is a domain (4.7%) | 0.1% | 19.0% | 19.4% |
| State conflict (1.1%) | 0.6% | 15.5% | 18.4% |
| Names similar, ≥ 90 (74.4%) | 0.5% | 9.1% | 12.1% |

**Common false positives (wrong merges):**
- **Same name, nearby house number** on the same street: `Katz Pelican Inc, 190 Wall St` vs `192 Wall St`, and `Adams Holding Company, 330` vs `334 Waterwheel Way`. The ground truth treats these as different entities, but they are nearly indistinguishable from name and address alone.
- **Candidate with no address:** the name matches but there's nothing to confirm location (`Slater Generating`, `Continenta1 Apex`). This category has the highest FP rate, 4.7%.
- **Same address, different business** (co-located businesses): `Bombay Ventures` vs `Umbradelta` at the same Bihar address.

**Common false negatives (missed matches):**
- **Lost at blocking (14%, the dominant error).**
  - Records with **empty addresses** only have name tokens, and common name words are dropped by the df cap.
  - **Completely different trade names** at the same address (`Excellent & Co` vs `Vioumbraumbra`) depend on address tokens that are too frequent at full scale.
  - **Heavy typos** in rare tokens (`Naridoanl`, `Dhbaoon`) plus a changed house number (`11674` vs `11666`) leave too few shared tokens.
- **Rejected by the model (2.4%).**
  - House numbers differing by one digit (`100` vs `200 Pheasant Lane`), which the model has learned to distrust because of the false-positive pattern above.
  - Names written in Indian scripts whose transliteration shares little with the English name (`यूनिक टेक्नोलॉजीज` vs `Unique Technologies`).

---

## 6. Conclusion

- A scalable pipeline (normalization, IDF token blocking, LightGBM with ranking features, and a one-owner, expected-F0.5 decision rule) resolves the full dataset on a 16 GB laptop without a GPU. It reaches validation macro F0.5 = 0.919 at 99.4% pair precision, and transfers to the unseen French data.
- The error analysis shows the matcher is within 0.017 of the ceiling set by its candidates. The largest remaining gain is in **blocking**: 86% recall, with too many candidates per entity.
- Next steps:
  - composite tokens that stay rare at scale (house number + street, name bigrams, postcode + name);
  - a document-frequency cap that scales with the data;
  - an adaptive number of candidates;
  - multilingual sentence embeddings (e.g. `multilingual-e5-small`, MIT) as an extra blocking signal and extra feature, which would help native-script and very different names.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`:
- `README.md`: exact run commands;
- `requirements.txt`: pinned environment (Python 3.13.1);
- `src/`: all source code.

| File | Role |
|---|---|
| `00_download_data.py` | Downloads the challenge zip and arranges `data/train`, `data/test` |
| `config.py` | Paths, all hyper-parameters, fold hashing, sample mode |
| `normalize.py` | Every cleaning rule (`python src/normalize.py` prints worked examples) |
| `features.py` | Pair features, ranking features |
| `metrics.py` | Macro F0.5, one-owner rule, threshold and expected-F0.5 decision rules |
| `01_clean.py` | Raw TSV → cleaned Parquet per source (multiprocessing) |
| `02_labels_folds.py` | Ground-truth lookup + 5 folds grouped by S1 entity |
| `03_block.py` | Per-country TF-IDF token blocking, top-5 per S2/S3 record; reports blocking recall |
| `04_features.py` | Features for every candidate pair, in 1M-pair chunks |
| `05_train.py` | LightGBM 5-fold training, out-of-fold predictions, feature importance |
| `06_tune.py` | Decision-rule search on out-of-fold macro F0.5 |
| `07_predict.py` | Test scoring, average of the 5 models |
| `08_write_submission.py` | Writes `matching_results.tsv` + `candidate_pairs.tsv`, runs the official validator |
| `09_error_analysis.py` | This section's confusion tables, causes and examples |
| `10_make_zip.py` | Builds the submission package |

To reproduce both output files:

```
python src/00_download_data.py
python src/01_clean.py && python src/02_labels_folds.py && python src/03_block.py
python src/04_features.py && python src/05_train.py && python src/06_tune.py
python src/01_clean.py --split test && python src/03_block.py --split test
python src/04_features.py --split test && python src/07_predict.py && python src/08_write_submission.py
```

### B. Additional Results

**Runtime on an Intel i5-13420H laptop (8 cores / 12 threads, 16 GB RAM, CPU only):**

| Step | Train | Test |
|---|---|---|
| Cleaning (12.5M / 11.7M records) | 6.5 min | ~4 min |
| Blocking | 4.1 min | 3.9 min |
| Features (48.9M / 48.4M pairs) | 13.8 min | 14.0 min |
| LightGBM, 5 folds (train + out-of-fold predict) | 81 min | – |
| Test scoring (5 models × 48.4M pairs) | – | 106 min |

**Reproducibility:**
- Deterministic hash-based folds and sampling.
- Fixed seeds (`SEED = 42`).
- Pinned package versions.
- A `--sample 0.05` mode reruns the whole training pipeline on 5% of the entities in about 10 minutes.

**Fair play:**
- No external data, APIs or geocoding are used.
- The only built-in knowledge is small hand-written tables in `normalize.py`: abbreviation expansions, legal-suffix lists, and state/region names and codes (including native-script spellings of Indian state names and the French department → region mapping). This is general language and geography knowledge written into the code, not a lookup of any business or address.
- The model is LightGBM (MIT licence); no pretrained models are used.
