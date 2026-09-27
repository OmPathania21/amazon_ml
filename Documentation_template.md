# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Overclocked
**Team Members:** Garvit Singh Rathore (Team Leader), Om Pathania, Sanskriti Rastogi, Shaurya Kesarwani
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

We built a CPU-only pipeline that scales to the full dataset (12.5M training records and 11.7M test records on a 16 GB laptop). It has four parts:

- **Normalization:** rule-based cleaning of names and addresses that works for any country.
- **Three-pass blocking:** separate TF-IDF searches over combined, name and address tokens, whose results are merged and then pruned.
- **Matching model:** a LightGBM classifier on 73 pair features, trained with 5-fold cross-validation grouped by Source-1 entity.
- **Decision rule:** a small second-stage LightGBM re-scores each pair from its probability in context (rank and gap among the other candidates of the same entity and record). Each S2/S3 record then goes to at most one S1 entity (the one-owner rule), and matches are selected per entity by maximizing expected F0.5.

Results on held-out training entities:

| Measure | Value |
|---|---|
| Macro F0.5 | **0.975** (0.97536) |
| Pair precision | 0.995 |
| Pair recall | 0.939 |
| Blocking recall | 96.7% |
| Candidates per S1 entity (mean) | 16.6 |

The three-pass blocking design was the single largest improvement. It came directly out of our error analysis of a first version, which had one blocking search: blocking recall rose from 86.0% to 96.7%, the candidate set shrank by 25%, and macro F0.5 rose from 0.919 to 0.974. A second-stage re-scoring model then added +0.0008, for 0.975.

---

## 2. Methodology

### 2.1 Problem Analysis

Key facts found during exploratory analysis (training set):

| Fact | Value | Consequence for the design |
|---|---|---|
| Records | S1 2.21M, S2 5.03M, S3 5.29M (test: 1.73M / 4.89M / 5.08M) | All-pairs comparison (11.8 × 10^12 same-country pairs in train) is impossible, so blocking is required |
| Matches per S1 entity | mean 3.46; **5.6% singletons** | Recall matters, but a false match on a singleton scores 0 |
| **Each S2/S3 record matches at most one S1 entity** | 7,638,365 matched IDs, all unique | Enables the **one-owner rule** and blocking from the S2/S3 side |
| Unmatched S2/S3 records ("decoys") | ~26% | Blocking pulls in look-alikes, and the model must reject them |
| Matches across countries | 0 | Blocking and features are computed within each country label |
| Countries | train US + India; **test adds France (15% of test S1)** | No country-specific model inputs; country is only used to group records |
| Name noise | abbreviations, legal suffixes, "doing business as" names, typos with digits (`C0mpany`, `8usiness`), word order, junk prefixes (`--`, `<<`, `***`), bracketed words, domain names (`ipower.com`), appended generic words (`Services`, `Partners`, `(India)`), **names written in Indian scripts** | Normalization, consonant-skeleton tokens, multiple fuzzy views |
| Address noise | abbreviations, reordered components, zero-padded house numbers (`00709`), `H.No/Door No/Plot No` labels, landmarks (`Near SBI ATM`), states written as names, codes or native script (`महाराष्ट्र`, `TG`), `CDP` suffixes, broken characters (`Â\x80\x93`), **~3.4% empty addresses** | Address parsed into components (house number, postcode, state, landmark, tokens) |
| Phone numbers | **not present** in the data (only name, address and country) | No phone features |

### 2.2 Solution Strategy

**Approach Type:** Blocking + classifier + constrained decision rule (hybrid rule/ML pipeline)

**Core Innovation:**

1. **Three-pass blocking with combined tokens.** Single words stop being selective at millions of records. Tokens such as *house number × name skeleton* or *postcode × name skeleton* stay rare, so they pin down the right business even at full scale. Separate name-only and address-only passes cover records with an empty address or a completely different trade name.
2. **Blocking from the S2/S3 side.** Each S2/S3 record belongs to at most one S1 entity, so every S2/S3 record retrieves only a handful of S1 candidates. This keeps the candidate set small.
3. **Relative and cross-source features.** Features describe each candidate relative to the other candidates of the same record (rank, score gap). They also compare it with the S1 entity's strongest other candidate: S2 and S3 records of the same business agree with each other.
4. **One-owner assignment plus expected-F0.5 selection.** After scoring, each S2/S3 record keeps only its best S1 entity. For each S1 entity we then choose the number of top candidates that maximizes expected F0.5, and predicting nothing is one of the options.

Pipeline:

```
raw TSV ──► 01 clean ──► 03 block (3 passes: combo / name / addr, union + pruning, per country)
        ──► 04 features (73) ──► 05 LightGBM (5-fold, grouped by S1) ──► 06 tune decision rule on OOF F0.5
        ──► 07 score test pairs (average of 5 models) ──► 06c second-stage re-scoring (5-fold)
        ──► 08 one-owner + decision rule ──► TSVs
```

---

## 3. Candidate Generation (Blocking)

**Normalization first (`normalize.py`, identical for train and test and for every country):**

- **Names:**
  - convert to ASCII (`unidecode`; Indian scripts and accents become Latin letters), remove broken characters, and flag names written in a non-Latin script;
  - collapse dotted acronyms (`L.L.C.` → `llc`);
  - `&`/`+` → `and`;
  - fix digit-for-letter typos inside words;
  - strip `M/s`;
  - split "doing business as" names (`dba`, `d/b/a`, `trading as`, …) into a main and an alternative name;
  - detect domain names;
  - map words to canonical forms (`limited→ltd`, `private→pvt`, `corporation→corp`, `centre→center`, …);
  - pull legal suffixes into a separate field (incl. French `SARL/SAS/EURL`), also when written in Indian scripts, via their consonant skeleton;
  - produce: core name, key name (core without generic words), consonant skeleton (`raam maarketting` → `rm mrktng` = `ram marketing`), and the name with spaces removed.
- **Addresses:**
  - convert to ASCII;
  - rejoin 6-digit PINs split by a space;
  - strip `cdp`;
  - split on commas, map state/region names and codes to one code (US states, Indian states incl. native-script spellings, French regions and departments), and remove them from the address text;
  - move landmark phrases into their own field;
  - remove number labels, rejoin compound house numbers (`5-513/4`), and strip leading zeros;
  - canonicalize street words (`street/str/saint→st`, `road→rd`, `boulevard/bd→blvd`, `rue`, `allee`, …);
  - extract house number, postcode and all numbers.

**Blocking keys used.** Each record gets three token sets, stored as stable 64-bit hashes, one per pass:

| Pass | Tokens | What it catches |
|---|---|---|
| **combo** | house/unit number × name skeleton, postcode × name skeleton, street word × name skeleton, pairs of name skeletons, joined name | Most matches, even when every single word is common |
| **name** | name words, consonant skeletons, skeleton pairs, joined name | Records with an **empty address** |
| **addr** | address words, house number, postcode, numbers, number × street-word pairs | **Different trade names** at the same address |

**How candidates are retrieved:**

1. Per country and per pass, tokens are weighted by IDF over Source 1.
2. Tokens present in more S1 records than `max(400, 0.3% of the country's S1 count)` are dropped. The cap grows with the data, so a cap tuned on a small sample does not over-prune at full scale.
3. Every S2/S3 record retrieves its **top-3 S1 records per pass** by TF-IDF cosine, using a multithreaded sparse top-k matrix product (`sparse_dot_topn`) in chunks of 50k queries.
4. Within a pass, a candidate that is not the pass's best is kept only if its score is ≥ 0.75 × the best score.
5. The three lists are merged. Each pair keeps its three pass scores, their sum, its rank and the number of passes that found it; all of these become model features.

**Candidate pairs generated:**

| | Train | Test |
|---|---|---|
| Candidate pairs | 36,642,595 | 36,809,656 |
| All same-country S1 × S2/S3 pairs | 11.84 × 10^12 | 6.72 × 10^12 |
| Reduction ratio | 0.9999969 | 0.9999945 |
| Candidates per S1 entity (mean / median / p95) | 16.6 / 9 / 42 | 21.2 / 12 / 58 |
| S1 entities without any candidate | 0.04% | 0.02% |
| Blocking time on a 16 GB laptop | ~20 min | ~10 min |

The test set has more S2/S3 records per S1 entity than the training set (5.75 vs 4.68), which explains its larger candidate count per entity.

**How we ensured true matches were not lost.** We measured blocking recall on the training ground truth and iterated on the design using error analysis:

| Blocking version | Recall | Candidates per S1 (mean) | F0.5 ceiling |
|---|---|---|---|
| v1: one TF-IDF pass, fixed df cap 400, top-5 | 0.860 | 22.2 | 0.936 |
| **v2: three passes, combined tokens, scaling cap, pruning (submitted)** | **0.967** (India 0.953, US 0.977) | **16.6** | **0.989** |

- Recall per pass on its own: combo 0.903, addr 0.806, name 0.594.
- The union adds the pairs that only one pass can find.
- The error analysis of v1 showed why its recall was low. For 40% of the missed pairs, all shared tokens had been removed by the fixed cap (for example `k_invstmnts`, found in 7,667 S1 records). The other 60% were outranked by look-alikes.

---

## 4. Matching Model

**Features used (73, computed with `rapidfuzz`, vectorized and multithreaded):**

- **Name features:**
  - token-set, token-sort, plain and partial ratios, and Jaro-Winkler on the core name;
  - token-set ratio on the full name and on the key name (generic words removed);
  - token-set and plain ratio on consonant skeletons;
  - ratio and partial ratio on the name with spaces removed (domains);
  - best score over the "doing business as" alternatives;
  - legal suffix equal / conflicting / missing;
  - first key word equal;
  - domain flags;
  - non-Latin-script flags;
  - name lengths and token counts.
- **Address features:**
  - token-set, token-sort, plain and partial ratios;
  - state equal / conflicting / missing;
  - postcode equal / conflicting / missing;
  - house number equal / conflicting / missing / partial;
  - **numeric house-number difference** (absolute and relative) and numeric equality;
  - Jaccard similarity and count of shared numbers;
  - empty-address flags;
  - landmark flag.
- **Blocking and ranking features:**
  - summed and per-pass scores (combo, name, addr);
  - number of passes that found the pair;
  - rank among the S2/S3 record's candidates;
  - gap to the next and to the best candidate;
  - ratio to the best;
  - number of candidates;
  - on the S1 side: number of candidates, rank, ratio to the best, and number of records for which it is the top candidate;
  - source (S2 or S3);
  - relative string features: name, address and combined similarity minus the best value among the same record's candidates, plus the combined-similarity rank.
- **Cross-source ("partner") features:** name and address similarity between the candidate and the S1 entity's strongest other candidate, plus that candidate's score. These help records with an empty address, whose partner in the other source confirms the business.

The country label is **never** used as a feature.

Most important features by gain:

| Feature | Gain share |
|---|---|
| combined similarity relative to best candidate | 0.42 |
| blocking score (sum of passes) | 0.20 |
| shared-number Jaccard | 0.08 |
| house-number numeric difference | 0.04 |
| address token-set ratio | 0.04 |
| combined-similarity rank | 0.04 |

**Model type:** LightGBM binary classifier (MIT licence, no pretrained models, far below the 8B-parameter limit).

| Parameter | Value |
|---|---|
| num_leaves | 127 |
| learning_rate | 0.1 |
| min_data_in_leaf | 200 |
| feature_fraction | 0.8 |
| bagging_fraction | 0.8 |
| lambda_l2 | 1.0 |
| Early stopping | 50 rounds, up to 2000 rounds |

- **5-fold cross-validation grouped by Source-1 entity.** Folds come from a deterministic hash of the entity ID, so they are identical on every machine.
- Each fold trains on 6M uniformly sampled pairs (the probabilities stay calibrated). Out-of-fold predictions are produced for **all** 36.6M training pairs.
- Best iterations were 869–1163, with validation AUC 0.99975.
- The test set is scored by averaging the 5 fold models.

**Threshold selection method:** Directly optimize macro F0.5 on the **out-of-fold** predictions of all 2.2M training entities:

1. **One-owner rule:** each S2/S3 record keeps only its highest-probability S1 entity.
2. We compared two decision rules on the out-of-fold predictions:
   - **(a) a global probability threshold** (best: 0.70, F0.5 0.97434);
   - **(b) per-entity expected-F0.5 selection** (F0.5 0.97448). Tuning the floor separately for India and the US gave 0.97457. For each S1 entity, sort its candidates by probability p and choose k to maximize `1.25·Σ_{i≤k} p_i / (0.25·Σ p + k)`. Compare this with the expected score of predicting nothing, `Π(1 − p_i)`, which protects singletons. A probability floor of 0.6 is applied first.

**Second-stage re-scoring (`06c_stack.py`).** A small LightGBM (63 leaves, about 100–150 trees) is trained with the same 5 folds on the out-of-fold probabilities. It uses 17 context features:
- the probability and its logit;
- for the pair's S1 entity and for its S2/S3 record: its rank, the gap to the best probability, the number of candidates, the maximum, the sum, and the count above 0.5;
- the second-best probability of the record;
- the pair's share of the entity's total probability.

These features correct decisions where several candidates compete. The expected-F0.5 rule is then re-tuned (probability floor 0.525), and validation macro F0.5 rises from **0.97457 to 0.97536**. The stage is applied only because it improved the out-of-fold score.

---

## 5. Results & Error Analysis

**Validation (out-of-fold, all 2,206,821 training S1 entities):**

| Metric | v1 (one-pass blocking) | **Submitted (v2)** |
|---|---|---|
| **Macro F0.5** | 0.919 | **0.97536** (0.97448 before re-scoring) |
| Macro F0.5, India / US | 0.921 / 0.918 | **0.967 / 0.980** |
| Pair precision | 0.994 | **0.995** |
| Pair recall | 0.831 | **0.939** |
| Singletons correctly predicted empty | 96.8% | **97.0%** |
| F0.5 ceiling given our candidates | 0.936 | **0.989** |

**Test-set sanity check (submitted file):**
- 3.29 matches per S1 entity. Validation predicts 3.46 × 0.939 ÷ 0.995 ≈ 3.27.
- 5.8% of entities predicted empty, against a true singleton rate of 5.6%.
- **France 3.38**, India 3.23, US 3.33 matches per entity. France, which never appears in training, behaves like the training countries.

**Pair-level confusion (out-of-fold):**

| Outcome | Pairs | Share |
|---|---|---|
| TP (true match, predicted) | 7,172,391 | 93.9% of true pairs |
| FP (wrong merge) | 36,694 | 0.51% of predicted pairs |
| FN: lost at blocking | 249,654 | 3.3% of true pairs |
| FN: one-owner rule gave the record to another S1 | 44,814 | 0.6% |
| FN: below the decision threshold | 171,506 | 2.2% |

**Entity-level outcomes:**

| Outcome | S1 entities |
|---|---|
| Singleton, predicted empty (TN) | 119,581 |
| Singleton, given a wrong match (FP) | 3,666 |
| All matches found, no wrong ones | 1,677,197 |
| Partly found, no wrong ones | 362,734 |
| At least one wrong merge | 32,029 |
| Has matches, predicted empty | 11,614 |

**Error rates by cause:**

| Property | FP rate among predictions | True pairs lost at blocking | True pairs missed overall |
|---|---|---|---|
| Candidate address missing (4.4% of true pairs) | **3.3%** | **29.1%** | **54.2%** |
| Names very different, token-set < 60 (9.6%) | 0.8% | 10.1% | 14.0% |
| House-number conflict (22.0%) | 1.1% | 5.0% | 10.3% |
| Candidate name is a domain (4.7%) | 0.1% | 7.6% | 8.0% |
| State conflict (1.1%) | 0.5% | 2.8% | 5.9% |
| Names similar, ≥ 90 (74.4%) | 0.45% | 1.6% | 4.3% |
| India / US | 0.7% / 0.4% | 4.7% / 2.3% | 7.7% / 5.1% |

Recall by name similarity of the true pair rises from 0.85 (token-set < 40) to 0.96 (≥ 90). Compared with v1, recall on very dissimilar names went from 0.66 to 0.85.

**Common false positives (wrong merges):**
- **Near-identical business at the same address with a different legal form or an extra word**: `Mint Brush Public Limited` vs `Public Mint Brush Overseas Ltd`, `North Producer Limited` vs `North Producer Private Limited`, `Archana Brothers` vs `Archana Brothers Group`. The ground truth treats these as different entities, although name and address alone barely distinguish them.
- **Candidate with no address:** the name matches but nothing confirms the location (`Bright Software Brands L.L.C.`, `Advanced (india) (Private)`). This category has the highest FP rate, 3.3%.
- **Native-script names that share only a generic word:** `ஃபர்ஸ்ட் ஹாஸ்பிடாலிட்டி` ("First Hospitality") matched to `Tirupati Hospitality` at the same address.

**Common false negatives (missed matches):**
- **Empty addresses** remain the hardest case: 29% are lost at blocking, because only name tokens are available and typos (`Hutton Nrvona` for `Hutton Nevada`) or truncated names (`Siddhi &`) leave too few rare tokens.
- **Names written in Indian scripts:** `न्यू सॉल्यूशंस प्राइवेट लिमिटेड` for `New Solutions Pvt Ltd`, and `ஸ்டார் ஃபுட்` for `Star Food`. After unidecode, their words share little with the English name. This is the main reason India (0.967) trails the US (0.980).
- **Completely different trade names at the same address** (`Big Coffee` vs `Zephcirajax`, `Grand Infrastructure Studios` vs `Avionyxiri`). The model keeps them below the threshold because the same pattern (same address, different name) also produces false positives.

---

## 6. Conclusion

- A scalable, CPU-only pipeline resolves the full dataset on a 16 GB laptop and reaches validation macro F0.5 = 0.975 at 99.5% pair precision. The pipeline combines normalization, three-pass combined-token blocking, LightGBM with ranking and cross-source features, a second-stage re-scoring model, and a one-owner, expected-F0.5 decision rule.
- The blocking design came from error analysis:
  - blocking recall rose from 86.0% to 96.7%;
  - the candidate set shrank to 16.6 per S1 entity;
  - the matcher is now within 0.014 of the 0.989 ceiling set by its candidates.
- Remaining errors concentrate on empty addresses and names written in Indian scripts. Planned next steps:
  - the learned transliteration dictionary already included in `02b_translit.py` (native-script word → English word, learned from training pairs only; not used in the submitted run);
  - multilingual sentence embeddings (e.g. `multilingual-e5-small`, MIT) as a fourth blocking pass and an extra feature.

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
| `config.py` | Paths, all hyper-parameters (blocking passes, caps, LightGBM), fold hashing, sample mode |
| `normalize.py` | Every cleaning rule and the three blocking token sets (`python src/normalize.py` prints worked examples) |
| `features.py` | Pair, ranking, house-number and partner features |
| `metrics.py` | Macro F0.5, one-owner rule, threshold and expected-F0.5 decision rules |
| `01_clean.py` | Raw TSV → cleaned Parquet per source (multiprocessing, written in 500k-row pieces) |
| `02_labels_folds.py` | Ground-truth lookup + 5 folds grouped by S1 entity |
| `02b_translit.py` | *Optional, not used for the submitted results:* learned native-script → English word dictionary |
| `03_block.py` | Three-pass per-country TF-IDF blocking, union + pruning; reports blocking recall |
| `04_features.py` | Features for every candidate pair, in 1M-pair chunks |
| `05_train.py` | LightGBM 5-fold training, out-of-fold predictions, feature importance |
| `06_tune.py` | Decision-rule search on out-of-fold macro F0.5 (global and per country) |
| `06c_stack.py` | Second-stage re-scoring model on the out-of-fold / test probabilities; used only if it improves validation F0.5 |
| `07_predict.py` | Test scoring, average of the 5 models |
| `08_write_submission.py` | Writes `matching_results.tsv` + `candidate_pairs.tsv`, runs the official validator |
| `09_error_analysis.py` | This section's confusion tables, causes and examples |
| `10_make_zip.py` | Builds the submission package |

To reproduce the submitted output files:

```
python src/00_download_data.py
python src/01_clean.py && python src/02_labels_folds.py && python src/03_block.py
python src/04_features.py && python src/05_train.py && python src/06_tune.py
python src/01_clean.py --split test && python src/03_block.py --split test
python src/04_features.py --split test && python src/07_predict.py
python src/06c_stack.py && python src/08_write_submission.py
```

### B. Additional Results

**Runtime on an Intel i5-13420H laptop (8 cores / 12 threads, 16 GB RAM, CPU only).**
Step times below are the core compute time printed by the pipeline's own timers. They exclude loading and saving the intermediate Parquet files (several GB per step), the blocking-recall report, script start-up and the gaps between steps, so the real end-to-end wall-clock time was noticeably longer than the totals shown.

| Step | Train | Test |
|---|---|---|
| Cleaning (12.5M / 11.7M records) | 4.2 min | 4.1 min |
| Labels + folds | 0.2 min | – |
| Blocking, 3 passes | ~20 min | 9.9 min |
| Features (36.6M / 36.8M pairs) | 14.2 min | 13.8 min |
| LightGBM, 5 folds (46 min training + 10 min out-of-fold prediction) | 56 min | – |
| Decision-rule search | 3.0 min | – |
| Test scoring (5 models × 36.8M pairs) | – | 45.7 min |
| **Total measured compute** | **~1 h 40 min** | **~1 h 15 min** |

The whole train + test pipeline needs roughly **3 hours of measured compute**, plus file loading and saving, on a single 16 GB laptop. Test scoring became 2.3× faster than in v1 (106 → 46 min), because the candidate set is smaller and the models use fewer trees.

**Reproducibility:**
- Deterministic hash-based folds, sampling and blocking-token hashing.
- Fixed seeds (`SEED = 42`).
- Pinned package versions.
- Cleaning and blocking are written crash-safely (temporary file, then rename).
- A `--sample 0.05` mode reruns the whole training pipeline on 5% of the entities in about 10 minutes.

**Fair play:**
- No external data, APIs or geocoding are used.
- The only built-in knowledge is small hand-written tables in `normalize.py`: abbreviation expansions, legal-suffix lists, and state/region names and codes (including native-script spellings of Indian state names and the French department → region mapping). This is general language and geography knowledge written into the code, not a lookup of any business or address.
- The model is LightGBM (MIT licence); no pretrained models are used.
