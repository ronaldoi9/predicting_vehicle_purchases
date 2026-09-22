# Data profile: train.csv / test.csv

Resolves research ticket [#2](https://github.com/ronaldoi9/predicting_vehicle_purchases/issues/2).
Read-only analysis; no model was trained on the target. Scripts were ad hoc and live outside the repo.

## TL;DR — the four facts that change downstream decisions

1. **Zero exact duplicate feature rows anywhere.** All 668,665 train rows are distinct on the 13 features, all 286,571 test rows are distinct, and the train/test feature-vector intersection is **empty (0 rows)**. Plain `StratifiedKFold` on `Will_Buy_EV` does **not** leak through duplicated rows. Near-duplicates exist but are weak (see §5).
2. **Single-feature AUC is dominated by three columns**: `Environmental_Concern_Level` 0.8435, `Subsidy_Available` 0.7179, `Annual_Income_USD` 0.6704. Everything else is ≤ 0.551. The two categorical leaders behave like near-deterministic gates.
3. **No train/test drift.** Adversarial validation (HistGradientBoosting, 5-fold CV, all 13 features) gives **AUC 0.49877** — a classifier cannot tell the files apart. Largest column-wise difference: TVD 0.00372 (`Charging_Stations_Near_Work`); largest KS 0.00229 (`Age`).
4. **`id` is pure row order and carries nothing.** Train = 0…668,664, test = 668,665…955,235, both contiguous, sorted, non-overlapping, perfectly partitioned. Single-feature AUC of `id` = 0.50001; target rate by `id` decile ranges 0.17142–0.17568. Drop `id` from the feature set.

## 1. Shapes, schema, missing values

| file | rows | cols |
|---|---|---|
| `data/train.csv` | **668,665** | 15 (id + 13 features + target) |
| `data/test.csv` | **286,571** | 14 (id + 13 features) |
| `data/sample_submission.csv` | **286,571** | 2 |

**Correction to the map:** the map (issue #1) states 286,572 test rows. The true count is **286,571**, confirmed independently by `sample_submission.csv`. Off-by-one, probably a header miscount.

`sample_submission.csv` ids equal `test.csv` ids exactly, in the same order, and the prediction column is the constant `0.1746450016076809` — i.e. the train positive rate.

**Missing values: zero in every column of both files** (confirmed, 13/13 feature columns + id + target in train, 14/14 in test). The source dataset description's promised ~2 % missingness did not survive the synthetic regeneration. Nothing to impute; no missingness indicator is available as a feature.

Dtypes: `id`, `Age`, `Number_of_Cars_Owned`, `Charging_Stations_Near_Home`, `Charging_Stations_Near_Work` are int64; `Annual_Income_USD`, `Daily_Commute_km`, `Environmental_Concern_Level` are float64; `Gender`, `City_Type`, `Current_Car_Type`, `Home_Charging_Possible`, `Subsidy_Available`, `Range_Anxiety_Level`, `Will_Buy_EV` are strings. `Environmental_Concern_Level` is float but holds only the 5 integers 1.0–5.0 — the float dtype is a leftover of the original column having had NaNs.

`Range_Anxiety_Level` is present in test.csv and is a **feature**, as the map already records.

## 2. Per-column profile (train, with test alongside)

### Numeric

| column | min | 1 % | 25 % | 50 % | 75 % | 99 % | max | distinct (train) |
|---|---|---|---|---|---|---|---|---|
| `Age` | 25 | 25 | 36 | 47 | 58 | 69 | 69 | 45 |
| `Annual_Income_USD` | 30,000 | 30,000 | 67,376 | 84,880 | 102,753 | 151,159 | 188,549 | 13,214 |
| `Daily_Commute_km` | 5.0 | 5.0 | 17.2 | 33.6 | 47.4 | 68.5 | 98.7 | 805 |

Test matches closely (median income 84,926, median commute 33.6, median age 47). Test maxima run slightly past train: commute 103.9 vs 98.7, income 186,936 vs 188,549 (income max is *lower* in test).

**Three hard clipping floors**, identical in both files:
- `Age` ∈ [25, 69], 1.98 % of train rows sit exactly at 25 (1.99 % in test).
- `Annual_Income_USD` floored at exactly 30,000: **9.21 %** of train rows (9.17 % of test).
- `Daily_Commute_km` floored at exactly 5.0: **21.58 %** of train rows (21.52 % of test).

These floors are the largest single value masses in the data and are worth an explicit "is at floor" flag, since a tree has to spend a split to isolate them.

Grid: income is integral (no cents), commute lives on a 1-decimal grid.

### Discrete / categorical (train counts)

| column | values |
|---|---|
| `Number_of_Cars_Owned` | 2: 298,287 · 1: 288,012 · 3: 68,877 · 4: 13,489 |
| `Charging_Stations_Near_Home` | 0–14; mode 2 (101,547), then 0 (69,865), 1 (68,177) |
| `Charging_Stations_Near_Work` | 0–19; mode 3 (79,739), then 1 (53,736) |
| `Environmental_Concern_Level` | 1: 147,476 · 2: 135,133 · 3: 127,351 · 4: 130,469 · 5: 128,236 |
| `Gender` | Male 367,954 · Female 295,427 · Other 5,284 |
| `City_Type` | Urban 289,305 · Suburban 255,377 · Rural 123,983 |
| `Current_Car_Type` | Sedan 303,459 · SUV 246,545 · Hatchback 79,438 · Truck 39,223 |
| `Home_Charging_Possible` | Yes 462,677 · No 205,988 |
| `Subsidy_Available` | Yes 419,909 · No 248,756 |
| `Range_Anxiety_Level` | Low 603,972 · Medium 62,499 · **High 2,194** |
| `Will_Buy_EV` | No 551,886 · Yes 116,779 |

Every category level present in train is present in test and vice versa — no unseen categories. `Range_Anxiety_Level = High` is rare (0.33 % of train, 1,029 rows in test) and, as §4 shows, almost perfectly negative.

## 3. Target

Positive rate: **116,779 / 668,665 = 0.174645**. Moderate imbalance; nothing that requires resampling for an AUC metric.

## 4. What separates the classes

### Single-feature AUC (train, categoricals ordered by their target mean — i.e. the best monotone encoding)

| rank | feature | AUC |
|---|---|---|
| 1 | `Environmental_Concern_Level` | **0.84352** |
| 2 | `Subsidy_Available` | **0.71794** |
| 3 | `Annual_Income_USD` | **0.67037** |
| 4 | `Home_Charging_Possible` | 0.55082 |
| 5 | `Range_Anxiety_Level` | 0.54510 |
| 6 | `Daily_Commute_km` | 0.53376 (inverse: raw AUC 0.46624) |
| 7 | `City_Type` | 0.52345 |
| 8 | `Charging_Stations_Near_Home` | 0.51589 (inverse) |
| 9 | `Charging_Stations_Near_Work` | 0.51062 (inverse) |
| 10 | `Current_Car_Type` | 0.51037 |
| 11 | `Age` | 0.50554 (inverse) |
| 12 | `Gender` | 0.50461 |
| 13 | `Number_of_Cars_Owned` | 0.50393 |
| — | `id` | 0.50001 |

Note that `Subsidy_Available` reaches 0.718 as a *binary* feature — that is the ceiling a single yes/no split can reach and it means the split is nearly deterministic in one direction.

### Target rate by level

**`Environmental_Concern_Level`** — monotone and enormous:
1 → 0.57 % · 2 → 2.14 % · 3 → 11.09 % · 4 → 24.88 % · 5 → **51.83 %**

**`Subsidy_Available`** — near-gate: No → **0.58 %** (248,756 rows), Yes → 27.47 %

**`Range_Anxiety_Level`** — near-gate: High → **0.14 %** (2,194 rows), Medium → 4.17 %, Low → 18.90 %

**`Home_Charging_Possible`**: No → 12.71 %, Yes → 19.58 %
**`City_Type`**: Urban 16.11 % · Suburban 18.09 % · Rural **19.34 %** (note: *rural* is the most EV-positive, which is counter-intuitive and a sign the generator's dependency structure is not domain-realistic)
**`Current_Car_Type`**: Truck 15.64 % · Sedan 17.20 % · Hatchback 17.43 % · SUV 18.10 %
**`Gender`**: Male 17.23 % · Other 17.37 % · Female 17.76 % (essentially flat)
**`Number_of_Cars_Owned`**: 4 → 16.57 % · 1 → 17.14 % · 3 → 17.48 % · 2 → 17.82 % (flat, non-monotone)
**`Charging_Stations_Near_Home`**: 1 → 19.80 %, 2 → 18.80 %, 0 → 18.51 %, rest 15.8–17.5 % — weak, non-monotone, slightly *decreasing* in station count
**`Charging_Stations_Near_Work`**: 0 → 19.14 %, 2 → 18.84 %, rest 16.3–18.2 % — same shape

**`Annual_Income_USD`** by decile — the strongest numeric, monotone:
| decile | rate |
|---|---|
| ≤ 45,980 | 4.45 % |
| 45,980–62,671 | 8.46 % |
| 62,671–71,832 | 10.66 % |
| 71,832–78,915 | 13.17 % |
| 78,915–84,880 | 15.61 % |
| 84,880–91,619 | 17.97 % |
| 91,619–96,748 | 19.77 % |
| 96,748–107,798 | 21.96 % |
| 107,798–122,349 | 28.90 % |
| > 122,349 | **33.73 %** |

**`Daily_Commute_km`** — mildly *inverse*: the 5.0-floor bucket (201,796 rows) is 19.63 %, the top bucket (55.8–98.7) is 12.75 %.
**`Age`** — non-monotone and weak: lowest at 60–65 (15.96 %), highest at 51–56 (**20.08 %**) with a secondary bump at 25–29 (18.95 %). A tree will find this; a linear term will not.

### Interactions worth carrying into feature engineering

`Environmental_Concern_Level` × `Subsidy_Available` is strongly multiplicative, not additive:

| Env level | Subsidy=No | Subsidy=Yes |
|---|---|---|
| 1 | 0.017 % | 0.94 % |
| 2 | 0.049 % | 3.48 % |
| 3 | 0.30 % | 18.66 % |
| 4 | 0.81 % | 39.94 % |
| 5 | 2.48 % | **69.33 %** |

`Subsidy_Available = Yes AND Environmental_Concern_Level ≥ 3` is 37.4 % of train rows, has a 44.7 % positive rate, and captures **95.6 % of all positives**. The remaining 4.4 % of positives scattered over the other 62.6 % of rows are where the ranking contest actually happens — this is the region the rank-1 gap most likely lives in.

`Environmental_Concern_Level` × `Range_Anxiety_Level`: `High` anxiety zeroes the target out at every concern level (0.0 % for levels 1–4, 1.32 % at level 5); `Medium` cuts the `Low` rate by roughly 3–4×.

## 5. Duplicates — StratifiedKFold verdict

| check | result |
|---|---|
| distinct feature vectors in train | **668,665 / 668,665** (all unique) |
| train rows in a duplicated feature group | **0 (0.0000 %)** |
| duplicate groups with conflicting targets | 0 (none exist) |
| distinct feature vectors in test | 286,571 / 286,571 (all unique) |
| train ∩ test feature vectors | **0** |
| union of distinct vectors | 955,236 = 668,665 + 286,571 |

**Verdict: plain `StratifiedKFold` on `Will_Buy_EV` does not leak.** No group-aware CV is needed. This is the ticket's headline blocker and it comes back clean.

The uniqueness is driven by the two high-cardinality continuous columns. Dropping them exposes the near-duplicate structure:

| key | distinct combos | rows in groups > 1 | largest group |
|---|---|---|---|
| 11 discrete columns (drop income + commute) | 546,134 | **205,065** (30.7 % of train) | 15 |
| all features except `Annual_Income_USD` | 659,868 | 16,623 (2.5 %) | 6 |
| all features except `Daily_Commute_km` | 667,025 | 3,189 (0.5 %) | 5 |

So a third of train rows share their entire discrete profile with at least one other row, but the largest such cluster is 15 rows out of 668,665. That is far too diffuse to leak across folds: a fold never memorises a cluster, because no cluster is large enough to be memorised. Near-duplicates are a *feature-engineering* observation (the discrete space is densely sampled), not a validation hazard.

Theoretical size of the discrete grid: 45 × 4 × 15 × 20 × 5 × 3 × 3 × 4 × 2 × 2 × 3 = **116,640,000** cells, of which train occupies 546,134 (0.47 %). The discrete space is sparse in absolute terms but the *marginal* structure is fully covered.

## 6. Train vs test drift

**Adversarial validation** — HistGradientBoostingClassifier (200 iters, lr 0.1, native categoricals), 5-fold stratified CV, label = "is this row from test", all 13 features, all 955,236 rows:

> **AUC = 0.49877** (chance is 0.5)

The two files are statistically indistinguishable. No adversarial-weighting, no drift-aware validation, no test-informed feature selection is warranted. A random `StratifiedKFold` split of train is a faithful proxy for the public/private test split.

Column-by-column, all differences are in the third decimal or smaller:

| column | statistic | value |
|---|---|---|
| `Charging_Stations_Near_Work` | TVD | 0.00372 |
| `Charging_Stations_Near_Home` | TVD | 0.00260 |
| `Age` | KS | 0.00229 |
| `Environmental_Concern_Level` | TVD | 0.00180 |
| `Range_Anxiety_Level` | TVD | 0.00171 |
| `Number_of_Cars_Owned` | TVD | 0.00166 |
| `Annual_Income_USD` | KS | 0.00164 |
| `Subsidy_Available` | TVD | 0.00145 |
| `City_Type` | TVD | 0.00135 |
| `Daily_Commute_km` | KS | 0.00107 |
| `Current_Car_Type` | TVD | 0.00075 |
| `Home_Charging_Possible` | TVD | 0.00066 |
| `Gender` | TVD | 0.00020 |

(TVD = total variation distance between the discrete distributions; KS = two-sample Kolmogorov–Smirnov statistic for the continuous ones. Standardised mean shifts are all ≤ 0.0025 σ.)

**Negative result, stated plainly: no train/test drift detected. Maximum column-wise distribution difference is TVD 0.00372, and a gradient-boosted adversarial classifier scores AUC 0.49877.**

## 7. The `id` column

| | train | test |
|---|---|---|
| range | 0 … 668,664 | 668,665 … 955,235 |
| unique | yes | yes |
| contiguous | yes | yes |
| sorted ascending | yes | yes |
| overlap with the other file | 0 ids | 0 ids |

The two files **partition** a single contiguous 0…955,235 block with no interleaving: train is the prefix, test is the suffix. Together they cover 955,236 ids for 955,236 rows.

Target rate by `id` decile (train): 0.17466, 0.17466, 0.17445, 0.17429, 0.17541, 0.17558, 0.17505, **0.17142**, 0.17568, 0.17526. Range 0.0043, entirely consistent with sampling noise at n ≈ 66,867 per decile. Single-feature AUC of `id` = 0.50001.

**Conclusion: `id` is a row index, not a hidden feature.** Exclude it from the model matrix. Do not build ordering-based or rolling features on it. There is no evidence of block structure, no "the last N rows are different" effect, and no chronological signal.

## 8. Row-count sanity and the shape of the synthetic expansion

668,665 + 286,571 = 955,236 rows generated from a source dataset stated at 10,000 records — a **95.5×** expansion, with the split 70.0 % / 30.0 %.

The expansion is visible in the value-frequency spectrum of the only two high-cardinality columns:

- `Age`, `Number_of_Cars_Owned`, `Charging_Stations_Near_*`, `Environmental_Concern_Level` and all six string columns are **saturated**: every possible level appears in both files (45/45 ages, 15/15 and 20/20 station counts, etc.). These columns carry no fingerprint of the original 10k.
- `Daily_Commute_km` is nearly saturated on its grid: 833 distinct values out of the 990 possible 1-decimal values in [5.0, 103.9].
- `Annual_Income_USD` is the informative one: **14,667 distinct values** across all 955,236 rows, out of 158,550 possible integers in range. Excluding the 30,000 floor (87,881 rows), the spectrum is sharply **bimodal**:

| income values with count ≥ … | # values | rows covered |
|---|---|---|
| 500 | 122 | 90,307 |
| 200 | 1,257 | 395,836 |
| 100 | 3,171 | 672,590 |
| 50 | 4,877 | 795,259 |
| **20** | **6,224** | **842,260** |
| 10 | 6,793 | 850,047 |
| 5 | 7,679 | 855,801 |
| 2 | 10,480 | 863,169 |
| 1 (all) | 14,666 | 867,355 |

A core of **~6,200 income values covers 97.1 %** of the non-floor rows (842,260 / 867,355), while a tail of 8,442 values — 4,186 of which appear exactly once — accounts for only ~25,000 rows. The most frequent single non-floor value (86,095) appears 1,552 times.

That shape is the signature of a generator resampling a finite source table: ~6,200 heavily reused income values plus ~9 % of rows pinned at the 30,000 floor is exactly what a 10,000-row original with a clipped income column would produce, with a thin tail of genuinely novel values mixed in. Nothing here recovers the original 10k rows directly — the joint (age, income, commute) triple has 818,397 distinct values across 955,236 rows, so rows are *not* verbatim copies.

**Practical consequence:** a count/frequency encoding of `Annual_Income_USD` is a legitimate candidate feature, since value frequency is not a function of value magnitude. Its standalone AUC is weak (0.461 excluding the floor, i.e. rarer income values are slightly *more* positive), but the effect survives conditioning on income level in the upper deciles:

| income decile (floor excluded) | rate, rarest tercile | rate, most-common tercile |
|---|---|---|
| 9 (highest) | **0.3880** | 0.3192 |
| 8 | 0.3100 | 0.2873 |
| 7 | 0.2471 | 0.2322 |
| 0 (lowest) | 0.0526 | 0.0986 (reversed) |

Small, but real and cheap. Compute it on train+test combined.

## 9. Environment finding (map open risk: Python 3.14 wheels)

The map flags "Python 3.14 is new enough that wheel availability for the GBDT stack is an open risk". **That risk is closed — negatively.** Using `uv` (available at `/opt/homebrew/bin/uv`), a fresh Python **3.14.7** environment installs and imports the whole stack:

> scikit-learn 1.9.1 · lightgbm 4.7.0 · xgboost 3.4.1 · catboost 1.2.10 · numpy 2.5.3 · pandas 3.0.6

The real problem is narrower: **`pip` inside the project `.venv` is broken**, not the wheels. The fix is to install with `uv pip install --python .venv/bin/python …` (or rebuild the venv with `uv venv`) instead of repairing pip.

Two pandas-3.0 gotchas found while profiling, worth knowing before pipeline code is written:
- String columns load as the new `str` extension dtype, **not** `object`. `df[c].dtype == object` is False for every categorical column here; use `pd.api.types.is_numeric_dtype` or check for `'str'`.
- Reductions on that dtype raise `TypeError: Cannot perform reduction 'mean' with string dtype`, so `(df['Will_Buy_EV'] == 'Yes').mean()` is required rather than a direct mean.

## 10. Recommendations flowing from these facts

- **Validation:** plain `StratifiedKFold(n_splits=5, shuffle=True)` on `Will_Buy_EV`. No groups, no adversarial weighting, no time-based split. Train-CV should track the leaderboard tightly given zero drift.
- **Features:** drop `id`. Treat `Environmental_Concern_Level` as ordinal (it is monotone in the target). Add explicit floor flags for income = 30,000 and commute = 5.0. The `Env × Subsidy` interaction is the single most productive engineered feature; `Range_Anxiety = High` is an effective hard-zero gate.
- **Where the gap is:** `Subsidy=Yes & Env≥3` already isolates 95.6 % of positives into 37.4 % of rows. Ranking effort spent outside that region — the 5,100-odd positives hiding in the low-concern / no-subsidy mass — is where a 0.0027 AUC edge over the ~0.9466 plateau plausibly comes from.
