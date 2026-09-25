# Multi-key income TE, the 173-feature XGB, and the sources never read

Research, retrieved 2026-09-25. Reading only: nothing was trained or submitted, and no
`src/` file, ledger entry or pipeline file was touched. It answers three questions: what
Marc Maldonado Lorca's multi-key income target encoding actually is (discussion 742385);
what the 173 features and the protocol of `heuljax/kps6e09-xgb-sample` are and which of
them our Frame could take; and what najiama's notebooks and threads 742324 and 741755
contain.

The yardstick throughout is our standing best: **LightGBM, `baseline` Frame, nested
cross-fit TE of exact income at prior weight 20, 5-fold Canonical Fold Partition
(`random_state=0`), 2341 fixed rounds, OOF 0.94557 / LB 0.94567**
(`hpsearch_lightgbm_best_confirm`, `ledger/runs.jsonl`).

---

## Headline answers

**Q1 — Marc's multi-key TE.** The code is public and its run log is readable, so this
answer rests on code, not the post. The post lists four keys. **The code target-encodes
six:** exact income, `income // 100`, `income // 1000`, *nearest original income*,
**exact `Daily_Commute_km`, and `floor(commute)`** (**Verified**, `TE_COLS`; the run log
says "6 encoded inside the folds"). "Nearest original" is the **original value itself used
as a key**, not a distance. For each income, the code takes the closest value among the
sorted unique incomes of the 10,000-row original
(`itzzomkar/ev-adoption-behavior-and-range-anxiety`) by absolute distance, and a tie goes
to the lower value. For 97.9% of rows that is the income itself, so the key only changes
the 2.1% of rows (13,742) whose income was invented. The TE is a **plain smoothed mean
at prior weight α = 1** (ours is 20), fitted on the competition training fold only (no
original rows are appended). It is cross-fitted by an inner `KFold(5, shuffle,
random_state=42)` inside an outer `StratifiedKFold(5, shuffle, random_state=42)`, so it
is a genuine Nested Cross-Fit. The LightGBM **early-stops on the scored fold**
(`early_stopping(200)`, best iterations 1340–1809). **OOF 0.94603 is pooled and
verified in the run log.** LB 0.94617 is the author's own report. His ablation table
(self-report, not re-run on Kaggle) prices the pieces on an additive model: prior 20 → 1
**+0.00025**, coarse keys **+0.00012**, nearest-original **+0.00005**. The regularised
params (`feature_fraction 0.3`, depth 5, `min_data_in_leaf 10`, `max_bin 1024`) add
**+0.00014** over that additive model. Those params are najiama's. **The model has no
digits and no count encoding.** It is 5-fold and pooled like ours, and early stopping
probably costs it only ~0.00001–0.00002 of optimism. So it is the published number
nearest to a Comparison Run, and it sits **+0.00046 above our 0.94557**.

**Q2 — heuljax's 173 features.** **Verified by recount.** The 173 columns are: 13 raw,
4 original-data income priors, 7 exact-value TEs and supports, 38 neighbour or
hierarchical rates, 9 income-group model priors and uncertainty, 28 per-income-value
means of the Recipe's latent variables, 10 single digits, 2 "worry-key" TEs, 18
gate-mixture columns, 12 composition-inversion columns, 4 GAM coordinates and 28
multi-scale TEs. **Our Frame has only the 13 raw columns and one of the 7 exact TEs.**
Our digits are a different decomposition, so even those do not match. The protocol is
confirmed (**Verified**): 10 outer folds `random_state=42`, 5 inner donor folds per outer
fold, XGBoost with `lr 0.015`, depth 5, `max_bin 256`, `colsample_bynode 0.8`,
`reg_alpha 10`, `reg_lambda 15`, and **+1 monotone constraints on 78 rate columns**. It
runs up to 6000 rounds with `EarlyStopping(rounds=750, save_best=True)` **on the scored
fold** and takes 693–1631 trees. Pooled OOF is 0.946308910839. The GAM enters both as
four columns and as `base_margin`. It is never scored alone. The per-fold AUC after
round 0 averages **0.938**, which is roughly the GAM margin on its own. That makes it a
Recipe-level signal, not a Member (**Inferred**). The logged AUC curves bound the
early-stopping optimism at **≈ +0.000013 per fold** against a fixed 1000 rounds
(**Inferred**). Add 10-fold ≈ +0.00015 (739354, self-report), and **about 0.00017 of the
0.00074 gap to us is protocol. Roughly 0.0006 is features and model.**

**Q3 — the unread sources.** **najiama** has six notebooks. The load-bearing one is *Pure
LGBM CV 0.94607 LB 0.94638* (**OOF 0.94607 verified in its log**, 5-fold,
early-stopped on the scored fold). It is the ancestor of Marc's model: same LightGBM
params, same "smooth keys" (`income`, `//100`, `//1000`, `floor(commute)`). It is wrapped
in digits of every numeric at 10^-4..10^3, `_org_mean` of every column, global frequency
encoding, and sklearn `TargetEncoder` at three smoothings over 41 columns, for 109
features. **Techniques not in our ledger:** income TE at a light prior; `//100` and
floor-commute TE keys; the najiama/Marc LightGBM parameter point; LightGBM
`interaction_constraints`; and, reported only in the hub notebook, a **RealMLP at CV
0.94601**. **742324** is a one-sentence question ("Bayes error rate") with 0 replies. It
makes no claim. **741755** is Rugved Bane's one-at-a-time FE ablation on a 255-bin
LightGBM, and its numbers are **verified by the notebook's own log**: digit
decomposition +0.00182, Subsidy×Income +0.00026, triple TE on 4 low-cardinality
categoricals +0.00008, and ≤ +0.00004 for everything once digits are in. Its one reply
of substance (Mamarin, self-report) shows the digit gain is mostly `max_bin` resolution
(+0.00201 either way). It also puts `interaction_constraints` at +0.00111.

---

## How these facts were obtained

1. **`kaggle kernels pull <owner>/<slug> -m`** for notebook source and metadata (the
   `dataset_sources` field names the original dataset). Pulled into
   `/private/tmp/claude-501/research-te/`, not the repo.
2. **`kaggle kernels output <owner>/<slug> --file-pattern '.*\.log$'`** for the
   **executed run log** of each notebook's current version. The log contains every
   `print`, per-fold AUC and early-stopping curve. It is the highest-trust channel used:
   what the code actually printed when it ran on Kaggle. Pulled sources carry no cell
   outputs, so **a number counts as Verified here only if it appears in the log.**
3. **Discussion threads** through `r.jina.ai` with `x-engine: browser`,
   `x-no-cache: true`, **and `x-respond-with: text`**. The default markdown mode returns
   only the largest block (one reply of 741755 and nothing else). Text mode returns the
   whole thread, comments included. Comment counts were cross-checked against
   `GetForumTopicById` (`totalMessages`). `GetForumMessagesInTopic` exists but returns
   `PERMISSION_DENIED` even with the CLI's OAuth token.
4. Nothing comes from search-engine summaries or memory.

---

## 1. Marc Maldonado Lorca's multi-key income TE (742385)

### 1.1 The post, and where the code lives

The first post of <https://www.kaggle.com/competitions/playground-series-s6e9/discussion/742385>
(author `marcmaldonado`, 2026-09-21, **0 comments**) says, verbatim:

> "A cross-fitted target encoding of the exact value (prior 1, keys value / //100 /
> //1000 / nearest original) under a well-regularised LightGBM gives OOF 0.94603 / LB
> 0.94617; the original's label on top adds 0.00003; keying by income × recipe column
> costs 0.0004; as a tie-breaker it recovers nothing."

It links two notebooks, both pulled and read in full:

- **companion**: <https://www.kaggle.com/code/marcmaldonado/s6e9-the-generator-remembers-the-original-rows>.
  It holds the model, the ablation table (§5) and the submission.
- **AUC harness**: <https://www.kaggle.com/code/marcmaldonado/s6e9-is-the-row-memory-worth-anything-in-auc>.
  It re-runs the negative readings with noise bars.

**Code outranks the post, and the two differ in one place (Verified).** The companion's
cell 18:

```python
df["income_100"] = (df.Annual_Income_USD // 100).astype("float32")
df["income_1000"] = (df.Annual_Income_USD // 1000).astype("float32")
df["commute_km"] = np.floor(df.Daily_Commute_km).astype("float32")
df["income_nearest_original"] = nearest_original(df.Annual_Income_USD.values)
TE_COLS = ["Annual_Income_USD", "Daily_Commute_km", "income_100", "income_1000", "commute_km", "income_nearest_original"]
TE_ALPHA = 1.0
```

The run log confirms it: *"17 columns before target encoding, + 6 encoded inside the
folds (prior weight 1)"*. **The post's four keys omit the two commute keys.** Cell 17's
prose does name them ("… of `Annual_Income_USD` and `Daily_Commute_km` — at the value, at
coarser keys (`// 100`, `// 1000`, whole kilometres …)").

### 1.2 "Nearest original": exactly defined (Verified, cell 18)

```python
orig_values = np.sort(orig.Annual_Income_USD.dropna().unique())
def nearest_original(values):
    j = np.clip(np.searchsorted(orig_values, values), 1, len(orig_values) - 1)
    left, right = orig_values[j - 1], orig_values[j]
    return np.where(np.abs(values - left) <= np.abs(right - values), left, right).astype("float32")
```

- **The key is the original income value itself**, not a distance and not a flag. It is
  then target-encoded like any other key.
- **Absolute distance**, and **a tie goes to the lower value** (`<=` picks `left`).
  Values outside the original's range clamp to the extreme original values through the
  `np.clip`.
- It is computed from the original's **incomes only**. No label is read, so in our
  vocabulary it is a **Frame transform**. The target encoding of it is an Adapter
  concern.
- **What it changes (Verified, run log):** 97.9% of train rows already carry an original
  income value, so for them this key equals exact income. It differs only on the 2.1%:
  *"6,125 distinct, 13,742 rows (2.1%), 2.24 rows per value; median $2 from the nearest
  original value"*. The author's stated purpose: "so the 2.1% invented incomes borrow
  their neighbour's memory." Cell 20 (self-report, output not in log) says the key "takes
  more gain than the exact value".
- **The original dataset is Kaggle dataset `itzzomkar/ev-adoption-behavior-and-range-anxiety`,
  file `EV_Adoption_and_Range_Anxiety_Dataset.csv`, 10,000 rows** (**Verified**:
  `kernel-metadata.json` `dataset_sources`, the `find()` call, and
  `assert orig.shape == (10000, 15)`). Our `data/` directory does **not** contain it.

### 1.3 TE parameters (Verified, cell 18 `target_encode`)

```python
prior = ytr.mean()
for c in cols:
    for a, b in KFold(n_inner, shuffle=True, random_state=seed).split(Xtr):   # n_inner=5, seed=42
        gk = ...groupby("k")["y"].agg(["sum", "size"])   # on inner-train rows a
        enc[b] = map((gk["sum"] + alpha * prior) / (gk["size"] + alpha)).fillna(prior)
    full = (sum + alpha*prior) / (size + alpha)  # on the whole outer-train fold -> valid and test
```

| parameter | Marc | ours (`src/adapter.py`) |
|---|---|---|
| form | smoothed **mean** (not log-odds) | smoothed mean — same algebra |
| prior weight α | **1.0** | **20.0** (`PRIOR_WEIGHT`) |
| prior | outer-train-fold mean | fold prior — same |
| unseen key | `fillna(prior)` | fold prior — same |
| min count | none | none |
| inner split | `KFold(5, shuffle, random_state=42)`, **not stratified, same seed in every outer fold** | `StratifiedKFold(5)`, seed `100 + outer_fold` — ours is stricter |
| held-out and test rows | encoded from the whole outer-train fold | same for held-out; our test is encoded once on full train |
| data | **competition train only**; the original is used only for the nearest-original key | same (we do not use the original) |

**Fold discipline: Verified honest.** Training rows are encoded inner out-of-fold, and
the scored fold's labels never enter any encoding. That is our Nested Cross-Fit contract
in all but the inner seed.

The prior was chosen by a sweep (cell 17 table, self-report): from α = 20 (0.94552),
α = 10 / 5 / 2 / 1 give 0.94563 / 0.94570 / 0.94575 / 0.94577, and α = 0.5 gives back 0.00005. Cell 17
attributes to megayak's sweep the observation that "heavy smoothing washes the effect
out". megayak's notebook was not read (see Gaps).

### 1.4 Model, rounds, early stopping, pooled OOF (Verified, cell 19 and run log)

```python
PARAMS = dict(objective="binary", metric="auc", learning_rate=0.02, num_leaves=32, max_depth=5, min_data_in_leaf=10,
              feature_fraction=0.3, bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0, lambda_l1=0.071, max_bin=1024, seed=42)
mdl = lgb.train(params, dtr, 50000, valid_sets=[dva], callbacks=[lgb.early_stopping(200, verbose=False)])
oof[va] = mdl.predict(Xva, num_iteration=mdl.best_iteration)
```

- **Outer folds:** `StratifiedKFold(5, shuffle=True, random_state=42)` over `train.csv`
  in file order. The same scheme as ours except for the seed (ours is 0).
- **Early stopping uses the scored fold.** `dva` is the outer validation fold, and the
  prediction uses `best_iteration`. Rounds are selected on scored rows. **This is not a
  Comparison Run under our frozen protocol.**
- **Pooled:** `roc_auc_score(y, oof)`. Run log, verbatim:

  ```
    fold 0: AUC 0.94502  (1587 rounds)
    fold 1: AUC 0.94576  (1809 rounds)
    fold 2: AUC 0.94698  (1669 rounds)
    fold 3: AUC 0.94627  (1340 rounds)
    fold 4: AUC 0.94615  (1723 rounds)
  out-of-fold AUC 0.94603   (5.1 min)
  ```

- **The 6 categoricals enter as pandas `Categorical`** (LightGBM native categorical
  splits). There are **no digits, no count/frequency encoding, no interactions, no
  blend.**
- **How optimistic is 0.94603? Inferred small.** Patience 200 with `best_iteration` on a
  flat plateau selects among near-equal rounds. §2.3 measures the same mechanism on
  heuljax's printed curves at ≈ +0.00001–0.00002. Marc's own statement (cell 17, self-report):
  "seed replicas of the final additive model differ by 0.000002". **So 0.94603 is the
  closest published analogue to our Comparison Run.** It is 5-fold and pooled; it
  differs from ours only in fold seed and a small upward bias. Treat it as ≈ 0.9460 ±
  partition noise.
- **LB 0.94617 is a self-report** (cell 22 prose: "0.94585 → 0.94595 → 0.94617 on the
  public board"). The Kaggle API exposes no per-kernel public score for this notebook.

### 1.5 Ablation: who contributes what

**Status: self-report.** The table is markdown in cell 17, labelled "Measured with
LightGBM 4.7.0 … local". Only two rows were re-run on Kaggle: the final model (0.94603)
and the additive reference (log: **0.94581**, table 0.94582). Base: LightGBM, 5 folds,
seed 42, **lr 0.1**, early stopping on the scored fold.

| step (cumulative unless ↳) | OOF | Δ |
|---|---|---|
| raw 13 columns, `max_bin` 255 | 0.94180 | — |
| + `max_bin` 4095 | 0.94358 | +0.00178 |
| raw + TE of exact income and commute, prior 20 (255 bins) | 0.94484 | +0.00304 vs raw |
| + `max_bin` 4095 | 0.94504 | +0.00020 |
| + `interaction_constraints` (additive model) | 0.94540 | +0.00036 |
| ↳ noise bar (two random columns, twice) | 0.94534 / 0.94534 | −0.00006 |
| + **coarser keys** (`//100`, `//1000`, whole km) | 0.94552 | **+0.00012** |
| + **prior 20 → 10 / 5 / 2 / 1** | … / 0.94577 | **+0.00025** |
| + **nearest-original key**, prior 1 = *additive reference* | 0.94582 | **+0.00005** |
| ↳ prior 0.5 | 0.94577 | −0.00005 |
| ↳ noise bar for the reference | 0.94580 / 0.94581 | −0.00002 / −0.00001 |
| ↳ + original-row witness (label cols) | 0.94585 | +0.00003 |
| ↳ + income × subsidy/concern/anxiety TE keys | 0.94538 | −0.00044 |
| ↳ + `_org_mean` / `%1000` digit encodings / 11 other-column TEs / triple prior | 0.94585 / 0.94583 / 0.94577 / 0.94583 | +0.00003 / +0.00001 / −0.00004 / +0.00001 |
| **no constraint**, 32 leaves / depth 5 / 10 per leaf / colsample 0.3 / L2 2 / L1 0.071 / `max_bin` 1024 = *final* | 0.94596 | **+0.00014** vs reference |
| ↳ noise bar for the final model | 0.94591 / 0.94588 | −0.00005 / −0.00008 |
| ↳ colsample 0.3 → 0.8 / 0.5 / 0.2 | 0.94588 / 0.94592 / 0.94599 | −0.00008 / −0.00004 / +0.00003 |
| ↳ `max_bin` 1024 → 255 / 4095 | 0.94597 / 0.94595 | ±0.00001 |
| ↳ + digit and other-column encodings | 0.94599 | +0.00003 |
| final at lr 0.02 | **0.94603** | +0.00007 |

The AUC-harness notebook **re-ran** the negative rows on Kaggle, and its log (**Verified**)
agrees:

```
reference (additive, lr 0.1)       OOF AUC 0.94581
+ two random columns (noise bar, A) OOF AUC 0.94581
+ two random columns (noise bar, B) OOF AUC 0.94580
+ original row as witness          OOF AUC 0.94584
+ income × recipe-column keys      OOF AUC 0.94537   (0/5 folds up)
+ income × full-cell key           OOF AUC 0.94524
linear reader on income + cell     OOF AUC 0.94319
```

**Reading:**

- **Verified directly:** the witness is worth +0.00003 (4/5 folds). Composite income ×
  recipe TE keys cost −0.00044 on all 5 folds. This matches our #29 C1/C2 (−0.00014,
  0/5).
- **Inferred:** keying the TE by income resolution is what pays. The remaining steps
  (prior and extra keys) are **worth ~+0.0004 on top of prior-20 exact TE in his
  pipeline.** That pipeline has **no digits**. His final-model row "+ `%1000` digit
  encodings +0.00001" suggests that once the encoding is light and multi-key, digits add
  little. The converse is unmeasured: how much prior 1 and extra keys add **on top of our
  digits** is not known.
- **Self-report:** interaction constraints are worth +0.00036 with the plain encoding but
  only +0.00005 with the full encoding. A regularised free model beats the constrained
  one by +0.00014, 5/5 folds. The author's summary: "the constraint was standing in for
  regularisation".

### 1.6 Is 0.94603 comparable to our Comparison Run?

It is the most comparable number in the forum, with two caveats. **Verified:** it is
5-fold, pooled, and uses an honest Nested Cross-Fit. **Verified:** it early-stops on the
scored fold. **Inferred:** that bias is ~0.00001–0.00002. **Not measured:** the effect of
fold seed 42 against our 0. **So Marc's 0.94603 against our 0.94557 is a gap of about
+0.0004, and it cannot be explained by protocol.**

---

## 2. `heuljax/kps6e09-xgb-sample` (Apache-2.0, 173 features)

<https://www.kaggle.com/code/heuljax/kps6e09-xgb-sample>. The current version was pulled
(last run 2026-09-24, T4 GPU, dataset source `itzzomkar/ev-adoption-behavior-and-range-anxiety`),
and its log reprints `CV POOLED_OOF_AUC=0.946308910839`. It is the same run the earlier
note read through `scriptcontent/352387662`. The ridge-Newton GAM is already covered in
[`linear-model-representation.md`](linear-model-representation.md) §2 and is not
repeated here.

### 2.1 The 173 features, grouped (Verified by re-counting `FEATURE_COLS`, cell 2)

The code asserts `len(FEATURE_COLS) == 173`. The groups below were recomputed from the
constants in cell 1 and sum to 173. "Fold-fitted" means built by `DonorState` on donor
rows only: inner donors for training rows, the full outer-train fold for valid and test.

| # | group (code name) | n | what it is (cell 5 `DonorState.transform`) | reads target? | in our Frame? |
|---|---|---|---|---|---|
| 1 | raw (`RAW_COLS`) | 13 | 7 numerics as floats, 6 categoricals as integer codes declared categorical (`'c'`) | no | **yes** (nominals one-hot) |
| 2 | original-data income priors (`SOURCE_COLS`) | 4 | from the **original**: mean label at the exact income, `log1p` count, seen-flag, ±25-neighbour smoothed rate (α 20). Original rows identical to any train/test row are dropped first (feature hash) | original labels only | no |
| 3 | exact-value TE (`TE_INC_A{5,20,50}`, `TE_CMT_A{20,100}`, `SUPPORT_{INC,CMT}_LOG`) | 7 | `(S+αμ)/(n+α)` on exact income (3 priors) and exact commute (2), plus `log1p(n)` | yes, fold-fitted | **1 of 7** (income at α = 20) |
| 4 | neighbour and hierarchical rates (`NBR_INC_W*` 8, `NBR_DETAIL_*` 4, `HIER_INC_*` 15, `INC_SURPRISE_*` 3, `NBR_CMT_W*` 6, `DCMT_LOCAL_W5`, `P_DCMT_LOCAL_W5`) | 38 | rate of the **adjacent distinct values, own value left out** (box half-widths 1…250 on the sorted value axis, α 20); differences between widths; exact TE shrunk toward the neighbour rate; commute neighbour rates; a commute residual effect against a main-effects GAM | yes, fold-fitted | no |
| 5 | income-group model prior (`GXP_*` 6, `GXP_POST_*` 3) | 9 | an `XGBRegressor` (depth 3, λ 100, 600 trees) fit at the **income-value level** on per-value means of ~30 covariates, weighted by √count; its prediction as a prior, 5 shrinkages, ±1 SD | yes, fold-fitted | no |
| 6 | latent composition (`LAT_INC_*`) | 28 | per-income-value smoothed **means of the Recipe's inputs** (concern, subsidy, anxiety penalty, buy score) at 6 α, plus the row's deviation from them | **no** (feature means only) | no |
| 7 | digits (`DIG_INC_10E0..5`, `DIG_CMT10_10E0..3`) | 10 | single decimal digits of income and of `round(10·commute)`, **declared categorical** | no | **no**: we carry `//1000`, `%1000`, `%100` of income, and no commute digits |
| 8 | worry key (`WTE05_A5`, `WHIER05_W25_A20`) | 2 | TE of `round((commute − 5·home − 5·work − 150·[no home charging]) / 0.5)` | yes, fold-fitted | no |
| 9 | gate mixtures (`MIX_W{16,64,256}_*`) | 18 | Gaussian-smoothed (in income) class-conditional distributions over the 31 concern × subsidy × anxiety cells; a mixture-weight fit per income value, posteriors, row probability, information, gain | yes, fold-fitted | no |
| 10 | composition inversion (`CSHIFT_*`, `CCORR_*`) | 12 | inverts `E_donor[σ(nuisance + s)] = p` per income value for 6 rate channels; then `s − 0.25·GAM_income` | yes, fold-fitted | no |
| 11 | GAM coordinates (`GAM_*_COORD`) | 4 | the ridge-Newton GAM's gate, income, commute and other components | yes, fold-fitted | no (#33 tested the margin form) |
| 12 | multi-scale TE (`MSTE_{INC10,INC100,INC1K,CMTINT}_A{1,2,5,10,50,200}` + `_LOGN`) | 28 | TE on `floor(income/10)`, `/100`, `/1000`, `floor(commute)` at 6 priors, plus `log1p(n)` | yes, fold-fitted | no (we have the `//1000` column raw, never encoded) |
| | **total** | **173** | 13 + 4 + 7 + 38 + 9 + 28 + 10 + 2 + 18 + 12 + 4 + 28 | | |

Two observations the counts make plain:

- **Everything beyond the raw 13 is about per-value income structure, looked at in
  several ways.** Of the 160 other columns, 157 are keyed on income or commute at some
  resolution. The exception is the 3 worry-key and gate-mixture interactions with the
  Recipe cells, and even those are indexed by income value. This is the Resolution axis,
  exhausted.
- **The 28 `LAT_INC_*` columns read no target.** They describe how the Recipe's other
  inputs are distributed at a given income value, which is the "copied row" effect in
  Marc §4 seen from the feature side. They are the only large group that is a pure Frame
  transform.

### 2.2 Protocol (Verified, cells 1, 6, 8 and run log)

- **Outer:** `StratifiedKFold(10, shuffle=True, random_state=42)`.
- **Inner donor folds:** `StratifiedKFold(5, shuffle=True, random_state=42+3000+fold)`.
  Each inner donor state encodes its receiving rows. Valid and test are encoded by a
  state fitted on the whole outer-train fold. `fit_donor_state` asserts the donor rows
  are disjoint from the receiving rows and from the outer valid rows, and it asserts
  complete, non-duplicated inner coverage. **This is honest Nested Cross-Fit.**
- **XGBoost:** `binary:logistic`, `hist` on CUDA, `learning_rate 0.015`, `max_depth 5`,
  `max_bin 256`, `colsample_bynode 0.8`, `colsample_bytree 1.0`, `subsample 1.0`,
  `min_child_weight 1`, `reg_alpha 10`, `reg_lambda 15`, `base_score 0.5`, seed
  `42+fold`. **`monotone_constraints` = +1 on 78 target-rate columns** (recounted).
  Digit and categorical columns use `feature_types='c'`.
- **Rounds and early stopping — confirmed:** `num_boost_round=6000`,
  `EarlyStopping(rounds=750, metric_name='auc', data_name='valid', maximize=True,
  save_best=True)` with `evals=[(dvalid,'valid')]`, **where `dvalid` is the scored outer
  fold**. OOF is `booster.predict(dvalid)` from the saved best booster. The log's trees
  per fold: 721, 1233, 1215, 1631, 844, 1353, 906, 693, 1257, 1262.
- **Pooled:** `roc_auc_score(y, oof)` → `CV POOLED_OOF_AUC=0.946308910839`. Per-fold
  0.94521–0.94733. Test is the mean of the 10 fold models.

### 2.3 Is the GAM a feature, a margin, or a Member?

- **Verified:** it is a feature (the 4 `GAM_*_COORD` columns) **and** the XGBoost
  `base_margin` (`gate + other + 0.25·(income + commute)`) on train, valid and test. No
  cell scores it alone, and no ablation removes it.
- **Inferred standalone AUC ≈ 0.938.** XGBoost logs `valid-auc` after the first tree.
  With `lr 0.015`, one depth-5 tree barely perturbs the margin, so `[0]` is
  approximately the margin's own ranking. Across the ten folds it reads
  0.93653–0.93972, mean **0.93805**. That is the Recipe's level (our #4 free-coefficient
  refit 0.937772), **~0.0075 under our OOF.** As a Member it would enter a Blend at a
  weight near zero. #33 already killed the margin as `init_score` in both
  configurations. **Nothing here reopens it.**

### 2.4 How much of 0.94631 is protocol?

- **Early stopping on the scored fold — Inferred ≈ +0.00001.** The log prints each
  fold's AUC every 100 rounds. For every fold, best-iteration AUC minus AUC at a fixed
  round 1000: mean **+0.000013**, max +0.000041. Against round 500 it is +0.000050,
  because the curves are still rising. Fixed round 1000 is itself read off the same
  curves, so this is an upper-leaning estimate of what a sensible frozen count would
  cost. The curves are flat (±0.00002 over hundreds of rounds), which is why
  `save_best` buys little.
- **10-fold versus 5-fold ≈ +0.00015.** This is a self-report from
  <https://www.kaggle.com/competitions/playground-series-s6e9/discussion/739354>, as
  quoted in `linear-model-representation.md` §1. It was not re-read for this note.
- **Therefore (Inferred):** the gap is 0.946309 − 0.94557 = 0.00074. **≈ 0.00016 is
  protocol and ≈ 0.00058 is features and model.** On a 5-fold footing heuljax would sit
  near **0.9461–0.9462**, only ~+0.0001 above Marc's 23-column model (0.94603) and
  najiama's (0.94607). **So the 150 extra columns buy about a ten-thousandth over a
  light multi-key TE.** That is consistent with the Plateau, and a cross-pipeline
  comparison, so noisy.

### 2.5 Portability, group by group

| group | Frame or Adapter | what it would need | expected worth (anchor) |
|---|---|---|---|
| 3: exact TE at more priors, commute TE | Adapter: config (`prior_weight` is already an `Adapter.__init__` argument; `Experiment` does not expose it) | expose `prior_weight` per TE column, or list-valued | prior 20 → 1: **+0.00025** (Marc, self-report); triple prior: +0.00001 (Marc) and #29 B2 never ran; commute TE at prior 20: **+0.00000 (our #29 B1)** |
| 12: multi-scale TE keys `//10`, `//100`, `//1000`, floor commute | **Frame** (key columns; `//1000` already exists) + Adapter `target_encode` list: single-field once the columns exist | new Frame spec adding `income_div100` (and `div10`, `commute_floor`) | coarse keys **+0.00012** (Marc, self-report, includes whole km) |
| nearest-original key (Marc's, not in heuljax's 173) | **Frame** (reads the original's incomes, no label) | the original CSV in `data/`; one Frame column; then TE it | **+0.00005** (Marc, self-report) — at his noise bar |
| 7: single digits as categorical, commute digits | Frame | new spec | #29 `digits7` −0.00004 (commute and other columns); income single digits vs our `//1000,%1000,%100`: no source compares them |
| 6: `LAT_INC_*` latent means per income value | **Frame** (no target; could be computed over train+test) | 28 columns (or 4 at one α) | unablated anywhere |
| 4: neighbour, leave-own-value-out rates | Adapter (target-reading, fold-fitted) | sorted value axis, box sums, α 20 | unablated anywhere; the most distinctive idea in the file |
| 2: original-label priors | Frame | original CSV | **≈ 0**: Marc +0.00003/−0.00005, and he cites Dvorkin, megayak and Mamarin §16 at zero |
| 5, 9, 10, 11: GXP, mixtures, composition, GAM | Adapter, each a model inside the Adapter | substantial code | unablated; GAM as margin dead (#33) |
| 8: worry key TE | Adapter (composite key) | Frame key + TE | unablated; composite-key TEs have been negative everywhere measured |
| monotone +1 on rate columns | model param | LightGBM `monotone_constraints` on `_te` columns | unablated |

---

## 3. The never-read sources

### 3.1 najiama (Naji Ama)

`kaggle kernels list --competition playground-series-s6e9 --user najiama` lists six
notebooks. All six were pulled, and every log was read.

| notebook | CV (log) | CV and LB (title/markdown) | what it is |
|---|---|---|---|
| [`pure-lgbm-model-cv-0-94607-lb-0-94638`](https://www.kaggle.com/code/najiama/pure-lgbm-model-cv-0-94607-lb-0-94638) | **0.94607** (**Verified**) | 0.94607 / 0.94638 | the source of Marc's params and smooth keys; described below |
| [`xgboost-triple-te-dynamic-pruning-lb-0-94639`](https://www.kaggle.com/code/najiama/xgboost-triple-te-dynamic-pruning-lb-0-94639) | **0.94614** (**Verified**, 5-fold) | header says **10-fold** 0.94624 / LB 0.94639 — **the code runs `Folds = 5`**, a header/code mismatch | same features; XGB `lossguide`, depth 4, 16 leaves, γ 3.67, `max_bin` 1024, lr 0.01, ES(500) on scored fold; prunes 85 low-gain columns from a hard-coded list |
| [`catboost-triple-te-dynamic-pruning-lb-0-94624`](https://www.kaggle.com/code/najiama/catboost-triple-te-dynamic-pruning-lb-0-94624) | **0.94610** (**Verified**, 5-fold) | LB 0.94624 | same features; CatBoost depth 6, lr 0.015, ES(800) on scored fold |
| [`oof-power-two-single-models-blend-lb-0-94638`](https://www.kaggle.com/code/najiama/oof-power-two-single-models-blend-lb-0-94638) | **run failed** (`FileNotFoundError` in log) | CV 0.94613 / LB 0.94638 | rank blend of the LGBM with Sergey Qt2024's focal-loss `linear_tree` LGBM, weight grid-searched **on the same OOF rows it is scored on**: +0.00007 CV / +0.00001 LB (self-report) |
| [`s6e9-electric-vehicle-oof-cv-0-94618-lb-0-94633`](https://www.kaggle.com/code/najiama/s6e9-electric-vehicle-oof-cv-0-94618-lb-0-94633) | 0.94618 (log; recomputed from a private CSV) | 0.94618 / 0.94633 | loads a private blend's OOF and submission; no training code |
| [`honest-model-directory-oof-hub`](https://www.kaggle.com/code/najiama/honest-model-directory-oof-hub) | n/a | n/a | a hand-typed table of 31 public notebooks (CV, LB, folds); **now fully readable** as source |

**The Pure LGBM pipeline (Verified, cells 6 and 8; log: "Dropping 85 redundant/constant
features / Total Features: 109 / Columns to Target Encode: 41"):**

- drops `Number_of_Cars_Owned`;
- digits `(x // 10**k) % 10` for k = −4…3 on **every** numeric column;
- `_org_mean` (the original's mean label per value) of every raw and digit column;
- every numeric and digit column also as a string category, with **global frequency
  encoding (train+test)** of every category;
- flags `is_30k_spike`, `is_millionaire_cliff` (≥ 170,537), `is_dead_zone` (38–42k),
  `is_env_hater` (concern = 1);
- **Markus.JM's "smooth keys"**: `floor(income)`, `floor(income/100)`,
  `floor(income/1000)`, `floor(commute)` as strings;
- drops perfectly correlated and constant columns;
- `sklearn.preprocessing.TargetEncoder(shuffle=True, cv=5, random_state=42)` at
  `smooth='auto'`, `10.0`, `100.0` over all 41 category columns (`fit_transform` on the
  training fold is sklearn's internal cross-fitting: a Nested Cross-Fit), **replacing**
  the string columns;
- LightGBM `lr 0.02, max_depth 5, num_leaves 32, min_child_samples 10, subsample 0.81,
  colsample_bytree 0.30, reg_alpha 0.071, reg_lambda 2.03, max_bin 1024`, 20,000 trees,
  **`early_stopping(500)` on the scored fold** (best trees 1188–1564);
  `StratifiedKFold(5, shuffle, 42)`, pooled OOF.

Log per fold: 0.94513, 0.94573, 0.94694, 0.94640, 0.94617 → **0.94607**. Marc's folds on
the same split: 0.94502, 0.94576, 0.94698, 0.94627, 0.94615 → 0.94603. **Marc's 23
columns reproduce najiama's 109-feature result within 0.00004, fold by fold (Verified,
both logs).** That is the strongest evidence in this note that najiama's extra breadth
(digits of every column, `_org_mean`, global frequency, flags, triple smoothing) is worth
≈ 0 once the light multi-key income/commute TE and the regularised params are in place.

**Techniques not in our ledger** (checked against every `experiment` name in
`ledger/runs.jsonl`):

1. income TE at a **light prior** (α = 1, or sklearn's `'auto'`, which the #27 resolution
   inferred to be ≈ unsmoothed); we have only ever run α = 20;
2. TE on the **`//100`, `//1000` and floor-commute keys**;
3. the **najiama/Marc LightGBM point**: colsample 0.3, depth 5, 10 per leaf,
   `max_bin` 1024. Our hp search settled at `feature_fraction` 0.60, depth 4,
   `min_child_samples` 273, `max_bin` 511;
4. LightGBM `interaction_constraints` (Mamarin +0.00111 in 741755; Marc +0.00005 with
   the full encoding, −0.00010 on the regularised model) — no ledger entry;
5. focal loss plus `linear_tree` as an orthogonal Member (self-report via the blend
   notebook);
6. from the hub table (**self-report, not opened**): **RealMLP (Vladimir Demidov) CV
   0.94601 / LB 0.94621**, 5-fold, and a **competition-only LGBM** (Mizushima Toshihiko)
   CV 0.946099 / LB 0.94636. If the RealMLP number is honest, it contradicts 739354's
   "nothing that is not a boosted tree came closer than about 0.0027". That is worth
   knowing before the Arena is declared closed.

### 3.2 Discussion 742324 — "Bayes error rate"

<https://www.kaggle.com/competitions/playground-series-s6e9/discussion/742324>, by
`badlandsram` (Shaun the Sheep), 2026-09-21, **0 comments** (**Verified**, rendered page
and topic API). Full text:

> "Has anyone tried estimating the Bayes error rate (or irreducible noise floor) on this
> dataset, and what ceiling did you run into?"

**It is a question, not a claim.** No number, no mechanism, no code. Nothing to verify.

### 3.3 Discussion 741755 — "I Tested 15 FE Candidates One at a Time"

<https://www.kaggle.com/competitions/playground-series-s6e9/discussion/741755>, by
Rugved Bane, 3 comments. The post's claims **are verified by the linked notebook's run
log** ([`rugvedbane/i-tested-15-features-1-did-all-the-work`](https://www.kaggle.com/code/rugvedbane/i-tested-15-features-1-did-all-the-work)).
The log's numbers match the post within 0.00001.

- **Setup (Verified):** 17-column baseline (13 raw, nominals one-hot); LightGBM lr 0.05,
  2000 trees, **`early_stopping(50)` on the scored fold**, default 255 bins;
  `StratifiedKFold(5, shuffle, 42)`; baseline **0.94160**.
- **One at a time (Verified, log):** digit decomposition (16 columns from 7 numerics)
  **+0.00182**; Subsidy × Income +0.00026; Income Bin +0.00016; Income per Car +0.00016;
  triple TE of `Gender`, `City_Type`, `Current_Car_Type`, `Range_Anxiety_Level` at
  smoothing 1, 5, 20 **+0.00008**; everything else +0.00001 to +0.00008. **No noise bar
  was measured.**
- **Forward selection after digits (Verified, log):** Income Bin +0.00001, Subsidy ×
  Income −0.00002, Home Charging × Range Anxiety +0.00004, Subsidy × Range Anxiety
  −0.00004, Income per Car +0.00002. **Once digits are in, nothing else moves.**
- **Mechanism claimed:** the generator preserves digit-level patterns. Not tested by the
  author.
- **The substantive reply — Georgy Mamarin, self-report, no code linked in the thread:**
  - a random column moves his baseline by −0.00011 / −0.00015, and he takes that as the
    noise bar;
  - by column, **`Annual_Income_USD` digits alone +0.00192**, commute +0.00002, Age
    −0.00008, fractional digits −0.00004;
  - **raising `max_bin` to the distinct-value count alone gives +0.00201**, the same as
    the digits, and both together +0.00238. His reading: "a resolution lever rather than
    a generator-specific trick";
  - **`interaction_constraints` (no column combinations) at the raised cap: 0.94522 vs
    0.94411 (+0.00111)**, "the board paid that one +0.00123".
- The other two replies (Bogdan Doicin, Tilii) carry no numbers.

**Relevance to us:** it replicates our own finding (digits pay only on income and only
as Resolution). It adds one untried lever, `interaction_constraints`, whose value Marc
shows collapses once the TE is light and multi-key.

---

## Implications for the next turn

These are suggestions; the driving dev decides. Each is one change against the Incumbent
(`hpsearch_lightgbm_best_confirm`, OOF 0.94557). They are ranked by expected value, which
weighs the size of the published anchor against how much our Frame already overlaps it.
**Note: the standing promotion rule (+0.0003 on the canonical seed) is larger than every
single anchor below.** Only #1 and a #1 + #2 chain plausibly clear it. Under that rule,
the honest expectation is that the whole axis gets **at most one promotion**.

1. **`income_te_prior1`: prior weight 20 → 1 on the existing income TE.** Anchor:
   **+0.00025** (Marc, cell 17, α 20 → 1, self-report, additive lr-0.1 model, no digits).
   Change: expose `prior_weight` on `Experiment` and pass it to the Adapter. No Frame
   change. Why first: it is the largest anchor, the cheapest change, and α has never been
   varied in our ledger. The risk is overlap with our digits: Marc's pipeline had none.
   A second configuration at α = 5 (Marc 0.94570, i.e. +0.00018 of the +0.00025) makes it
   a two-point Axis. **Kill:** both α = 1 and α = 5 below +0.0001 or under 4/5 folds →
   the prior axis is dead. Promote on the standing +0.0003 rule.
2. **`te_income_coarse_keys`: TE of `income // 100` and `income // 1000`.** This is
   Marc's and najiama's coarse keys, minus commute. Anchor: **+0.00012** (Marc, prior 20,
   additive model; that figure also includes whole-km commute). Change: new Frame spec
   adding `Annual_Income_USD_div100`, then `target_encode` on it and on the existing
   `_div1000`. Run on top of #1's winner if #1 promotes, since Marc's order is keys then
   prior. **Kill:** below +0.00008 → dead. The raw `_div1000` column already gives the
   tree that grouping, so the prior is weaker than the anchor.
3. **`lgbm_najiama_point`: the published LightGBM params** (colsample 0.3, depth 5,
   `min_child_samples` 10, `max_bin` 1024, L1 0.071, L2 2.0, lr 0.02) at our fixed-round
   protocol. First find the round count on a non-scored inner split, or reuse the ~1600
   rounds both logs show. Anchor: **+0.00014** (Marc: regularised free model vs additive
   reference), and two independent pipelines reach 0.94603–0.94607 with it. It is a
   tuning change, not representation. It sits in a region our hp search did not end in
   (colsample 0.6, 273 per leaf), which is why it is worth one run. Best run *after* #1,
   because its value was measured on a light-prior TE. **Kill:** below +0.0001 → dead;
   the hp search stands.
4. **`te_nearest_original`**: a Frame column from the original's incomes (download
   `itzzomkar/ev-adoption-behavior-and-range-anxiety`), then TE it. Anchor: **+0.00005**
   (Marc, self-report). That sits at his additive noise bar and almost certainly under
   our Paired-Delta Noise Floor, because it touches only 2.1% of rows. **Only run if #1
   promotes; kill below +0.00005.** Otherwise record it as dead-on-anchor without
   running.
5. **Commute at prior 1** (`Daily_Commute_km` exact plus `floor(commute)` TE). Our #29
   `commute_te` was **+0.00000** at α = 20. Marc's only commute-specific evidence is
   bundled into the +0.00012 coarse-key step, and his §4b calls commute's fingerprint
   "the same kind of signal, a fraction of the size". **Reopen only if #1 shows α
   matters on our Frame**, as a single run. Kill below +0.00005.
6. **Not recommended (dead on published or own evidence):** original-label witnesses and
   `_org_mean` (≈ 0 in Marc's tests, three other authors cited at zero); composite income ×
   Recipe TE keys (−0.00044 Marc, −0.00014 ours); triple smoothing (+0.00001 Marc);
   frequency-encoding breadth (#29 dead); GAM margin (#33 dead); `interaction_constraints`
   on a regularised model (−0.00010 in Marc's final model). heuljax's neighbour rates
   (group 4) and latent composition (group 6) are unablated anywhere. They are the only
   genuinely new ideas in §2, but each is a multi-column Adapter build with no anchor.
   With five days to the deadline they are not worth the slot.

**Where the gap is, in one line (Inferred):** Marc and najiama reach 0.9460 with
**fewer** features than we have, and nothing they use that we lack is a feature family
we have not tested. What we have not tested is **how light the income prior is, which
coarse keys are encoded, and one parameter region.** Together these account for
~+0.0004 in their pipeline. How much survives on top of our digits is the one number
worth measuring.

---

## Gaps — stated plainly

1. **Marc's ablation table is a self-report.** Only the final model (0.94603) and the
   additive reference (0.94581) were re-run on Kaggle and appear in a log. The +0.00025
   (prior), +0.00012 (keys), +0.00005 (nearest original) and +0.00014 (params) are
   markdown from a local run on LightGBM 4.7.0. The negative readings (witness, composite
   keys) **are** verified by the AUC-harness log.
2. **LB 0.94617 (Marc) and LB 0.94638 (najiama, heuljax) are not verified.** The Kaggle
   API returned no public score per kernel, and the leaderboard endpoints still return
   the HTML shell.
3. **The early-stopping optimism in Marc's and najiama's runs was not measured.** Neither
   log prints an AUC curve (LightGBM `verbose=False`; najiama logs every 1000 rounds).
   The ~0.00001–0.00002 figure is carried over from heuljax's printed XGBoost curves and
   is an inference.
4. **None of the published gains was measured on a Frame that already has income
   digits.** Marc has no digits. najiama has digits and everything else at once, with no
   ablation. So the published anchors could shrink to zero on our Frame, as #29's did.
5. **Fold-seed effect unknown.** Marc and najiama use `random_state=42`; we use 0. Pooled
   OOF over the same 668,665 rows differs by partition noise that nobody here has sized
   for this seed pair.
6. **The 10-fold ≈ +0.00015 figure** is from 739354 as quoted in the earlier note. It was
   not re-read for this one.
7. **The GAM's standalone AUC is still not printed anywhere.** The ≈ 0.938 is the
   round-0 AUC (margin + one tree), an inference.
8. **heuljax's groups 4, 5, 6, 8, 9, 10 have no ablation anywhere.** Their individual
   worth is unknown, and so is whether the 173 beat Marc's 23 by more than protocol
   (§2.4 says ~+0.0001 at best, cross-pipeline).
9. **Not opened, though cited as primary by the sources read:** megayak's `%1000` / prior
   sweep notebook (the "heavy smoothing washes it out" claim); Markus.JM's CTBoost
   "smooth keys" notebook (the origin of the keys); Mamarin's ledger notebook (the source
   of the +0.00111 constraint number and the noise-bar method); Demidov's RealMLP (CV
   0.94601, which would matter for the Arena); Mizushima's competition-only LGBM (CV
   0.946099). Each is a self-report from here.
10. **The najiama blend notebook's current run failed** (`FileNotFoundError` in its log),
    so its +0.00007 CV and +0.00001 LB are unverified markdown. Its weights were chosen
    on the rows they are scored on, which our Blend definition calls a leak.
11. **najiama's XGB header claims 10-fold 0.94624; its code and log are 5-fold 0.94614.**
    The hub table repeats the header. Treat the hub's fold counts as unreliable.
12. **Discussion message API:** `GetForumMessagesInTopic` refuses both anonymous and
    CLI-token requests. Thread bodies here come from the rendered page (jina text mode)
    and could in principle miss collapsed replies. The comment counts match the topic
    API's `totalMessages` for 741755 (3). For 742385 and 742324 the rendered page says
    0 comments.

---

## Sources

Primary, read directly (source pulled with `kaggle kernels pull`, run log with `kaggle kernels output`):

- <https://www.kaggle.com/competitions/playground-series-s6e9/discussion/742385> — Marc Maldonado Lorca, first post (0 comments).
- <https://www.kaggle.com/code/marcmaldonado/s6e9-the-generator-remembers-the-original-rows> — cells 17 (ablation table), 18 (keys, `nearest_original`, `target_encode`), 19 (params, folds, early stopping), 22 (LB history); run log (OOF 0.94603, per-fold, 97.9% / 2.1% counts).
- <https://www.kaggle.com/code/marcmaldonado/s6e9-is-the-row-memory-worth-anything-in-auc> — cell 1 harness; run log (reference, noise bars, witness, composite keys, linear reader, ties).
- <https://www.kaggle.com/datasets/itzzomkar/ev-adoption-behavior-and-range-anxiety> — the original dataset, as named in both notebooks' `dataset_sources`.
- <https://www.kaggle.com/code/heuljax/kps6e09-xgb-sample> — cells 0–8; run log (per-fold AUC curves every 100 rounds, trees, pooled OOF).
- <https://www.kaggle.com/code/najiama/pure-lgbm-model-cv-0-94607-lb-0-94638>, `/xgboost-triple-te-dynamic-pruning-lb-0-94639`, `/catboost-triple-te-dynamic-pruning-lb-0-94624`, `/oof-power-two-single-models-blend-lb-0-94638`, `/s6e9-electric-vehicle-oof-cv-0-94618-lb-0-94633`, `/honest-model-directory-oof-hub` — sources and logs.
- <https://www.kaggle.com/competitions/playground-series-s6e9/discussion/742324> — Bayes error rate (0 comments).
- <https://www.kaggle.com/competitions/playground-series-s6e9/discussion/741755> — Rugved Bane plus 3 replies (Mamarin, Doicin, Tilii).
- <https://www.kaggle.com/code/rugvedbane/i-tested-15-features-1-did-all-the-work> — source and run log verifying 741755.
- Topic metadata via `POST https://www.kaggle.com/api/i/discussions.DiscussionsService/GetForumTopicById`.

Ours: `CONTEXT.md`; `docs/research/linear-model-representation.md`; issues #3, #26, #27, #29; `src/frame.py`, `src/adapter.py`, `src/experiments.py`; `ledger/runs.jsonl`, `ledger/promotions.jsonl`, `ledger/submissions.jsonl`.
