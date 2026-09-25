# The linear model at CV 0.94640: what representation does it need?

Research for issue #26 (map #22). Retrieved 2026-09-24. Reading only — nothing was
trained, no `src/` file, ledger entry or pipeline file was touched.

**Bottom line.** The 0.94640 claim is real, traceable to a named participant, and
**one sentence long** — its configuration is not published anywhere and the
author has only two comments and one notebook on Kaggle. But that same author's
*published* notebook contains a **penalised additive logistic model** (a
ridge-Newton GAM) fitted inside the folds and fed to XGBoost as an initial
margin, and its representation is almost certainly the answer to the ticket's
question. That representation is **not linear in any column of our Baseline
Frame**: every term is a table of one free coefficient per observed value, plus
one coefficient per `Concern × Subsidy × Anxiety` cell. The identification of
that GAM with the "logistic regression" number is an **inference**, clearly
marked as such below.

---

## How these facts were obtained

Kaggle's rendered pages are a JavaScript shell, so three unauthenticated
channels were used:

1. **Kaggle's JSON RPC API** for the forum index —
   `POST https://www.kaggle.com/api/i/discussions.DiscussionsService/GetTopicListByForumId`
   with body `{"forumId":9538219}`, paged with `{"page":n}`. This returns the
   competition's **50 topics** (the API's own `count` field says 50). The
   per-topic *message* endpoints are not exposed: `GetForumTopicById` with
   `{"forumTopicId":n}` returns the topic record with no messages, and every
   plausible message-service name returns the HTML shell. Thread bodies are
   therefore not obtainable from the API.
2. **A text-rendering proxy** (`r.jina.ai`) of the exact Kaggle discussion URLs,
   with `x-engine: browser` and `x-no-cache: true` — without both, the proxy
   returns the cookie banner and nothing else. It is flaky; several threads
   needed 3–6 attempts and two never rendered (noted under **Gaps**).
3. **Kaggle's notebook source endpoint**,
   `https://www.kaggle.com/kernels/scriptcontent/<scriptVersionId>/download`,
   which returns the full `.ipynb` **including saved cell outputs** without
   login. This is the highest-trust channel found and it is where the
   load-bearing evidence in §2 comes from — actual source code, not a
   description of it.

Nothing below rests on a search-engine summary. One web search returned a
confident paragraph about a "winning pipeline" that matches no primary source
read here; it is discarded.

---

## 1. The 0.94640 claim: located, and it is one sentence

**Source.** Philipp Singer's thread *"Is there a leak?"*,
<https://www.kaggle.com/competitions/playground-series-s6e9/discussion/742317>,
comment by user `heuljax`, verbatim and in full:

> "For me, my current best single models (10-fold) are XGB (0.94652) and
> logistic regression (0.94640), and am also now stuck within the CV
> 0.94650-0.94660 range."

**Author identity — verified.** `heuljax` is **Paul Bryan Elefante**
(<https://www.kaggle.com/heuljax>), Competitions Expert, global rank 2,204.
His profile pages report, in Kaggle's own counters:

| | |
|---|---|
| total discussion posts | **2** (0 topics, 2 comments) |
| public notebooks | **1** — `kps6e09-xgb-sample`, public score 0.94638 |

So the quoted sentence is **one of only two things this participant has ever
written on Kaggle about this competition**, and the other is a pointer to that
notebook. There is no follow-up, no code, no OOF file, and nobody asked him.
Issue #3 recorded him as "rank 5"; that specific rank is not confirmed by
anything read here (his profile shows the global tier rank, not the S6E9
standing, and the leaderboard API endpoints all return the HTML shell).

**Is 0.94640 a comparable OOF number? No.** Three reasons, in decreasing
certainty:

1. **Verified: it is 10-fold, ours is 5-fold.** He says "10-fold". Our
   Canonical Fold Partition is `StratifiedKFold(n_splits=5, shuffle=True,
   random_state=0)`. A 10-fold number is measured on a different instrument —
   each fit sees 90% of train instead of 80%. The same forum thread that
   measures this puts the effect at **≈ +0.00015** for an otherwise identical
   recipe (739354, §3), which is already half the distance from our Incumbent
   at 0.94528 to the Plateau's lower edge.
2. **Verified for his XGB, inferred for his LR: the published harness early-stops
   on the fold it scores.** His notebook's fold loop passes the validation fold
   as `evals=[(dvalid,'valid')]` with `xgb.callback.EarlyStopping(rounds=750,
   metric_name='auc', maximize=True, save_best=True)` and then computes the OOF
   AUC from `booster.predict(dvalid)`. The number of rounds — and `save_best`,
   the *choice of booster* — is selected on the scored rows. Its saved output
   reads `CV POOLED_OOF_AUC=0.946308910839`. That figure is optimistically
   biased by an unknown amount. It is not a Comparison Run under our frozen
   protocol, and if the logistic regression was measured in the same harness,
   neither is 0.94640.
3. **Not stated: pooled or mean-of-folds.** He does not say. 739354 notes that
   quoting a best fold rather than a pooled OOF flatters you by about +0.0006 on
   this data.

**So the honest reading of 0.94640 is: a self-report, on a 10-fold split we do
not share, from a harness whose published sibling selects rounds on the scored
rows, with no code and no OOF vector.** It is not above our Incumbent in any
sense our instrument can read. It is not evidence of a ceiling either — it is
one sentence.

---

## 2. What the same author's published code actually fits

This is the substance of the ticket, and it is verified source code:
`https://www.kaggle.com/kernels/scriptcontent/352387662/download`
(notebook <https://www.kaggle.com/code/heuljax/kps6e09-xgb-sample>, Apache-2.0,
10-fold, pooled OOF 0.946309, public LB 0.94638, 173 features).

The notebook's own header cell names the mechanism:

> "**Additive coordinates and composition inversion:** ridge-Newton GAM
> components; … Initial margin: $GAM_{gate}+GAM_{other}+0.25(GAM_{income}+GAM_{commute})$."

### 2.1 The additive logistic model, term by term

`class AdditiveState` builds a term list and fits it with `gam_kernel`, a
numba-compiled **penalised Newton coordinate ascent on the Bernoulli
log-likelihood** — `GAM_ITERS = 12` passes, `GAM_STEP = 0.5`, update
`delta = step * grad / (hess + ridge)`, intercept initialised to
`log(prior/(1-prior))`. That is a logistic regression: one coefficient per
level of each term, an L2 (ridge) penalty per term, Newton solver.

The terms, verbatim from the source:

```python
self.terms  = [('gate', 0, 5.0, 0)]
self.terms += [('income', w, 25.0, 1) for w in (0, 2, 8, 32, 128)]
self.terms += [('commute', w, 25.0, 2) for w in (0, 5)]
self.terms += [(c, 0, 5.0, 3) for c in OTHER_COLS]
```

(tuple = `key, smoothing half-width, ridge, coordinate group`.)

| term | design | size | ridge | smoothing |
|---|---|---|---|---|
| `gate` | **one indicator per `Concern × Subsidy × Anxiety` cell** | 31 | 5.0 | none |
| `income` × 5 | one coefficient per **exact income value**, at five resolutions | the income alphabet (~13k) | 25.0 | box half-widths 0, 2, 8, 32, 128 |
| `commute` × 2 | one coefficient per exact commute value | the commute alphabet | 25.0 | widths 0, 5 |
| `Age` | one coefficient per **value** | 45 | 5.0 | none |
| `Number_of_Cars_Owned`, `Charging_Stations_Near_Home`, `Charging_Stations_Near_Work`, `Gender`, `City_Type`, `Current_Car_Type`, `Home_Charging_Possible` | one coefficient per level | each column's cardinality | 5.0 | none |

The gate code is the explicit interaction, verbatim:

```python
def gate_codes(raw):
    concern = np.rint(raw['Environmental_Concern_Level']).astype(np.int32)-1
    anxiety = raw['Range_Anxiety_Level'].astype(np.int32)
    subsidy = (raw['Subsidy_Available'] == 1).astype(np.int32)
    valid = (concern >= 0) & (concern < 5) & (anxiety < 3) & (raw['Subsidy_Available'] < 2)
    return np.where(valid, concern*6 + subsidy*3 + anxiety, 30).astype(np.int32)
```

5 concern levels × 2 subsidy × 3 anxiety = 30 cells, plus one catch-all. **That
answers the ticket's interaction question directly: yes, the interaction is
entered explicitly, and not as a product term but as a saturated categorical
over the cells of the Recipe's three discrete drivers.** It is exactly the
`Subsidy × Concern` structure #2 measured (95.6% of positives in
`Subsidy=Yes & Env>=3`), generalised to include anxiety.

### 2.2 Scaling, regularisation, solver — the direct answers

- **Scaling: none, and none is needed.** Every term is a code table indexed by
  value. There is no continuous column entering as a slope anywhere in the GAM,
  so there is nothing for a scaler to equalise. (Scaling *does* appear elsewhere
  in the notebook — `IncomeGroupPrior` standardises its design matrix with
  `(matrix-self.mean)/self.scale` before a ridge-penalised XGBoost on
  income-group aggregates — but not in the additive logistic model.)
- **Regularisation: ridge, per block, two strengths.** 5.0 on the gate and the
  eight low-cardinality columns; **25.0 on income and commute**, the
  high-cardinality blocks. Plus a second, more interesting regulariser: the
  **box smoothing across adjacent values** (`width` in the term tuple), which
  pools the gradient and Hessian of neighbouring income values before the Newton
  step. That is what makes a 13,000-coefficient income block estimable — it is a
  smoothness prior in value space, i.e. a piecewise-constant spline basis at
  five resolutions simultaneously. There is no scikit-learn equivalent.
- **Solver: 12 Newton coordinate passes at step 0.5**, hand-written and
  numba-compiled. Not lbfgs, not liblinear, not SAGA.
- **No penalty search.** The five constants (`GAM_ITERS`, `GAM_STEP`, the two
  ridges, the width grid) are literals.

### 2.3 How the additive model reaches the GBDT — the hybrid, verified

Two ways at once, in `DonorState.transform`:

```python
components = self.gam.components(raw)
result[:, [COL[c] for c in COORD_COLS]] = components
...
margin = (components[:, 0]+components[:, 3]+0.25*(components[:, 1]+components[:, 2])).astype(np.float32)
return result, margin
```

- the four coordinate sums (`GAM_GATE_COORD`, `GAM_INCOME_COORD`,
  `GAM_COMMUTE_COORD`, `GAM_OTHER_COORD`) enter **as columns**; and
- a weighted recombination of them — gate + other at full weight, income +
  commute at **0.25** — enters as XGBoost's **`base_margin`** on train, valid
  and test `QuantileDMatrix`.

The 0.25 shrinkage on the income and commute coordinates is the one judgement
call visible in the file, and it is unexplained.

**Fold discipline: honest, and stricter than ours.** `build_fold_bundle` fits
the whole donor state — GAM included — on **five inner folds inside each of ten
outer folds** (`INNER_FOLDS = 5`), and `fit_donor_state` asserts
`np.intersect1d(donor, receiving).size or np.intersect1d(donor, outer_valid).size`
is empty before fitting. This is the same Nested Cross-Fit invariant our Adapter
holds, with the assert in the same place. The early-stopping problem in §1 is a
separate defect and does not touch this.

### 2.4 The rest of the 173 features, in one paragraph

For context on what a *competitive* representation looks like here, from the
notebook's own header and its `FEATURE_COLS` assembly: multi-scale target
encoding `TE(k) = (S_k + αμ)/(n_k + α)` on `income//10`, `//100`, `//1000` and
`floor(commute)` at α ∈ {1, 2, 5, 10, 50, 200} with `log1p(n_k)` support
columns; adjacent-value neighbourhood rates at half-widths 1…250; hierarchical
smoothed rates; per-value lift and priors joined against the original
10,000-row dataset; income digits `//10**k` for k = 0…5 declared as
**categorical** (`FEATURE_TYPES` marks `DIGIT_COLS` as `'c'`, not `'q'`);
Gaussian-smoothed class-conditional mixtures over concern/subsidy/anxiety;
posterior uncertainty columns; and monotone constraints pinned to `+1` on every
target-rate column. The Recipe's closed form is present as
`latent_values(...)[:,3]` = `1.2*income/1e5 + 0.6*concern + 2*subsidy - penalty`
with `penalty ∈ {0,1,3}` — the published constants, unmodified.

---

## 3. The one fully specified linear model in the forum — and it is on the wrong dataset

`siukeitin`, *"Logistic regression beats advanced models on the 'original'
dataset"*,
<https://www.kaggle.com/competitions/playground-series-s6e9/discussion/739142>.
Complete code is posted. The pipeline, verbatim in structure:

- features: **exactly four** — `Subsidy_Available`, `Environmental_Concern_Level`,
  `Annual_Income_USD/100000` (renamed `Annual_Income_100kUSD`),
  `Range_Anxiety_Level`;
- encoding: `category_encoders.OneHotEncoder()` on the categoricals, income as a
  single rescaled continuous column;
- imputation: `SimpleImputer(strategy='constant', fill_value=0, add_indicator=True)`;
- **no scaler**, **no explicit interaction**, `LogisticRegression(max_iter=1000)`
  — i.e. scikit-learn defaults: **L2, C = 1.0, lbfgs**;
- split: `StratifiedKFold(5, shuffle=True, random_state=0)`, pooled OOF.

Results, as printed:

| model | 5-fold OOF AUC |
|---|---|
| **logreg** | **0.90631** |
| tabpfn | 0.90558 |
| lgb | 0.90223 |
| xgb | 0.90146 |

**This is on the 10,000-row original `EV_Adoption_and_Range_Anxiety_Dataset.csv`,
not the competition data.** The author's own conclusion is about the generator:
"the data generation model is possibly a simple logistic model". So the one place
a plain scikit-learn logistic regression is shown to beat GBDTs is a dataset
sixty-seven times smaller than ours, and the numbers are four hundredths of AUC
below the Plateau.

The reason the ranking inverts is stated by another primary source and it is the
single most useful sentence found for this ticket (739354, quoted verbatim):

> "The original is a small, smooth problem that a linear model handles fine, and
> nearly everything a GBDT wins by on the competition data is the generator's
> sampling artefacts, which the original by definition does not contain."

---

## 4. Does anyone report a linear + GBDT blend, and what is it worth?

**Nobody publishes a linear + GBDT blend delta.** What exists, in decreasing
relevance:

| evidence | number | status |
|---|---|---|
| Deotte, *Fable 5.1 - XGB Starter* — Recipe's closed form as XGBoost `base_margin` vs the same 17-feature baseline (5-fold CV) | 0.94194 → **0.94200** (+0.00006); as an extra column 0.94201; equal blend of all three **0.94209** | **verified**, read from <https://www.kaggle.com/competitions/playground-series-s6e9/discussion/739321> |
| the same, on a stronger frame — commenter `starkhushi` | base_margin / LightGBM `init_score` **+0.00005 to +0.00007**, replicating across XGB and LGB; as extra columns 0.946209 → 0.946216 (nothing) | self-report, recorded in issue #3 |
| Elefante's notebook — additive logistic model as `base_margin` **and** as four columns, inside a 173-feature XGB | OOF 0.946309 / LB 0.94638 for the whole thing | **verified**; the GAM's own contribution is **not ablated anywhere**, so its worth is unknown |
| 739354 — every combiner tried over pools of up to ~80 members: rank average, hill climbing, stacking, NN meta, GBDT meta, bagged hill climbing | **all within ~0.0002 of the best single member**; rank averaging ~0.0014 *below* it | self-report, ablation published |
| 739354 — 20 model families on one frozen 5-fold split | "**Nothing that is not a boosted tree came closer than about 0.0027 below plain xgboost.**" TabM −0.0031, MLP −0.0027, RealMLP −0.0044 | self-report, ablation published |
| 739233, `starkhushi` — five algorithms, identical features, `StratifiedKFold(5, seed 42)` | LightGBM 0.94587, CatBoost 0.94579, XGBoost 0.94577, TabM 0.94546, MLP-with-embeddings 0.92760; rank-correlations 0.996–0.999 | self-report, independent replication of the above |
| 739354 — the generator's formula as a feature | "Adding both formulas to the model changed nothing." The buy score alone sits ~0.008 below their full model | self-report |
| Tilii (16th), in 740769 | "In numerous tries I never got Ridge to be better than any of these models, not even logistic regression. **So LR is my choice for quick blending**" | self-report — note this is LR **as the meta-learner**, not as a base model. Worth keeping the two senses apart; it is an easy source of confusion about "a logistic regression scoring 0.9464" |
| our own #4 | free-coefficient logistic refit on the Recipe's four variables: 0.937772, i.e. **+0.00008** over the closed form. A GBDT on the same four columns beats the linear refit by **+0.0029** | our measurement |

Read together: **the published field has no evidence that a linear model is
worth anything on this dataset beyond about +0.00006 as a base margin**, and two
independent ablations say every non-GBDT family lands 0.0027 or more below a
plain GBDT. Elefante's sentence is the only datapoint pointing the other way,
and §1 explains why it cannot be compared to ours.

There is a mechanism that reconciles them, and it is the answer to the ticket's
real question: **a linear model can reach the Plateau here, but only if the
representation carries the non-linearity for it.** Every gain measured in this
competition lives in *per-value* income structure (the $30,000 spike at 9.2% of
rows and a quarter of the base rate; the $170,537 cliff; 41.7% of rows on values
the generator over-produced; 97.9% exact-match against the original's income
alphabet). A per-value coefficient table *is* a linear model — in a design matrix
of 13,000 indicators. That is precisely what Elefante's GAM is, and it is why the
word "logistic regression" is not a claim about simplicity.

---

## 5. Which parts of our Baseline Frame a linear model cannot use

Our Frame (`src/frame.py`, spec `baseline`) is 27 columns: 6 raw numerics, 2
ordinals as single numerics, 3 income digits, 2 count encodings, 14 one-hot
indicators. Verdicts, with the mechanism and our own evidence:

| Frame element | usable by a linear model? | why |
|---|---|---|
| `Annual_Income_USD_mod1000`, `_mod100` | **No. Delete them.** | As a linear term, a coefficient on `income % 100` asserts that buy probability rises monotonically with the last two digits of income. It is a sawtooth in income, and the assertion is meaningless. Our own #4 measured `income % 100` at **AUC 0.50001 on non-floor rows** — all of its apparent signal is the $30,000 floor mass. It carries *zero* linear signal. Its entire value is **Resolution**: it lets a histogram splitter address individual income values. A linear model has no splitter. |
| `Annual_Income_USD_div1000` | **Technically yes, but pointless.** | `income // 1000` is income rescaled and floored — correlation ≈ 1 with raw income. It contributes one nearly-duplicate column, which only hurts the conditioning of the design matrix and splits one coefficient across two collinear terms. |
| `Age` raw (45 values, one slope) | **No — it silently discards the finding the Frame exists to protect.** | The column carries a **9-sigma non-monotone residual** (13.02% positive at age 68 against 21.77% at 27). A tree gets all 45 values for free, which is why the Frame asserts `AGE_DISTINCT_VALUES == 45`. A single linear coefficient fits one slope through a non-monotone curve and recovers almost none of it. The assert still passes — this failure is silent, which makes it exactly the class of bug ADR-0003 is about. |
| `Range_Anxiety_Level` as `{Low:0, Medium:1, High:2}` | **No.** | The Recipe's penalty is **0, −1, −3** — non-linear *and* unequally spaced in code order. A single slope forces equal spacing and misprices `High`. 739354 flags this as a trap in the same terms: "the range-anxiety category is **not** monotone in its code order (medium −1, high −3, low 0)." Note `High` is only 0.33% of rows (#4), so the mispricing is cheap in AUC but free to fix. |
| `Environmental_Concern_Level` as 1–5 numeric | **Yes, and it is the one ordinal a linear term gets right** | the Recipe is `0.6 * concern`, genuinely linear. But `Env == 1 → a 39× lower buy rate across 22% of rows` (starkhushi's "Edge 3", quoted in 739354 as "the strongest single-feature fact anyone has posted here") suggests a threshold on top; dummies cost four columns and are strictly safer. |
| `Annual_Income_USD` raw (one slope) | **Yes, and it captures the Recipe term — and nothing else.** | `1.2 * income/1e5` is linear, so one coefficient gets the Recipe right. But every measured gain in this competition is per-value income structure, and a slope expresses none of it. This is the single biggest representational hole. |
| `*_count` (count encoding of income and commute over train+test) | **Usable, but wrong scale.** | Raw integer frequencies with a long tail; as a linear term this asserts a monotone, unit-per-count effect. Needs `log1p` — which is exactly what Elefante's notebook does (`SUPPORT_INC_LOG`, `MSTE_*_LOGN`, `np.log1p(count)` throughout). |
| one-hot nominals, `drop_first=False` | **Yes.** | The Frame's #7 resolution already noted this choice "leaves the Frame usable by a scale-sensitive family without a second representation", and it does. With an L2 penalty the redundant level is harmless (the penalty breaks the tie); without one it is singular. |
| **no interaction terms at all** | **The binding gap.** | ADR-0001's fourth consequence forbids hand-engineered interactions, and the reason was measured and correct: a GBDT on the Recipe's own four inputs beats the Recipe's closed form by **+0.00269**, more than twice what all nine other columns are worth (+0.00120). But that reason is a fact about **trees**. A linear model finds no interaction it is not handed. The Recipe's gate — 95.6% of positives in `Subsidy=Yes & Env>=3` (#2) — is exactly what the one verified competitive additive model enters explicitly, as a 30-cell indicator. |
| nested cross-fit income target encoding (Adapter, `income_te`) | **Yes — and it is the one element that transfers unchanged.** | A per-value target rate is a number a linear model can take a coefficient on. It is also the only thing in our pipeline that speaks to per-value income structure. |

### What it would need instead

A linear entry into the Arena is not a new model on the existing Frame. It is a
**second representation behind the Model Adapter**, and it needs six things:

1. **Income as a basis, not a slope.** Cheapest honest version: keep raw income
   *and* the existing nested cross-fit TE of the exact value, and add the two
   hard edges as indicators — `income == 30000` and `income >= 170537`. Those
   flags were measured at **+0.00002** for a GBDT (najiama's ablation, cited in
   739354) precisely because a tree isolates them in one split; for a linear
   model they are not redundant, because it cannot. The stronger version is
   Elefante's: a smoothed per-value coefficient table at several resolutions,
   which has no scikit-learn equivalent and would have to be written.
2. **Age as 45 dummies** (or a natural cubic spline with enough knots to bend
   twice). This is non-negotiable: it is the only way the 9-sigma non-monotone
   residual survives into a linear model.
3. **`Range_Anxiety_Level` as three dummies.** Let the fit find −1 and −3
   instead of asserting −1 and −2.
4. **An explicit gate block.** One indicator per `Concern × Subsidy` cell (10
   columns), or per `Concern × Subsidy × Anxiety` cell (30 columns) to match the
   one verified competitive design. This is the term the Frame deliberately does
   not have, and the one a linear model cannot do without.
5. **Drop `_mod1000` and `_mod100`; drop `_div1000` if raw income stays.**
   `log1p` the two count columns.
6. **Scaling, and the penalty that makes it matter.** `needs_scaling()` already
   returns `True` for non-tree families (`src/models.py`), and ADR-0001 already
   fixes the two rules that go wrong here — **digits are computed from the
   original integer before any scaling**, and the scaler is **fitted inside the
   training fold**. Both still bind. One thing to state that the ADR does not:
   an L2 penalty is **not** scale-invariant, so without standardisation the
   penalty is effectively a different strength per column, and a saturated
   dummy design *requires* a penalty to be estimable at all. Scaling here is not
   cosmetic; it is part of the regulariser.

The seam exists, so this is representation work, not architecture work. But note
what it costs in our own terms: a linear candidate changes **six things** about
the Baseline Frame at once. It cannot be a Paired Delta against `income_te_tuned`
— it is a new Incumbent chain in the **Arena**, comparable only through its
out-of-fold vector, which is precisely what the Arena was defined for.

---

## Gaps — what is not findable, and what was not read

Stated plainly, because a "not findable" is a finding here.

1. **The configuration behind 0.94640 is not published, and there is strong
   evidence it never will be.** Its author has two Kaggle comments in total and
   neither describes the model. This is the ticket's kill criterion, and it is
   met: **no primary source describes that logistic regression.** Its features,
   encoding, penalty, solver, and whether it entered interactions are all
   unknown. Everything in §2 is *his published XGB notebook*, which is the best
   available proxy and is not the same artifact.
2. **The identification of §2's ridge-Newton GAM with the "logistic regression"
   of §1 is an inference.** It is a strong one — same author, same week, same
   feature machinery, and the GAM *is* a penalised logistic regression — but he
   never says so. If it is wrong, §2 still stands on its own as a verified
   description of a competitive additive logistic model on this data.
3. **The GAM's own AUC is never printed.** The notebook scores only the final
   XGBoost. So even the verified artifact does not tell us what an additive
   logistic model scores alone on this data, and no ablation of the `base_margin`
   or the four coordinate columns exists.
4. **No linear + GBDT blend delta exists in the forum.** The nearest published
   numbers are Deotte's +0.00006 for the *closed-form Recipe* as a base margin
   and starkhushi's +0.00005–0.00007 replication. Neither is a fitted linear
   model.
5. **Two threads never rendered** through any available channel and were not
   read: 742324 (*"Bayes error rate"* — unanswered per #3) and 741755 (*"I
   Tested 15 FE Candidates One at a Time"*). Also unread: the tables inside
   najiama's *Honest Model Directory & OOF Hub* notebook, which catalogues 28
   public models and is the one source that could settle whether **any** public
   notebook fits a linear model on the competition data. Its discussion thread
   (742697) renders; the notebook's tables do not. I therefore cannot say
   "nobody published a competitive linear model" — only that neither of the two
   published multi-family ablations (739354's 20 families, 739233's five)
   contains one, and both report every non-GBDT family at 0.0027 or worse.
6. **"Rank 5" is not verified.** Kaggle's leaderboard endpoints all return the
   HTML shell to an unauthenticated client. The comment is by a Competitions
   Expert at global rank 2,204; his S6E9 standing was not readable. Issue #3's
   "rank 5, 0.94667" should be treated as unconfirmed.
7. **Elefante's `0.25` margin weight is unexplained**, and his ridge constants
   are literals with no search behind them. If this axis is ever built, those are
   free parameters, not a recipe.

---

## Sources

Primary, read directly:

- <https://www.kaggle.com/competitions/playground-series-s6e9/discussion/742317> — "Is there a leak?" (Psi). The 0.94640 comment.
- <https://www.kaggle.com/heuljax>, `/code`, `/discussion` — author identity and post counts.
- `https://www.kaggle.com/kernels/scriptcontent/352387662/download` — full source **and saved outputs** of <https://www.kaggle.com/code/heuljax/kps6e09-xgb-sample>. The load-bearing evidence in §2.
- <https://www.kaggle.com/competitions/playground-series-s6e9/discussion/739142> — siukeitin, the fully specified logistic regression, with code and printed scores.
- <https://www.kaggle.com/competitions/playground-series-s6e9/discussion/739321> — Deotte / Fable 5.1, base-margin vs extra-column vs blend table.
- <https://www.kaggle.com/competitions/playground-series-s6e9/discussion/739354> — the 20-family and every-combiner ablations, and the sentence explaining why the linear/GBDT ranking inverts between the original and the competition data.
- <https://www.kaggle.com/competitions/playground-series-s6e9/discussion/739233> — starkhushi, five algorithms on identical features.
- <https://www.kaggle.com/competitions/playground-series-s6e9/discussion/740769> — Tilii on LR as a *blender*.
- <https://www.kaggle.com/competitions/playground-series-s6e9/discussion/742697>, `/739049`, `/742904` — read, nothing load-bearing.
- Forum index via `GetTopicListByForumId` (forumId 9538219): 50 topics.

Ours: issues #2, #3, #4, #5, #7; `src/frame.py`, `src/columns.py`, `src/models.py`;
ADR-0001; `CONTEXT.md`; `README.md`.
