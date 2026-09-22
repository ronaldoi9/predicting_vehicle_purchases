# Residual signal: what the published generator recipe does not explain

Resolves research ticket [#4](https://github.com/ronaldoi9/predicting_vehicle_purchases/issues/4).
Read-only analysis on `data/train.csv` (668,665 rows). Classifiers were fitted only as
measurement instruments (cross-fitted 5-fold OOF); no pipeline code and no deliverable model.
All scripts were ad hoc, run in a throwaway `uv` venv outside the repo.

Throughout, **`d`** is the published generator score

```
d = 1.2*(Annual_Income_USD/100000) + 0.6*Environmental_Concern_Level
    + 2*(Subsidy_Available=="Yes") - 1*(Range_Anxiety=="Medium") - 3*(Range_Anxiety=="High")
```

with the published buy rule `d + noise > 5.5`, and **the region** is
`Subsidy_Available == "Yes" AND Environmental_Concern_Level >= 3`
(37.4 % of rows, 95.6 % of positives). "Outside" means the complement:
**418,875 rows, 5,119 positives, 1.222 % positive rate**.

---

## TL;DR

1. **The published recipe is not complete — but the part it misses is worth ~+0.0012 AUC, and
   essentially none of it lives in the residual positives.** Chasing the ~5,100 outside-region
   positives is the wrong target: they carry only 4.4 % of the pairwise AUC weight, and the
   whole measured gain of a full model over the recipe (+0.00389) comes from re-ranking the
   *in-region* positives.
2. **The recipe's own functional form is the largest single miss.** A GBDT given only the
   recipe's four columns scores **0.94038** against the closed-form recipe's **0.93769**
   (+0.00269) — more than twice what all nine non-recipe columns are worth put together.
3. **Income count encoding is a real value-level artifact but dies under the recipe score.**
   Conditional on `d`, its AUC is 0.4948.
4. **Both "floors" are isolated point masses with a hole beside them**, so a floor flag is
   information-free: a single split on the raw value already isolates them. `Age` is not
   clipped at all.
5. **There is no income grid, and there is no univariate digit signal.** `income % 100` scores
   AUC 0.5414 over all rows and **0.50001** once the 30,000 mass is removed. The published
   digit-decomposition gain is a resolution/binning effect, not a recoverable grid.

---

## 1. Calibrating the published recipe

Fitting a probit link to the empirical `P(y=1 | d)` curve (563 bins of width 0.01 with n >= 200,
657,563 rows covered):

| quantity | published | measured |
|---|---|---|
| threshold | 5.5 | **5.652** |
| noise scale `sigma` | (unstated) | **0.894** |

Weighted RMSE of the probit fit in z units: **0.394** — the shape is right, the constants are
not exactly the published ones. This is expected: the recipe is the rule that labelled the
*original 10,000 rows*; the competition's 955k labels were resampled by Kaggle's deep tabular
generator, so the recipe survives as an approximation, not as ground truth.

**Closed-form ranking power of the recipe: AUC(d) = 0.93769.**
That is already ~0.009 below the published plateau, so the recipe was never the ceiling.

---

## 2. Priority 1 — the residual positives (the one that matters)

### 2.1 They are structurally forced to be noise draws

`max(d)` inside the outside region is **5.3594**, and **zero** outside rows have `d > 5.5`.
Under the published rule every one of the 5,119 outside positives therefore requires a
*positive* noise draw. There is no sub-population the recipe would have called positive.

Their distribution across the two ways of being outside:

| Subsidy | Env | rows | positives | rate |
|---|---|---|---|---|
| Yes | 2 | 82,317 | 2,863 | 3.478 % |
| Yes | 1 | 87,802 | 824 | 0.939 % |
| No | 5 | 33,567 | 831 | 2.476 % |
| No | 4 | 50,199 | 406 | 0.809 % |
| No | 3 | 52,500 | 159 | 0.303 % |
| No | 2 | 52,816 | 26 | 0.049 % |
| No | 1 | 59,674 | 10 | 0.017 % |

### 2.2 They are *not* pure noise — but only just

Cochran–Mantel–Haenszel style stratified test **inside the outside region**, conditioning on 30
strata of `d` (n = 418,875, 5,119 positives):

| feature | chi2 | df | p |
|---|---|---|---|
| `Charging_Stations_Near_Home` | 47.2 | 14 | 1.8e-05 |
| `Age` (9 bins) | 51.1 | 8 | **2.6e-08** |
| `Daily_Commute_km` (10 bins) | 43.1 | 7 | **3.1e-07** |
| `Home_Charging_Possible` | 22.3 | 1 | **2.4e-06** |
| `City_Type` | 19.1 | 2 | 7.0e-05 |
| `Charging_Stations_Near_Work` | 34.3 | 19 | 0.017 |
| `Current_Car_Type` | 11.1 | 3 | 0.011 |
| `Gender` | 6.6 | 2 | 0.037 |
| `Number_of_Cars_Owned` | 4.3 | 3 | 0.23 |
| *(control)* income tens-digit | 21.7 | 9 | 0.0098 |

The tens-digit control — which must be noise — lands at p = 0.0098, so treat anything at
p > 1e-3 as within this test's own noise floor. Four columns clear that bar decisively.
**Verdict: the recipe is incomplete, and the residual positives do cluster.**

### 2.3 …and the clustering is worth almost nothing

Cross-fitted 5-fold OOF, LightGBM used purely as a measuring instrument:

| score | overall AUC | in-region | outside-region |
|---|---|---|---|
| closed-form `d` | 0.93769 | 0.80787 | 0.83856 |
| isotonic(`d`) (cross-fitted) | 0.93780 | 0.80839 | 0.83798 |
| isotonic(`d`) as `init_score` + the 9 non-recipe columns | **0.93875** | 0.81169 | 0.84093 |
| LightGBM, 4 recipe columns only | **0.94038** | — | — |
| LightGBM, all 13 raw columns | **0.94158** | 0.82123 | 0.84522 |

So everything the nine non-recipe columns add on top of a perfectly calibrated recipe is
**+0.00095**; within the outside region they lift AUC by only 0.83798 → 0.84093.

### 2.4 The decisive measurement: where the AUC gap actually lives

Decomposing the overall AUC into the four positive-class x negative-class cells
(in-region / outside-region), with the pair weights they carry:

| pair cell | weight | LightGBM (13 raw) | recipe `d` | delta | contribution to overall |
|---|---|---|---|---|---|
| pos **in** x neg **in** | 0.2393 | 0.82123 | 0.80787 | +0.01336 | **+0.00320** |
| pos **in** x neg **out** | 0.7168 | 0.99923 | 0.99808 | +0.00114 | **+0.00082** |
| pos **out** x neg **in** | 0.0110 | 0.08863 | 0.12028 | −0.03165 | **−0.00035** |
| pos **out** x neg **out** | 0.0329 | 0.84522 | 0.83856 | +0.00665 | **+0.00022** |
| | | 0.94158 | 0.93769 | | +0.00389 |

**This is the answer to the ticket's priority 1.** The outside positives carry 4.4 % of the
pairwise weight. A model that already beats the recipe by +0.00389 gets **−0.00013 net** from
those two cells — it ranks the residual positives *below* in-region negatives more often than
the recipe does (0.0886 vs 0.1203) and is right to, because it correctly assigns them low
probability. Every measured point of headroom over the recipe is in the in-region cell.

Note how brutal the `pos_out x neg_in` number is: **0.0886**. Outside positives sit below
in-region negatives ~91 % of the time under the best model, and the modelled `P(y=1)` says they
should. To gain +0.001 on the overall metric from that cell alone you would have to lift its AUC
by ~0.09, which would require knowing the noise draw.

### 2.5 Implied ceiling

Treating the cross-fitted calibrated OOF probability as the truth and computing the AUC of the
Bayes-optimal ranking it implies:

| calibrated score | implied Bayes AUC | its own realised AUC | gap |
|---|---|---|---|
| closed-form `d` | 0.93801 | 0.93769 | 0.00032 |
| LightGBM, 13 raw columns | **0.94167** | 0.94158 | **0.00009** |

The gap of **0.00009** says the raw 13-column representation is exhausted: given the
probabilities this model assigns, no re-ranking of it is worth more than a tenth of a
thousandth. Remaining headroom must come from features that sharpen `P(y=1|x)` itself — which
is exactly what the published income target-encoding / digit path does, and it is bounded by
the plateau it has already reached.

---

## 3. Where the recipe *is* wrong (the live lead, such as it is)

Two separable misses, measured on the same OOF frame.

### 3.1 Functional form, on the recipe's own four columns — +0.00269

| model | OOF AUC |
|---|---|
| closed-form `d` | 0.93769 |
| LightGBM on `Annual_Income_USD`, `Environmental_Concern_Level`, `Subsidy_Available`, `Range_Anxiety_Level` | **0.94038** |

The published linear form with coefficients 1.2 / 0.6 / 2 / −1 / −3 and a 5.5 threshold leaves
**+0.00269** on the table *without any extra column*. This is the largest single quantity in
this whole ticket, and it is consistent with the field's finding that feeding the recipe as
`init_score` buys only +0.00005–0.00007: the recipe is a slightly wrong parameterisation of
something the GBDT learns better by itself.

### 3.2 `Age` carries a large, highly significant, non-monotone residual — worth ~0

Stratified test over **all** 668,665 rows, conditioning on 60 strata of `d`:

| feature | chi2 | df | p | max relative lift |
|---|---|---|---|---|
| **`Age`** | **1271.0** | 44 | **1.5e-237** | 0.150 |
| `Daily_Commute_km` (20 bins) | 477.3 | 15 | 3.5e-92 | 0.141 |
| `Charging_Stations_Near_Home` | 189.9 | 14 | 6.2e-33 | 0.045 |
| `Home_Charging_Possible` | 119.4 | 1 | 8.4e-28 | 0.038 |
| `Charging_Stations_Near_Work` | 114.3 | 19 | 1.3e-15 | 0.053 |
| `City_Type` | 38.2 | 2 | 5.2e-09 | 0.023 |
| `Current_Car_Type` | 30.9 | 3 | 8.8e-07 | 0.039 |
| `Number_of_Cars_Owned` | 21.8 | 3 | 7.1e-05 | 0.011 |
| `Gender` | 10.9 | 2 | 0.0042 | 0.021 |

`Age` does not appear in the recipe at all, and `corr(Age, d) = −0.0043` — it is orthogonal to
the recipe score, so this is not confounding. Per-age target rates and stratified z-scores:

| age | rate | z | | age | rate | z |
|---|---|---|---|---|---|---|
| 27 | 0.2177 | **+10.08** | | 68 | 0.1302 | **−10.32** |
| 46 | — | +8.24 | | 65 | — | −7.86 |
| 69 | 0.2103 | +8.09 | | 34 | — | −7.43 |
| 53 | — | +8.07 | | 25 | 0.1445 | −7.23 |
| 55 | — | +6.89 | | 31 | 0.1451 | −6.64 |
| 28 | 0.2045 | +6.66 | | 38 | — | −6.47 |

The swing is 13.0 % to 21.8 % on ~14,000 rows per age (binomial SE ≈ 0.0032) — a ~9-sigma
effect. It is **non-monotone and non-smooth**, which is why the profile's single-feature AUC for
`Age` was 0.5055: as a ranking direction it is worthless, as a 45-level lookup it is enormous.
This is generator memorisation — the deep model reproduced age-specific label noise from the
original 10,000 rows.

**And yet**: all nine non-recipe columns together, including this, are worth **+0.00095** on top
of a calibrated recipe and **+0.00120** inside a raw-feature model. A tree already picks the age
lookup up for free. There is no unexploited edge here; there is an explanation for why
`Age` must be treated as a categorical/high-`max_bin` lookup rather than a smooth numeric.

---

## 4. Priority 2 — income count encoding

**Verdict: a real value-level generator artifact, not an artifact of the count — but it is
subsumed by the recipe score and by income target encoding.**

| measurement (non-floor rows) | AUC |
|---|---|
| count over train+test | 0.46115 |
| **count computed from `test.csv` rows only**, evaluated on train labels | **0.46199** |
| log-count minus a local-density baseline (rolling median over 201 neighbouring income values) | 0.45105 |
| mean within-income-decile AUC, raw count | 0.48844 |
| mean within-income-decile AUC, density residual | 0.47756 |
| **conditional on 60 strata of `d`** | **0.49482** |

Two controls matter:

- **Self-reference is ruled out.** A count built *only* from the 286,571 test rows — which share
  no label information with train — reproduces the effect exactly (0.46199 vs 0.46115). The
  effect is a property of the income *value*, not of the row's own presence in the count.
- **Local density is ruled out.** Subtracting a 201-value rolling-median log-count leaves
  AUC 0.45105 and a within-decile mean of 0.47756. Rare values really are more positive at high
  income, e.g. decile 9 density-residual terciles: rarest **0.3913**, mid 0.3251, commonest
  0.3125; and the sign flips in decile 0 (rarest 0.0620 vs commonest 0.0890), which a pure
  density confound would not do.

But conditional on `d` it is **0.4948** — 0.005 from chance. The count is largely a proxy for
"this income value was copied from the original table many times", and the copied rows' label
information is already in the income value itself.

---

## 5. Priority 3 — the clipping floors

**Verdict: dead. Both "floors" are isolated point masses separated by a hole, so membership is
a function of the value and a flag adds no information. `Age` is not clipped at all.**

### 5.1 `Annual_Income_USD` — the 30,000 mass sits behind an 8,000-wide hole

Distinct income values in train+test, ascending: **30000, 31003, 32910, 36345, 38174, 38209,
38250, 38262, …** — dense from 38,174 onward.

| | rows |
|---|---|
| income exactly 30,000 (train) | 61,605 |
| income strictly between 30,000 and 38,174 (train) | **2** |
| same, test | **4** |

The largest gaps in the entire income support are 30000→32910→36345→38174. So `income == 30000`
is *identical* to `income < 38174`: one threshold split recovers the flag exactly. A separate
"at floor" indicator is redundant by construction, and the earlier "extrapolate the rate back to
the floor" test is invalid — there is nothing to extrapolate across.

(This also relocates the forum's reported "$38k–$42k dead zone": the real hole is
**30,000 → 38,174**, and 38,174 is where the distribution *begins*.)

### 5.2 `Daily_Commute_km` — same shape

| | rows (train) |
|---|---|
| commute exactly 5.0 | 144,280 |
| commute in (5.0, 10.0] | 1,705 |
| commute in (5.0, 8.0] | 1,064 |

Conditional on 60 strata of `d`:

| score | conditional AUC |
|---|---|
| commute **floor flag** | **0.49906** (chance) |
| commute **value** | 0.48545 |

The floor flag is exactly chance once the recipe score is controlled, while the value is not.
**The signal is in the commute value, not in floor membership.**

### 5.3 `Age` — there is no floor

`Age` is uniform on 25…69: 13,264 rows at 25 against a 14,859 per-level average (668,665/45).
That is *below* average, not a clipping spike. `Age == 25` has a stratified z of −7.23 — real,
but entirely unremarkable on an axis whose z-scores run from −10.32 (age 68) to +10.08 (age 27).
**The "age floor" is not a floor; §3.2 replaces this hypothesis.**

---

## 6. Priority 4 — value grid and rounding artifacts

**Verdict: dead. There is no grid, and the apparent digit signal is 100 % the 30,000 mass.**

Income is integral (no cents), but of the **14,667 distinct income values**:

| multiple of | distinct values | rows (train+test) |
|---|---|---|
| 1,000 | **7** | 87,978 |
| 500 | 15 | 88,449 |
| 100 | 98 | 96,242 |
| 10 | 1,480 | 190,309 |

The 87,978 rows on multiples of 1,000 are essentially *only* the 30,000 mass (87,881 rows across
both files). Income values are arbitrary integers; there is no recoverable sampling grid and
therefore no "distance from grid" to compute.

The digit statistics collapse the same way once the floor is removed:

| statistic | AUC, all rows | AUC, non-floor rows |
|---|---|---|
| `income % 100` | 0.54137 | **0.50001** |
| `income % 1000` | 0.52706 | 0.48326 |
| `income % 10 == 0` flag | — | 0.50038 (rate 0.18885 vs 0.18775) |
| `income % 100 == 0` flag | — | 0.49898 |

Conditional on 60 strata of `d`, non-floor rows: `income % 100` → **0.49441**,
`income % 1000` → **0.50040**, `is multiple of 100` → **0.49872**.

`Daily_Commute_km` lies exactly on a 0.1 grid (max deviation from `round(x*10)/10` is **0.0**),
195,588 rows are whole kilometres, and the sub-decimal digit scores AUC **0.50210**
marginally and 0.499 conditional on `d`. Nothing.

**Consequence for the pipeline.** The published digit-decomposition gain (+0.00143) is *not* a
digit signal — the digits carry no marginal information. It is a **resolution** effect: splitting
income into `//1000`, `%1000`, `%100` lets a histogram-binned tree address individual income
values that `max_bin` would otherwise merge, which is the same mechanism as exact-income target
encoding. Raising `max_bin` and target-encoding the income value is the honest version of the
same idea; adding digit columns on top is cheap and harmless but should not be expected to add
beyond it.

---

## 7. What this means for the project

### 7.1 The achievable-ceiling question, answered

The ticket asked whether the residual positives are noise draws and what that implies for the
ceiling. The precise answer:

- They are **not** pure noise — four columns show residual dependence inside the outside region
  at p < 1e-4, against a p = 0.0098 noise control.
- **But it does not matter.** The two AUC cells involving them carry 4.4 % of the pairwise
  weight, and a model that beats the recipe by +0.00389 gets **−0.00013** from those cells. The
  whole gain lives in `pos_in x neg_in`.
- Given a well-calibrated model on the raw 13 columns, the implied Bayes AUC is **0.94167**
  against a realised **0.94158** — a 0.00009 re-ranking margin. The raw representation is
  exhausted; only sharper estimation of `P(y=1|income value)` moves the number, and that is
  precisely what the published ~0.9459–0.9460 path already does.

**Therefore: the consensus ceiling of ~CV 0.9464 / LB 0.9466 is consistent with everything
measured here, and none of the four hypotheses in this ticket opens a route above it.** The
rank-1 gap of 0.0027 is not explained by any structure in the residual positives, by income
counts, by floors, or by a value grid — which strengthens the public-probing explanation in #3.

### 7.2 Concrete, cheap consequences for the pipeline

1. **Treat `Age` as a 45-level lookup**, not a smooth numeric: categorical, or a cross-fitted
   target encoding, or simply `max_bin >= 64` so every age gets its own bin. The residual is
   9-sigma and non-monotone (§3.2). This is free and a smoothing/monotone treatment would
   destroy it.
2. **Drop the "floor flag" items from the feature backlog.** `income == 30000` is
   `income < 38174` and `commute == 5.0` is `commute < 5.1`; the tree already has them, and
   conditional on the recipe score the commute flag is exactly chance (§5).
3. **Do not expect income count encoding to earn its place next to income target encoding.** Keep
   it if it is free (compute on train+test), but budget it at ~0 and measure it against the
   TE-bearing model, not standalone (§4).
4. **Do not build grid / distance-from-grid features.** There is no grid (§6).
5. **The recipe as `init_score` remains worth its published +0.00005–0.00007 at most.** Its
   closed form is 0.00269 *worse* than a GBDT on its own four inputs (§3.1), so it should be
   offered as a hint, never as a constraint, and never as a monotone prior.

### 7.3 What this closes on the map

Four of the ticket's hypotheses resolve negative, with measurements. Combined with the four
already killed in #2/#3, **every structural leak hypothesis raised so far on this competition is
now closed.** The gap-strategy ticket (#11) should be phrased as "confirm the plateau and hedge
the two final submissions", not as "find the differentiator".

---

## 8. Method notes

- All AUCs marked OOF are 5-fold `StratifiedKFold(shuffle=True, random_state=0)` on
  `Will_Buy_EV`, which #2 established is leak-free here (no duplicate feature vectors).
- LightGBM was used only as a measuring instrument: `binary` objective, `lr=0.05`,
  600–700 rounds, `num_leaves` 63–127, `feature_fraction`/`bagging_fraction` 0.8. No tuning,
  no early stopping on the evaluated fold, no model persisted. Absolute levels are therefore
  ~0.004 below the published tuned plateau; only the *differences between rows of the same
  table* are being read.
- Target encodings inside the ablation are fitted on the training folds only, with additive
  smoothing (prior weight 20) and the fold prior as the fallback for unseen keys.
- The stratified chi2 tests are Cochran–Mantel–Haenszel-style: within each `d` stratum the
  expected positive count per level is `n_level * rate_stratum`, summed across strata, and the
  statistic is the sum of squared standardised deviations with `df = levels − 1`. The income
  tens-digit row is a deliberate noise control and calibrates the test's own floor.
- The implied Bayes AUC is computed by isotonically calibrating the score against `y`, then
  evaluating `P(p_i > p_j)` weighted by `p_i (1 − p_j)` over all ordered pairs, with ties split.
  It is a *self-consistent* ceiling: it bounds what re-ranking the given probability estimate can
  achieve, not what a better probability estimate could achieve.
- Environment: throwaway `uv` venv, Python 3.14.7, pandas 3.0.6, numpy 2.5.3, scikit-learn 1.9.1,
  LightGBM 4.7.0, scipy 1.18.1. Nothing was installed into the project `.venv`.
