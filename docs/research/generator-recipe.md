# Is the reverse-engineered generator worth using?

Research for issue #5. Measured 2026-09-22 on `data/train.csv` (668,665 rows, 17.4645 % positive).
Read-only analysis plus throwaway LightGBM fits used strictly as measurement instruments — no
pipeline code, no model kept.

## Headline: the recipe is real, but it is NOT the ceiling

Scoring every train row with the published generator formula

```
buy_score = 1.2*(income/1e5) + 0.6*concern + 2*subsidy - 1*(anxiety == Medium) - 3*(anxiety == High)
positive if buy_score > 5.5
```

and ranking by the **noiseless** deterministic part (ROC AUC only needs a ranking, and additive
independent noise makes `P(y=1|x)` a monotone transform of `buy_score`, so the noiseless score *is*
the Bayes ranking under the published model) gives:

| quantity | AUC |
|---|---|
| **published formula, noiseless, as a ranking** | **0.937690** |
| free-coefficient logistic refit on the same 4 variables | 0.937772 |
| published plateau (field consensus) | ~0.9464 CV / 0.9466 LB |
| **gap** | **-0.0089** |

The ticket's hypothesis was that if the formula lands near the plateau, the residual is provably
noise and the ceiling is arithmetic. **It does not land near the plateau.** 0.0089 of AUC — roughly
45 times the LB noise floor of 0.0002 — sits between the published recipe and what the field
routinely reaches. The residual is *not* noise; it is structure the published formula does not
express.

### The recipe's coefficients are nevertheless confirmed

A free logistic regression on exactly the recipe's variables (income/1e5, concern, subsidy,
anxiety dummies), renormalised so the concern coefficient equals the published 0.6:

| term | published | free refit |
|---|---|---|
| income / 1e5 | 1.2 | **1.157** |
| concern | 0.6 | 0.6 (fixed by normalisation) |
| subsidy | 2 | **2.072** |
| anxiety = Medium | -1 | **-0.841** |
| anxiety = High | -3 | **-1.728** (High is 0.33 % of rows — weakly identified) |
| threshold | 5.5 | **5.647** |

The refit buys **+0.00008 AUC** over the published constants. The recipe is arithmetically correct
to within the fifth decimal of AUC. Re-deriving it would recover nothing.

## Where the missing 0.0089 lives

All numbers below are 5-fold `StratifiedKFold(shuffle=True, random_state=42)` OOF AUC, LightGBM
`lr=0.05, num_leaves=63, min_child_samples=200, ff/bf=0.9`, early stopping 50–200. This frame is a
deliberately plain baseline (no digit decomposition, no target encoding), so its absolute level sits
below the published plateau; the **deltas** are what the ticket needs.

| model | OOF AUC | delta |
|---|---|---|
| published formula as a score (no model) | 0.937690 | — |
| LightGBM on the **4 recipe variables only** | **0.940632** | **+0.00294** |
| LightGBM on all 13 features | 0.941665 | +0.00103 |
| published feature engineering the field measured on top (digit decomp +0.00143, triple TE +0.00129, freq enc +0.00112) | — | ~+0.0038 |

Read the decomposition:

- **+0.0029 comes from the recipe's own four variables, non-linearly.** A GBDT given nothing but
  income, concern, subsidy and anxiety beats the linear recipe on those same four columns by far
  more than the free linear refit did (+0.00008). The label in *our* 668k rows is therefore not a
  monotone function of any linear index of the recipe inputs. This is the mechanism the field
  already documented from the other side: exact income values are resampled from the original
  10,000 and "remember" their labels, so income carries fine-grained, non-monotone information that
  a coefficient cannot. It is exactly why digit decomposition, `max_bin` and target encoding of
  income pay.
- **+0.0010 comes from the nine features the formula does not mention at all.**
- The remainder to the plateau is the published encoding work, which our plain frame does not have.

The arithmetic roughly closes: 0.9377 + 0.0029 + 0.0010 = 0.9417 (measured), + ~0.0038 published FE
≈ 0.9455, + tuning ≈ the 0.9464–0.9466 plateau.

**Interpretation.** The published recipe is the label rule of the **original 10,000-row dataset**.
Our 668k rows were regenerated from it by a learned generative model, and that regeneration only
approximately preserves the rule while adding its own resampling artifacts. Treat the formula as a
faithful description of the *source*, not as the data-generating process of the competition data.

## Sanity checks

### 1. Does the threshold reproduce the 17.4645 % positive rate? Nearly, but not exactly.

The deterministic rule `buy_score > 5.5` with **zero** noise flags **17.794 %** of train rows —
0.33 pp above the observed 17.4645 %.

Adding symmetric zero-mean noise only makes it worse. Implied positive rate by Gaussian noise scale:

| sigma | 0.01 | 0.25 | 0.5 | 0.75 | 1.0 | 1.5 | 2.0 |
|---|---|---|---|---|---|---|---|
| implied rate | 0.1781 | 0.1807 | 0.1818 | 0.1886 | 0.2012 | 0.2340 | 0.2671 |

The rate is **monotonically increasing in sigma and never reaches 0.17465** — its infimum is
0.1778. So no zero-mean symmetric noise scale reproduces the observed positive rate with the
published threshold; the threshold would have to sit at ~5.65 (which is what the free refit
recovered).

**Inferred noise scale.** By maximum likelihood on the margin `buy_score - 5.5`:

- Gaussian noise: **sigma = 0.802**, mean log-loss 0.23622
- Logistic noise: **scale = 0.444** (equivalent sigma 0.805), mean log-loss 0.23610

The two are indistinguishable in fit (logistic marginally better), and both imply a positive rate of
~0.191 — over-predicting by 1.6 pp, again pointing at a threshold slightly above 5.5. `sigma ≈ 0.8`
is the honest estimate; a plain `N(0,1)` is rejected (rate 0.2012, worse log-loss).

### 2. Do the coefficient magnitudes match the observed single-feature AUCs? Yes.

Standard deviation of each term's contribution to `buy_score`, against #2's single-feature AUCs:

| term | sd in score units | single-feature AUC |
|---|---|---|
| subsidy (2 × 0/1) | 0.967 | 0.718 |
| concern (0.6 × 1–5) | 0.858 | **0.844** |
| income (1.2 × inc/1e5) | 0.344 | 0.670 |
| anxiety (0 / -1 / -3) | 0.335 | 0.545 |

The ordering matches once granularity is accounted for: subsidy has the largest spread but is
binary, and a binary feature's AUC is capped by its own prevalence, so 5-level concern out-ranks it.
Income and anxiety have nearly identical spread (0.344 vs 0.335) but income is continuous and
monotone while anxiety is 90.3 % Low / 9.3 % Medium / **0.33 % High**, so the large -3 coefficient
touches almost nobody. Independent corroboration of the recipe from a direction the recipe was not
fitted on.

### 3. Does the formula mispredict systematically on columns it does not contain? Yes — commute above all.

Residual = observed label − recipe probability at sigma = 0.802. Baseline residual is **-0.0162**
everywhere (the global over-prediction from check 1); deviations below are read against that.

**`Daily_Commute_km` — the big one.**

| band (km) | n | residual | buy rate |
|---|---|---|---|
| = 5.0 (floor) | 144,280 | -0.0145 | 0.1844 |
| (5, 10] | 1,705 | -0.0193 | 0.1871 |
| **(10, 15]** | **9,024** | **+0.0384** | **0.3020** |
| (15, 20] | 29,031 | +0.0062 | 0.2348 |
| (20, 40] | 226,556 | -0.0180 | 0.1749 |
| (40, 60] | 226,058 | -0.0186 | 0.1649 |
| **(60, 70]** | **27,967** | **-0.0334** | **0.1066** |
| (70, 200] | 4,044 | -0.0004 | 0.1088 |

This is not a confound of the recipe inputs: `corr(commute, income) = 0.008`,
`corr(commute, concern) = -0.019`. And it survives conditioning on the recipe score — observed buy
rate (%) by recipe-score decile × commute band:

| decile | ≤5 | (5,20] | (20,40] | (40,60] | >60 |
|---|---|---|---|---|---|
| 5 | 2.72 | **4.18** | 2.93 | 2.84 | 2.63 |
| 6 | 9.72 | **12.71** | 9.26 | 9.50 | 7.68 |
| 7 | 28.04 | **35.80** | 27.36 | 26.30 | 21.73 |
| 8 | 54.50 | **61.00** | 53.13 | 52.94 | 45.25 |
| 9 | 79.56 | **84.19** | 77.96 | 77.66 | 72.77 |

A consistent ~+6 pp lift for the 5–20 km band and ~-6 pp for >60 km, inside every score decile.
`Daily_Commute_km` is absent from the published formula and is worth real AUC.

**`Age` — a smaller but clean bump.** Residual by 5-year band: 50–55 is **-0.0032** (buy rate
0.2030) against a -0.016 baseline, while 60–65 is -0.0235 (0.1596). A non-monotone age effect the
formula has no term for.

**City type, home charging, chargers, car type, gender, cars owned** — all deviations within
±0.01 of baseline and Spearman correlation with the residual under |0.03| (`Home_Charging_Possible`
-0.028 is the largest). `Charging_Stations_Near_Work` drifts to -0.024 in the 16–19 range, and
`Charging_Stations_Near_Home` shows the Simpson pattern the field flagged. Secondary next to
commute.

**Calibration.** The recipe over-predicts monotonically across the whole upper half of the score
range (ventile 16: predicted 0.384 vs observed 0.327; ventile 20: 0.879 vs 0.845) and under-predicts
in the far-left tail (ventile 1: 0.00000 vs 0.00003). A recipe-derived probability is not calibrated
for our data and would need a link refit before any use as a probability.

## Decisions

### Recipe as `init_score` / `base_margin` — **NO for turn 1.**

Measured in our own frame (fold 0 of the same split, LightGBM, `metric='auc'`, early stopping 200;
no-init baseline on that fold = **0.940414**):

| init_score variant | best_iter | AUC | delta |
|---|---|---|---|
| none | 434 | 0.940414 | — |
| logit of recipe prob, clip 1e-6 | 132 | 0.937663 | **-0.00275** |
| logit, clip 1e-4 | 243 | 0.937865 | -0.00255 |
| logit, clip 1e-3 | 345 | 0.937940 | -0.00247 |
| logit, clip 1e-3, shrunk ×0.5 | 344 | 0.938566 | -0.00185 |
| raw margin `(s-5.5)/0.802` | 337 | 0.938420 | -0.00199 |
| full 5-fold OOF, clip 1e-6 | — | 0.938763 | -0.00290 vs 0.941665 |

Every variant **hurts**, by 0.002–0.003, and the effect is robust to clipping and shrinkage. The
mechanism is visible in `best_iter`: an anchored margin saturates the extreme score regions and the
booster cannot recover the within-region income fine structure that supplies most of its edge.

Even taking the field's number at face value, +0.00005–0.00007 is **below the 0.0002 LB noise
floor** and only ~2.5–3.5× a 0.00002 paired floor. It is an extra moving part in the baseline for a
gain we cannot measure on the leaderboard and that we measure as *negative* locally. Revisit only
if turn 2 reaches a plateau-class frame and has spare slots.

### Original 10,000 rows as extra training data — **NO. Rule it out.**

10,000 rows against 668,665 is a 1.5 % bump with a matching distribution; the field measured
label-joining at **+0.00003**, an order of magnitude under the LB noise floor, and measured the
income × recipe-column keying variant as *costing* 0.0004. Our own numbers reinforce it from a
different angle: the value that the original rows carry is the **exact-income identity**, and that
is already extractable from our own 668k rows by target/count encoding of income without any join
(#2 measured the encoding hook; the field measured the join on top at +0.00003). Nothing here
justifies the external-data plumbing.

### Reproducing the generator from seed 101 — **NO. Take the formula as given.**

Two measurements settle it. (a) Free refit of the formula's coefficients on our 668k rows buys
+0.00008 AUC over the published constants, so the recipe is already right and a byte-exact
reproduction cannot sharpen it. (b) The only thing a reproduction adds beyond the formula is the
original row identity, which the field measured at +0.00003. Against an 8-day budget, a
reproduction is a multi-hour task with a measured ceiling of ~0.00003–0.00008 — under the noise
floor in both directions. Verified empirically against our train set, which is what this document
did, is enough.

## What this changes downstream

1. **The ceiling is not arithmetic.** The published recipe accounts for 0.9377 of the ~0.9466
   plateau. The remaining 0.0089 is recoverable structure — most of it (+0.0029) living in
   *non-linear* use of the recipe's own inputs, chiefly the exact-income artifacts. This confirms
   digit decomposition / target encoding / high `max_bin` on income as the priority feature work,
   and it explains *why* they work.
2. **`Daily_Commute_km` is the most promising non-recipe feature**, and is currently under-rated:
   single-feature AUC 0.534 (#2) hides a non-monotone effect worth ~6 pp of buy rate inside every
   recipe-score decile. Bin it, or give the model enough splits on it; the 5–20 km band and the
   >60 km band are the two edges.
3. **Do not add recipe-derived quantities to turn 1** in any form — `init_score` measured negative,
   as a plain column it measured +0.00004 (0.941665 → 0.941704), inside noise. Deotte's own
   conclusion holds: a GBDT rediscovers the recipe from 669k rows on its own.

## Reproduction

Throwaway scripts in a `mktemp`-style scratch dir (`/tmp/genrecipe`), `uv` venv on Python 3.14.7
with numpy 2.x / pandas 3.x / scikit-learn 1.9.1 / lightgbm 4.7.0 / scipy 1.18.1. Nothing was
installed into the project `.venv`. The measurements are: score every row with the formula; AUC
against `Will_Buy_EV`; MLE of the noise scale on the margin; free logistic refit; 5-fold LightGBM
with and without the recipe as `init_score` and as a column; residual tabulation by every non-recipe
column.
