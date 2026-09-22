# Competition rules and public discussion — Kaggle S6E9

Research for issue #3. Retrieved 2026-09-22.

**Competition slug: `playground-series-s6e9`** — "Predicting Electric Vehicle Purchases",
Playground Series Season 6 Episode 9, hosted by Kaggle, sponsored by Google LLC.
<https://www.kaggle.com/competitions/playground-series-s6e9>

## How these facts were obtained

No Kaggle credentials were available. Two unauthenticated primary channels worked:

1. **Kaggle's own JSON RPC API**, which serves the competition record without login:
   `POST https://www.kaggle.com/api/i/competitions.CompetitionService/GetCompetition`
   with body `{"competitionName":"playground-series-s6e9"}`, and
   `.../discussions.DiscussionsService/GetTopicListByForumId` / `GetForumTopicById`
   for the competition forum (`forumId` 9538219).
2. **The rendered Rules and Overview pages.** The HTML served to a plain HTTP client is a
   JavaScript shell, so the pages were read through a text-rendering proxy
   (`r.jina.ai`) of the exact Kaggle URLs. Every number below that came from that channel
   is independently corroborated by the JSON API field of the same name, so nothing here
   rests on a single unverifiable read.

Nothing in this document is inferred from a search-engine summary. One early search summary
claimed "5 submissions per day" — it is wrong; Kaggle's own record says 10.

---

## 1. External data — ALLOWED

**Confidence: high.** Verbatim, Rules section 2.6:

> **6. EXTERNAL DATA AND TOOLS**
>
> a. You may use data other than the Competition Data ("External Data") to develop and test
> your Submissions. However, you will ensure the External Data is either publicly available
> and equally accessible to use by all Participants of the Competition for purposes of the
> competition at no cost to the other Participants, or satisfies the Reasonableness criteria
> as outlined in Section 2.6.b below. [...]
>
> b. The use of external data and models is acceptable unless specifically prohibited by the
> Host. [...] their use must be "reasonably accessible to all" and of "minimal cost".
>
> c. Automated Machine Learning Tools ("AMLT") [...] may be used [...] provided that the
> Participant or Team ensures that they have an appropriate license.

Source: <https://www.kaggle.com/competitions/playground-series-s6e9/rules>

**Conditions:** public availability and equal, no-cost access for all participants. There is
**no separate disclosure/registration obligation** anywhere in these rules — no "you must post
your external data in the forum" clause. The Host has not prohibited anything specifically.

**What this unblocks:** the original 10,000-row *EV Adoption Behavior* Kaggle dataset that the
competition data was generated from is public and free, so it is squarely inside the rule. The
field is already using it openly and Kaggle grandmasters are publishing analyses built on it
(see §6), which is strong behavioural confirmation that the Host is not treating it as a
violation. **Competition Data itself is CC BY 4.0** (`license` field in the API record), and
Rules 2.4 allows using it "for any purpose, whether commercial or non-commercial".

## 2. Submission limits

**Confidence: high — two independent sources agree exactly.**

| | Value | Source |
|---|---|---|
| Submissions per day | **10** | Rules 2.2.a "a maximum of ten (10) Submissions per day"; API `maxDailySubmissions: 10` |
| Final submissions selected for judging | **2** | Rules 2.2.b "up to two (2) Final Submissions"; API `numScoredSubmissions: 2` |
| Max team size | **3** | Rules 2.1.a; API `maxTeamSize: 3` |
| Submission size limit | 20480 MB | API `submissionSizeLimitMb` |

Note on the final-submission rule: you **select** 2 of your submissions to be scored on the
private leaderboard. Kaggle's default when nothing is selected is the best-public-LB
submissions — with a 20% public split and the noise analysis in §5, deliberately selecting one
best-CV and one best-public-LB submission is the standard hedge.

## 3. Deadline

**Confidence: high.** Overview > Timeline, verbatim:

> **Start Date** — September 1, 2026
> **Entry Deadline** — Same as the Final Submission Deadline
> **Team Merger Deadline** — Same as the Final Submission Deadline
> **Final Submission Deadline** — September 30, 2026
>
> All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted.

API `deadline` field agrees: `2026-09-30T23:59:00Z`.

**In the dev's timezone (America/Sao_Paulo, UTC−3, no DST):
Wednesday 2026-09-30 at 20:59 BRT.**

From 2026-09-22 that is **8 days**, and at 10/day roughly **80 remaining submissions**
(fewer if any were already spent today). Entry deadline is the same day, so joining the
competition late is not a risk as long as the rules are accepted before submitting.

## 4. Evaluation split — public LB is 20% of test

**Confidence: high.**

- API record: `leaderboardPercentage: 20`, `totalSolutionRows: 286571`.
- 20% of 286,571 = **57,314 rows** on the public leaderboard; the remaining **229,257 rows
  (80%)** decide the private leaderboard.
- Corroborated from the forum: a public post computing leaderboard standard errors uses
  "the *same* 57,314 public rows" as the public-LB size, which matches exactly.
  <https://www.kaggle.com/competitions/playground-series-s6e9/discussion/740049>

Metric is ROC AUC (API `evaluationAlgorithm: Roc Auc Score`, `isMax: true`), scores truncated
to 5 decimals (`scoreTruncationNumDecimals: 5`).

**How much can a public-LB delta be trusted?** The best public analysis in the forum (below)
answers this precisely, and the answer is better than "not at all": because every team is
scored on the *same* 57,314 rows, the sampling noise is largely common and cancels in a
*difference*.

## 5. Leaderboard noise — the number that sets the submission strategy

**Confidence: medium-high** (a participant's analysis, but published with DeLong covariance,
a paired-bootstrap cross-check, and a public correction after a grandmaster-level challenge).

Source: "Correction: my LB standard-error analysis was wrong — the paired SE is 10x smaller",
Senih Bayankulu, <https://www.kaggle.com/competitions/playground-series-s6e9/discussion/740049>

- Marginal SE of a single AUC on 57,314 rows: **0.00101**.
- But two submissions on the same rows are correlated (ρ measured 0.9919–0.9986, median
  0.9954), so `SE(diff) = σ·√(2(1−ρ))` gives **0.00005–0.00013** for models sharing a pipeline,
  and ~0.00020 as a conservative cross-team figure.
- Practical reading, stated in the post: *"your own local neighbourhood is tied, the leaders
  are not."* Ranks 25–164 sit within ~1.2 SE of each other. Rank 1's lead is ~2–4 SE — real.
- Finite-population correction for the 20% draw without replacement: ×√(1−0.2) = 0.894,
  making gaps ~11% *more* significant.

**Implication for us:** a public-LB improvement of **+0.0002 or more** against our own previous
submission is a real signal; anything under ~0.0001 is indistinguishable from noise and should
not be trusted over CV. This also means public-LB probing is a poor use of the 10 daily slots
unless the expected effect is large.

## 6. What top teams disclosed

The forum has 45 topics; the substantive ones are unusually open for a Playground competition.

### Chris Deotte (rank 3, 0.94672 in 3 submissions) — published essentially everything

Deotte ran AI agents ("Fable 5.1", "GPT-6 Astra") on the data and published the results.
**Confidence: high — these are his own posts.**

**a) The generator was reverse-engineered.**
<https://www.kaggle.com/competitions/playground-series-s6e9/discussion/739303>
(notebook: <https://www.kaggle.com/code/cdeotte/fable-5-1-eda-original-data-insights>)

- The original 10,000-row dataset is fully synthetic and its RNG **seed was recovered (101)**;
  re-rolling reproduces all 10,000 ages, genders, incomes, cities, cars, car types and
  commutes **exactly**.
- Age is uniform 25–69. Income is a clipped normal with a **hard floor spike at exactly
  $30,000**. Charging stations are uniform with **city-dependent ranges** (rural 0–2,
  suburban 0–7, urban 2–14). `Home_Charging_Possible` is a city-conditional Bernoulli
  (rural ~90% yes, urban ~40%).
- **The label rule (the "recipe"):**
  `buy_score = 1.2·(income/100000) + 0.6·concern + 2·subsidy − 1·(anxiety=medium) − 3·(anxiety=high) + noise`,
  and the person buys if `buy_score > 5.5`.
- Consequence stated by Deotte: the noise term is irreducible, so **no model can be
  100% right — there is a hard Bayes ceiling.**

**b) How to feed the recipe to a GBDT — and how little it buys.**
<https://www.kaggle.com/competitions/playground-series-s6e9/discussion/739321>
(notebook: <https://www.kaggle.com/code/cdeotte/fable-5-1-xgb-starter>)

| model (17 features, 5-fold CV) | AUC |
|---|---|
| baseline XGB | 0.94194 |
| recipe as XGBoost **base margin** (init_score) | 0.94200 |
| recipe as an extra column | 0.94201 |
| equal blend of all three | **0.94209** |

His own conclusion: XGBoost rediscovers the recipe on its own from 669k rows. A commenter
running a stronger frame (`starkhushi`) reports the recipe as extra columns adds nothing
(0.946209 → 0.946216) but as **base_margin / LightGBM `init_score` adds +0.00005 to +0.00007,
replicating across XGB and LGB** — "best single idea in this competition".

**c) A Simpson's paradox to beware of.**
<https://www.kaggle.com/competitions/playground-series-s6e9/discussion/738991>
More chargers near home correlates with a *lower* raw buy rate (19.00% at 0–2 stations vs
17.27% at 11–14), which reverses inside both `Home_Charging_Possible` strata (9.20%→14.70%
and 20.16%→21.07%). The confounder is that 89.4% of the 0–2 group can charge at home vs 40.4%
of the 11–14 group. Relevant to any univariate feature screening we do.

**d) On agent workflow** (Deotte's own reply in 739303): the instruction that mattered most was
forcing agents to produce **leak-free, honest CV/OOF** — he cites agents caught doing target
encoding outside the k-fold and using validation targets to build features.

### The commodity path to ~0.9459–0.9463 is published in full

**Confidence: high — author-reported ablations, two independent authors agree on the ranking.**

Rugved Bane, "0.94590 LB | Digit Decomposition + Triple Target Encoding":
<https://www.kaggle.com/competitions/playground-series-s6e9/discussion/741117>
(notebook: <https://www.kaggle.com/code/rugvedbane/0-94590-lb-digit-decomp-triple-te-full-guide>)

| technique | AUC gain |
|---|---|
| Digit decomposition | +0.00143 |
| Triple target encoding | +0.00129 |
| Frequency encoding | +0.00112 |
| **total over tuned-XGB baseline** | **+0.00384** |

Score progression: solo Optuna-tuned XGB 0.94206 → 7-model stack + digit + freq 0.94461 →
**single well-tuned LGBM + digit + freq + triple TE 0.94590**. His stated surprise: *the
stacking framework was beaten by a single LGBM once the feature engineering was solid.*

Corroborating, independently: amirhossein karimiee reports digit decomposition as the single
biggest OOF gain at **+0.0015**
(<https://www.kaggle.com/competitions/playground-series-s6e9/discussion/739596>), and
ANSHIKA JOISAR reports 0.94217 → 0.94311 from it alone
(<https://www.kaggle.com/competitions/playground-series-s6e9/discussion/741457>).

**Why digit features work** — the generator's quantisation artifacts. Beyond the $30,000 floor,
the field has found a **hard income ceiling around $170,537** (above which everyone buys:
<https://www.kaggle.com/competitions/playground-series-s6e9/discussion/738968>) and a
reported "dead zone" at $38k–$42k. The `% 1000` remainder of income is repeatedly cited.

### What the generator "remembers" about the original rows — and why it is worth ~nothing

**Confidence: high for the measurements (published with code and controls); the AUC verdict is
the author's own and is the most decision-relevant claim in the whole forum.**

Marc Maldonado Lorca, "S6E9: the generator copied the original incomes and remembers their
labels": <https://www.kaggle.com/competitions/playground-series-s6e9/discussion/742385>

- **97.9% of train rows carry an income value that exists in the 10,000-row original** (8.4%
  if drawn independently); each original value recurs ~60 times.
- The buy rate of an exact income value has **2.4× the variance** binomial sampling would give,
  net of the other twelve columns (±0.044 on a 17.5% base rate). This is the mechanism behind
  `max_bin`, digit features, the `%1000` remainder and target encoding all working.
- The original row's own label leaks into the residual: **+0.029 (original bought) vs −0.009
  (did not)**, bootstrap CIs far apart. The generator also copies city, charging stations and
  car type from the original row at 1.2–1.4× chance.
- **Verdict: "What it is worth in AUC: nothing beyond the encoding."** A cross-fitted target
  encoding of exact income (keys `value`, `//100`, `//1000`, nearest-original) under a
  well-regularised LightGBM gives **OOF 0.94603 / LB 0.94617**; adding the original's label on
  top gains **+0.00003**; keying by income × recipe-column *costs* 0.0004; as a tie-breaker it
  recovers nothing.

This matters a lot for us: **joining the original dataset for label information is a dead end
that several people have already measured.** The external-data rule permits it; the evidence
says it does not pay.

### Where the field says the ceiling is

**Confidence: medium — consistent, independent participant reports, not an official statement.**

- Oleksii Zhukov (Master): one frozen 10-fold scheme, ~300 logged runs; best single model
  CV 0.946274, best blend CV 0.946378 → public **0.94652**. *"reachable signal seems to stop
  around CV 0.9464"*.
- Paul Bryan Elefante (rank 5, 0.94667): best single models 10-fold XGB **0.94652**,
  **logistic regression 0.94640**, stuck in CV 0.94650–0.94660.
  (Note the striking one: a plain logistic regression is within 0.0001 of a tuned XGB — exactly
  what a linear-recipe-plus-noise generator predicts.)
- DefiAudit: multi-seed ensemble OOF ~0.9463, *"no model family escapes it"*.
- A dedicated "Bayes error rate" thread exists and is so far unanswered:
  <https://www.kaggle.com/competitions/playground-series-s6e9/discussion/742324>

## 7. The rank-1 gap — yes, the field noticed; no, nobody has explained it

**Confidence: high that the field noticed and has no explanation. No source offers a mechanism.**

Philipp Singer ("Psi", Kaggle Grandmaster) opened a thread titled **"Is there a leak?"** on
2026-09-21: *"Top score looks suspicious? Any secret sauce?"*
<https://www.kaggle.com/competitions/playground-series-s6e9/discussion/742317>

Seven messages in, nobody has an answer. The strongest reply (Oleksii Zhukov) does the
arithmetic rather than guessing, and it is worth reproducing because it reframes the gap:

- `megayak` (rank 7) published a notebook showing that **the public 20% can be partially
  *read* with paired submissions** — two exact AUC identities let a pair of submissions recover
  a group's labels, moving a file from 0.94651 to 0.94656 **with no model at all**:
  <https://www.kaggle.com/code/megayak/s6e9-0-94656-reading-the-public-split>
- By that arithmetic **one public positive is worth about two units of the fifth decimal**, so
  the 0.9465–0.9466 band is a matter of a few hundred specific public rows.
- Zhukov's conclusion: a score that far above the pack would mean *"roughly 85 public positives
  above the pack — a lot of rows to be right about when nobody's reported CV suggests the
  signal is there."* He expects **"the private board to look quite different from this one."**

Rank 1 (`URAD`, user `uradkr`) has 39 submissions, has not replied in the thread, and has
published nothing. **No public notebook or post accounts for 0.94945.**

### The three live hypotheses, as the field frames them

1. **Public-split probing.** 10 submissions/day × ~30 days is ~300 slots; megayak *proved* the
   public 20% is partially readable by paired submissions. Rank 1 has only 39 submissions,
   which is few for systematic probing, but the technique is cheap per bit. If this is the
   cause, **the lead evaporates on the private 80%.**
2. **Genuine signal nobody else found** — some further generator artifact past income.
   Nothing in the forum supports this, and several people have looked hard.
3. **Luck on the 20%.** Fits the DeLong numbers poorly: 0.0027 is ~13–27 SE(diff), far too
   large for chance alone.

**Our reading: (1) is by far the best-supported hypothesis, and it is not a target worth
chasing.** The honest ceiling the field reports is CV ≈ 0.9464 / LB ≈ 0.9466, and that is what
turn 1 should aim at. The rank-1 gap is most likely a public-leaderboard artifact that will not
survive the private split — which is exactly why the 2-final-submission rule should be spent on
best-CV rather than best-public-LB.

---

## Open items / gaps

- **Rank 1's actual method is unknown and unknowable from public sources.** `uradkr` has
  published nothing and did not answer the grandmaster who asked directly.
- **No official Kaggle statement on the gap.** The Host has not posted in the forum about it.
- The megayak public-split notebook's full code body was not read (Kaggle's notebook source is
  behind the JS app); the summary above is from its rendered table of contents plus Zhukov's
  description of it in the forum. The *existence* and *headline numbers* of the technique are
  well-sourced; the implementation detail is not.
- Ceiling estimates in §6 are participant self-reports. No independent replication was done
  here; they are consistent across at least four people, which is why confidence is medium
  rather than low.

## Sources

- <https://www.kaggle.com/competitions/playground-series-s6e9/rules>
- <https://www.kaggle.com/competitions/playground-series-s6e9/overview>
- Kaggle JSON API: `POST /api/i/competitions.CompetitionService/GetCompetition`
- <https://www.kaggle.com/competitions/playground-series-s6e9/discussion/742317> (Is there a leak?)
- <https://www.kaggle.com/competitions/playground-series-s6e9/discussion/739303> (Deotte — generator/EDA)
- <https://www.kaggle.com/competitions/playground-series-s6e9/discussion/739321> (Deotte — XGB starter / base margin)
- <https://www.kaggle.com/competitions/playground-series-s6e9/discussion/738991> (Deotte — Simpson's paradox)
- <https://www.kaggle.com/competitions/playground-series-s6e9/discussion/742385> (generator remembers original rows)
- <https://www.kaggle.com/competitions/playground-series-s6e9/discussion/741117> (digit decomp + triple TE ablation)
- <https://www.kaggle.com/competitions/playground-series-s6e9/discussion/740049> (DeLong LB standard errors)
- <https://www.kaggle.com/competitions/playground-series-s6e9/discussion/738968> (income ceiling $170,537)
- <https://www.kaggle.com/competitions/playground-series-s6e9/discussion/739596> (digit features +0.0015)
- <https://www.kaggle.com/code/megayak/s6e9-0-94656-reading-the-public-split>
- <https://www.kaggle.com/code/rugvedbane/0-94590-lb-digit-decomp-triple-te-full-guide>
