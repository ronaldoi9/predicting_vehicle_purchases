# Predicting Vehicle Purchases

A [Kaggle Playground Series S6E9](https://www.kaggle.com/competitions/playground-series-s6e9) entry — predicting `Will_Buy_EV` from 668,665 rows of synthetic survey data, scored by ROC AUC.

The interesting part of this repo is not the model. It is the **instrument**: a measurement apparatus built so that a difference of 0.0003 in AUC can be told apart from noise, and so that a number which comes out *plausible but biased* fails loudly instead of being believed.

That emphasis is not stylistic. The competition's entire open question is a gap of 0.0027, the honest ceiling is ~0.9466, and 840 of 2,704 teams were already within 0.0006 of each other. At that resolution, a pipeline you cannot trust is worse than no pipeline, because its results look like knowledge.

## Results

| | CV (OOF AUC) | Public LB | CV→LB Offset |
| --- | --- | --- | --- |
| Turn-1 Incumbent (the **Floor**) | 0.943577 | 0.94394 | +0.00036 |
| `income_te` — nested cross-fit income target encoding | 0.94472 | — | — |
| **`income_te_tuned`** — the above + regularised parameters | **0.94528** | **0.94560** | **+0.00032** |

The two Offsets agree to **0.00004** across configurations that differ by a whole target encoding and a whole parameter surface. That agreement — not the leaderboard score — is the result that matters: it is the evidence that CV can rank candidates which the 57,314-row public split cannot resolve.

**The public leaderboard is read as a thermometer, never as a ranking.** Its noise floor is 0.0002; promotion is by CV only.

## Architecture

Six modules, one seam each. Two rules make the seams load-bearing: **`runner` is the only writer of the runs ledger**, and **`adapter` is the only module that sees `y`** outside a model's own `fit`. The two places a leak can enter are one file each.

```
 train.csv ──┐
             │  data.load_train()          ← reads in FILE ORDER, asserts zero missing
             │  data.fold_ids_for(y, seed) ← split immediately after the read
             │        │
             │        └──► Canonical Fold Partition ──[assert sha256]──► frozen
  test.csv ──┤
             │
             ▼
      frame.build_frame(train, test)              ← the Baseline Frame
             │   one-hot nominals · numeric ordinals · Age raw (45 values)
             │   income raw + digits · count encoding over train+test
             │   [assert] 45 Age values · zero NaN · one-hot vocab matches
             ▼
   ┌────────────────── per outer fold k ──────────────────┐
   │                                                      │
   │   adapter.fit_transform(X[train], y[train])          │ ← the ONLY code that sees y
   │      · nested cross-fit target encoding              │   [assert] refuses validation rows
   │      · scaling (off for trees, ADR-0001)             │
   │   adapter.transform(X[validation])                   │
   │                      │                               │
   │                      ▼                               │
   │   models.fit(...) ──► fold model ──► predictions     │ ← fixed rounds, no early stopping
   │                                                      │
   └──────────────────────────────────────────────────────┘
             │
             ▼
     runner.run(config)  ← OOF AUC · Paired Delta vs Incumbent · kill criterion
             │              prints the verdict and STOPS
             ├──► ledger/runs.jsonl        (committed, append-only)
             └──► runs/<run_id>/oof.npy    (gitignored, ~5 MB)
                         │
      promote <run_id> ──┤  ledger/promotions.jsonl   ← the one write that is a judgement
                         │
      submit <run_id> ───┴──► mean of the 5 fold models ──► Kaggle
                              ledger/submissions.jsonl
```

| module | owns | interface |
| --- | --- | --- |
| `data` | reading the CSVs in file order; the Canonical Fold Partition and its sha256 assert | `load_train()`, `fold_ids(seed)` |
| `frame` | the Baseline Frame — one-hot, ordinals, digit decomposition, count encoding | `build_frame(train, test)` |
| `adapter` | the Model Adapter — scaling, nested cross-fit TE, everything fitted inside the training fold | `fit_transform(X_tr, y_tr)` / `transform(X_va)` |
| `models` | model families behind one signature (currently LightGBM only) | `fit(X, y, params)` |
| `runner` | executing a Comparison Run: 5 folds, OOF, Paired Delta, the ledger write | `run(config) -> RunRecord` |
| `submission` | the Submission Fit (mean of the 5 fold models), the CSV, the Kaggle call | `submit(run_id, message)` |
| `experiments` | one frozen dataclass per Experiment, each docstring stating its hypothesis | `resolve(name)` |

`src/` is a **flat import root** — no nested package — installed editable, so modules import by bare name.

## The instrument

Every design choice below exists to make a 0.0003 difference readable.

**One partition, forever.** `StratifiedKFold(n_splits=5, shuffle=True, random_state=0)`, built *immediately* after `read_csv` in file order — `shuffle=True` assigns by input row order, so no transform may touch the rows first. A **sha256 of the fold-id vector** is committed and asserted on every run, catching reordered rows, rows dropped before the split, and a scikit-learn upgrade changing the algorithm.

**Everything is a Paired Delta.** No result is ever reported as a standalone AUC. A candidate changes exactly one field against the **Incumbent** and is scored on the same rows, so the comparison is paired and its error bar is roughly an order of magnitude tighter than an absolute OOF AUC's ~0.0006.

**The verdict rule is staggered by magnitude**, and the 4-of-5 gate applies first regardless:

| mean paired delta | verdict |
| --- | --- |
| ≥ 0.0003 | accept on the canonical seed |
| 0.0001 – 0.0003 | accept only if a Confirmation Run holds the sign on seeds 0/1/2 |
| < 0.0001 | reject as noise, whatever the sign |
| positive in < 4/5 folds | reject regardless of magnitude |

**Kill criteria are declared before the run.** Each Experiment's docstring names the threshold that kills it. Reopening a criterion after seeing the number is the failure this discipline exists to prevent.

**Promotion is manual.** `runner` prints the verdict and stops. Advancing the Incumbent is the only ledger write that is a judgement rather than an observed fact, and it is made under deadline pressure — exactly when an automatic rule turns a noise-sized delta into the reference every later measurement is taken against.

**Determinism is part of the instrument.** `num_threads=10` (performance cores only), `deterministic=true`, `force_row_wise=true`, all four LightGBM seeds explicit. Four consecutive runs of the Incumbent returned OOF `0.943577` with identical fold AUCs.

## Correctness is asserted at runtime, not unit-tested

The failure mode here is not "the function returns the wrong value" — it is "the number comes out plausible and biased". A leaked target encoding does not throw; it makes the score go **up**. So there is no unit-test suite. There are asserts on the execution path that cannot be skipped:

1. **sha256 of the fold-id vector** — the partition is the one every banked measurement used.
2. **45 distinct addressable `Age` values** — the column carries a 9-sigma *non-monotone* residual (13.02% positive at age 68 vs 21.77% at age 27) that is invisible to single-feature AUC. A tree gets it for free; a "clean up the numerics" refactor destroys it silently.
3. **Zero `NaN` in the Baseline Frame** — catches a silently failed lookup or join.
4. **The Adapter refuses a `fit` that receives validation rows** — the fold-boundary violation, caught where it is attempted.

Plus the one `pytest` that cannot be expressed as a runtime assert: `test_target_encoding_never_sees_its_own_row`, running the Adapter over a tiny synthetic frame. That property is **invisible from the top** — when it breaks, the end-to-end health check still passes, and passes *better*.

> **Where the repo diverges from its own ADR.** ADR-0003 specified *exactly one* pytest. What exists is 127 across 9 files. The reason is the same one behind [the five bugs](#five-bugs-and-why-the-tests-missed-all-of-them): the agents that wrote the pipeline had neither the data nor a network, so pure structural tests were the only verification available to them. They are cheap and they do catch refactors — but not one of them caught a single one of the five bugs, which is the evidence the ADR was arguing from in the first place.

This was not theory. See [Five bugs](#five-bugs-and-why-the-tests-missed-all-of-them).

## Quick start

Requires [`uv`](https://docs.astral.sh/uv/) and the competition CSVs in `data/` (gitignored).

```bash
uv venv --python 3.12          # 3.12 exactly; pandas is held below 3.0
uv sync --extra dev
uv run pytest -q               # 127 tests

kaggle competitions download -c playground-series-s6e9 -p data && unzip -o 'data/*.zip' -d data
./scripts/setup-kaggle.sh      # interactive OAuth wizard, if not already set up
```

```bash
uv run run-experiment --queue              # the declared queue, in run order
uv run run-experiment baseline             # the turn-1 Incumbent  (~80s)
uv run run-experiment income_te            # a candidate, as a Paired Delta
uv run run-experiment income_te --confirm  # Confirmation Run across seeds 0/1/2

uv run render-ledger                       # every run, sorted by Paired Delta
uv run promote <run_id>                    # advance the Incumbent, recording the rule
uv run submit <run_id> --confirmed -m "…"  # mean of the 5 fold models → Kaggle
uv run record-score <run_id> <score>       # fill in the public score later
```

A Comparison Run costs **~80 seconds** on an M5 Pro (10 threads). A Confirmation Run is six of them.

## The Experiment Ledger

The memory of the cycle, committed with the repo so a Run Record and the `src/` that produced it are one commit.

| file | holds |
| --- | --- |
| `ledger/runs.jsonl` | one **Run Record** per Comparison Run |
| `ledger/submissions.jsonl` | one line per submission, written *before* the score is known |
| `ledger/promotions.jsonl` | every Incumbent advance, with the rule that authorised it |
| `runs/<run_id>/oof.npy` | the OOF prediction vector — **gitignored** |

JSONL because two runs appending never conflict in git. Two files because a submission carries a public score and no Paired Delta, and merging the schemas destroys the grep.

A Run Record carries the run id, config hash, timestamp, git sha, dirty-tree flag, the full config, the five fold AUCs, the OOF AUC, the Paired Delta and its per-fold breakdown, the folds-positive count, the fold-partition sha256, the seeds, wall time and a pointer to the OOF vector.

**The OOF vectors are the reason this works across time**: a Paired Delta stays computable between two configurations that never ran against each other. The ledger holds the numbers; the vectors let new questions be asked of old runs without retraining.

`run_id` is `<timestamp>-<experiment>`, **not** a config hash — a Confirmation Run deliberately runs one config three times, and hash-as-identity would collide exactly on the repetitions the protocol depends on.

## What was measured

Every candidate below is a single-field change, scored on the same partition, with its kill criterion declared beforehand.

| Paired Delta | Experiment | OOF | Folds + | Outcome |
| --- | --- | --- | --- | --- |
| +0.00114 | `income_te` | 0.94472 | 5/5 | **promoted** |
| +0.00063 | `conservative_tuning` | 0.94421 | 5/5 | accepted (standalone) |
| +0.00056 | `income_te_tuned` | 0.94528 | 5/5 | **promoted**, confirmed on 3 seeds |
| +0.00032 | `seed_bag` | 0.94390 | 5/5 | accepted (standalone) |
| +0.00012 | `max_bin_1023` | 0.94369 | 3/5 | dead — 4/5 rule |
| +0.00011 | `income_te_tuned_bag` | 0.94539 | — | dead — below cost break-even |
| +0.00008 | `max_bin_2047` | 0.94366 | 3/5 | dead — 4/5 rule |
| +0.00001 | `age_te` | 0.94359 | 3/5 | dead — below threshold |
| −0.00002 | `max_bin_255` | 0.94356 | 2/5 | dead — 4/5 rule |
| −0.00172 | `smotenc` | 0.94186 | 0/5 | dead — below threshold |

Three findings worth carrying forward:

- **The nested cross-fit holds.** The income target encoding landed at +0.00114. The same feature measured **−0.00187** in earlier research when the encoding was fitted on the rows it was applied to — a 0.003 swing from fold discipline alone, and the most expensive mistake available in this project.
- **The Resolution axis is saturated, which contradicts the project's own premise.** All three `max_bin` values fail the 4/5 rule and `age_te` lands at +0.00001. Digit decomposition had already bought the resolution; raising the bin count buys nothing on top.
- **Tuning and the encoding are near-orthogonal.** Tuning paid +0.00063 alone and +0.00056 stacked — an overlap of ~0.00007. The seed bag was the opposite: +0.00032 alone, +0.00011 on the stack, because the variance it reduced had already been absorbed.

## Decisions

Architecture Decision Records live in [`docs/adr/`](docs/adr/). Each exists because the decision is hard to reverse, surprising without context, and the result of a real trade-off — and because it would otherwise look like an omission and get "fixed".

| ADR | Decision |
| --- | --- |
| [0001](docs/adr/0001-transforms-belong-to-the-model-adapter.md) | Transforms belong to the Model Adapter, fitted inside the training fold |
| [0002](docs/adr/0002-the-turn-1-incumbent-is-a-reproduction.md) | The turn-1 Incumbent is a reproduction, not a fresh start |
| [0003](docs/adr/0003-the-ledger-is-the-memory-and-correctness-is-asserted-at-runtime.md) | The Experiment Ledger is the cycle's memory; correctness is asserted at runtime |
| [0004](docs/adr/0004-the-rank-1-gap-is-not-chased.md) | The Rank-1 Gap is not chased |

The project's vocabulary — **Canonical Fold Partition**, **Paired Delta**, **Incumbent**, **Baseline Frame**, **Resolution**, **Health Gate**, **Floor**, **Winner's Curse** — is defined in [`CONTEXT.md`](CONTEXT.md). Terms in capitals throughout the docs point there.

### Why the leaderboard's top score is not a target

ADR-0004 rules the 0.0027 above the plateau out of scope, and the arithmetic is the reason. At AUC 0.9466 with a 17.5% positive rate, the Hanley–McNeil SE is 0.00158 on the 57,314-row public split and 0.00079 on the private remainder, so a single model's public and private scores differ with SD **0.00177**. Taking the maximum over 2,704 teams with an independent noise fraction of ρ = 0.90–0.95 gives an expected **winner's curse of +0.0012 to +0.0017** — a majority of the gap.

Two corrections that fell out of doing the arithmetic: the top team's own best-of-39 selection explains only ~0.0002 (near-identical models have highly correlated public noise), and the curse explains *most* of the gap but not all of it — roughly 0.001 is unaccounted for, and no evidence available to this project can say whether it is a tail draw or something real.

The decision does not depend on resolving that. It depends on there being **no mechanism to act on**, which is cheaper to establish and a stronger reason to stop.

## How the work was organised

The repo is driven by agent skills, documented in [`docs/agents/`](docs/agents/) and indexed in [`CLAUDE.md`](CLAUDE.md).

```
/wayfinder  ── a map issue + decision tickets, worked one per session
     │           11 tickets: data profile, competition rules, generator recipe,
     │           leak hunt, validation, representation, model, code layout,
     │           submission mechanics, gap strategy
     ▼
/to-spec    ── one spec issue: 74 user stories, implementation and testing decisions
     ▼
/to-tickets ── 9 vertical slices with native GitHub blocking edges
     ▼
 sandcastle ── parallel Docker agents, one per ticket, merged into `dev`
     ▼
  this repo ── executed, debugged, measured, submitted
```

The wayfinder map (issue #1) produced **decisions**, not code: what to validate against, what the Frame contains, which model family, where the seams go, and how much of the budget the leaderboard gap deserves (none). Every one of those decisions is traceable from the ADRs and the glossary back to a closed ticket with its evidence.

### Five bugs, and why the tests missed all of them

The sandcastle containers had **no network and no `data/`** — the CSVs are gitignored. The agents built and structurally tested the whole pipeline without ever executing it. 127 tests passed. Then it met the data:

| # | Bug | Caught by |
| --- | --- | --- |
| 1 | Both ordinals encoded with the same Low/Medium/High map; `Environmental_Concern_Level` is float 1.0–5.0 | Frame assert |
| 2 | `Will_Buy_EV` handed to LightGBM as the strings `"Yes"`/`"No"` | LightGBM |
| 3 | The dirty-tree flag saw the previous run's own uncommitted ledger line, marking every run after the first as dirty | Running twice in a row |
| 4 | The Submission Fit built the Adapter without the target encoding, the oversampler or the validation index — a promoted candidate would have submitted predictions from a different model than the one scored | Adapter assert |
| 5 | The Confirmation Run re-ran the candidate alone and reported the spread of its own OOF — a gate that could not fail, sitting in front of the only irreversible action in the project | Reading the output |

All five are the same thing: **code written against assumptions about data it had never seen.** Bug 1's false premise was even written down as a comment — *"the two ordinals share this ordering"* — and then implemented.

Four broke loudly. The fifth **passed**, which is why the runtime-assert bet in ADR-0003 is the design choice this repo would defend hardest.

## Known gaps

Carried forward rather than patched under deadline, because changing the instrument mid-measurement is its own failure mode:

- **Promotion does not redirect comparisons.** Every Experiment hardcodes `incumbent="baseline"`; `promotions.jsonl` is written and nothing reads it. Stacking therefore requires declaring a new Experiment naming its own Incumbent — which is what `income_te_tuned` does.
- **The Health Gate cries wolf on distribution-changing candidates.** It is a reproduction check for the Incumbent, but it runs on every experiment, so `smotenc` was reported as "a bug in how the partition or the Frame was assembled" when a lower OOF was the expected result of the experiment.
- **The kill criterion and the verdict rule can disagree** without the runner resolving it. On `income_te_tuned_bag`, the cost-based kill said dead while the magnitude-based verdict asked for a Confirmation Run. The kill criterion governs — it was declared first — but the code should say so.

## Layout

```
├── CONTEXT.md              glossary — the project's vocabulary, nothing else
├── CLAUDE.md               agent skill index
├── docs/adr/               architecture decision records
├── docs/agents/            issue tracker, triage labels, domain doc conventions
├── ledger/                 the Experiment Ledger (committed)
├── notebooks/              EDA, tracked WITH outputs as evidence
├── scripts/                the Kaggle setup wizard, stdlib test harnesses
├── src/                    flat import root — the six modules above
├── tests/                  127 structural tests + the one that matters (see above)
├── data/                   competition CSVs (gitignored)
└── runs/                   OOF vectors (gitignored)
```

**Nothing produced in a notebook may enter the ledger.** Notebooks read data and call into `src/`; they never redefine a transform. That rule is what makes it safe to commit them with their outputs — a stale output documents what was seen on a given day and can never contaminate a measurement.
