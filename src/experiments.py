"""Experiments as declarations: a named frozen config that states one hypothesis.

An Experiment is a declaration, not a script — it cannot forget to record itself,
because the single entry point (``run-experiment``) resolves it by name and hands
it to the runner, where the ledger write, the fold assert and the git capture all
happen in one place. A candidate is expressed as a single-field change against
the Incumbent, so the one-change discipline is visible in the declaration.

The tracer bullet (#13) ships exactly one Experiment: ``tracer_raw13``, the
narrow path through every layer on the raw 13 columns. Later tickets widen the
Frame (digit decomposition, count encoding, nested cross-fit target encoding)
and add queue candidates as single-field replacements of this one.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

# Turn 1's LightGBM parameters, taken verbatim from the leak hunt's ablation so
# the first number the pipeline produces is a reproduction that can be checked.
# Determinism is part of the instrument: performance cores only, deterministic
# histogram construction, all four seeds explicit. bagging_fraction is inert at
# bagging_freq=0 and stays inert — it is the first divergence hypothesis if the
# reproduction misses.
LGBM_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 127,
    "max_bin": 511,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 0,
    "num_threads": 10,
    "deterministic": True,
    "force_row_wise": True,
    "seed": 0,
    "bagging_seed": 0,
    "feature_fraction_seed": 0,
    "data_random_seed": 0,
    "verbosity": -1,
}

# The four seeds the instrument pins, surfaced into the Run Record.
SEED_KEYS = ("seed", "bagging_seed", "feature_fraction_seed", "data_random_seed")


def seeded_params(params: Mapping[str, Any], seed: int) -> dict[str, Any]:
    """A copy of ``params`` with all four instrument seeds set to ``seed``.

    The seed-averaged bag fits the same config once per seed and averages the
    fold predictions; each member differs only in the four seeds pinned in
    :data:`SEED_KEYS`. The source mapping is never mutated, so the Incumbent
    keeps its seed 0.
    """
    out = dict(params)
    for k in SEED_KEYS:
        if k in out:
            out[k] = int(seed)
    return out


@dataclass(frozen=True)
class Experiment:
    """A frozen, named configuration stating one hypothesis."""

    name: str
    hypothesis: str
    frame: str
    model: str
    params: Mapping[str, Any]
    num_boost_round: int
    fold_seed: int = 0
    scale: bool = False
    # Columns the Model Adapter nested-cross-fit target-encodes inside each
    # training fold. Empty for the Incumbent; a candidate adds exactly this one
    # field, so the one-change discipline is visible in the declaration itself.
    target_encode: tuple[str, ...] = ()
    # The Incumbent this candidate is a Paired Delta against, and the kill
    # criterion declared *before* it runs (a mean paired delta below this on the
    # canonical seed kills the candidate). Both ``None`` for the Incumbent runs.
    incumbent: str | None = None
    kill_delta: float | None = None
    # The Resolution sweep's kill criterion is folds-positive, not a mean-delta
    # threshold: a max_bin value survives only if it beats the Incumbent in at
    # least this many of five folds. ``None`` for a candidate judged on
    # ``kill_delta`` instead (the two are mutually exclusive per declaration).
    kill_min_folds_positive: int | None = None
    # The seed-averaged bag (#19): the seeds whose fold predictions are averaged.
    # Empty for the Incumbent; a candidate adds exactly this one field. Its kill
    # criterion is cost-versus-gain: each extra fit must earn ``kill_value_per_run``
    # AUC, so the break-even gain is ``kill_value_per_run * (len(seed_bag) - 1)``.
    seed_bag: tuple[int, ...] = ()
    kill_value_per_run: float | None = None
    # SMOTENC (#19): the categorical-aware oversampler, fitted inside the training
    # fold only, with income digits regenerated from the synthetic income. ``None``
    # for the Incumbent; the candidate sets it to ``adapter.OVERSAMPLE_SMOTENC``.
    oversample: str | None = None
    # Conservative tuning (#19): the 2h machine-time box its kill criterion is
    # scored against — 2h without +0.0003 -> dead. ``None`` for a candidate not
    # time-boxed (the tuning candidate combines it with ``kill_delta``).
    kill_time_budget_s: float | None = None
    health_gate: float = 0.9434
    target_oof: float = 0.94167

    def seeds(self) -> dict[str, int]:
        return {k: self.params[k] for k in SEED_KEYS if k in self.params}

    def as_config(self) -> dict[str, Any]:
        """The JSON-serialisable config that lands in the ledger.

        Carries the modelling fields (a change to any of them is a change to the
        config hash); the Incumbent pointer and kill criterion are recorded on
        the Run Record, not here, so they do not perturb "have I tried this?".
        """
        return {
            "name": self.name,
            "frame": self.frame,
            "model": self.model,
            "num_boost_round": self.num_boost_round,
            "fold_seed": self.fold_seed,
            "scale": self.scale,
            "target_encode": list(self.target_encode),
            "seed_bag": list(self.seed_bag),
            "oversample": self.oversample,
            "params": dict(self.params),
        }


TRACER_RAW13 = Experiment(
    name="tracer_raw13",
    hypothesis=(
        "One Comparison Run end to end on the raw 13 columns reproduces the "
        "leak hunt's raw-13 ablation OOF of 0.94167, proving the Canonical Fold "
        "Partition and the read order are right before any feature engineering "
        "can obscure the question."
    ),
    frame="raw13",
    model="lightgbm",
    params=LGBM_PARAMS,
    num_boost_round=700,
    fold_seed=0,
    scale=False,
    health_gate=0.9434,
    target_oof=0.94167,
)


BASELINE = Experiment(
    name="baseline",
    hypothesis=(
        "The complete Baseline Frame — the raw-13 columns plus income digit "
        "decomposition (from the original integer) and train+test count encoding "
        "of income and commute — reproduces the ablation's OOF 0.94372 and clears "
        "the Health Gate at 0.9434, proving the partition and the Frame were "
        "assembled correctly. Digit decomposition takes the raw-13 0.94167 to "
        "0.94342 (+0.00175); count encoding adds +0.00030. This is the turn-1 "
        "Incumbent every band (ii) candidate is a Paired Delta against."
    ),
    frame="baseline",
    model="lightgbm",
    params=LGBM_PARAMS,
    num_boost_round=700,
    fold_seed=0,
    scale=False,
    health_gate=0.9434,
    target_oof=0.94372,
)


# The band (ii) queue's first and highest-value candidate: nested cross-fit
# target encoding of the exact income value. Expressed as a single-field change
# against the Incumbent (``target_encode``) via ``replace``, so the one-change
# discipline is literal — every modelling field but that one is the Incumbent's.
INCOME_TE = replace(
    BASELINE,
    name="income_te",
    hypothesis=(
        "Adding the Model Adapter's nested cross-fit target encoding of the "
        "exact income value to the Baseline Frame beats the Incumbent. The leak "
        "hunt published this feature at +0.00129 standalone and measured it at "
        "-0.00187 when mis-fitted on its own rows — the most expensive mistake "
        "in the project. On top of the Frame the delta is expected to be smaller "
        "than +0.00129, because the target encoding and income digit "
        "decomposition both buy Resolution and overlap partially. Single-field "
        "change against the Incumbent: target_encode = (Annual_Income_USD,). "
        "Kill criterion, declared before running: paired delta < +0.0005 on one "
        "seed -> the candidate is dead."
    ),
    target_encode=("Annual_Income_USD",),
    incumbent="baseline",
    kill_delta=0.0005,
    # The gate/target the Incumbent already cleared; the candidate is judged on
    # its Paired Delta and kill criterion, not on clearing a new gate.
    target_oof=0.94372,
)


# --------------------------------------------------------------------------- #
# The Resolution axis (#18): the only axis this dataset's representation has been
# shown to pay on. Two candidates, each a single-field change against the
# Incumbent, each with a kill criterion declared before it runs.
# --------------------------------------------------------------------------- #

# The max_bin sweep. max_bin is the Resolution control, not an ordinary tuning
# knob: it decides whether a histogram-binned tree can address individual income
# values or merges them into a range. This is why the published digit
# decomposition works at all — the leak hunt showed the apparent digit signal is
# a resolution effect (income % 100 scores 0.50001 on non-floor rows). Swept
# around the Incumbent's 511; every value stays >= models.MIN_MAX_BIN (64), so
# the 45 Age values remain individually addressable at each resolution tested.
MAX_BIN_SWEEP: tuple[int, ...] = (255, 1023, 2047)

# The sweep's kill criterion, declared before running: a value survives only if
# it beats the Incumbent in at least this many of five folds. No value clearing
# it freezes the Resolution axis at the Incumbent's 511.
MAX_BIN_KILL_MIN_FOLDS_POSITIVE = 4


def _max_bin_candidate(max_bin: int) -> Experiment:
    """One max_bin value as a Paired-Delta Experiment against the Incumbent.

    Single-field change: the Incumbent's params with ``max_bin`` replaced, so the
    one-change discipline is literal — every other param, and every other
    modelling field, is the Incumbent's. Fixed rounds and early stopping off are
    inherited from the Incumbent and the Comparison-Run protocol.
    """
    return replace(
        BASELINE,
        name=f"max_bin_{max_bin}",
        hypothesis=(
            f"Setting max_bin={max_bin} lets the histogram-binned tree address "
            "income at a different Resolution than the Incumbent's 511. max_bin "
            "is the Resolution control, not a tuning knob: it decides whether the "
            "tree sees individual income values or a merged range. Single-field "
            f"change against the Incumbent: params['max_bin'] = {max_bin}. Kill "
            "criterion, declared before running: it must beat the Incumbent in at "
            "least 4 of 5 folds, else the Resolution axis freezes at 511."
        ),
        params={**LGBM_PARAMS, "max_bin": max_bin},
        incumbent="baseline",
        kill_min_folds_positive=MAX_BIN_KILL_MIN_FOLDS_POSITIVE,
        target_oof=0.94372,
    )


MAX_BIN_EXPERIMENTS: tuple[Experiment, ...] = tuple(
    _max_bin_candidate(mb) for mb in MAX_BIN_SWEEP
)


# Age as a 45-level target-encoded lookup — the honest alternative to relying on
# max_bin to deliver the lookup. Age carries a 9-sigma non-monotone residual
# invisible to single-feature AUC. The TE runs under the exact nested cross-fit
# contract the income TE uses (Adapter, inner K=5, seed 100+fold, prior weight
# 20), and adds an `Age_te` column rather than replacing raw Age, so all 45
# values stay individually addressable — the Age invariant respected either way.
AGE_TE = replace(
    BASELINE,
    name="age_te",
    hypothesis=(
        "Target-encoding Age as a 45-level lookup under the Adapter's nested "
        "cross-fit contract beats the Incumbent. Age carries a 9-sigma "
        "non-monotone residual invisible to single-feature AUC; the TE is the "
        "honest alternative to relying on max_bin for the lookup. The TE adds an "
        "Age_te column, never replacing raw Age, so all 45 values stay "
        "individually addressable (the Age invariant). Single-field change "
        "against the Incumbent: target_encode = (Age,). Kill criterion, declared "
        "before running: paired delta < +0.0001 -> the candidate is dead."
    ),
    target_encode=("Age",),
    incumbent="baseline",
    kill_delta=0.0001,
    target_oof=0.94372,
)


# --------------------------------------------------------------------------- #
# The tail of the queue (#19): three candidates nothing measured so far suggests
# will pay, each a single-field change against the Incumbent and each boxed by a
# kill criterion declared before it runs so that none can absorb the week. A
# multi-family blend is NOT built here — it enters only if these three exhaust
# before 29/09, and competes for time rather than being assumed (a published
# finding for this competition is that a single tuned LightGBM beat a 7-model
# stack).
# --------------------------------------------------------------------------- #

# Seed-averaged bag over three seeds. Variance reduction, expected +0.0002. The
# member configs differ only in the four instrument seeds (see seeded_params);
# the bag averages their fold predictions before scoring.
SEED_BAG: tuple[int, ...] = (0, 1, 2)

# The bag's cost-versus-gain kill: each *extra* full-model fit must earn at least
# this much AUC. Three seeds = two extra fits, so break-even is the expected
# +0.0002. Below it, the compute cost exceeds the measured gain and the bag dies.
SEED_BAG_VALUE_PER_RUN = 0.0001

SEED_BAG_EXPERIMENT = replace(
    BASELINE,
    name="seed_bag",
    hypothesis=(
        "Averaging the fold predictions of three seeds (0, 1, 2) reduces variance "
        "for an expected +0.0002, bought only if it is cheaper than it is worth. "
        "The three members differ only in the four instrument seeds; every other "
        "field is the Incumbent's. Single-field change against the Incumbent: "
        "seed_bag = (0, 1, 2). Kill criterion, declared before running: cost "
        "versus gain — each of the two extra fits must earn +0.0001, so a paired "
        "delta below the break-even +0.0002 means compute cost exceeds measured "
        "gain and the bag is dead."
    ),
    seed_bag=SEED_BAG,
    incumbent="baseline",
    kill_value_per_run=SEED_BAG_VALUE_PER_RUN,
    target_oof=0.94372,
)


# SMOTENC — categorical-aware oversampling, fitted inside the training fold only,
# with income digits regenerated from the synthetic income so the Frame stays
# self-consistent. sampling_strategy=0.5. Plain SMOTE is excluded outright (see
# adapter): interpolating Age to 43.7 and producing digits that no longer derive
# from their own income destroys Resolution, the one axis that pays. No class
# weighting is used anywhere else — at 17.5% positive there are ~117,000
# positives and ROC AUC is rank-based. The contract constants live in adapter.
from adapter import OVERSAMPLE_SMOTENC, SMOTENC_SAMPLING_STRATEGY  # noqa: E402

SMOTENC_KILL_DELTA = 0.0003

SMOTENC = replace(
    BASELINE,
    name="smotenc",
    hypothesis=(
        "Oversampling the minority class with SMOTENC at sampling_strategy=0.5, "
        "fitted inside the training fold only and with income digits regenerated "
        "from the synthetic income so the Frame stays self-consistent, beats the "
        "Incumbent. Plain SMOTE is excluded outright: interpolating Age to 43.7 "
        "and producing digits that no longer derive from their own income "
        "destroys Resolution, the one axis that pays. Oversampling before the "
        "split is the failure mode that would make this test appear to succeed, "
        "so it is fitted strictly inside the fold. Single-field change against "
        "the Incumbent: oversample = smotenc. Kill criterion, declared before "
        "running: paired delta < +0.0003 -> the candidate is dead."
    ),
    oversample=OVERSAMPLE_SMOTENC,
    incumbent="baseline",
    kill_delta=SMOTENC_KILL_DELTA,
    target_oof=0.94372,
)


# Conservative tuning of the depth/leaves/regularisation surface. max_bin is
# explicitly NOT part of this — it was reclassified as the Resolution control
# and has its own allocated experiment — so it is held at the Incumbent's 511.
CONSERVATIVE_PARAMS: dict[str, Any] = {
    **LGBM_PARAMS,
    "num_leaves": 63,          # fewer leaves than the Incumbent's 127
    "max_depth": 8,            # an explicit depth cap (the Incumbent leaves it -1)
    "min_child_samples": 200,  # larger leaves — regularisation
    "min_split_gain": 0.01,    # a minimum gain to split — regularisation
    "lambda_l1": 1.0,          # L1 regularisation
    "lambda_l2": 1.0,          # L2 regularisation
    # max_bin is untouched: it stays at the Incumbent's 511 above.
}

CONSERVATIVE_KILL_DELTA = 0.0003
CONSERVATIVE_TIME_BUDGET_S = 7200.0  # the 2h machine-time box

CONSERVATIVE_TUNING = replace(
    BASELINE,
    name="conservative_tuning",
    hypothesis=(
        "Regularising the depth/leaves/regularisation surface — fewer leaves, an "
        "explicit depth cap, larger leaves and L1/L2 penalties — beats the "
        "Incumbent. max_bin is explicitly NOT part of this: it was reclassified "
        "as the Resolution control and has its own allocated experiment, so it "
        "stays at the Incumbent's 511. Single-field change against the Incumbent: "
        "params replaced on the depth/leaves/regularisation surface only. Kill "
        "criterion, declared before running: 2h of machine time without +0.0003 "
        "-> dead."
    ),
    params=CONSERVATIVE_PARAMS,
    incumbent="baseline",
    kill_delta=CONSERVATIVE_KILL_DELTA,
    kill_time_budget_s=CONSERVATIVE_TIME_BUDGET_S,
    target_oof=0.94372,
)


# --------------------------------------------------------------------------- #
# Stacking the band (ii) survivors. Three candidates beat the turn-1 Incumbent
# on the canonical seed — income_te +0.00114, conservative_tuning +0.00063,
# seed_bag +0.00032 — but each was measured against the SAME reference, so the
# deltas are not additive and adding them up would be exactly the arithmetic a
# Paired Delta exists to forbid. Stacking is a fresh measurement, declared here
# as a chain: each candidate changes one field against the candidate below it,
# and names that candidate as its Incumbent.
#
# These are follow-ups to the queue, not members of it: BAND_II_QUEUE keeps its
# declared order untouched, because rewriting it after the fact would make the
# runs already in the ledger unreadable against their own declaration.
# --------------------------------------------------------------------------- #

INCOME_TE_TUNED = replace(
    INCOME_TE,
    name="income_te_tuned",
    hypothesis=(
        "The regularised depth/leaves/regularisation surface still pays once the "
        "nested cross-fit income target encoding is in the Frame. Both beat the "
        "turn-1 Incumbent alone (+0.00114 and +0.00063), and the open question is "
        "whether they buy the same thing: a tuned tree may already be extracting "
        "what the encoding supplies, in which case the stack lands well under the "
        "sum. Single-field change against the income_te Incumbent: params "
        "replaced on the depth/leaves/regularisation surface only, max_bin left "
        "at 511 where the Resolution sweep froze it. Kill criterion, declared "
        "before running: paired delta < +0.0003 -> the stack does not pay and "
        "income_te stands alone."
    ),
    params=CONSERVATIVE_PARAMS,
    incumbent="income_te",
    kill_delta=0.0003,
)


INCOME_TE_TUNED_BAG = replace(
    INCOME_TE_TUNED,
    name="income_te_tuned_bag",
    hypothesis=(
        "Seed-averaging still earns its compute on top of the tuned, "
        "target-encoded stack. The bag is the most likely of the three survivors "
        "to stack cleanly, because it is orthogonal to both — it changes only the "
        "four instrument seeds and averages three members — but 'most likely' is "
        "not measured. Single-field change against the income_te_tuned "
        "Incumbent: seed_bag = (0, 1, 2). Kill criterion, declared before "
        "running: the same break-even as the standalone bag — each of the two "
        "extra fits must earn +0.0001, so a paired delta below +0.0002 means "
        "compute cost exceeds measured gain and the bag is dead here too."
    ),
    seed_bag=SEED_BAG,
    incumbent="income_te_tuned",
    kill_delta=None,
    kill_value_per_run=SEED_BAG_VALUE_PER_RUN,
)


# --------------------------------------------------------------------------- #
# Representation axis (#29): frequency-encoding breadth and what is left of
# the TE axis, on evidence from #27's reading of the published stack's own
# source. Each candidate is a single-field change (the frame spec, or
# target_encode) against the standing Incumbent income_te_tuned. Grouped A-D
# to match the ticket's priority groups and kill criteria.
# --------------------------------------------------------------------------- #

# Group A: frequency-encoding breadth. We count 2 columns; the published stack
# counts ~79. Kill criterion for the group: dead only once A1, A2 and A3 have
# ALL been measured, per the Axis rule (#27 amended ADR-0005's premise, not
# this rule).
COUNT_COMPOSITES = replace(
    INCOME_TE_TUNED,
    name="count_composites",
    hypothesis=(
        "Count-encoding the five composite keys the published stack itself "
        "counts (income x subsidy, income x concern, income-per-car, charging "
        "total, commute-per-concern) beats the Incumbent. #27 found these are "
        "the ONLY composite keys anywhere in the published stack — as counts, "
        "not target encodings, contradicting this project's earlier composite-"
        "key-TE hypothesis. Single-field change against income_te_tuned: "
        "frame = 'count_composites'. Kill criterion, declared before running: "
        "paired delta < +0.0001 -> dead (part of Group A; the axis is not "
        "declared dead until A1-A3 have all run)."
    ),
    frame="count_composites",
    incumbent="income_te_tuned",
    kill_delta=0.0001,
    target_oof=0.94528,
)

COUNT_ALL13 = replace(
    INCOME_TE_TUNED,
    name="count_all13",
    hypothesis=(
        "Count-encoding all 13 raw columns, not just income and commute, "
        "beats the Incumbent. The published stack's frequency encoding covers "
        "~79 surfaces against our 2 (+0.00112 published vs +0.00030 measured "
        "here) -- this candidate is the direct width test. Single-field change "
        "against income_te_tuned: frame = 'count_all13'. Kill criterion, "
        "declared before running: paired delta < +0.0001 -> dead (Group A)."
    ),
    frame="count_all13",
    incumbent="income_te_tuned",
    kill_delta=0.0001,
    target_oof=0.94528,
)

COUNT_DIGITS = replace(
    INCOME_TE_TUNED,
    name="count_digits",
    hypothesis=(
        "Count-encoding the three income digit columns beats the Incumbent. "
        "The published stack's frequency encoding is applied to digit columns "
        "too (56 of its ~79 surfaces are digits); this measures whether that "
        "specific slice pays here. Single-field change against income_te_tuned: "
        "frame = 'count_digits'. Kill criterion, declared before running: "
        "paired delta < +0.0001 -> dead (Group A)."
    ),
    frame="count_digits",
    incumbent="income_te_tuned",
    kill_delta=0.0001,
    target_oof=0.94528,
)

COUNT_ALL13_COMPOSITES = replace(
    INCOME_TE_TUNED,
    name="count_all13_composites",
    hypothesis=(
        "count_all13 and count_composites stack. Run ONLY if both land "
        "individually -- see #29. Single-field change against income_te_tuned: "
        "frame = 'count_all13_composites'. Kill criterion, declared before "
        "running: paired delta < +0.0003 -> the stack does not pay and the "
        "better of the two stands alone."
    ),
    frame="count_all13_composites",
    incumbent="income_te_tuned",
    kill_delta=0.0003,
    target_oof=0.94528,
)

# Group B: what is left of the target-encoding axis after #27. Kill criterion
# for the group: dead after B1 and B3 -- #27 established our one-column TE
# already banks nearly the whole published TE step, so triple-smoothing
# variants (B2/B4) do not earn their runs unless one of these two lands.
COMMUTE_TE = replace(
    INCOME_TE_TUNED,
    name="commute_te",
    hypothesis=(
        "Target-encoding Daily_Commute_km, found non-monotone and worth ~6pp "
        "inside every score decile (#5) and never target-encoded here, beats "
        "the Incumbent. Single-field change against income_te_tuned: "
        "target_encode = (Annual_Income_USD, Daily_Commute_km). Kill "
        "criterion, declared before running: paired delta < +0.0001 -> dead "
        "(Group B)."
    ),
    target_encode=("Annual_Income_USD", "Daily_Commute_km"),
    incumbent="income_te_tuned",
    kill_delta=0.0001,
    target_oof=0.94528,
)

# The seven columns the published stack's own source target-encodes (#27):
# Age, income, commute, cars owned, both charging-station counts, concern.
TE_SEVEN_COLUMNS: tuple[str, ...] = (
    "Annual_Income_USD",
    "Age",
    "Daily_Commute_km",
    "Number_of_Cars_Owned",
    "Charging_Stations_Near_Home",
    "Charging_Stations_Near_Work",
    "Environmental_Concern_Level",
)

TE_SEVEN = replace(
    INCOME_TE_TUNED,
    name="te_seven",
    hypothesis=(
        "Target-encoding the same seven columns the published stack's own "
        "source encodes, at our prior weight of 20, beats the Incumbent. "
        "age_te alone already measured +0.00001 (dead) -- this tests whether "
        "the other five columns carry what age_te alone did not. Single-field "
        "change against income_te_tuned: target_encode = the seven columns. "
        "Kill criterion, declared before running: paired delta < +0.0001 -> "
        "dead (Group B)."
    ),
    target_encode=TE_SEVEN_COLUMNS,
    incumbent="income_te_tuned",
    kill_delta=0.0001,
    target_oof=0.94528,
)

# Group C: the two composite-key target encodings #27's own source contradicts
# (a published composite-key TE measured -0.0004) -- run to put the number on
# our own record rather than leave it as folklore. Both run regardless of
# result, per #29.
TE_INCOME_X_SUBSIDY = replace(
    INCOME_TE_TUNED,
    name="te_income_x_subsidy",
    hypothesis=(
        "Target-encoding the composite key income x subsidy beats the "
        "Incumbent. Declared prior: negative -- #27 found the published stack "
        "uses no composite-key TE anywhere (only as counts), and a published "
        "composite-key TE measured -0.0004. Single-field change against "
        "income_te_tuned: frame = 'composite_te_keys', target_encode = "
        "(Annual_Income_USD, income_x_subsidy). Kill criterion, declared "
        "before running: paired delta < +0.0001 -> dead. Runs regardless of "
        "result, to put the number on this project's own record."
    ),
    frame="composite_te_keys",
    target_encode=("Annual_Income_USD", "income_x_subsidy"),
    incumbent="income_te_tuned",
    kill_delta=0.0001,
    target_oof=0.94528,
)

TE_INCOME_X_CITY = replace(
    INCOME_TE_TUNED,
    name="te_income_x_city",
    hypothesis=(
        "Target-encoding the composite key income x city beats the Incumbent. "
        "Same declared-negative prior as te_income_x_subsidy. Single-field "
        "change against income_te_tuned: frame = 'composite_te_keys', "
        "target_encode = (Annual_Income_USD, income_x_city). Kill criterion, "
        "declared before running: paired delta < +0.0001 -> dead. Runs "
        "regardless of result."
    ),
    frame="composite_te_keys",
    target_encode=("Annual_Income_USD", "income_x_city"),
    incumbent="income_te_tuned",
    kill_delta=0.0001,
    target_oof=0.94528,
)

# Group D: digit decomposition of the six numeric columns income's own digits
# do not cover. Lowest expected value on the axis -- these are already
# low-cardinality and fully addressable raw, unlike income.
DIGITS7 = replace(
    INCOME_TE_TUNED,
    name="digits7",
    hypothesis=(
        "Units/tens digit decomposition of Age, commute, cars owned, both "
        "charging-station counts and concern beats the Incumbent. Expected "
        "close to a no-op: unlike income, these columns are already "
        "low-cardinality and fully addressable raw, so a tree already has "
        "individual-value resolution on them without decomposition. "
        "Single-field change against income_te_tuned: frame = 'digits7'. Kill "
        "criterion, declared before running: paired delta < +0.0001 -> dead."
    ),
    frame="digits7",
    incumbent="income_te_tuned",
    kill_delta=0.0001,
    target_oof=0.94528,
)

# The representation axis (#29), in the ticket's declared priority order.
REPRESENTATION_AXIS: tuple[Experiment, ...] = (
    COUNT_COMPOSITES,
    COUNT_ALL13,
    COUNT_DIGITS,
    COMMUTE_TE,
    TE_SEVEN,
    TE_INCOME_X_SUBSIDY,
    TE_INCOME_X_CITY,
    DIGITS7,
)


# --------------------------------------------------------------------------- #
# XGBoost as a second Arena family (#24, turn 2). Not a Paired-Delta candidate
# against income_te_tuned — a parallel Incumbent chain on the same Frame (the
# Baseline Frame plus the income target encoding, so both families are scored
# on identical rows). ``incumbent`` still points at income_te_tuned so each run
# gets a Paired Delta printed for readability, but the family's own kill
# criterion (#24) is judged on absolute OOF across all three configs together,
# not per-run: dead only if all three land below OOF 0.9445 (>0.0008 behind
# the standing Incumbent). That aggregate judgement is made once all three
# have run, not encoded as a per-run kill_delta.
#
# Pinned for determinism the same way LGBM_PARAMS is: tree_method=hist (the
# only method that takes max_bin, keeping the Age-resolution invariant
# expressible), nthread on the ten performance cores, one explicit seed.
XGB_BASE_PARAMS: dict[str, Any] = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "max_bin": 511,
    "nthread": 10,
    "seed": 0,
    "verbosity": 0,
}

# Three distinct configurations, not one port of LightGBM's parameters (#24's
# own kill criterion excludes that): idiomatic depth-wise XGBoost defaults,
# LightGBM-style leaf-wise growth (a genuine cross-library hypothesis, not a
# parameter port — XGBoost's lossguide implementation differs internally),
# and a deeper regularised depth-wise tree.
XGBOOST_BASELINE = replace(
    BASELINE,
    name="xgboost_baseline",
    hypothesis=(
        "XGBoost, fit with idiomatic depth-wise defaults (max_depth=6, "
        "eta=0.05, subsample/colsample_bytree=0.8) on the same Frame as the "
        "standing Incumbent (baseline + income target encoding), opens its own "
        "Arena chain. This is the family's first reproduction number, not a "
        "port of LightGBM's leaf-wise parameters. incumbent points at "
        "income_te_tuned so a Paired Delta prints, but survival is judged on "
        "absolute OOF across all three declared configs (#24): dead only if "
        "all three land below 0.9445."
    ),
    model="xgboost",
    target_encode=("Annual_Income_USD",),
    params={**XGB_BASE_PARAMS, "max_depth": 6, "eta": 0.05, "subsample": 0.8, "colsample_bytree": 0.8},
    incumbent="income_te_tuned",
    target_oof=0.94528,
)

XGBOOST_LOSSGUIDE = replace(
    XGBOOST_BASELINE,
    name="xgboost_lossguide",
    hypothesis=(
        "XGBoost with grow_policy=lossguide, max_leaves=127 and max_depth=0 "
        "(unlimited -- lossguide is meant to be leaf-bounded, not depth-bounded; "
        "xgboost's max_depth default of 6 would otherwise silently cap it to "
        "depth-wise's own shape) -- leaf-wise growth, LightGBM's default shape, "
        "under XGBoost's own histogram implementation -- beats xgboost_baseline's "
        "depth-wise growth on this Frame. A genuine hypothesis about growth "
        "policy, not a parameter port: the two libraries' lossguide/leaf-wise "
        "splitters differ internally. One of the family's three declared "
        "configs (#24)."
    ),
    params={
        **XGB_BASE_PARAMS,
        "grow_policy": "lossguide",
        "max_leaves": 127,
        "max_depth": 0,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
)

XGBOOST_TUNED = replace(
    XGBOOST_BASELINE,
    name="xgboost_tuned",
    hypothesis=(
        "A deeper, regularised depth-wise XGBoost tree (max_depth=10, "
        "min_child_weight=5, lambda=1.0) trades depth for regularisation "
        "instead of chasing LightGBM's leaf count. Third of the family's three "
        "declared configs (#24) — the aggregate kill criterion (all three "
        "below OOF 0.9445) is judged once this one has run too."
    ),
    params={
        **XGB_BASE_PARAMS,
        "max_depth": 10,
        "min_child_weight": 5,
        "lambda": 1.0,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
)

XGBOOST_ARENA: tuple[Experiment, ...] = (XGBOOST_BASELINE, XGBOOST_LOSSGUIDE, XGBOOST_TUNED)


# --------------------------------------------------------------------------- #
# The band (ii) experiment queue, in its declared order (#12, PRD #12).
# --------------------------------------------------------------------------- #
# Story 62: the queue is run in its declared order, so the candidate with a
# published gain is measured before the ones that merely sound promising. Each
# closed child ticket added its own candidates; the *order* lived only as prose
# in the spec table, so this is the single ordered source of truth turn 2 (and a
# queue runner) reads instead of reconstructing.
#
#   1. income_te                              — the largest published gain, first
#   2. the Resolution sweep (ascending max_bin)
#   3. age_te
#   4. seed_bag
#   5. conservative_tuning
#
# ``smotenc`` is deliberately absent: it is a *separately scheduled* candidate
# that "keeps its own slot" (see SEPARATELY_SCHEDULED). The multi-family blend
# (candidate 6) is gated behind the queue exhausting early and is not built at
# all, so it appears in neither structure and does not resolve.
BAND_II_QUEUE: tuple[Experiment, ...] = (
    INCOME_TE,
    *MAX_BIN_EXPERIMENTS,
    AGE_TE,
    SEED_BAG_EXPERIMENT,
    CONSERVATIVE_TUNING,
)

# Candidates that carry their own submission slot rather than the main queue's
# ordered track — SMOTENC keeps its own slot per the spec.
SEPARATELY_SCHEDULED: tuple[Experiment, ...] = (SMOTENC,)


def queue_names() -> tuple[str, ...]:
    """The band (ii) queue's Experiment names in declared run order."""
    return tuple(exp.name for exp in BAND_II_QUEUE)


_REGISTRY: dict[str, Experiment] = {
    TRACER_RAW13.name: TRACER_RAW13,
    BASELINE.name: BASELINE,
    INCOME_TE.name: INCOME_TE,
    AGE_TE.name: AGE_TE,
    SEED_BAG_EXPERIMENT.name: SEED_BAG_EXPERIMENT,
    SMOTENC.name: SMOTENC,
    CONSERVATIVE_TUNING.name: CONSERVATIVE_TUNING,
    INCOME_TE_TUNED.name: INCOME_TE_TUNED,
    INCOME_TE_TUNED_BAG.name: INCOME_TE_TUNED_BAG,
    **{exp.name: exp for exp in MAX_BIN_EXPERIMENTS},
    **{exp.name: exp for exp in REPRESENTATION_AXIS},
    COUNT_ALL13_COMPOSITES.name: COUNT_ALL13_COMPOSITES,
    **{exp.name: exp for exp in XGBOOST_ARENA},
}


def resolve(name: str) -> Experiment:
    """Look an Experiment up by name for the single entry point."""
    try:
        return _REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(f"unknown experiment {name!r}; known: {known}") from None
