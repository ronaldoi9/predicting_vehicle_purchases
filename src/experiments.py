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
    **{exp.name: exp for exp in MAX_BIN_EXPERIMENTS},
}


def resolve(name: str) -> Experiment:
    """Look an Experiment up by name for the single entry point."""
    try:
        return _REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(f"unknown experiment {name!r}; known: {known}") from None
