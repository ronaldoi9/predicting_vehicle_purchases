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


_REGISTRY: dict[str, Experiment] = {
    TRACER_RAW13.name: TRACER_RAW13,
    BASELINE.name: BASELINE,
    INCOME_TE.name: INCOME_TE,
    AGE_TE.name: AGE_TE,
    **{exp.name: exp for exp in MAX_BIN_EXPERIMENTS},
}


def resolve(name: str) -> Experiment:
    """Look an Experiment up by name for the single entry point."""
    try:
        return _REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(f"unknown experiment {name!r}; known: {known}") from None
