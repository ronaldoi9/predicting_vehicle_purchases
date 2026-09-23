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

from dataclasses import dataclass
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
    health_gate: float = 0.9434
    target_oof: float = 0.94167

    def seeds(self) -> dict[str, int]:
        return {k: self.params[k] for k in SEED_KEYS if k in self.params}

    def as_config(self) -> dict[str, Any]:
        """The JSON-serialisable config that lands in the ledger."""
        return {
            "name": self.name,
            "frame": self.frame,
            "model": self.model,
            "num_boost_round": self.num_boost_round,
            "fold_seed": self.fold_seed,
            "scale": self.scale,
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


_REGISTRY: dict[str, Experiment] = {
    TRACER_RAW13.name: TRACER_RAW13,
    BASELINE.name: BASELINE,
}


def resolve(name: str) -> Experiment:
    """Look an Experiment up by name for the single entry point."""
    try:
        return _REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(f"unknown experiment {name!r}; known: {known}") from None
