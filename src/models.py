"""Model families behind one signature. Turn 1 is LightGBM only.

Comparison Runs use a **fixed round count with early stopping disabled** —
early stopping on the evaluated fold biases OOF upward and breaks the pairing.
Early stopping is permitted only in the Submission Fit, which lives elsewhere.

An assert binds ``max_bin`` from below against the ``Age`` value count, so a
parameter change cannot violate the representation invariant that all 45 ages
stay individually addressable. lightgbm is imported lazily so the module
imports by bare name anywhere.
"""

from __future__ import annotations

from typing import Any, Mapping

# max_bin must stay at least this large so the 45 distinct Age values remain
# individually addressable (45 < 64, with headroom).
MIN_MAX_BIN = 64

# The tree families: a GBDT chooses splits by gain and is invariant to any
# monotone transform, so scaling is a no-op and stays off (ADR-0001). Any family
# not listed here is scale-sensitive (linear, SVM, neural net) and the Adapter's
# scaling hook turns on for it. Turn 1 is LightGBM alone, so scaling never runs.
TREE_FAMILIES = frozenset({"lightgbm", "xgboost", "catboost"})


def needs_scaling(family: str) -> bool:
    """Whether the Adapter's scaling hook applies to this model family.

    ``False`` for a tree family (a no-op there), ``True`` for a scale-sensitive
    one — the family-conditional switch that keeps scaling out of turn 1 while
    leaving it wired for a linear family that arrives later.
    """
    return family not in TREE_FAMILIES


def fit(X, y, params: Mapping[str, Any], num_boost_round: int = 700):
    """Fit one model family on ``X``/``y`` under ``params`` for fixed rounds.

    Turn 1 is LightGBM. No validation set is passed and no early stopping is
    used, so the model cannot peek at the fold it is scored on.
    """
    import lightgbm as lgb

    max_bin = int(params.get("max_bin", 255))
    if max_bin < MIN_MAX_BIN:
        raise AssertionError(
            f"max_bin={max_bin} < {MIN_MAX_BIN}: too coarse to address the 45 "
            "distinct Age values the representation depends on"
        )

    dtrain = lgb.Dataset(X, label=y, free_raw_data=False)
    booster = lgb.train(
        dict(params),
        dtrain,
        num_boost_round=num_boost_round,
    )
    return booster


def predict(model, X):
    """Probability predictions for the positive class."""
    return model.predict(X)
