"""Model families behind one signature. Turn 2 adds XGBoost (#24) and CatBoost
(#25).

Comparison Runs use a **fixed round count with early stopping disabled** —
early stopping on the evaluated fold biases OOF upward and breaks the pairing.
Early stopping is permitted only in the Submission Fit, which lives elsewhere.

An assert binds ``max_bin`` from below against the ``Age`` value count, so a
parameter change cannot violate the representation invariant that all 45 ages
stay individually addressable — checked for every family, not just LightGBM.
Each family's library is imported lazily inside :func:`fit`/:func:`predict` so
the module imports by bare name anywhere, and ``fit``/``predict`` dispatch on
an explicit ``family`` argument rather than inspecting ``params`` — the caller
(an :class:`experiments.Experiment`) already knows its family in ``.model``.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

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


def _assert_max_bin(value: int) -> int:
    value = int(value)
    if value < MIN_MAX_BIN:
        raise AssertionError(
            f"max_bin={value} < {MIN_MAX_BIN}: too coarse to address the 45 "
            "distinct Age values the representation depends on"
        )
    return value


def fit(
    X,
    y,
    params: Mapping[str, Any],
    num_boost_round: int = 700,
    family: str = "lightgbm",
    cat_features: Sequence[str] = (),
):
    """Fit one model family on ``X``/``y`` under ``params`` for fixed rounds.

    No validation set is passed and no early stopping is used for any family,
    so the model cannot peek at the fold it is scored on. ``cat_features``
    names columns CatBoost should treat as categorical, computing its own
    ordered target statistics rather than reading them as plain numerics; it
    is empty (and ignored) for every family but ``catboost``.
    """
    if family == "lightgbm":
        import lightgbm as lgb

        _assert_max_bin(params.get("max_bin", 255))
        dtrain = lgb.Dataset(X, label=y, free_raw_data=False)
        return lgb.train(dict(params), dtrain, num_boost_round=num_boost_round)

    if family == "xgboost":
        import xgboost as xgb

        # xgboost's own default max_bin (256) already clears MIN_MAX_BIN, but
        # the assert stays explicit rather than assumed, same as lightgbm.
        _assert_max_bin(params.get("max_bin", 256))
        dtrain = xgb.DMatrix(X, label=y)
        return xgb.train(dict(params), dtrain, num_boost_round=num_boost_round)

    if family == "catboost":
        import catboost as cb

        # catboost's own default border_count (254, its name for max_bin)
        # already clears MIN_MAX_BIN, but the assert stays explicit rather
        # than assumed, same as the other two families.
        border_count = _assert_max_bin(params.get("border_count", 254))
        cb_params = {k: v for k, v in params.items() if k != "border_count"}
        pool = cb.Pool(X, label=y, cat_features=list(cat_features) or None)
        model = cb.CatBoostClassifier(border_count=border_count, iterations=num_boost_round, **cb_params)
        model.fit(pool)
        return model

    raise ValueError(
        f"unknown model family {family!r}; models.fit supports lightgbm, xgboost, catboost"
    )


def predict(model, X, family: str = "lightgbm", cat_features: Sequence[str] = ()):
    """Probability predictions for the positive class."""
    if family == "lightgbm":
        return model.predict(X)
    if family == "xgboost":
        import xgboost as xgb

        return model.predict(xgb.DMatrix(X))
    if family == "catboost":
        import catboost as cb

        pool = cb.Pool(X, cat_features=list(cat_features) or None)
        return model.predict_proba(pool)[:, 1]
    raise ValueError(
        f"unknown model family {family!r}; models.predict supports lightgbm, xgboost, catboost"
    )
