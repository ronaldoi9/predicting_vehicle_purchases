"""Model families behind one signature. Turn 2 adds XGBoost (#24), CatBoost
(#25) and a linear family (#34).

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
# scaling hook turns on for it. ``linear`` (#34) is the first family that
# actually exercises that switch.
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
    init_score=None,
):
    """Fit one model family on ``X``/``y`` under ``params`` for fixed rounds.

    No validation set is passed and no early stopping is used for any family,
    so the model cannot peek at the fold it is scored on. ``cat_features``
    names columns CatBoost should treat as categorical, computing its own
    ordered target statistics rather than reading them as plain numerics; it
    is empty (and ignored) for every family but ``catboost``. ``init_score``
    (#28) is a per-row log-odds margin the model is boosted on top of — LightGBM
    only; a non-``None`` value for any other family is a caller bug, not a
    silent no-op.
    """
    if family == "lightgbm":
        import lightgbm as lgb

        _assert_max_bin(params.get("max_bin", 255))
        dtrain = lgb.Dataset(X, label=y, init_score=init_score, free_raw_data=False)
        return lgb.train(dict(params), dtrain, num_boost_round=num_boost_round)

    if init_score is not None:
        raise ValueError(
            f"init_score is only supported for family='lightgbm', got {family!r}"
        )

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

    if family == "linear":
        # No max_bin concept -- a linear model has no splitter, so there is
        # nothing to bind against the 45 Age values here; that invariant is
        # the Adapter's job for this family (it one-hots Age instead, #34).
        # ``num_boost_round`` doubles as the solver's max_iter, the same
        # "how long to fit" knob every other family reads it as.
        from sklearn.linear_model import LogisticRegression

        # penalty defaults to L2 (ridge); passing it explicitly is deprecated
        # in sklearn>=1.8 in favour of l1_ratio, so C alone selects the ridge
        # strength -- same convention adapter.py's fitted-margin fit uses.
        clf = LogisticRegression(
            C=float(params.get("C", 1.0)),
            solver=params.get("solver", "lbfgs"),
            max_iter=num_boost_round,
        )
        clf.fit(X, y)
        return clf

    raise ValueError(
        f"unknown model family {family!r}; models.fit supports lightgbm, xgboost, catboost, linear"
    )


def predict(model, X, family: str = "lightgbm", cat_features: Sequence[str] = (), init_score=None):
    """Probability predictions for the positive class.

    ``init_score`` (#28): LightGBM's ``Booster.predict`` returns the trees'
    output alone, ignorant of the margin training was boosted on top of — so a
    model fitted with ``init_score`` must have it added back by hand, in raw
    (pre-sigmoid) space, before predictions are comparable to a model fitted
    without one.
    """
    if family == "lightgbm":
        if init_score is None:
            return model.predict(X)
        import numpy as np

        raw = model.predict(X, raw_score=True)
        return 1.0 / (1.0 + np.exp(-(raw + np.asarray(init_score))))
    if init_score is not None:
        raise ValueError(
            f"init_score is only supported for family='lightgbm', got {family!r}"
        )
    if family == "xgboost":
        import xgboost as xgb

        return model.predict(xgb.DMatrix(X))
    if family == "catboost":
        import catboost as cb

        pool = cb.Pool(X, cat_features=list(cat_features) or None)
        return model.predict_proba(pool)[:, 1]
    if family == "linear":
        return model.predict_proba(X)[:, 1]
    raise ValueError(
        f"unknown model family {family!r}; models.predict supports lightgbm, xgboost, catboost, linear"
    )
