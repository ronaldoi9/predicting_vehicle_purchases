"""XGBoost as a second Arena family (#24, turn 2).

Declaration tests are dependency-free, mirroring test_resolution.py: they check
the three configs are genuinely distinct, share the Frame with the standing
Incumbent, and open their own chain rather than expressing a single-field
change against ``income_te_tuned``. A small dispatch test exercises
``models.fit``/``models.predict`` against xgboost on synthetic data (fast,
no CSVs) to prove the family-conditional dispatch works end to end, separate
from the live Comparison Runs that need the gitignored CSVs.
"""

from __future__ import annotations

import experiments
import models


def test_xgboost_arena_covers_three_distinct_configs() -> None:
    assert [e.name for e in experiments.XGBOOST_ARENA] == [
        "xgboost_baseline",
        "xgboost_lossguide",
        "xgboost_tuned",
    ]
    param_sets = [tuple(sorted(e.params.items())) for e in experiments.XGBOOST_ARENA]
    assert len(set(param_sets)) == 3, "the three configs must be genuinely distinct"


def test_xgboost_arena_shares_the_incumbent_frame_not_its_params() -> None:
    income_te_tuned = experiments.resolve("income_te_tuned")
    for exp in experiments.XGBOOST_ARENA:
        assert exp.model == "xgboost"
        # Same Frame and target encoding as the standing Incumbent, so both
        # families are scored on identical rows (#24).
        assert exp.frame == income_te_tuned.frame
        assert exp.target_encode == income_te_tuned.target_encode
        # But not the Incumbent's LightGBM params -- a family, not a port.
        assert exp.params != income_te_tuned.params


def test_xgboost_arena_opens_its_own_chain() -> None:
    # incumbent points at income_te_tuned for a printed Paired Delta, but the
    # family's own kill criterion (#24) is judged on absolute OOF across all
    # three, not a per-run kill_delta.
    for exp in experiments.XGBOOST_ARENA:
        assert exp.incumbent == "income_te_tuned"
        assert exp.kill_delta is None
        assert exp.kill_min_folds_positive is None


def test_xgboost_arena_params_clear_the_min_max_bin_assert() -> None:
    for exp in experiments.XGBOOST_ARENA:
        assert exp.params["max_bin"] >= models.MIN_MAX_BIN


def test_xgboost_arena_pins_determinism_inputs() -> None:
    for exp in experiments.XGBOOST_ARENA:
        assert exp.params["tree_method"] == "hist"
        assert exp.params["nthread"] == 10
        assert exp.params["seed"] == 0


def test_lossguide_and_tuned_are_genuine_growth_hypotheses_not_a_port() -> None:
    lossguide = experiments.resolve("xgboost_lossguide")
    assert lossguide.params["grow_policy"] == "lossguide"
    assert lossguide.params["max_leaves"] == 127
    # max_depth must be unlimited, else xgboost's default (6) silently caps
    # lossguide to depth-wise's own shape and the hypothesis is never exercised.
    assert lossguide.params["max_depth"] == 0

    tuned = experiments.resolve("xgboost_tuned")
    assert tuned.params["max_depth"] == 10
    assert "lambda" in tuned.params


def test_unknown_model_family_is_rejected() -> None:
    import pytest

    with pytest.raises(ValueError):
        models.fit([[0.0]], [0], {}, family="not-a-family")
    with pytest.raises(ValueError):
        models.predict(None, [[0.0]], family="not-a-family")


def test_models_fit_dispatches_xgboost_on_synthetic_data() -> None:
    xgboost = __import__("xgboost")  # skip cleanly if not installed
    assert xgboost.__version__

    import numpy as np

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    y = (X[:, 0] + rng.normal(scale=0.1, size=200) > 0).astype(int)

    params = {
        "objective": "binary:logistic",
        "tree_method": "hist",
        "max_bin": 511,
        "nthread": 1,
        "seed": 0,
        "verbosity": 0,
    }
    booster = models.fit(X, y, params, num_boost_round=10, family="xgboost")
    preds = models.predict(booster, X, family="xgboost")
    assert len(preds) == len(y)
    assert all(0.0 <= p <= 1.0 for p in preds)
