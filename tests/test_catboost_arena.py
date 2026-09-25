"""CatBoost as a third Arena family (#25, turn 2).

Declaration tests are dependency-free, mirroring test_xgboost_arena.py: they
check the three configs are genuinely distinct, share the Frame with the
standing Incumbent, and open their own chain rather than expressing a
single-field change against ``income_te_tuned``. A small dispatch test
exercises ``models.fit``/``models.predict`` against catboost on synthetic
data (fast, no CSVs) to prove the family-conditional dispatch and the
``cat_features`` plumbing work end to end, separate from the live Comparison
Runs that need the gitignored CSVs.
"""

from __future__ import annotations

import experiments
import models


def test_catboost_arena_covers_three_distinct_configs() -> None:
    assert [e.name for e in experiments.CATBOOST_ARENA] == [
        "catboost_baseline",
        "catboost_ordered_ts",
        "catboost_tuned",
    ]
    # ordered_ts's distinguishing field is cat_features, not params -- it and
    # catboost_baseline are deliberately the same tree shape so the ordered
    # target statistics' own effect is isolated, not confounded with a depth
    # or learning-rate change.
    config_sets = [
        (tuple(sorted(e.params.items())), e.cat_features) for e in experiments.CATBOOST_ARENA
    ]
    assert len(set(config_sets)) == 3, "the three configs must be genuinely distinct"


def test_catboost_arena_shares_the_incumbent_frame_not_its_params() -> None:
    income_te_tuned = experiments.resolve("income_te_tuned")
    for exp in experiments.CATBOOST_ARENA:
        assert exp.model == "catboost"
        # Same Frame and target encoding as the standing Incumbent, so both
        # families are scored on identical rows (#25).
        assert exp.frame == income_te_tuned.frame
        assert exp.target_encode == income_te_tuned.target_encode
        # But not the Incumbent's LightGBM params -- a family, not a port.
        assert exp.params != income_te_tuned.params


def test_catboost_arena_opens_its_own_chain() -> None:
    # incumbent points at income_te_tuned for a printed Paired Delta, but the
    # family's own kill criterion (#25) is judged on absolute OOF across all
    # three, not a per-run kill_delta.
    for exp in experiments.CATBOOST_ARENA:
        assert exp.incumbent == "income_te_tuned"
        assert exp.kill_delta is None
        assert exp.kill_min_folds_positive is None


def test_catboost_arena_params_clear_the_min_max_bin_assert() -> None:
    for exp in experiments.CATBOOST_ARENA:
        assert exp.params["border_count"] >= models.MIN_MAX_BIN


def test_catboost_arena_pins_determinism_inputs() -> None:
    for exp in experiments.CATBOOST_ARENA:
        assert exp.params["thread_count"] == 10
        assert exp.params["random_seed"] == 0


def test_ordered_ts_exercises_cat_features_on_adapter_untouched_ordinals() -> None:
    baseline = experiments.resolve("catboost_baseline")
    ordered_ts = experiments.resolve("catboost_ordered_ts")
    tuned = experiments.resolve("catboost_tuned")

    # Only the ordered_ts config sets cat_features; the other two run on the
    # Frame's plain numeric/one-hot columns like every other family.
    assert baseline.cat_features == ()
    assert tuned.cat_features == ()
    assert ordered_ts.cat_features == (
        "Environmental_Concern_Level",
        "Range_Anxiety_Level",
    )
    # Neither ordinal is the column the Model Adapter target-encodes, so the
    # mechanism carries no overlap with income_te to state (#25).
    assert set(ordered_ts.cat_features).isdisjoint(ordered_ts.target_encode)


def test_tuned_is_a_genuine_regularisation_hypothesis_not_a_port() -> None:
    tuned = experiments.resolve("catboost_tuned")
    assert tuned.params["depth"] == 10
    assert "l2_leaf_reg" in tuned.params


def test_unknown_model_family_is_still_rejected() -> None:
    import pytest

    with pytest.raises(ValueError):
        models.fit([[0.0]], [0], {}, family="not-a-family")
    with pytest.raises(ValueError):
        models.predict(None, [[0.0]], family="not-a-family")


def test_models_fit_dispatches_catboost_on_synthetic_data() -> None:
    catboost = __import__("catboost")  # skip cleanly if not installed
    assert catboost.__version__

    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "a": rng.normal(size=200),
            "b": rng.normal(size=200),
            "cat": rng.integers(0, 3, size=200),
        }
    )
    y = (X["a"] + rng.normal(scale=0.1, size=200) > 0).astype(int)

    params = {
        "loss_function": "Logloss",
        "border_count": 64,
        "thread_count": 1,
        "random_seed": 0,
        "verbose": False,
        "allow_writing_files": False,
        "depth": 3,
        "learning_rate": 0.1,
    }
    model = models.fit(X, y, params, num_boost_round=20, family="catboost")
    preds = models.predict(model, X, family="catboost")
    assert len(preds) == len(y)
    assert all(0.0 <= p <= 1.0 for p in preds)


def test_models_fit_dispatches_catboost_with_cat_features() -> None:
    catboost = __import__("catboost")  # skip cleanly if not installed
    assert catboost.__version__

    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "a": rng.normal(size=200),
            "cat": rng.integers(0, 3, size=200),
        }
    )
    y = (X["a"] + rng.normal(scale=0.1, size=200) > 0).astype(int)

    params = {
        "loss_function": "Logloss",
        "border_count": 64,
        "thread_count": 1,
        "random_seed": 0,
        "verbose": False,
        "allow_writing_files": False,
        "depth": 3,
        "learning_rate": 0.1,
    }
    model = models.fit(X, y, params, num_boost_round=20, family="catboost", cat_features=["cat"])
    preds = models.predict(model, X, family="catboost", cat_features=["cat"])
    assert len(preds) == len(y)
    assert all(0.0 <= p <= 1.0 for p in preds)
