"""heuljax's pipeline as an Arena family: the tracer bullet (#37, turn 3).

Declaration tests are dependency-free, mirroring test_xgboost_arena.py and
test_linear_arena.py. The family tests run ``models.fit``/``models.predict``
and one runner fold on a small synthetic frame shaped like the raw CSVs (no
CSVs needed): the family consumes the raw columns through the ``raw_columns``
Frame spec and builds its own representation inside the fold, so these check
that it trains on CPU for exactly the declared rounds, returns one probability
per validation row, is deterministic under a fixed seed, and that its donor
state never counts the rows it encodes -- the overlap assert fires when forced.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

import experiments
import models

SRC = Path(__file__).resolve().parent.parent / "src"

CATEGORY_LEVELS = {
    "Gender": ("Male", "Female", "Other"),
    "City_Type": ("Suburban", "Rural", "Urban"),
    "Current_Car_Type": ("Sedan", "SUV", "Hatchback", "Truck"),
    "Home_Charging_Possible": ("Yes", "No"),
    "Subsidy_Available": ("No", "Yes"),
    "Range_Anxiety_Level": ("Low", "Medium", "High"),
}


def _synthetic_raw(n: int = 600, seed: int = 0, with_target: bool = True, id_start: int = 0):
    """A frame with the raw CSV schema: id, the 13 raw columns, the 0/1 target."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    income = rng.integers(200, 260, size=n) * 100.0  # repeated keys, so TE has support
    concern = rng.integers(1, 6, size=n).astype(float)
    subsidy = rng.choice(CATEGORY_LEVELS["Subsidy_Available"], size=n)
    df = pd.DataFrame(
        {
            "id": np.arange(id_start, id_start + n),
            "Age": rng.integers(25, 70, size=n),
            "Annual_Income_USD": income,
            "Daily_Commute_km": np.round(rng.uniform(1, 60, size=n), 1),
            "Number_of_Cars_Owned": rng.integers(0, 4, size=n),
            "Charging_Stations_Near_Home": rng.integers(0, 5, size=n),
            "Charging_Stations_Near_Work": rng.integers(0, 8, size=n),
            "Environmental_Concern_Level": concern,
            **{
                c: rng.choice(levels, size=n)
                for c, levels in CATEGORY_LEVELS.items()
                if c != "Subsidy_Available"
            },
            "Subsidy_Available": subsidy,
        }
    )
    if with_target:
        score = income / 1e4 + 0.6 * concern + 2.0 * (subsidy == "Yes") + rng.normal(scale=1.0, size=n)
        df["Will_Buy_EV"] = (score > np.median(score)).astype("int8")
    return df


def _small_config(num_boost_round: int = 20):
    exp = experiments.resolve("heuljax_tracer")
    return replace(exp, num_boost_round=num_boost_round, params={**exp.params, "nthread": 1})


# --------------------------------------------------------------------------- #
# The declaration.
# --------------------------------------------------------------------------- #
def test_heuljax_tracer_is_declared_as_its_own_family_on_raw_columns() -> None:
    exp = experiments.resolve("heuljax_tracer")
    assert exp.model == "heuljax"
    # Its representation is its own (ADR-0001): no Baseline Frame, no Adapter
    # target encoding, no scaling -- only the raw columns reach the family.
    assert exp.frame == "raw_columns"
    assert exp.target_encode == ()
    assert exp.scale is False
    assert exp.fold_seed == 0


def test_heuljax_tracer_pins_the_notebook_booster_on_cpu_at_fixed_rounds() -> None:
    exp = experiments.resolve("heuljax_tracer")
    assert exp.num_boost_round == 1000  # fixed, mid-plateau; no early stopping
    p = exp.params
    assert p["learning_rate"] == 0.015
    assert p["max_depth"] == 5
    assert p["max_bin"] == 256
    assert p["tree_method"] == "hist"
    assert p["device"] == "cpu"
    assert p["seed"] == 0
    assert p["max_bin"] >= models.MIN_MAX_BIN


def test_heuljax_tracer_opens_its_own_chain_against_the_turn3_incumbent() -> None:
    exp = experiments.resolve("heuljax_tracer")
    # Correlation and a reference Paired Delta are read against the turn-3
    # Incumbent (ADR-0006 §4); the family's gate is absolute, not a kill_delta.
    assert exp.incumbent == "hpsearch_lightgbm_best_confirm"
    assert exp.kill_delta is None
    assert exp.kill_min_folds_positive is None


def test_heuljax_module_carries_the_apache_attribution_and_source_link() -> None:
    text = (SRC / "heuljax.py").read_text(encoding="utf-8")
    assert "Apache License, Version 2.0" in text
    assert "https://www.kaggle.com/code/heuljax/kps6e09-xgb-sample" in text


def test_heuljax_is_a_tree_family_so_scaling_stays_off() -> None:
    assert models.needs_scaling("heuljax") is False


# --------------------------------------------------------------------------- #
# The raw_columns Frame spec: the raw CSV columns, id and target dropped.
# --------------------------------------------------------------------------- #
def test_raw_columns_spec_hands_over_the_13_raw_columns_untouched() -> None:
    import frame
    from columns import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS

    train = _synthetic_raw(200)
    test = _synthetic_raw(50, seed=1, with_target=False, id_start=200)
    X_train, X_test = frame.build_frame(train, test, "raw_columns")
    assert sorted(X_train.columns) == sorted(NUMERIC_COLUMNS + CATEGORICAL_COLUMNS)
    assert list(X_train.columns) == list(X_test.columns)
    assert len(X_train) == 200 and len(X_test) == 50
    # Untouched: strings stay strings, numerics keep their values.
    assert X_train["Gender"].tolist() == train["Gender"].tolist()
    assert X_train["Annual_Income_USD"].tolist() == train["Annual_Income_USD"].tolist()
    assert frame.extra_columns("raw_columns") == []


# --------------------------------------------------------------------------- #
# The family through models.fit / models.predict.
# --------------------------------------------------------------------------- #
def test_models_fit_trains_heuljax_for_exactly_the_fixed_rounds() -> None:
    config = _small_config(num_boost_round=17)
    df = _synthetic_raw()
    X = df.drop(columns=["id", "Will_Buy_EV"])
    y = df["Will_Buy_EV"].to_numpy()

    model = models.fit(X, y, config.params, num_boost_round=17, family="heuljax")
    assert model.booster.num_boosted_rounds() == 17


def test_models_predict_returns_one_probability_per_validation_row() -> None:
    config = _small_config()
    df = _synthetic_raw()
    X = df.drop(columns=["id", "Will_Buy_EV"])
    y = df["Will_Buy_EV"].to_numpy()

    model = models.fit(X.iloc[:450], y[:450], config.params, num_boost_round=20, family="heuljax")
    preds = models.predict(model, X.iloc[450:], family="heuljax")
    assert preds.shape == (150,)
    assert ((preds >= 0.0) & (preds <= 1.0)).all()


def test_heuljax_is_deterministic_under_a_fixed_seed() -> None:
    import numpy as np

    config = _small_config()
    df = _synthetic_raw()
    X = df.drop(columns=["id", "Will_Buy_EV"])
    y = df["Will_Buy_EV"].to_numpy()

    runs = [
        models.predict(
            models.fit(X.iloc[:450], y[:450], config.params, num_boost_round=20, family="heuljax"),
            X.iloc[450:],
            family="heuljax",
        )
        for _ in range(2)
    ]
    assert np.array_equal(runs[0], runs[1])


# --------------------------------------------------------------------------- #
# The donor state: fitted only on donor rows, never on the rows it encodes.
# --------------------------------------------------------------------------- #
def test_donor_overlap_assert_fires_on_a_forced_overlap() -> None:
    import numpy as np

    import heuljax

    df = _synthetic_raw(100)
    X = df.drop(columns=["id", "Will_Buy_EV"])
    y = df["Will_Buy_EV"].to_numpy()
    with pytest.raises(AssertionError, match="overlaps"):
        heuljax.fit_donor_state(X, y, donor=np.arange(0, 60), receiving=np.arange(50, 100))


def test_a_training_row_never_counts_towards_its_own_encoding() -> None:
    import numpy as np

    import heuljax

    df = _synthetic_raw(400)
    # One row with an income no other row carries: if its own label reached its
    # encoding, its support would be 1, not 0.
    df.loc[7, "Annual_Income_USD"] = 999_999.0
    X = df.drop(columns=["id", "Will_Buy_EV"])
    y = df["Will_Buy_EV"].to_numpy()

    features = heuljax.donor_features(X, y, seed=0)
    support = features[:, heuljax.FEATURE_COLUMNS.index("SUPPORT_INC_LOG")]
    assert support[7] == 0.0
    assert features.shape == (400, len(heuljax.FEATURE_COLUMNS))
    assert np.isfinite(features).all()


# --------------------------------------------------------------------------- #
# One runner fold, end to end on the synthetic raw frame.
# --------------------------------------------------------------------------- #
def test_one_runner_fold_trains_on_outer_training_rows_only() -> None:
    import numpy as np

    import frame
    import runner

    config = _small_config()
    train = _synthetic_raw(600)
    test = _synthetic_raw(100, seed=1, with_target=False, id_start=600)
    y = train["Will_Buy_EV"].to_numpy()
    X_train, X_test = frame.build_frame(train, test, config.frame)

    va_idx = np.arange(0, 120)
    tr_idx = np.arange(120, 600)
    adapter = runner.fold_adapter(config, 0, va_idx.tolist())
    X_tr = adapter.fit_transform(X_train.iloc[tr_idx], y[tr_idx])
    X_va = adapter.transform(X_train.iloc[va_idx])
    preds = runner.fold_predict(config, X_tr, y[tr_idx], X_va)
    assert preds.shape == (120,)

    # The fold boundary is the Adapter's: a validation row reaching the fit is
    # refused before the family ever sees it.
    leaky = runner.fold_adapter(config, 0, va_idx.tolist())
    with pytest.raises(AssertionError, match="validation-fold rows"):
        leaky.fit_transform(X_train.iloc[np.arange(100, 600)], y[100:600])


# --------------------------------------------------------------------------- #
# The Run Record: per-fold time and the OOF correlation with the Incumbent.
# --------------------------------------------------------------------------- #
def test_oof_correlation_is_pearson_between_two_vectors() -> None:
    import runner

    assert runner.oof_correlation([0.1, 0.2, 0.3], [0.2, 0.4, 0.6]) == pytest.approx(1.0)
    assert runner.oof_correlation([0.1, 0.2, 0.3], [0.3, 0.2, 0.1]) == pytest.approx(-1.0)
    with pytest.raises(ValueError):
        runner.oof_correlation([0.1, 0.2], [0.1, 0.2, 0.3])


def test_run_record_carries_fold_times_and_incumbent_correlation() -> None:
    import runner

    record = runner.build_run_record(
        experiment="heuljax_tracer",
        config={"name": "heuljax_tracer"},
        fold_aucs=[0.94] * 5,
        oof_auc=0.94,
        fold_sha256="abc",
        oof_path="runs/x/oof.npy",
        wall_time=10.0,
        git_info={"git_sha": "deadbeef", "dirty": False},
        fold_wall_times=[1.0, 2.0, 3.0, 4.0, 5.0],
        incumbent_oof_corr=0.98,
    )
    assert record["fold_wall_times"] == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert record["incumbent_oof_corr"] == 0.98
