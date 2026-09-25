"""The linear representation as a fourth Arena family (#34, turn 2).

Declaration tests are dependency-free, mirroring test_xgboost_arena.py and
test_catboost_arena.py: they check the three configs are genuinely distinct,
share the Frame with the standing Incumbent (the linear representation lives
in the Model Adapter, not a new frame spec -- ADR-0001), and open their own
chain rather than expressing a single-field change against income_te_tuned.
A dispatch test exercises models.fit/models.predict against sklearn's
LogisticRegression on synthetic data (fast, no CSVs). The Adapter tests below
exercise adapter.Adapter's linear_design representation directly on a tiny
synthetic frame shaped like the real one, checking the structural properties
#34 requires: the mod/div income digits are gone, the two threshold flags and
the log1p'd counts are present, Age and Range_Anxiety_Level are one-hot (not a
single ordinal slope), the gate block is present only when linear_gate is
set, and fit/transform produce an identical column layout.
"""

from __future__ import annotations

import experiments
import models


def test_linear_arena_covers_three_distinct_configs() -> None:
    assert [e.name for e in experiments.LINEAR_ARENA] == [
        "linear_baseline",
        "linear_no_gate",
        "linear_strong_ridge",
    ]
    config_sets = [
        (tuple(sorted(e.params.items())), e.linear_gate) for e in experiments.LINEAR_ARENA
    ]
    assert len(set(config_sets)) == 3, "the three configs must be genuinely distinct"


def test_linear_arena_shares_the_incumbent_frame_not_its_params() -> None:
    income_te_tuned = experiments.resolve("income_te_tuned")
    for exp in experiments.LINEAR_ARENA:
        assert exp.model == "linear"
        # Same Baseline Frame spec and target encoding as the standing
        # Incumbent -- the linear representation is an Adapter transform on
        # top of it (#34), not a new frame spec.
        assert exp.frame == income_te_tuned.frame
        assert exp.target_encode == income_te_tuned.target_encode
        assert exp.params != income_te_tuned.params
        assert exp.linear_design is True
        assert exp.scale is True


def test_linear_arena_opens_its_own_chain() -> None:
    # incumbent points at income_te_tuned for a printed Paired Delta, but the
    # family's own kill criterion (#34) is judged on absolute OOF across all
    # three, not a per-run kill_delta.
    for exp in experiments.LINEAR_ARENA:
        assert exp.incumbent == "income_te_tuned"
        assert exp.kill_delta is None
        assert exp.kill_min_folds_positive is None


def test_no_gate_isolates_the_gate_block_alone() -> None:
    baseline = experiments.resolve("linear_baseline")
    no_gate = experiments.resolve("linear_no_gate")
    assert baseline.linear_gate is True
    assert no_gate.linear_gate is False
    # Every other field matches, so the ablation isolates the gate alone.
    assert no_gate.params == baseline.params


def test_strong_ridge_only_changes_the_penalty() -> None:
    baseline = experiments.resolve("linear_baseline")
    strong = experiments.resolve("linear_strong_ridge")
    assert strong.linear_gate == baseline.linear_gate
    assert strong.params["C"] < baseline.params["C"]


def test_unknown_model_family_is_rejected() -> None:
    import pytest

    with pytest.raises(ValueError):
        models.fit([[0.0]], [0], {}, family="not-a-family")
    with pytest.raises(ValueError):
        models.predict(None, [[0.0]], family="not-a-family")


def test_models_fit_dispatches_linear_on_synthetic_data() -> None:
    import numpy as np

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    y = (X[:, 0] + rng.normal(scale=0.1, size=200) > 0).astype(int)

    clf = models.fit(X, y, {"C": 1.0}, num_boost_round=200, family="linear")
    preds = models.predict(clf, X, family="linear")
    assert len(preds) == len(y)
    assert all(0.0 <= p <= 1.0 for p in preds)


def test_needs_scaling_is_true_for_linear() -> None:
    assert models.needs_scaling("linear") is True
    assert "linear" not in models.TREE_FAMILIES


# --------------------------------------------------------------------------- #
# adapter.Adapter's linear_design representation, on a tiny synthetic frame
# shaped like the real Baseline Frame (no CSVs needed).
# --------------------------------------------------------------------------- #
def _synthetic_baseline_frame(n: int = 300, seed: int = 0):
    import numpy as np
    import pandas as pd

    import adapter as adapter_mod

    rng = np.random.default_rng(seed)
    income = rng.integers(20_000, 200_000, size=n)
    concern = rng.integers(1, 6, size=n).astype(float)
    anxiety = rng.integers(0, 3, size=n)
    subsidy_yes = rng.integers(0, 2, size=n)
    age = rng.integers(25, 70, size=n)
    commute = rng.integers(1, 60, size=n)

    X = pd.DataFrame(
        {
            "Age": age,
            adapter_mod.RECIPE_INCOME_COLUMN: income,
            "Daily_Commute_km": commute,
            "Number_of_Cars_Owned": rng.integers(0, 4, size=n),
            "Charging_Stations_Near_Home": rng.integers(0, 5, size=n),
            "Charging_Stations_Near_Work": rng.integers(0, 5, size=n),
            adapter_mod.RECIPE_CONCERN_COLUMN: concern,
            adapter_mod.RECIPE_ANXIETY_COLUMN: anxiety,
            f"{adapter_mod.RECIPE_INCOME_COLUMN}_div1000": income // 1000,
            f"{adapter_mod.RECIPE_INCOME_COLUMN}_mod1000": income % 1000,
            f"{adapter_mod.RECIPE_INCOME_COLUMN}_mod100": income % 100,
            f"{adapter_mod.RECIPE_INCOME_COLUMN}_count": rng.integers(1, 50, size=n),
            "Daily_Commute_km_count": rng.integers(1, 50, size=n),
            adapter_mod.RECIPE_SUBSIDY_YES_COLUMN: subsidy_yes,
            "Gender_Female": (rng.integers(0, 2, size=n)),
        }
    )
    y = (
        1.2 * (income / 1e5)
        + 0.6 * concern
        + 2.0 * subsidy_yes
        + rng.normal(scale=0.5, size=n)
        > 3.0
    ).astype(int)
    return X, y


def test_linear_design_drops_mod_div_income_columns() -> None:
    import adapter as adapter_mod

    X, y = _synthetic_baseline_frame()
    a = adapter_mod.Adapter(linear_design=True)
    out = a._apply_linear_design(X)
    for col in adapter_mod.LINEAR_MOD_COLUMNS:
        assert col not in out.columns


def test_linear_design_adds_income_flags_and_log1p_counts() -> None:
    import numpy as np

    import adapter as adapter_mod

    X, y = _synthetic_baseline_frame()
    a = adapter_mod.Adapter(linear_design=True)
    out = a._apply_linear_design(X)
    assert adapter_mod.LINEAR_INCOME_FLOOR_COLUMN in out.columns
    assert adapter_mod.LINEAR_INCOME_CEIL_COLUMN in out.columns
    assert set(out[adapter_mod.LINEAR_INCOME_FLOOR_COLUMN].unique()) <= {0, 1}
    expected_log1p = np.log1p(X[f"{adapter_mod.RECIPE_INCOME_COLUMN}_count"].astype("float64"))
    assert np.allclose(out[f"{adapter_mod.RECIPE_INCOME_COLUMN}_count"].to_numpy(), expected_log1p.to_numpy())


def test_linear_design_one_hots_age_and_anxiety_not_a_single_slope() -> None:
    import adapter as adapter_mod

    X, y = _synthetic_baseline_frame()
    a = adapter_mod.Adapter(linear_design=True)
    out = a._apply_linear_design(X)
    assert "Age" not in out.columns
    assert adapter_mod.RECIPE_ANXIETY_COLUMN not in out.columns
    age_cols = [c for c in out.columns if c.startswith("Age_")]
    anxiety_cols = [c for c in out.columns if c.startswith(f"{adapter_mod.RECIPE_ANXIETY_COLUMN}_")]
    assert len(age_cols) == len(adapter_mod.LINEAR_AGE_VALUES) == 45
    assert len(anxiety_cols) == 3


def test_linear_design_gate_block_toggles_with_linear_gate() -> None:
    import adapter as adapter_mod

    X, y = _synthetic_baseline_frame()
    with_gate = adapter_mod.Adapter(linear_design=True, linear_gate=True)._apply_linear_design(X)
    without_gate = adapter_mod.Adapter(linear_design=True, linear_gate=False)._apply_linear_design(X)
    gate_cols = [c for c in with_gate.columns if c.startswith("gate_")]
    assert len(gate_cols) == len(adapter_mod.LINEAR_GATE_CODES) == 30
    assert not any(c.startswith("gate_") for c in without_gate.columns)
    assert set(with_gate.columns) - set(without_gate.columns) == set(gate_cols)


def test_linear_design_fit_and_transform_share_a_column_layout() -> None:
    import numpy as np

    import adapter as adapter_mod

    X, y = _synthetic_baseline_frame(n=400)
    tr_idx = np.arange(300)
    va_idx = np.arange(300, 400)

    a = adapter_mod.Adapter(
        scale=True,
        target_encode=(adapter_mod.RECIPE_INCOME_COLUMN,),
        outer_fold=0,
        validation_index=set(va_idx.tolist()),
        scale_columns=adapter_mod.linear_scale_columns((adapter_mod.RECIPE_INCOME_COLUMN,)),
        linear_design=True,
        linear_gate=True,
    )
    X_tr = a.fit_transform(X.iloc[tr_idx], y[tr_idx])
    X_va = a.transform(X.iloc[va_idx])
    assert list(X_tr.columns) == list(X_va.columns)
    assert not X_tr.isna().any().any()
    assert not X_va.isna().any().any()


def test_linear_scale_columns_includes_target_encoded_columns() -> None:
    import adapter as adapter_mod

    cols = adapter_mod.linear_scale_columns((adapter_mod.RECIPE_INCOME_COLUMN,))
    assert f"{adapter_mod.RECIPE_INCOME_COLUMN}{adapter_mod.TE_SUFFIX}" in cols
    assert adapter_mod.RECIPE_INCOME_COLUMN in cols
    assert "Age" not in cols  # dummy columns are already 0/1, no scaling needed
