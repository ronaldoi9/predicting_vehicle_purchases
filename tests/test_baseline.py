"""Dependency-free tests for the complete Baseline Frame (#14, PRD #12).

The Health Gate itself — reproducing OOF 0.94372 on the full Baseline Frame —
needs pandas/scikit-learn/lightgbm and the gitignored CSVs, so it runs via
``run-experiment baseline`` on a provisioned machine, not here. What *is*
checkable without the stack are the pieces the Frame widening adds: the digit
decomposition arithmetic (which must operate on the original integer), the
count-encoding lookup (computed over train+test combined), and the ``baseline``
Experiment's declaration of the turn-1 Incumbent config and the 0.94372 target.
"""

from __future__ import annotations

import experiments
import frame


def test_baseline_experiment_declares_the_full_frame_and_gate() -> None:
    exp = experiments.resolve("baseline")
    assert exp.frame == "baseline"  # the complete Frame, not raw13
    assert exp.model == "lightgbm"
    assert exp.fold_seed == 0
    assert exp.num_boost_round == 700  # fixed rounds, early stopping disabled
    assert exp.scale is False  # scaling off for a tree family
    # The two measured steps land the raw-13 0.94167 on 0.94372, clearing 0.9434.
    assert exp.target_oof == 0.94372
    assert exp.health_gate == 0.9434
    # Turn 1's Incumbent, verbatim.
    p = exp.params
    assert p["objective"] == "binary"
    assert p["learning_rate"] == 0.05
    assert p["num_leaves"] == 127
    assert p["max_bin"] == 511
    assert p["feature_fraction"] == 0.8
    assert p["bagging_freq"] == 0  # bagging_fraction stays inert; no class weighting
    assert "scale_pos_weight" not in p and "is_unbalance" not in p
    assert p["max_bin"] >= 64  # clears the 45-distinct-Age representation floor


def test_income_digits_are_computed_from_the_original_integer() -> None:
    """`(scaled_income) // 1000` is meaningless (ADR-0001) — digits come from the
    raw integer, and the transforms are the plain integer ops applied verbatim."""
    names = [name for name, _ in frame.INCOME_DIGIT_TRANSFORMS]
    assert names == [
        "Annual_Income_USD_div1000",
        "Annual_Income_USD_mod1000",
        "Annual_Income_USD_mod100",
    ]
    ops = {name: op for name, op in frame.INCOME_DIGIT_TRANSFORMS}
    # 123456 -> 123 / 456 / 56, on the original integer.
    assert ops["Annual_Income_USD_div1000"](123456) == 123
    assert ops["Annual_Income_USD_mod1000"](123456) == 456
    assert ops["Annual_Income_USD_mod100"](123456) == 56
    # A value below 1000 keeps its low digits and floors to zero thousands.
    assert ops["Annual_Income_USD_div1000"](742) == 0
    assert ops["Annual_Income_USD_mod1000"](742) == 742
    assert ops["Annual_Income_USD_mod100"](742) == 42


def test_count_lookup_counts_over_the_combined_values() -> None:
    """Count encoding maps each value to its frequency over train+test combined."""
    combined = [1000, 1000, 2000, 1000, 3000, 2000]
    lookup = frame._count_lookup(combined)
    assert lookup == {1000: 3, 2000: 2, 3000: 1}


def test_count_encoded_columns_cover_income_and_commute() -> None:
    assert frame.COUNT_ENCODED_COLUMNS == (
        "Annual_Income_USD",
        "Daily_Commute_km",
    )


def test_frame_specs_are_declared_and_baseline_adds_the_measured_columns() -> None:
    # raw13 adds nothing beyond the raw-13 layout; baseline adds the five columns
    # the two measured steps introduce (3 income digits + 2 count encodings).
    assert frame.extra_columns("raw13") == []
    assert frame.extra_columns("baseline") == [
        "Annual_Income_USD_div1000",
        "Annual_Income_USD_mod1000",
        "Annual_Income_USD_mod100",
        "Annual_Income_USD_count",
        "Daily_Commute_km_count",
    ]


def test_age_distinct_value_count_is_pinned_at_45() -> None:
    assert frame.AGE_DISTINCT_VALUES == 45


def test_unknown_frame_spec_is_rejected() -> None:
    try:
        frame.extra_columns("does_not_exist")
    except ValueError as e:
        assert "does_not_exist" in str(e)
    else:  # pragma: no cover
        raise AssertionError("an unknown frame spec must be rejected")
