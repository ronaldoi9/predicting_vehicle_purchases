"""The Resolution axis (#18, PRD #12): the max_bin sweep and Age as a lookup.

Resolution is the only axis this dataset's representation has been shown to pay
on — the degree to which the model can address individual values of a
high-cardinality column rather than a merged range of them. This ticket declares
the two candidates on that axis, each as an Experiment against the standing
Incumbent and each carrying a kill criterion declared *before* it runs:

* **The ``max_bin`` sweep over {255, 1023, 2047}.** ``max_bin`` is the Resolution
  control, not an ordinary tuning knob: it decides whether a histogram-binned
  tree addresses individual income values or merges them into a range. Kill
  criterion: no value beats the Incumbent in at least 4 of 5 folds -> freeze at
  the Incumbent's 511. A folds-positive criterion, not a mean-delta one.
* **``Age`` as a 45-level target-encoded lookup**, under the same nested
  cross-fit contract as the income TE — the honest alternative to relying on
  ``max_bin`` to deliver the lookup. Kill criterion: paired delta < +0.0001 ->
  dead.

Everything here is the declaration and the folds-positive kill bookkeeping,
dependency-free so the stdlib harness (scripts/run_resolution_tests.py) runs it
where the ML stack is absent. The live Paired Deltas need the stack + the
gitignored CSVs and land on a provisioned machine.
"""

from __future__ import annotations

import adapter
import experiments
import models
import runner


# --------------------------------------------------------------------------- #
# The max_bin sweep: three named Experiments, each a single-field change.
# --------------------------------------------------------------------------- #
def test_max_bin_sweep_covers_the_declared_values() -> None:
    assert experiments.MAX_BIN_SWEEP == (255, 1023, 2047)
    for mb in experiments.MAX_BIN_SWEEP:
        exp = experiments.resolve(f"max_bin_{mb}")
        assert exp.params["max_bin"] == mb
        # Each is a Paired Delta against the standing Incumbent.
        assert exp.incumbent == "baseline"


def test_max_bin_sweep_is_a_single_field_change_against_the_incumbent() -> None:
    baseline = experiments.resolve("baseline")
    for mb in experiments.MAX_BIN_SWEEP:
        exp = experiments.resolve(f"max_bin_{mb}")
        # The only modelling field that differs is params["max_bin"].
        fields = ("frame", "model", "num_boost_round", "fold_seed", "scale", "target_encode")
        for f in fields:
            assert getattr(exp, f) == getattr(baseline, f), f"{f} must match the Incumbent"
        # Every param but max_bin is the Incumbent's.
        for key, value in baseline.params.items():
            if key == "max_bin":
                continue
            assert exp.params[key] == value, f"param {key} must match the Incumbent"
        assert exp.params["max_bin"] != baseline.params["max_bin"] or mb == baseline.params["max_bin"]


def test_max_bin_sweep_values_clear_the_min_max_bin_assert() -> None:
    # The max_bin >= 64 assert must hold for every value swept, so the 45 Age
    # values stay individually addressable at every resolution tested.
    for mb in experiments.MAX_BIN_SWEEP:
        assert mb >= models.MIN_MAX_BIN


def test_max_bin_sweep_uses_fixed_rounds_and_carries_the_folds_positive_kill() -> None:
    baseline = experiments.resolve("baseline")
    for mb in experiments.MAX_BIN_SWEEP:
        exp = experiments.resolve(f"max_bin_{mb}")
        # Fixed rounds (early stopping is disabled by the Comparison-Run protocol
        # in models.fit): the round count is the Incumbent's.
        assert exp.num_boost_round == baseline.num_boost_round
        # The sweep's kill line is folds-positive, not a mean-delta threshold.
        assert exp.kill_min_folds_positive == 4
        assert exp.kill_delta is None


def test_max_bin_candidate_config_hashes_differ_from_the_incumbent() -> None:
    base = experiments.resolve("baseline").as_config()
    for mb in experiments.MAX_BIN_SWEEP:
        cfg = experiments.resolve(f"max_bin_{mb}").as_config()
        assert cfg["params"]["max_bin"] == mb
        assert runner.config_hash(cfg) != runner.config_hash(base)


# --------------------------------------------------------------------------- #
# The folds-positive kill criterion (the sweep's rule), recorded either way.
# --------------------------------------------------------------------------- #
def test_folds_positive_kill_outcome_is_applied_and_recorded() -> None:
    # A value that beats the Incumbent in only 3/5 folds is dead (frozen at 511).
    dead = runner.folds_positive_kill_outcome(3, 4)
    assert dead["dead"] is True
    assert dead["min_folds_positive"] == 4
    assert dead["folds_positive"] == 3
    assert dead["n_folds"] == 5
    # 4/5 survives to the verdict rule; 5/5 too.
    assert runner.folds_positive_kill_outcome(4, 4)["dead"] is False
    assert runner.folds_positive_kill_outcome(5, 4)["dead"] is False


# --------------------------------------------------------------------------- #
# Age as a 45-level target-encoded lookup: same nested cross-fit contract.
# --------------------------------------------------------------------------- #
def test_age_te_is_a_single_field_change_under_the_income_te_contract() -> None:
    baseline = experiments.resolve("baseline")
    age_te = experiments.resolve("age_te")

    # Single-field change: target_encode = (Age,), everything else the Incumbent.
    assert age_te.target_encode == ("Age",)
    fields = ("frame", "model", "num_boost_round", "fold_seed", "scale", "params")
    for f in fields:
        assert getattr(age_te, f) == getattr(baseline, f), f"{f} must match the Incumbent"

    # Same nested cross-fit contract as the income TE — the Adapter's, by name.
    income_te = experiments.resolve("income_te")
    assert type(age_te.target_encode) is type(income_te.target_encode)
    assert age_te.incumbent == "baseline"
    assert "age" in age_te.hypothesis.lower()


def test_age_te_kill_criterion_is_declared_below_one_ten_thousandth() -> None:
    age_te = experiments.resolve("age_te")
    assert age_te.kill_delta == 0.0001
    # Applied via the mean-delta kill machinery: below +0.0001 -> dead.
    assert runner.kill_criterion_outcome(0.00005, age_te.kill_delta)["dead"] is True
    assert runner.kill_criterion_outcome(0.0002, age_te.kill_delta)["dead"] is False


def test_age_te_respects_the_age_invariant_by_adding_a_column() -> None:
    # The TE adds a `<col>_te` column rather than replacing the raw value, so all
    # 45 Age values stay individually addressable — never smoothed into the raw
    # column, never monotone-constrained. The invariant is respected by the
    # additive-suffix contract, whichever candidate survives.
    assert adapter.TE_SUFFIX == "_te"
    age_te = experiments.resolve("age_te")
    # The candidate encodes Age but never asks the Frame to drop or bin it.
    assert age_te.target_encode == ("Age",)
    assert age_te.frame == "baseline"  # the Frame still carries raw Age (45 values)


def test_age_te_config_carries_the_target_encoding() -> None:
    cfg = experiments.resolve("age_te").as_config()
    assert cfg["target_encode"] == ["Age"]
    base = experiments.resolve("baseline").as_config()
    assert runner.config_hash(cfg) != runner.config_hash(base)


# --------------------------------------------------------------------------- #
# Every candidate is registered and resolvable through the single entry point.
# --------------------------------------------------------------------------- #
def test_resolution_candidates_are_all_registered() -> None:
    for name in ("max_bin_255", "max_bin_1023", "max_bin_2047", "age_te"):
        exp = experiments.resolve(name)
        assert exp.name == name
