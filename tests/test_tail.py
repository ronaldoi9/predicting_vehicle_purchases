"""The tail of the band (ii) queue (#19, PRD #12): seed-averaging, SMOTENC,
conservative tuning — three candidates nothing measured so far suggests will
pay, each boxed by a kill criterion declared before it runs so that none can
absorb the week.

Everything here is the declaration and the kill bookkeeping, dependency-free so
the stdlib harness (scripts/run_tail_tests.py) runs it where the ML stack is
absent. The one SMOTENC property test reaches for imbalanced-learn / pandas and
is reported SKIP where those are absent; it runs for real under pytest on a
provisioned machine.
"""

from __future__ import annotations

import adapter
import experiments
import runner


# --------------------------------------------------------------------------- #
# Seed-averaged bag over three seeds: variance reduction, bought only if cheaper
# than it is worth.
# --------------------------------------------------------------------------- #
def test_seed_bag_is_a_single_field_change_against_the_incumbent() -> None:
    baseline = experiments.resolve("baseline")
    exp = experiments.resolve("seed_bag")
    assert exp.seed_bag == (0, 1, 2)
    # The only modelling field that differs is seed_bag; everything else is the
    # Incumbent's, so the one-change discipline is literal in the declaration.
    fields = (
        "frame", "model", "params", "num_boost_round", "fold_seed", "scale",
        "target_encode", "oversample",
    )
    for f in fields:
        assert getattr(exp, f) == getattr(baseline, f), f"{f} must match the Incumbent"
    assert exp.incumbent == "baseline"
    # The Incumbent carries no bag.
    assert baseline.seed_bag == ()
    assert "age" not in exp.name


def test_seeded_params_sets_all_four_seeds_without_mutating_the_source() -> None:
    p = experiments.seeded_params(experiments.LGBM_PARAMS, 2)
    for k in experiments.SEED_KEYS:
        assert p[k] == 2
    # A non-seed param is untouched.
    assert p["num_leaves"] == experiments.LGBM_PARAMS["num_leaves"]
    # The source params are never mutated (the Incumbent stays at seed 0).
    assert experiments.LGBM_PARAMS["seed"] == 0


def test_seed_bag_kill_weighs_compute_cost_against_measured_gain() -> None:
    exp = experiments.resolve("seed_bag")
    assert exp.kill_value_per_run == experiments.SEED_BAG_VALUE_PER_RUN
    # Three seeds = two extra fits; each must earn +0.0001, so break-even is the
    # expected +0.0002. At exactly the expected gain the bag survives.
    out = runner.seed_bag_kill_outcome(0.0002, 3, 0.0001)
    assert out["n_seeds"] == 3
    assert out["extra_runs"] == 2
    assert out["required_gain"] == 0.0002
    assert out["dead"] is False
    # Below break-even the compute cost exceeds the measured gain -> dead.
    assert runner.seed_bag_kill_outcome(0.00019, 3, 0.0001)["dead"] is True
    # A clear win survives.
    assert runner.seed_bag_kill_outcome(0.001, 3, 0.0001)["dead"] is False


def test_seed_bag_config_carries_the_bag_and_hashes_differently() -> None:
    base = experiments.resolve("baseline").as_config()
    cfg = experiments.resolve("seed_bag").as_config()
    assert cfg["seed_bag"] == [0, 1, 2]
    assert base["seed_bag"] == []
    assert runner.config_hash(cfg) != runner.config_hash(base)


# --------------------------------------------------------------------------- #
# SMOTENC, fitted inside the training fold only, income digits regenerated from
# the synthetic income. Plain SMOTE is excluded outright.
# --------------------------------------------------------------------------- #
def test_smotenc_is_a_single_field_change_and_names_smotenc_not_smote() -> None:
    baseline = experiments.resolve("baseline")
    exp = experiments.resolve("smotenc")
    assert exp.oversample == adapter.OVERSAMPLE_SMOTENC == "smotenc"
    fields = (
        "frame", "model", "params", "num_boost_round", "fold_seed", "scale",
        "target_encode", "seed_bag",
    )
    for f in fields:
        assert getattr(exp, f) == getattr(baseline, f), f"{f} must match the Incumbent"
    assert exp.incumbent == "baseline"
    # sampling_strategy=0.5 is the declared contract.
    assert adapter.SMOTENC_SAMPLING_STRATEGY == 0.5
    # The hypothesis names SMOTENC and states why plain SMOTE is excluded.
    h = exp.hypothesis.lower()
    assert "smotenc" in h
    assert "smote" in h  # the exclusion is stated


def test_smotenc_kill_criterion_declared_at_three_ten_thousandths() -> None:
    exp = experiments.resolve("smotenc")
    assert exp.kill_delta == 0.0003
    assert exp.kill_value_per_run is None
    assert exp.kill_time_budget_s is None
    # Applied via the mean-delta kill machinery: below +0.0003 -> dead.
    assert runner.kill_criterion_outcome(0.0002, exp.kill_delta)["dead"] is True
    assert runner.kill_criterion_outcome(0.0004, exp.kill_delta)["dead"] is False


def test_plain_smote_is_refused_before_any_import() -> None:
    # Constructing an Adapter with plain SMOTE fails immediately, without the ML
    # stack: plain SMOTE is excluded outright.
    try:
        adapter.Adapter(oversample="smote")
    except AssertionError as exc:
        assert "SMOTENC" in str(exc)
    else:
        raise AssertionError("plain SMOTE must be refused")
    # An unknown strategy is refused too.
    try:
        adapter.Adapter(oversample="rose")
    except AssertionError:
        pass
    else:
        raise AssertionError("an unknown oversample strategy must be refused")
    # SMOTENC and no oversampling are both accepted.
    adapter.Adapter(oversample="smotenc")
    adapter.Adapter(oversample=None)


def test_smotenc_config_carries_the_oversample_and_hashes_differently() -> None:
    base = experiments.resolve("baseline").as_config()
    cfg = experiments.resolve("smotenc").as_config()
    assert cfg["oversample"] == "smotenc"
    assert base["oversample"] is None
    assert runner.config_hash(cfg) != runner.config_hash(base)


def test_smotenc_regenerates_income_digits_inside_the_fold() -> None:
    # Property test (needs imbalanced-learn + pandas + numpy): SMOTENC oversamples
    # only the training rows, and the income digit columns of the synthetic rows
    # are regenerated from the synthetic income so the Frame stays self-consistent.
    import numpy as np  # noqa: F401
    import pandas as pd
    import imblearn  # noqa: F401

    import frame

    rng = np.random.default_rng(0)
    n = 400
    income = rng.integers(20_000, 90_000, size=n)
    X = pd.DataFrame({
        "Age": rng.integers(18, 63, size=n),          # 45 addressable values
        frame.INCOME_COLUMN: income,
        "Gender_Male": rng.integers(0, 2, size=n),     # a categorical indicator
    })
    for name, op in frame.INCOME_DIGIT_TRANSFORMS:
        X[name] = op(X[frame.INCOME_COLUMN]).astype("int32")
    # ~15% positive: the imbalance SMOTENC is asked to lift to 0.5.
    y = (rng.random(n) < 0.15).astype(int)

    continuous = (frame.INCOME_COLUMN,) + tuple(n for n, _ in frame.INCOME_DIGIT_TRANSFORMS)
    a = adapter.Adapter(
        oversample="smotenc",
        validation_index=set(),  # the outside-the-loop full-training fit
        oversample_continuous_columns=continuous,
        income_column=frame.INCOME_COLUMN,
        income_digit_transforms=frame.INCOME_DIGIT_TRANSFORMS,
    )
    X_res = a.fit_transform(X, y)

    # Oversampling grew the training rows and lifted the minority share.
    assert len(X_res) > n
    assert a.resampled_y is not None and len(a.resampled_y) == len(X_res)
    minority = float(a.resampled_y.mean())
    assert minority > y.mean()

    # Every income digit column derives from its own (possibly synthetic) income.
    for name, op in frame.INCOME_DIGIT_TRANSFORMS:
        regenerated = op(X_res[frame.INCOME_COLUMN]).astype(X_res[name].dtype)
        assert (X_res[name].to_numpy() == regenerated.to_numpy()).all(), name

    # The column layout is unchanged and Age stays integer-valued (never
    # interpolated to 43.7), so its 45-value Resolution survives oversampling.
    assert list(X_res.columns) == list(X.columns)
    assert (X_res["Age"] == X_res["Age"].round()).all()


# --------------------------------------------------------------------------- #
# Conservative tuning of the depth/leaves/regularisation surface — max_bin is
# explicitly NOT part of this, it is the Resolution control with its own axis.
# --------------------------------------------------------------------------- #
def test_conservative_tuning_does_not_touch_max_bin() -> None:
    baseline = experiments.resolve("baseline")
    exp = experiments.resolve("conservative_tuning")
    # max_bin is held at the Incumbent's value: the tuning surface never touches
    # the Resolution control.
    assert exp.params["max_bin"] == baseline.params["max_bin"] == 511
    # Only params differ among the modelling fields.
    fields = (
        "frame", "model", "num_boost_round", "fold_seed", "scale",
        "target_encode", "seed_bag", "oversample",
    )
    for f in fields:
        assert getattr(exp, f) == getattr(baseline, f), f"{f} must match the Incumbent"
    assert exp.incumbent == "baseline"


def test_conservative_params_are_the_depth_leaves_regularisation_surface() -> None:
    base = experiments.LGBM_PARAMS
    con = experiments.CONSERVATIVE_PARAMS
    # Fewer leaves and real regularisation, relative to the Incumbent.
    assert con["num_leaves"] < base["num_leaves"]
    assert con.get("lambda_l1", 0.0) > 0.0
    assert con.get("lambda_l2", 0.0) > 0.0
    # The only knobs that changed are on the depth/leaves/regularisation surface;
    # max_bin (the Resolution control) is not among them.
    changed = {k for k in con if con[k] != base.get(k)} | {k for k in base if base[k] != con.get(k)}
    assert "max_bin" not in changed
    assert changed, "conservative tuning must change something"


def test_conservative_tuning_kill_is_time_boxed_at_two_hours() -> None:
    exp = experiments.resolve("conservative_tuning")
    assert exp.kill_delta == 0.0003
    assert exp.kill_time_budget_s == experiments.CONSERVATIVE_TIME_BUDGET_S == 7200.0
    # Reached the +0.0003 target within the 2h box -> survives.
    assert runner.tuning_kill_outcome(0.0004, 3600, 0.0003, 7200)["dead"] is False
    # 2h of machine time without +0.0003 -> dead.
    dead = runner.tuning_kill_outcome(0.0002, 3600, 0.0003, 7200)
    assert dead["dead"] is True
    assert dead["threshold"] == 0.0003
    assert dead["time_budget_s"] == 7200.0
    # Blowing the budget kills it even if the target was reached.
    assert runner.tuning_kill_outcome(0.0005, 7300, 0.0003, 7200)["dead"] is True


# --------------------------------------------------------------------------- #
# Every candidate is registered and resolvable through the single entry point.
# --------------------------------------------------------------------------- #
def test_tail_candidates_are_all_registered() -> None:
    for name in ("seed_bag", "smotenc", "conservative_tuning"):
        exp = experiments.resolve(name)
        assert exp.name == name
        # Each is a Paired Delta against the standing Incumbent.
        assert exp.incumbent == "baseline"
