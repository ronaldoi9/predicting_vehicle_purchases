"""The Model Adapter (#15, PRD #12): fold discipline is invisible from above.

The failure mode this ticket guards is not "the function returns the wrong
value" — it is "the number comes out plausible and biased". A target encoding
fitted on the rows it is applied to does not throw; it makes the OOF score go
*up*. The leak hunt measured the same income encoding at **-0.00187** that way,
against a published **+0.00129** — a 0.003 swing from fold discipline alone —
and the Health Gate cannot see it, because the leaked score is higher and still
passes. So the one real ``pytest`` in the repo lives here: it looks directly at
the encoded column on a tiny synthetic frame and proves a target encoding never
sees its own row.

That single property test needs pandas / numpy / scikit-learn, so it runs under
pytest on a provisioned machine. Everything else in this file is the Adapter's
pure arithmetic and structural contract (the smoothing formula, the inner-seed
rule, the validation-row refusal, the income-TE Experiment declaration and the
Paired-Delta / kill-criterion bookkeeping), dependency-free so the stdlib
harness (scripts/run_adapter_tests.py) can run it where the ML stack is absent.
"""

from __future__ import annotations

import adapter
import experiments
import runner


# --------------------------------------------------------------------------- #
# THE one pytest: a target encoding never sees its own row.
# Needs the ML stack; skipped by the stdlib harness, run for real under pytest.
# --------------------------------------------------------------------------- #
def test_target_encoding_never_sees_its_own_row() -> None:
    """On a synthetic frame where every key is unique, the nested cross-fit
    encoding of every *training* row must fall back to the fold prior — because
    a unique key is never present in the inner-training folds that encode it.

    With smoothing off (``prior_weight=0``), an encoder that leaked by fitting on
    the row it encodes would return that row's own target exactly (count 1, sum
    ``y_i`` -> ``y_i``). Falling back to the prior instead of the raw target is
    the observable signature of the discipline holding.
    """
    import numpy as np
    import pandas as pd

    n = 20
    y = np.array([i % 2 for i in range(n)])  # 10 zeros, 10 ones, alternating
    X = pd.DataFrame({"key": np.arange(n)})  # every row is its own unique key
    prior = float(y.mean())  # 0.5

    ad = adapter.Adapter(
        target_encode=("key",),
        outer_fold=0,               # inner random_state = 100 + 0
        validation_index=set(),     # this fit sees no outer-validation rows
        prior_weight=0.0,           # no smoothing, so a leak would be unmistakable
    )
    out = ad.fit_transform(X, y)

    assert "key_te" in out.columns, "the target encoding must add its own column"
    te = out["key_te"].to_numpy()

    # Every training row falls back to the (inner-training) fold prior...
    assert np.allclose(te, prior), f"a unique key must encode to the prior, got {te}"
    # ...and never to its own target, which is what a leaked encoder would give.
    assert not np.allclose(te, y), "the encoding reproduced each row's own target — it leaked"


# --------------------------------------------------------------------------- #
# The smoothing formula (pure arithmetic).
# --------------------------------------------------------------------------- #
def test_smoothed_mean_pulls_a_thin_key_toward_the_prior() -> None:
    prior = 0.2
    # A key seen many times with a high rate stays near that rate.
    strong = adapter._smoothed_mean(count=1000, total=900.0, prior=prior, weight=20.0)
    assert abs(strong - 0.9) < 0.02
    # A key seen once is pulled almost all the way back to the prior.
    thin = adapter._smoothed_mean(count=1, total=1.0, prior=prior, weight=20.0)
    assert abs(thin - prior) < 0.05
    # Weight 0 is the unsmoothed empirical mean.
    assert adapter._smoothed_mean(count=4, total=3.0, prior=prior, weight=0.0) == 0.75


def test_target_encoding_contract_constants_match_the_ablation() -> None:
    # The exact parameters behind the banked ablation, so its numbers compare.
    assert adapter.INNER_SPLITS == 5
    assert adapter.INNER_SEED_BASE == 100  # inner random_state = 100 + outer_fold
    assert adapter.PRIOR_WEIGHT == 20.0


# --------------------------------------------------------------------------- #
# The validation-row refusal (the fold-boundary guard).
# --------------------------------------------------------------------------- #
def test_fit_refuses_when_a_validation_row_reaches_it() -> None:
    # A training block that overlaps the outer validation fold is refused.
    try:
        adapter._assert_no_validation_rows([0, 1, 2, 3], {3, 4, 5})
    except AssertionError as exc:
        assert "validation" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("a fit that receives a validation row must be refused")


def test_fit_requires_the_validation_index_to_be_declared() -> None:
    # ``None`` means the caller forgot to prove it is fitting inside the fold.
    try:
        adapter._assert_no_validation_rows([0, 1, 2], None)
    except AssertionError as exc:
        assert "validation index" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("an undeclared validation index must be refused")


def test_disjoint_training_rows_are_accepted() -> None:
    # The full-training test-time fit passes an empty set: no outer fold at all.
    adapter._assert_no_validation_rows([0, 1, 2], set())
    adapter._assert_no_validation_rows([0, 1, 2], {7, 8, 9})


def test_inner_seed_never_aligns_with_the_outer_fold() -> None:
    # inner random_state = 100 + outer_fold, so the inner partition of fold k is
    # seed 100+k and can never coincide with the outer seed (0..4).
    seeds = {adapter.inner_seed(k) for k in range(5)}
    assert seeds == {100, 101, 102, 103, 104}
    assert seeds.isdisjoint(range(5))


# --------------------------------------------------------------------------- #
# Scaling: a family-conditional hook, off for the tree family, and never on a
# digit child.
# --------------------------------------------------------------------------- #
def test_scaling_is_off_for_the_tree_family() -> None:
    import models

    assert models.needs_scaling("lightgbm") is False
    assert models.needs_scaling("logistic") is True


def test_scaling_targets_the_continuous_column_never_its_digit_children() -> None:
    ad = adapter.Adapter(
        scale=True,
        scale_columns=("Annual_Income_USD",),
        scale_exclude=(
            "Annual_Income_USD_div1000",
            "Annual_Income_USD_mod1000",
            "Annual_Income_USD_mod100",
        ),
    )
    targets = ad._scale_targets()
    assert targets == ["Annual_Income_USD"]
    assert "Annual_Income_USD_div1000" not in targets


# --------------------------------------------------------------------------- #
# The income-TE Experiment: a single-field change against the Incumbent.
# --------------------------------------------------------------------------- #
def test_income_te_is_a_single_field_change_against_the_incumbent() -> None:
    baseline = experiments.resolve("baseline")
    income_te = experiments.resolve("income_te")

    # It declares the Incumbent it is a Paired Delta against, and the kill line.
    assert income_te.incumbent == "baseline"
    assert income_te.kill_delta == 0.0005  # declared before running

    # The only modelling field that differs from the Incumbent is target_encode.
    assert income_te.target_encode == ("Annual_Income_USD",)
    assert baseline.target_encode == ()
    fields = ("frame", "model", "num_boost_round", "fold_seed", "scale", "params")
    for f in fields:
        assert getattr(income_te, f) == getattr(baseline, f), f"{f} must match the Incumbent"

    # The hypothesis is stated in the declaration.
    assert income_te.hypothesis
    assert "income" in income_te.hypothesis.lower()


def test_income_te_config_carries_the_target_encoding() -> None:
    cfg = experiments.resolve("income_te").as_config()
    assert cfg["target_encode"] == ["Annual_Income_USD"]
    # And it hashes differently from the Incumbent — a real, recorded change.
    base = experiments.resolve("baseline").as_config()
    assert runner.config_hash(cfg) != runner.config_hash(base)


# --------------------------------------------------------------------------- #
# Paired Delta and the kill criterion, recorded either way.
# --------------------------------------------------------------------------- #
def test_paired_delta_is_per_fold_and_counts_positive_folds() -> None:
    cand = [0.9440, 0.9441, 0.9439, 0.9442, 0.9438]
    inc = [0.9438, 0.9439, 0.9440, 0.9441, 0.9437]
    deltas = runner.fold_deltas(cand, inc)
    assert len(deltas) == 5
    mean, folds_positive = runner.paired_delta_summary(deltas)
    # Four folds improve, one regresses.
    assert folds_positive == 4
    assert abs(mean - (sum(deltas) / 5)) < 1e-12


def test_fold_deltas_requires_matching_fold_counts() -> None:
    try:
        runner.fold_deltas([0.1, 0.2, 0.3], [0.1, 0.2])
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("mismatched fold counts must be a programming error")


def test_kill_criterion_is_applied_and_recorded_either_way() -> None:
    # Declared: paired delta < +0.0005 on one seed -> the candidate is dead.
    dead = runner.kill_criterion_outcome(0.0003, 0.0005)
    assert dead["dead"] is True and dead["threshold"] == 0.0005
    survived = runner.kill_criterion_outcome(0.0006, 0.0005)
    assert survived["dead"] is False
    # Exactly at the threshold is not below it, so it survives.
    assert runner.kill_criterion_outcome(0.0005, 0.0005)["dead"] is False


def test_run_record_carries_paired_delta_and_kill_criterion() -> None:
    record = runner.build_run_record(
        experiment="income_te",
        config={"name": "income_te", "target_encode": ["Annual_Income_USD"]},
        fold_aucs=[0.9440, 0.9441, 0.9439, 0.9442, 0.9438],
        oof_auc=0.94400,
        fold_sha256="deadbeef",
        oof_path="runs/x/oof.npy",
        wall_time=1.0,
        incumbent_run_id="20260923T010000.000000Z-baseline",
        paired_delta=0.0006,
        folds_positive=4,
        fold_deltas=[0.0002, 0.0002, -0.0001, 0.0001, 0.0001],
        kill_criterion={"threshold": 0.0005, "dead": False},
        git_info={"git_sha": "abc123", "dirty": False},
    )
    assert record["paired_delta"] == 0.0006
    assert record["folds_positive"] == 4
    assert record["fold_deltas"] == [0.0002, 0.0002, -0.0001, 0.0001, 0.0001]
    assert record["kill_criterion"] == {"threshold": 0.0005, "dead": False}
