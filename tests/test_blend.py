"""The Blend harness: the rule, the weight search, and the leak (#23).

Dependency-free where the module claims it is: the rank transform, the pure
AUC, the weighted combination and the greedy weight search never import
pandas/lightgbm/sklearn, so they are tested here without the ML stack, exactly
like ``verdict`` and the pure half of ``runner``. ``honest_blend_oof`` and
``naive_blend_oof`` are numpy-only (not the full ML stack) and are exercised
with small synthetic vectors; :mod:`scripts.run_blend_tests` reports them SKIP
where numpy itself is absent.

The two leak-protocol functions are tested against a synthetic case
constructed so the honest and naive numbers provably differ: a Member whose
vector happens to correlate with the fold assignment itself would let a
weight fitted on all rows exploit that correlation, while a weight chosen on
the *other* folds cannot see fold k's own rows to exploit.
"""

from __future__ import annotations

import blend
import runner


# --------------------------------------------------------------------------- #
# The rank transform: fractional rank in [0, 1], average rank for ties.
# --------------------------------------------------------------------------- #
def test_rank_transform_of_distinct_values_is_evenly_spaced() -> None:
    ranks = blend.rank_transform([30.0, 10.0, 20.0, 40.0])
    assert ranks == [2 / 3, 0.0, 1 / 3, 1.0]


def test_rank_transform_shares_the_average_rank_across_ties() -> None:
    # Two lowest values tie for ranks 1-2 (average 1.5); highest two tie for
    # ranks 3-4 (average 3.5). Normalised over n=4: (1.5-1)/3, (3.5-1)/3.
    ranks = blend.rank_transform([5.0, 5.0, 9.0, 9.0])
    assert ranks[0] == ranks[1]
    assert ranks[2] == ranks[3]
    assert abs(ranks[0] - (1.5 - 1) / 3) < 1e-12
    assert abs(ranks[2] - (3.5 - 1) / 3) < 1e-12


def test_rank_transform_is_scale_free() -> None:
    # The whole point of the rule: two vectors on wildly different scales but
    # the same ORDER rank-transform to the identical vector.
    small_scale = [0.01, 0.02, 0.03]
    huge_scale = [100.0, 5000.0, 999999.0]
    assert blend.rank_transform(small_scale) == blend.rank_transform(huge_scale)


def test_rank_transform_of_a_single_value_is_zero() -> None:
    assert blend.rank_transform([7.0]) == [0.0]


# --------------------------------------------------------------------------- #
# The pure (sklearn-free) AUC.
# --------------------------------------------------------------------------- #
def test_pure_auc_of_a_perfect_separator_is_one() -> None:
    y = [0, 0, 0, 1, 1, 1]
    scores = [0.1, 0.2, 0.3, 0.8, 0.9, 0.95]
    assert abs(blend.pure_auc(y, scores) - 1.0) < 1e-12


def test_pure_auc_of_an_inverted_separator_is_zero() -> None:
    y = [0, 0, 0, 1, 1, 1]
    scores = [0.9, 0.8, 0.95, 0.1, 0.2, 0.3]
    assert abs(blend.pure_auc(y, scores) - 0.0) < 1e-12


def test_pure_auc_of_random_looking_ties_is_one_half() -> None:
    y = [0, 1, 0, 1]
    scores = [0.5, 0.5, 0.5, 0.5]  # no discrimination at all
    assert abs(blend.pure_auc(y, scores) - 0.5) < 1e-12


def test_pure_auc_matches_sklearn_on_a_larger_random_case() -> None:
    import random

    random.seed(0)
    y = [random.randint(0, 1) for _ in range(500)]
    scores = [random.random() for _ in range(500)]
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return  # SKIP: sklearn absent; the sklearn-free claim can't be checked here
    expected = roc_auc_score(y, scores)
    assert abs(blend.pure_auc(y, scores) - expected) < 1e-9


def test_pure_auc_rejects_mismatched_lengths() -> None:
    try:
        blend.pure_auc([0, 1], [0.1, 0.2, 0.3])
    except ValueError:
        return
    raise AssertionError("mismatched y/scores lengths must raise")


def test_pure_auc_rejects_a_single_class() -> None:
    try:
        blend.pure_auc([1, 1, 1], [0.1, 0.2, 0.3])
    except ValueError:
        return
    raise AssertionError("AUC is undefined with only one class present")


# --------------------------------------------------------------------------- #
# The weighted combination.
# --------------------------------------------------------------------------- #
def test_weighted_combine_is_the_plain_mean_at_equal_weights() -> None:
    combined = blend.weighted_combine([[0.0, 1.0], [1.0, 0.0]], [1.0, 1.0])
    assert list(combined) == [0.5, 0.5]


def test_weighted_combine_normalises_weights_that_do_not_sum_to_one() -> None:
    # Raw vote counts from the greedy search (e.g. [3, 1]) pass through
    # unnormalised; weighted_combine must normalise them itself.
    combined = blend.weighted_combine([[1.0, 1.0], [0.0, 0.0]], [3.0, 1.0])
    assert abs(combined[0] - 0.75) < 1e-12


def test_weighted_combine_rejects_a_non_positive_weight_total() -> None:
    try:
        blend.weighted_combine([[1.0], [2.0]], [0.0, 0.0])
    except ValueError:
        return
    raise AssertionError("a zero weight total must raise")


def test_weighted_combine_rejects_a_vector_weight_count_mismatch() -> None:
    try:
        blend.weighted_combine([[1.0], [2.0]], [1.0])
    except ValueError:
        return
    raise AssertionError("mismatched vector/weight counts must raise")


# --------------------------------------------------------------------------- #
# The greedy weight search: it must at least match, never dilute below, the
# single best Member's own AUC.
# --------------------------------------------------------------------------- #
def test_greedy_hillclimb_prefers_a_perfect_member_over_a_noisy_one() -> None:
    y = [0, 0, 0, 0, 1, 1, 1, 1]
    perfect = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
    noise = [0.9, 0.1, 0.8, 0.2, 0.1, 0.9, 0.2, 0.8]  # AUC 0.5-ish, no signal
    weights = blend.greedy_hillclimb_weights([perfect, noise], y, max_rounds=20)
    assert weights[0] > weights[1]
    combined = blend.weighted_combine([perfect, noise], weights)
    assert blend.pure_auc(y, combined) >= blend.pure_auc(y, perfect) - 1e-12


def test_greedy_hillclimb_falls_back_to_uniform_when_nothing_ever_improves() -> None:
    y = [0, 1, 0, 1]
    coin_flip_a = [0.5, 0.5, 0.5, 0.5]
    coin_flip_b = [0.5, 0.5, 0.5, 0.5]
    weights = blend.greedy_hillclimb_weights([coin_flip_a, coin_flip_b], y, max_rounds=10)
    assert weights == [0.5, 0.5]


def test_greedy_hillclimb_weights_sum_to_one() -> None:
    y = [0, 0, 1, 1, 0, 1]
    a = [0.1, 0.4, 0.6, 0.9, 0.2, 0.7]
    b = [0.3, 0.1, 0.8, 0.4, 0.6, 0.9]
    weights = blend.greedy_hillclimb_weights([a, b], y, max_rounds=15)
    assert abs(sum(weights) - 1.0) < 1e-9


def test_greedy_hillclimb_is_deterministic() -> None:
    y = [0, 0, 0, 1, 1, 1]
    a = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    b = [0.2, 0.1, 0.3, 0.9, 0.7, 0.8]
    c = [0.9, 0.8, 0.7, 0.1, 0.2, 0.3]  # inverted — should never be picked
    w1 = blend.greedy_hillclimb_weights([a, b, c], y, max_rounds=20)
    w2 = blend.greedy_hillclimb_weights([a, b, c], y, max_rounds=20)
    assert w1 == w2
    assert w1[2] == 0.0


def test_greedy_hillclimb_needs_at_least_one_member() -> None:
    try:
        blend.greedy_hillclimb_weights([], [0, 1])
    except ValueError:
        return
    raise AssertionError("an empty member list must raise")


# --------------------------------------------------------------------------- #
# The kill criterion is an AND of magnitude and folds-positive, not an OR.
# --------------------------------------------------------------------------- #
def test_blend_kill_outcome_dead_on_magnitude_alone() -> None:
    outcome = blend.blend_kill_outcome(0.0001, 5, kill_delta=0.0003, min_folds_positive=4)
    assert outcome["dead"] is True


def test_blend_kill_outcome_dead_on_folds_positive_alone() -> None:
    outcome = blend.blend_kill_outcome(0.0005, 3, kill_delta=0.0003, min_folds_positive=4)
    assert outcome["dead"] is True


def test_blend_kill_outcome_survives_only_when_both_clear() -> None:
    outcome = blend.blend_kill_outcome(0.0005, 4, kill_delta=0.0003, min_folds_positive=4)
    assert outcome["dead"] is False


# --------------------------------------------------------------------------- #
# The leak: honest (per-outer-fold nested) vs naive (in-sample) weight
# selection, on a case built so they provably diverge. numpy-only (not the
# full ML stack); reported SKIP by the stdlib harness where numpy is absent.
# --------------------------------------------------------------------------- #
def test_naive_weights_can_exploit_a_fold_shaped_member_but_honest_cannot() -> None:
    try:
        import numpy as np
    except ImportError:
        return  # SKIP under scripts/run_blend_tests.py: numpy absent

    rng = np.random.default_rng(0)
    n = 400
    fold = np.tile(np.arange(5), n // 5)
    y = rng.integers(0, 2, size=n).astype(float)

    # A genuine, fold-independent signal every Member sees a noisy copy of.
    signal = y + rng.normal(scale=0.6, size=n)
    honest_member = list(blend.rank_transform(signal.tolist()))

    # A "cheating" member that is uninformative in general but happens to spike
    # for the positive class *only inside fold 0* — a fold-shaped correlation a
    # weight fitted on ALL rows (including fold 0) can exploit, but a weight
    # fitted on the other four folds (never seeing fold 0's rows) cannot.
    cheat_raw = rng.normal(scale=1.0, size=n)
    cheat_raw[(fold == 0) & (y == 1)] += 6.0
    cheat_member = list(blend.rank_transform(cheat_raw.tolist()))

    rank_vectors = [honest_member, cheat_member]
    honest_oof, weights_per_fold = blend.honest_blend_oof(
        rank_vectors, y.tolist(), fold.tolist(), n_folds=5, max_rounds=30
    )
    naive_oof, naive_weights = blend.naive_blend_oof(rank_vectors, y.tolist(), max_rounds=30)

    honest_auc = blend.pure_auc(y.tolist(), list(honest_oof))
    naive_auc = blend.pure_auc(y.tolist(), list(naive_oof))

    # Fold 0's weights cannot lean on the cheat Member — it was invisible to
    # the training rows fold 0's weights were chosen on.
    assert weights_per_fold[0][1] < naive_weights[1] + 1e-6
    # And the leaked (naive) number reads at least as high as the honest one —
    # "the difference is the size of the trap".
    assert naive_auc >= honest_auc - 1e-9


def test_honest_blend_oof_covers_every_row_exactly_once() -> None:
    try:
        import numpy as np
    except ImportError:
        return  # SKIP: numpy absent

    y = [0, 1] * 10
    fold = [i % 5 for i in range(20)]
    a = blend.rank_transform([float(i) for i in range(20)])
    b = blend.rank_transform([float(20 - i) for i in range(20)])
    oof, weights_per_fold = blend.honest_blend_oof([a, b], y, fold, n_folds=5, max_rounds=5)
    assert len(oof) == 20
    assert len(weights_per_fold) == 5
    assert not np.isnan(oof).any()


# --------------------------------------------------------------------------- #
# Resolving Members and picking the best one — pure ledger reads.
# --------------------------------------------------------------------------- #
def _rec(run_id, experiment, oof_auc, fold_aucs=None, config=None):
    return {
        "run_id": run_id,
        "experiment": experiment,
        "oof_auc": oof_auc,
        "fold_aucs": fold_aucs or [oof_auc] * 5,
        "config": config or {"fold_seed": 0},
    }


def test_best_member_record_is_the_highest_oof_auc() -> None:
    records = [_rec("r1", "a", 0.940), _rec("r2", "b", 0.945), _rec("r3", "c", 0.942)]
    assert blend.best_member_record(records)["run_id"] == "r2"


def test_best_member_record_rejects_an_empty_set() -> None:
    try:
        blend.best_member_record([])
    except ValueError:
        return
    raise AssertionError("selecting a best Member from nothing must raise")


def test_resolve_members_raises_naming_every_missing_member() -> None:
    import json
    from pathlib import Path

    ledger = Path(runner.RUNS_LEDGER.parent, "test_blend_ledger.jsonl")
    ledger.parent.mkdir(parents=True, exist_ok=True)
    try:
        ledger.write_text(
            json.dumps(
                {
                    "run_id": "r1",
                    "experiment": "present_member",
                    "config": {"fold_seed": 0},
                    "oof_auc": 0.94,
                    "fold_aucs": [0.94] * 5,
                }
            )
            + "\n"
        )
        # Patch runner's ledger path for the duration of this call only.
        original = runner.RUNS_LEDGER
        runner.RUNS_LEDGER = ledger
        try:
            config = blend.BlendConfig(
                name="t",
                hypothesis="t",
                members=("present_member", "absent_member"),
            )
            try:
                blend.resolve_members(config)
            except ValueError as exc:
                assert "absent_member" in str(exc)
                assert "present_member" not in str(exc)
                return
            raise AssertionError("a missing Member must raise")
        finally:
            runner.RUNS_LEDGER = original
    finally:
        ledger.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# The declared registry (blends.py) resolves against the real ledger — proves
# the harness has real vectors to exercise on, per #23's own framing.
# --------------------------------------------------------------------------- #
def test_declared_blend_all_members_resolves_against_the_real_ledger() -> None:
    import blends

    try:
        records = blend.resolve_members(blends.BLEND_ALL_MEMBERS)
    except ImportError:
        return  # SKIP: the ML stack is absent in this environment
    except FileNotFoundError:
        return  # SKIP: the gitignored ledger/oof files are absent
    assert len(records) == len(blends.BLEND_ALL_MEMBERS.members)
    assert blend.best_member_record(records) is not None
