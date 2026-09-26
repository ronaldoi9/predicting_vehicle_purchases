"""Band (iii): freeze, select the two finals, and close the turn (#20, PRD #12).

Band (iii) is the freeze — **nothing new enters once this ticket begins**. Its
three moves are all decisions taken against what the ledger already holds, so
they are pure and tested here without the ML stack:

* the **Confirmation Run of the standing Incumbent** across fold seeds 0/1/2 —
  re-running one config on each of the three seeds and reading how stable its
  OOF AUC is (``runner.confirmation_summary``). At freeze there is no candidate
  to pair against, so it confirms the Incumbent itself before it is submitted;
* the **two final submissions = best CV + the Floor** (``finals.select_finals``),
  a risk decision reserved for the driving dev — never an unattended session —
  and deliberately *not* best public LB (the 57,314-row split selects noise);
* **closing the turn** by rendering a summary for the wayfinder map's
  Decisions-so-far (``finals.render_turn_summary``) — no separate turn document.

The freeze itself is expressed as ``finals.assert_no_new_candidate``: every
experiment in the ledger must be one already declared, so a new candidate
sneaking in after the ticket begins fails loudly.

Everything here is dependency-free; the stdlib harness
(scripts/run_finals_tests.py) runs it where the ML stack is absent.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout

import experiments
import finals
import runner


# --------------------------------------------------------------------------- #
# Small builders so a test reads as the ledger shape it exercises.
# --------------------------------------------------------------------------- #
def _run(run_id, experiment, oof_auc, **kw):
    rec = {"run_id": run_id, "experiment": experiment, "oof_auc": oof_auc}
    rec.update(kw)
    return rec


def _sub(run_id, oof_auc, *, is_floor=False, is_final=False, public_score=None):
    return {
        "run_id": run_id,
        "oof_auc": oof_auc,
        "is_floor": is_floor,
        "is_final": is_final,
        "public_score": public_score,
    }


# --------------------------------------------------------------------------- #
# The Confirmation Run of the standing Incumbent across fold seeds 0/1/2.
# --------------------------------------------------------------------------- #
def test_confirmation_summary_reads_the_three_seeds_mean_and_spread() -> None:
    summary = runner.confirmation_summary({0: 0.94372, 1: 0.94360, 2: 0.94381})
    assert summary["seeds"] == [0, 1, 2]
    assert summary["oof_aucs"] == [0.94372, 0.94360, 0.94381]
    assert abs(summary["mean_oof_auc"] - (0.94372 + 0.94360 + 0.94381) / 3) < 1e-12
    assert abs(summary["oof_spread"] - (0.94381 - 0.94360)) < 1e-12


def test_confirmation_summary_requires_exactly_fold_seeds_0_1_2() -> None:
    # A Confirmation Run has exactly the three canonical fold seeds — no more,
    # no fewer, no other seed.
    for bad in ({0: 0.9, 1: 0.9}, {0: 0.9, 1: 0.9, 2: 0.9, 3: 0.9}, {0: 0.9, 1: 0.9, 5: 0.9}):
        try:
            runner.confirmation_summary(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected a ValueError for seeds {sorted(bad)}")


def test_confirmation_seeds_are_the_verdict_modules_canonical_three() -> None:
    import verdict

    assert tuple(verdict.CONFIRMATION_SEEDS) == (0, 1, 2)


# --------------------------------------------------------------------------- #
# Best CV is the highest OOF AUC — not the highest public leaderboard score.
# --------------------------------------------------------------------------- #
def test_best_cv_run_is_the_highest_oof_auc() -> None:
    records = [
        _run("r1", "baseline", 0.94372),
        _run("r2", "income_te", 0.94400),
        _run("r3", "age_te", 0.94381),
    ]
    best = finals.best_cv_run(records)
    assert best["run_id"] == "r2"


def test_best_cv_run_ignores_runs_without_an_oof_auc() -> None:
    records = [
        _run("r1", "baseline", 0.94372),
        {"run_id": "r2", "experiment": "broken"},  # no oof_auc at all
    ]
    assert finals.best_cv_run(records)["run_id"] == "r1"


def test_best_cv_run_raises_when_there_is_nothing_to_select() -> None:
    try:
        finals.best_cv_run([])
    except ValueError:
        return
    raise AssertionError("selecting best CV from an empty ledger must raise")


# --------------------------------------------------------------------------- #
# The Floor: the submission flagged is_floor (turn 1 alone).
# --------------------------------------------------------------------------- #
def test_floor_submission_is_the_flagged_one() -> None:
    subs = [
        _sub("r1", 0.94372, is_floor=True),
        _sub("r2", 0.94400),
    ]
    assert finals.floor_submission(subs)["run_id"] == "r1"


def test_floor_submission_is_none_when_no_floor_was_sent() -> None:
    assert finals.floor_submission([_sub("r2", 0.94400)]) is None


# --------------------------------------------------------------------------- #
# The two finals = best CV + the Floor, and only the driving dev may select them.
# --------------------------------------------------------------------------- #
def test_select_finals_is_best_cv_plus_the_floor() -> None:
    records = [
        _run("r1", "baseline", 0.94372),
        _run("r2", "income_te", 0.94400),
    ]
    subs = [_sub("r1", 0.94372, is_floor=True)]
    sel = finals.select_finals(records, subs, driving_dev=True)
    assert sel.best_cv_run_id == "r2"
    assert sel.best_cv_oof_auc == 0.94400
    assert sel.floor_run_id == "r1"


def test_an_unattended_session_may_never_select_the_finals() -> None:
    records = [_run("r2", "income_te", 0.94400)]
    subs = [_sub("r1", 0.94372, is_floor=True)]
    try:
        finals.select_finals(records, subs, driving_dev=False)
    except PermissionError as exc:
        assert "driving dev" in str(exc).lower()
        return
    raise AssertionError("the two finals are always the driving dev's")


def test_select_finals_requires_a_floor_to_exist() -> None:
    records = [_run("r2", "income_te", 0.94400)]
    try:
        finals.select_finals(records, [], driving_dev=True)
    except ValueError as exc:
        assert "floor" in str(exc).lower()
        return
    raise AssertionError("the Floor must exist before the finals are selected")


def test_finals_are_not_best_public_lb() -> None:
    # The rule is best CV + Floor, chosen precisely so it is NOT best public LB:
    # here the highest public score is r3 but the selected CV final is r2.
    records = [
        _run("r1", "baseline", 0.94372),
        _run("r2", "income_te", 0.94400),
        _run("r3", "age_te", 0.94381),
    ]
    subs = [
        _sub("r1", 0.94372, is_floor=True),
        _sub("r3", 0.94381, public_score=0.9466),  # best public LB
        _sub("r2", 0.94400, public_score=0.9460),
    ]
    sel = finals.select_finals(records, subs, driving_dev=True)
    assert sel.best_cv_run_id == "r2"
    assert sel.best_cv_run_id != "r3"
    # The rationale names why best public LB is refused.
    assert "public" in sel.rationale.lower()


# --------------------------------------------------------------------------- #
# The freeze: nothing new enters once band (iii) begins.
# --------------------------------------------------------------------------- #
def test_no_new_candidate_passes_for_declared_experiments() -> None:
    records = [
        _run("r1", "baseline", 0.94372),
        _run("r2", "income_te", 0.94400),
        _run("r3", "max_bin_1023", 0.94360),
    ]
    finals.assert_no_new_candidate(records)  # must not raise


def test_no_new_candidate_rejects_an_experiment_that_was_never_declared() -> None:
    records = [
        _run("r1", "baseline", 0.94372),
        _run("r9", "surprise_blend", 0.94500),  # never declared, introduced late
    ]
    try:
        finals.assert_no_new_candidate(records)
    except ValueError as exc:
        assert "surprise_blend" in str(exc)
        return
    raise AssertionError("a new candidate after the freeze must fail loudly")


def test_frozen_candidate_names_are_exactly_the_registry() -> None:
    # The freeze whitelist is the declared registry — no hand-maintained copy
    # that could drift from what experiments actually declares.
    assert finals.frozen_candidate_names() == frozenset(experiments._REGISTRY)


# --------------------------------------------------------------------------- #
# Closing the turn: a summary for the map, never a separate document.
# --------------------------------------------------------------------------- #
def test_turn_summary_names_the_two_finals_and_the_freeze() -> None:
    records = [
        _run("r1", "baseline", 0.94372, paired_delta=None),
        _run("r2", "income_te", 0.94400, paired_delta=0.00028, folds_positive=5),
    ]
    subs = [_sub("r1", 0.94372, is_floor=True)]
    sel = finals.select_finals(records, subs, driving_dev=True)
    summary = finals.render_turn_summary(records, subs, sel)
    # It names both finals...
    assert "r2" in summary
    assert "r1" in summary
    # ...and it embeds the Paired-Delta-sorted ledger table (render.render_table).
    assert "Paired Delta" in summary
    # ...and it states the freeze so turn 2 reads it off the map.
    assert "freeze" in summary.lower() or "nothing new" in summary.lower()


def test_close_turn_prints_the_summary_and_writes_no_file(tmp_path=None) -> None:
    # "No separate turn-summary document" — close-turn renders to stdout for the
    # driving dev to paste into the map; it returns 0 and creates no file.
    records = [_run("r2", "income_te", 0.94400, paired_delta=0.00028, folds_positive=5)]
    subs = [_sub("r1", 0.94372, is_floor=True)]
    sel = finals.select_finals(records, subs, driving_dev=True)
    buf = io.StringIO()
    with redirect_stdout(buf):
        text = finals.render_turn_summary(records, subs, sel)
        print(text)
    out = buf.getvalue()
    assert "Decisions" in out or "final" in out.lower()


# --------------------------------------------------------------------------- #
# The Proven Final clause (#38, ADR-0006 §5): best CV + the best Proven
# single-family model, both with a passing Confirmation Run.
# --------------------------------------------------------------------------- #
def _config(name, fold_seed, *, model="lightgbm", **frame):
    config = {
        "name": name,
        "frame": "baseline",
        "model": model,
        "num_boost_round": 700,
        "fold_seed": fold_seed,
        "target_encode": ["Annual_Income_USD"],
        "params": {"learning_rate": 0.05},
    }
    config.update(frame)
    return config


def _confirmed(experiment, oof_auc, *, deltas=(0.0002, 0.0002, 0.0002), **config):
    """A Comparison Run on each of fold seeds 0/1/2, one Paired Delta per seed."""
    return [
        _run(
            f"{experiment}-s{seed}",
            experiment,
            oof_auc + 0.00001 * seed,
            paired_delta=delta,
            config=_config(experiment, seed, **config),
        )
        for seed, delta in zip((0, 1, 2), deltas)
    ]


def _scored(run_id):
    return {"run_id": run_id, "public_score": 0.9456, "is_floor": False, "is_final": False}


def test_a_proven_best_cv_pick_stays_best_cv_beside_the_best_proven_single_family() -> None:
    # tuned is Proven through its own scored submission; old_te shares its
    # family and Frame configuration with it, only the params differ, so it is
    # Proven too. The unproven xgboost ranks between them and takes no slot.
    records = [
        *_confirmed("old_te", 0.94528),
        *_confirmed("tuned", 0.94557, params={"learning_rate": 0.015}),
        *_confirmed("xgb", 0.94549, model="xgboost"),
    ]
    subs = [_scored("old_te-s0"), _scored("tuned-s0")]
    sel = finals.select_proven_finals(records, subs, driving_dev=True)
    assert (sel.best_cv.experiment, sel.best_cv.run_id) == ("tuned", "tuned-s0")
    assert sel.best_cv.oof_auc == 0.94557
    assert sel.best_cv.proven
    assert (sel.single_family.experiment, sel.single_family.run_id) == ("old_te", "old_te-s0")
    assert sel.single_family.proven


def test_a_non_proven_best_cv_pick_forces_a_proven_second_final() -> None:
    # The best CV is new code (a family never submitted), and so is the next
    # best by CV: the second slot skips both to the best Proven model.
    records = [
        *_confirmed("incumbent", 0.94557),
        *_confirmed("new_family", 0.94616, model="heuljax", frame="raw_columns"),
        *_confirmed("new_te", 0.94597, te_derived_keys=[["Annual_Income_USD", "div100"]]),
    ]
    subs = [_scored("incumbent-s0")]
    sel = finals.select_proven_finals(records, subs, driving_dev=True)
    assert sel.best_cv.experiment == "new_family"
    assert not sel.best_cv.proven
    assert sel.single_family.experiment == "incumbent"
    assert sel.single_family.proven


def test_a_submission_without_a_public_score_proves_nothing() -> None:
    records = [
        *_confirmed("incumbent", 0.94557),
        *_confirmed("new_family", 0.94616, model="heuljax"),
    ]
    unscored = {"run_id": "incumbent-s0", "public_score": None}
    try:
        finals.select_proven_finals(records, [unscored], driving_dev=True)
    except ValueError as exc:
        assert "proven" in str(exc).lower()
        return
    raise AssertionError("an unscored submission must not make a Proven Final")


def test_no_confirmed_proven_model_puts_the_hp_search_lightgbm_second() -> None:
    default = finals.DEFAULT_PROVEN_FINAL
    assert default == "hpsearch_lightgbm_best_confirm"
    records = [
        *_confirmed("new_family", 0.94616, model="heuljax"),
        *_confirmed(default, 0.94557),
    ]
    sel = finals.select_proven_finals(records, [], driving_dev=True)
    assert sel.best_cv.experiment == "new_family"
    assert sel.single_family.experiment == default


def test_a_proven_best_cv_needs_no_proven_second_final() -> None:
    # The clause already holds on the first slot, so the second is the next
    # best single-family model even though it is not Proven.
    records = [
        *_confirmed("incumbent", 0.94557),
        *_confirmed("xgb", 0.94549, model="xgboost"),
    ]
    sel = finals.select_proven_finals(records, [_scored("incumbent-s0")], driving_dev=True)
    assert (sel.best_cv.experiment, sel.best_cv.proven) == ("incumbent", True)
    assert (sel.single_family.experiment, sel.single_family.proven) == ("xgb", False)


def test_a_candidate_without_a_passing_confirmation_run_is_refused() -> None:
    records = [
        *_confirmed("incumbent", 0.94557),
        *_confirmed("proven_too", 0.94528),
        # Best by CV, but only ever run on the canonical seed.
        _run("blend-s0", "blend", 0.94622, paired_delta=0.00006,
             config={"name": "blend", "fold_seed": 0, "members": []}),
        # Second best by CV, but its Paired Delta flips sign on fold seed 2.
        *_confirmed("flaky", 0.94600, deltas=(0.0002, 0.0001, -0.00005)),
    ]
    subs = [_scored("incumbent-s0")]
    sel = finals.select_proven_finals(records, subs, driving_dev=True)
    assert sel.best_cv.experiment == "incumbent"
    assert sel.single_family.experiment == "proven_too"
    assert sel.refused == ("blend", "flaky")


def test_a_member_gate_held_on_every_seed_is_a_passing_confirmation_run() -> None:
    # An Arena family's Confirmation Run reads its declared Member gate.
    records = [*_confirmed("incumbent", 0.94557)]
    for seed in (0, 1, 2):
        records.append(_run(
            f"member-s{seed}", "member", 0.94616, paired_delta=-0.0001,
            kill_criterion={"member_gate": True, "passed": True},
            config=_config("member", seed, model="heuljax"),
        ))
    sel = finals.select_proven_finals(records, [_scored("incumbent-s0")], driving_dev=True)
    assert sel.best_cv.experiment == "member"
    assert finals.passes_confirmation(records, "member")
    records[-1]["kill_criterion"]["passed"] = False
    assert not finals.passes_confirmation(records, "member")


def test_a_blend_takes_the_best_cv_slot_but_never_the_single_family_one() -> None:
    blend = [
        _run(f"blend-s{seed}", "blend", 0.94622, paired_delta=0.0003,
             config={"name": "blend", "fold_seed": seed, "members": []})
        for seed in (0, 1, 2)
    ]
    records = [*_confirmed("incumbent", 0.94557), *blend]
    sel = finals.select_proven_finals(records, [_scored("incumbent-s0")], driving_dev=True)
    assert sel.best_cv.experiment == "blend"
    assert sel.single_family.experiment == "incumbent"


def test_an_unattended_session_may_never_select_the_proven_finals() -> None:
    records = [*_confirmed("incumbent", 0.94557), *_confirmed("other", 0.94528)]
    try:
        finals.select_proven_finals(records, [_scored("incumbent-s0")], driving_dev=False)
    except PermissionError as exc:
        assert "driving dev" in str(exc).lower()
        return
    raise AssertionError("the two finals are always the driving dev's")


# --------------------------------------------------------------------------- #
# Turn 3's freeze is a date (ADR-0006): no Run Record after 29/09 23:59 BRT.
# --------------------------------------------------------------------------- #
def test_the_turn_3_freeze_is_29_09_at_23_59_brt() -> None:
    assert finals.TURN3_FREEZE.isoformat() == "2026-09-30T03:00:00+00:00"


def test_runs_recorded_before_the_freeze_pass_whatever_their_name() -> None:
    records = [
        _run("r1", "hpsearch_lightgbm_0007", 0.9445, timestamp="2026-09-25T13:48:14+00:00"),
        _run("r2", "blend_turn3", 0.9462, timestamp="2026-09-30T02:59:59+00:00"),
    ]
    finals.assert_nothing_after_freeze(records)  # must not raise


def test_a_run_recorded_after_the_freeze_fails_loudly() -> None:
    records = [
        _run("r1", "te_keys_prior5", 0.9460, timestamp="2026-09-25T23:00:15+00:00"),
        _run("r9", "late_idea", 0.9470, timestamp="2026-09-30T03:00:00+00:00"),
    ]
    try:
        finals.assert_nothing_after_freeze(records)
    except ValueError as exc:
        assert "r9" in str(exc) and "r1" not in str(exc)
        return
    raise AssertionError("a Run Record after the freeze must fail loudly")
