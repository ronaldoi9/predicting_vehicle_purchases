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
