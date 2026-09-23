"""The submission path (#17): Submission Fit, the ledger, the Offset, the Floor.

Dependency-free by design, exactly like the promotion layer (#16): the Submission
Fit's *combining rule* is the mean of the five fold-model prediction vectors
(pure arithmetic), the submissions ledger is a JSON line, the CV->LB Offset is a
subtraction, the leak-hunt trigger is a comparison against three times the
leaderboard noise floor, and the unattended-slot policy is a count over the
ledger. None of it needs the ML stack, the gitignored CSVs or the network — the
actual Kaggle call and the mean-of-folds fit itself are the untested seam the
Floor submission covers live (PRD #12: proven end to end by a 0.50000 constant
submission). So all the logic below is pinned here and runs under the stdlib
harness as well as pytest.
"""

from __future__ import annotations

import json
from pathlib import Path

import submission

SRC = Path(__file__).resolve().parent.parent / "src"


# --------------------------------------------------------------------------- #
# The Submission Fit is the mean of the five fold models, never a refit.
# --------------------------------------------------------------------------- #
def test_submission_fit_is_the_mean_of_the_five_fold_models() -> None:
    fold_preds = [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5],
    ]
    mean = list(submission.mean_of_fold_predictions(fold_preds))
    assert mean == [0.5, 0.3, 0.7]


def test_submission_fit_refuses_anything_but_five_fold_vectors() -> None:
    # A single vector is exactly what a full-data refit would produce: refused,
    # because the CV->LB Offset must compare the SAME models on both sides.
    try:
        submission.mean_of_fold_predictions([[0.1, 0.2, 0.3]])
    except ValueError as exc:
        assert "refit" in str(exc) or "5" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("a full-data refit (one vector) must be refused")

    # Four fold vectors is a missing model, also refused.
    try:
        submission.mean_of_fold_predictions([[0.0]] * 4)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("fewer than five fold models must be refused")


def test_submission_fit_refuses_ragged_fold_vectors() -> None:
    try:
        submission.mean_of_fold_predictions([[0.0, 0.0], [0.0], [0.0], [0.0], [0.0]])
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("fold vectors of different lengths must be refused")


# --------------------------------------------------------------------------- #
# A submission is recorded at send time with a null public score.
# --------------------------------------------------------------------------- #
def test_submission_recorded_at_send_time_with_null_public_score(tmp_path) -> None:
    line = submission.build_submission_line(
        run_id="20260923T120000.000000Z-baseline",
        oof_auc=0.94372,
        message="Floor: turn 1 alone",
    )
    assert line["run_id"] == "20260923T120000.000000Z-baseline"
    assert line["oof_auc"] == 0.94372
    assert line["public_score"] is None
    assert line["cv_lb_offset"] is None
    assert line["competition"] == submission.COMPETITION_SLUG

    ledger = tmp_path / "submissions.jsonl"
    submission.append_submission(line, path=ledger)
    written = json.loads(ledger.read_text().splitlines()[0])
    assert written["run_id"] == line["run_id"]
    assert written["public_score"] is None


def test_the_floor_is_marked_and_is_never_a_final() -> None:
    floor = submission.build_submission_line(
        run_id="r-baseline", oof_auc=0.94372, message="Floor", is_floor=True
    )
    assert floor["is_floor"] is True
    assert floor["is_final"] is False


# --------------------------------------------------------------------------- #
# A separate command fills in the public score later; the Offset is recorded.
# --------------------------------------------------------------------------- #
def test_cv_lb_offset_is_public_minus_oof() -> None:
    assert submission.cv_lb_offset(0.94372, 0.94350) == pytest_approx(-0.00022)
    assert submission.cv_lb_offset(0.94372, 0.94372) == 0.0


def test_record_score_fills_the_score_later_and_records_the_offset(tmp_path) -> None:
    ledger = tmp_path / "submissions.jsonl"
    submission.append_submission(
        submission.build_submission_line(
            run_id="r1", oof_auc=0.94372, message="Floor"
        ),
        path=ledger,
    )
    # Scoring was slow; the score arrives later and is filled in.
    reading = submission.record_score_for("r1", 0.94350, path=ledger)

    filled = submission.find_submission("r1", path=ledger)
    assert filled["public_score"] == 0.94350
    assert filled["cv_lb_offset"] == pytest_approx(-0.00022)
    assert reading.offset == pytest_approx(-0.00022)
    assert reading.leak_hunt is False


def test_record_score_refuses_an_unknown_run(tmp_path) -> None:
    ledger = tmp_path / "submissions.jsonl"
    try:
        submission.record_score_for("nope", 0.9, path=ledger)
    except (KeyError, ValueError) as exc:
        assert "nope" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("scoring a run that was never submitted must fail")


def test_find_submission_last_write_wins(tmp_path) -> None:
    ledger = tmp_path / "submissions.jsonl"
    submission.append_submission(
        submission.build_submission_line(run_id="r1", oof_auc=0.9, message="sent"),
        path=ledger,
    )
    submission.record_score_for("r1", 0.88, path=ledger)
    assert submission.find_submission("r1", path=ledger)["public_score"] == 0.88


# --------------------------------------------------------------------------- #
# A divergence beyond 3x the leaderboard noise floor is a leak hunt, not a knob.
# --------------------------------------------------------------------------- #
def test_leak_hunt_threshold_is_three_times_the_lb_noise_floor() -> None:
    assert submission.LB_NOISE_FLOOR == 0.0002
    assert submission.LEAK_HUNT_THRESHOLD == pytest_approx(3 * 0.0002)


def test_divergence_beyond_three_times_the_noise_floor_is_a_leak_hunt() -> None:
    stable = submission.offset_reading(0.94372, 0.94340)  # |offset| = 0.00032
    assert stable.leak_hunt is False
    assert "thermometer" in stable.message.lower()

    diverged = submission.offset_reading(0.94372, 0.94300)  # |offset| = 0.00072
    assert diverged.leak_hunt is True
    assert "leak hunt" in diverged.message.lower()
    # It is surfaced as an instrument problem, never as something to tune on.
    assert "tuning" not in diverged.message.lower() or "not" in diverged.message.lower()


def test_exactly_three_times_the_noise_floor_is_not_yet_a_leak_hunt() -> None:
    # The trigger is a strict inequality: an offset exactly at 3x the floor is
    # still stable; only a divergence beyond it is a leak hunt. Tested on the
    # threshold directly so a float-cancellation artefact of the subtraction
    # cannot masquerade as the boundary.
    assert submission.is_leak_hunt(submission.LEAK_HUNT_THRESHOLD) is False
    assert submission.is_leak_hunt(-submission.LEAK_HUNT_THRESHOLD) is False
    assert submission.is_leak_hunt(submission.LEAK_HUNT_THRESHOLD + 1e-6) is True
    assert submission.is_leak_hunt(-(submission.LEAK_HUNT_THRESHOLD + 1e-6)) is True


# --------------------------------------------------------------------------- #
# Unattended runs: one slot per day, only after a Confirmation Run, never final.
# --------------------------------------------------------------------------- #
def test_unattended_never_spends_a_final_submission() -> None:
    try:
        submission.assert_unattended_allowed(
            confirmation_passed=True, is_final=True, submissions=[]
        )
    except PermissionError as exc:
        assert "final" in str(exc).lower()
    else:  # pragma: no cover
        raise AssertionError("an unattended session must never spend a final")


def test_unattended_requires_a_passing_confirmation_run() -> None:
    try:
        submission.assert_unattended_allowed(
            confirmation_passed=False, is_final=False, submissions=[]
        )
    except PermissionError as exc:
        assert "confirmation" in str(exc).lower()
    else:  # pragma: no cover
        raise AssertionError("an unattended submission needs a Confirmation Run")


def test_unattended_may_spend_at_most_one_slot_per_day() -> None:
    from datetime import datetime, timezone

    day = datetime(2026, 9, 25, 9, 0, 0, tzinfo=timezone.utc)
    # No unattended slot spent yet today: allowed.
    submission.assert_unattended_allowed(
        confirmation_passed=True, is_final=False, submissions=[], when=day
    )

    already = [
        submission.build_submission_line(
            run_id="earlier",
            oof_auc=0.9,
            message="unattended offset check",
            unattended=True,
            when=datetime(2026, 9, 25, 3, 0, 0, tzinfo=timezone.utc),
        )
    ]
    try:
        submission.assert_unattended_allowed(
            confirmation_passed=True, is_final=False, submissions=already, when=day
        )
    except PermissionError as exc:
        assert "one" in str(exc).lower() or "day" in str(exc).lower()
    else:  # pragma: no cover
        raise AssertionError("a second unattended slot in one day must be refused")


def test_unattended_slot_yesterday_does_not_block_today() -> None:
    from datetime import datetime, timezone

    yesterday = [
        submission.build_submission_line(
            run_id="y",
            oof_auc=0.9,
            message="unattended",
            unattended=True,
            when=datetime(2026, 9, 24, 12, 0, 0, tzinfo=timezone.utc),
        )
    ]
    # A slot spent yesterday leaves today's slot free.
    submission.assert_unattended_allowed(
        confirmation_passed=True,
        is_final=False,
        submissions=yesterday,
        when=datetime(2026, 9, 25, 12, 0, 0, tzinfo=timezone.utc),
    )


# --------------------------------------------------------------------------- #
# submission is the sole writer of the submissions ledger.
# --------------------------------------------------------------------------- #
def test_submission_is_the_sole_writer_of_the_submissions_ledger() -> None:
    writers = {
        path.name for path in SRC.glob("*.py") if "submissions.jsonl" in path.read_text()
    }
    assert writers == {"submission.py"}, writers


# --------------------------------------------------------------------------- #
# A tiny stand-in for pytest.approx so the stdlib harness needs no pytest.
# --------------------------------------------------------------------------- #
class _Approx:
    def __init__(self, expected: float, tol: float = 1e-9) -> None:
        self.expected = expected
        self.tol = tol

    def __eq__(self, other: object) -> bool:
        return abs(float(other) - self.expected) <= self.tol  # type: ignore[arg-type]

    def __repr__(self) -> str:  # pragma: no cover
        return f"~{self.expected}"


def pytest_approx(expected: float, tol: float = 1e-9) -> _Approx:
    return _Approx(expected, tol)
