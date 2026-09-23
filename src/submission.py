"""The Submission Fit, the CSV, the Kaggle call and the submissions ledger.

This is the submission path that closes band (i) of PRD #12: the **Floor**
submitted, and the **CV->LB Offset** measured. The pieces, and why each is the
way it is:

* **The Submission Fit is the mean of the five fold models — never a full-data
  refit.** The five models already exist once the Comparison Run has finished,
  ROC AUC is a rank metric so averaging their probabilities is legitimate, and
  — the reason that matters — the CV->LB Offset then compares *the same set of
  models* on both sides of the gap. A refit would change the models the OOF AUC
  was measured on, so :func:`mean_of_fold_predictions` refuses anything but five
  fold vectors.

* **A submission is recorded at the moment it is sent, with a null public
  score.** A submission that exists on Kaggle and not in the ledger is how one
  of the ten daily slots gets burned without knowing which config spent it. A
  separate command (:func:`record_score`) fills the score in later if scoring
  was slow.

* **The public leaderboard is a thermometer, not a ranking.** Its 57,314 rows
  cannot resolve 0.0002, so its job is to show whether the Offset is *stable*. A
  divergence beyond three times the leaderboard noise floor
  (:data:`LEAK_HUNT_THRESHOLD`) is surfaced as a **leak hunt**, not acted on as
  a tuning signal.

* **Unattended sessions may spend at most one slot per day, only after a passing
  Confirmation Run, and never a final submission** — the two finals are always
  the driving dev's, mirroring the manual promotion gate.

``submission`` is the **sole writer of** ``ledger/submissions.jsonl`` — the path
literal lives here and nowhere else — kept separate from the runs ledger because
a submission carries a public score and no Paired Delta. The Kaggle call and the
mean-of-folds fit itself are the one untested seam: the submission module touches
the network and spends a limited slot, so it is covered by the Floor submission
live (the path is proven end to end by a 0.50000 constant submission, #10), not
by a mock. Everything below the network — the combining rule, the ledger schema,
the Offset, the leak-hunt trigger and the unattended policy — is pure and tested.

Heavy imports (numpy, pandas, the ML stack, the runner's Comparison Run path)
are deferred into :func:`submit` so the module and its pure helpers import by
bare name in any environment.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from data import N_FOLDS


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


# submission is the sole writer of the submissions ledger; this literal lives
# here and nowhere else.
SUBMISSIONS_LEDGER = _repo_root() / "ledger" / "submissions.jsonl"

# The competition, proven end to end by a constant-prediction submission scoring
# exactly 0.50000 (#10). CLI 2.2.4, OAuth credentials already provisioned.
COMPETITION_SLUG = "playground-series-s6e9"

# The public leaderboard's noise floor: its 57,314 rows cannot resolve 0.0002.
# This is a distinct floor from the Paired-Delta one in :mod:`verdict` — naming
# which one is meant is required (CONTEXT.md, "Noise Floor").
LB_NOISE_FLOOR = 0.0002

# A CV->LB divergence beyond three times the leaderboard noise floor is a leak
# hunt, not a tuning signal.
DIVERGENCE_FACTOR = 3
LEAK_HUNT_THRESHOLD = DIVERGENCE_FACTOR * LB_NOISE_FLOOR  # 0.0006


# --------------------------------------------------------------------------- #
# The Submission Fit: the mean of the five fold models (pure arithmetic).
# --------------------------------------------------------------------------- #
def mean_of_fold_predictions(fold_preds: Iterable[Sequence[float]], n_folds: int = N_FOLDS):
    """Average the ``n_folds`` fold-model prediction vectors element-wise.

    The Submission Fit is the mean of the five fold models, **not a full-data
    refit** — so the CV->LB Offset compares the same models on both sides of the
    gap. Anything but exactly ``n_folds`` vectors is refused: one vector is what
    a refit would look like, fewer than five is a missing model.

    Uses numpy when it is present (the real path, on large float arrays) and
    falls back to pure Python otherwise, so the combining rule is testable
    without the ML stack.
    """
    preds = [list(p) for p in fold_preds]
    if len(preds) != n_folds:
        raise ValueError(
            f"the Submission Fit is the mean of the {n_folds} fold models, not a "
            f"full-data refit; got {len(preds)} prediction vector(s)"
        )
    lengths = {len(p) for p in preds}
    if len(lengths) != 1:
        raise ValueError(
            f"the {n_folds} fold prediction vectors differ in length: {sorted(lengths)}"
        )
    try:
        import numpy as np
    except ImportError:
        return [sum(col) / n_folds for col in zip(*preds)]
    return np.mean(np.stack([np.asarray(p, dtype=float) for p in preds]), axis=0)


# --------------------------------------------------------------------------- #
# The CV->LB Offset and the leak-hunt trigger.
# --------------------------------------------------------------------------- #
def cv_lb_offset(oof_auc: float, public_score: float) -> float:
    """The CV->LB Offset: public leaderboard score minus OOF AUC.

    Its *stability over time* is what the leaderboard is read for; its absolute
    value ranks nothing (CONTEXT.md, "CV->LB Offset"). A negative offset — the
    common case — means the public split scored below CV.
    """
    return float(public_score) - float(oof_auc)


@dataclass(frozen=True)
class OffsetReading:
    """A read of one submission's CV->LB Offset, and what to do about it."""

    offset: float
    leak_hunt: bool
    message: str


def is_leak_hunt(offset: float) -> bool:
    """True iff the offset diverges beyond three times the LB noise floor."""
    return abs(offset) > LEAK_HUNT_THRESHOLD


def offset_reading(oof_auc: float, public_score: float) -> OffsetReading:
    """Read the Offset and classify it: stable thermometer, or a leak hunt.

    A divergence beyond three times the leaderboard noise floor is surfaced as a
    **leak hunt** — an instrument problem to investigate — and explicitly *not*
    as a tuning signal to act on. Within the band, the leaderboard is doing its
    only job: a thermometer confirming the Offset is stable.
    """
    offset = cv_lb_offset(oof_auc, public_score)
    if is_leak_hunt(offset):
        message = (
            f"CV->LB divergence {offset:+.5f} exceeds {DIVERGENCE_FACTOR}x the "
            f"leaderboard noise floor ({LEAK_HUNT_THRESHOLD:.4f}): this is a LEAK "
            "HUNT, not a tuning signal. Investigate the instrument (a leak, a "
            "fold-boundary violation, a train/test skew) before trusting any "
            "score built on it."
        )
        return OffsetReading(offset=offset, leak_hunt=True, message=message)
    message = (
        f"CV->LB offset {offset:+.5f} is within {DIVERGENCE_FACTOR}x the leaderboard "
        f"noise floor ({LEAK_HUNT_THRESHOLD:.4f}): the Offset is stable. The public "
        "leaderboard is a thermometer, not a ranking."
    )
    return OffsetReading(offset=offset, leak_hunt=False, message=message)


# --------------------------------------------------------------------------- #
# The submissions ledger (submission is its sole writer).
# --------------------------------------------------------------------------- #
def build_submission_line(
    *,
    run_id: str,
    oof_auc: float,
    message: str,
    public_score: float | None = None,
    unattended: bool = False,
    is_final: bool = False,
    is_floor: bool = False,
    competition: str = COMPETITION_SLUG,
    when: datetime | None = None,
) -> dict[str, Any]:
    """One submission line, recorded at send time with a null public score.

    ``public_score`` is ``None`` at the moment of sending; :func:`record_score`
    fills it in later and, with it, the CV->LB Offset. The line always carries
    the run id it was fit from, so a burned daily slot always has a config
    attached to it.
    """
    when = when or datetime.now(timezone.utc)
    offset = None if public_score is None else cv_lb_offset(oof_auc, public_score)
    return {
        "run_id": run_id,
        "competition": competition,
        "message": message,
        "timestamp": when.isoformat(),
        "oof_auc": float(oof_auc),
        "public_score": None if public_score is None else float(public_score),
        "cv_lb_offset": offset,
        "unattended": bool(unattended),
        "is_final": bool(is_final),
        "is_floor": bool(is_floor),
    }


def append_submission(line: Mapping[str, Any], path: Path | None = None) -> None:
    """Append one submission as a single JSON line (jsonl never git-conflicts)."""
    path = Path(path) if path is not None else SUBMISSIONS_LEDGER
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(line) + "\n")


def load_submissions(path: Path | None = None) -> list[dict[str, Any]]:
    """Read every submission line, in the order it was written."""
    path = Path(path) if path is not None else SUBMISSIONS_LEDGER
    if not path.exists():
        return []
    lines: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if raw:
            lines.append(json.loads(raw))
    return lines


def find_submission(run_id: str, path: Path | None = None) -> dict[str, Any] | None:
    """The submission for this run id, or ``None``. Last write wins, so a
    score filled in later resolves over the null-score line it updated."""
    match: dict[str, Any] | None = None
    for line in load_submissions(path):
        if line.get("run_id") == run_id:
            match = line
    return match


def record_score_for(
    run_id: str,
    public_score: float,
    path: Path | None = None,
    when: datetime | None = None,
) -> OffsetReading:
    """Fill in a submission's public score after the fact and record the Offset.

    Appends an updated line (last-write-wins) carrying the public score and the
    computed CV->LB Offset, and returns the :class:`OffsetReading` so the caller
    can surface a divergence as a leak hunt.
    """
    existing = find_submission(run_id, path=path)
    if existing is None:
        raise ValueError(
            f"no submission recorded for run {run_id!r}; a score can only be "
            "filled in for a submission that was sent"
        )
    updated = dict(existing)
    updated["public_score"] = float(public_score)
    updated["cv_lb_offset"] = cv_lb_offset(existing["oof_auc"], public_score)
    updated["scored_at"] = (when or datetime.now(timezone.utc)).isoformat()
    append_submission(updated, path=path)
    return offset_reading(existing["oof_auc"], public_score)


# --------------------------------------------------------------------------- #
# The unattended-submission policy.
# --------------------------------------------------------------------------- #
def _utc_date(timestamp: str) -> str:
    """The UTC calendar date (YYYY-MM-DD) of an ISO-8601 timestamp."""
    return datetime.fromisoformat(timestamp).astimezone(timezone.utc).date().isoformat()


def unattended_slots_used_on(
    submissions: Iterable[Mapping[str, Any]], day: str
) -> int:
    """How many unattended slots were spent on the given UTC calendar date."""
    return sum(
        1
        for s in submissions
        if s.get("unattended") and _utc_date(s["timestamp"]) == day
    )


def assert_unattended_allowed(
    *,
    confirmation_passed: bool,
    is_final: bool,
    submissions: Iterable[Mapping[str, Any]],
    when: datetime | None = None,
) -> None:
    """Enforce the unattended-submission policy, or raise ``PermissionError``.

    An agent session may spend **at most one slot per day**, **only after a
    passing Confirmation Run**, and **never a final submission** — the two
    finals are always the driving dev's, mirroring the manual promotion gate.
    """
    if is_final:
        raise PermissionError(
            "an unattended session may never spend a final submission; the two "
            "finals are always the driving dev's (best CV + the Floor)"
        )
    if not confirmation_passed:
        raise PermissionError(
            "an unattended submission requires a passing Confirmation Run first; "
            "routine offset measurement does not get to skip the pairing check"
        )
    when = when or datetime.now(timezone.utc)
    day = when.astimezone(timezone.utc).date().isoformat()
    used = unattended_slots_used_on(submissions, day)
    if used >= 1:
        raise PermissionError(
            f"an unattended session may spend at most one slot per day; {used} "
            f"was already spent on {day}. Wait for the next UTC day."
        )


# --------------------------------------------------------------------------- #
# The Submission Fit + Kaggle call (the untested seam; needs the ML stack).
# --------------------------------------------------------------------------- #
def submit(
    run_id: str,
    message: str,
    *,
    unattended: bool = False,
    is_final: bool = False,
    is_floor: bool = False,
    confirmation_passed: bool = False,
    poll_seconds: float = 30.0,
) -> dict[str, Any]:
    """Produce the mean-of-folds predictions, send them to Kaggle, record them.

    The Submission Fit refits the five fold models under the frozen protocol and
    averages their test predictions (:func:`mean_of_fold_predictions`) — no
    full-data refit, so the CV->LB Offset compares the same models on both sides.
    The submission line is written with a null public score at the moment of
    sending, the score is polled briefly, and :func:`record_score` fills it in
    later if scoring was slow.

    Heavy imports are deferred here so the module imports by bare name; this
    path only runs where the ML stack and the gitignored CSVs are present.
    """
    import subprocess
    import tempfile
    import time

    import numpy as np
    import pandas as pd

    import adapter as adapter_mod
    import data
    import experiments
    import frame
    import models
    import runner
    from columns import ID_COLUMN, TARGET_COLUMN

    # Enforce the unattended policy before a slot is spent, never after.
    if unattended:
        assert_unattended_allowed(
            confirmation_passed=confirmation_passed,
            is_final=is_final,
            submissions=load_submissions(),
        )

    record = runner.find_run(run_id)
    if record is None:
        raise ValueError(f"no run with id {run_id!r} in the runs ledger")
    config = experiments.resolve(record["experiment"])

    train = data.load_train()
    test = data.load_test()
    y = train[TARGET_COLUMN].to_numpy()
    fold = data.fold_ids_for(y, config.fold_seed)
    data.assert_fold_partition(fold, config.fold_seed)

    X_train, X_test = frame.build_frame(train, test, config.frame)

    # The mean of the five fold models: fit each fold model exactly as the
    # Comparison Run did, predict the test set, and average — never a refit.
    fold_preds: list[Any] = []
    for k in range(data.N_FOLDS):
        tr_idx = np.where(fold != k)[0]
        va_idx = np.where(fold == k)[0]
        # Built through the runner's shared helpers, so the Submission Fit
        # applies exactly the transforms the Comparison Run measured.
        adapter = runner.fold_adapter(config, k, va_idx.tolist())
        X_tr = adapter.fit_transform(X_train.iloc[tr_idx], y[tr_idx])
        X_te = adapter.transform(X_test)
        y_fit = adapter.resampled_y if adapter.resampled_y is not None else y[tr_idx]
        fold_preds.append(runner.fold_predict(config, X_tr, y_fit, X_te))
    test_pred = mean_of_fold_predictions(fold_preds)

    # Write the submission CSV (id + probability) for the Kaggle CLI.
    out = pd.DataFrame({ID_COLUMN: test[ID_COLUMN].to_numpy(), TARGET_COLUMN: test_pred})
    csv_path = Path(tempfile.mkdtemp()) / f"submission-{run_id}.csv"
    out.to_csv(csv_path, index=False)

    # Record at send time with a null public score, THEN send — a burned slot
    # always has a config attached to it.
    line = build_submission_line(
        run_id=run_id,
        oof_auc=record["oof_auc"],
        message=message,
        unattended=unattended,
        is_final=is_final,
        is_floor=is_floor,
    )
    append_submission(line)
    subprocess.run(
        [
            "kaggle", "competitions", "submit",
            "-c", COMPETITION_SLUG,
            "-f", str(csv_path),
            "-m", message,
        ],
        check=True,
    )
    print(f"Submitted run {run_id} to {COMPETITION_SLUG}; recorded with null public score.")

    # Poll briefly for the score; if scoring is slow, record-score fills it later.
    time.sleep(poll_seconds)
    print(
        "If the public score has not landed yet, fill it in later with:\n"
        f"    record-score {run_id} <public-score>"
    )
    return line


# --------------------------------------------------------------------------- #
# CLI entry points.
# --------------------------------------------------------------------------- #
def build_submit_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="submit",
        description=(
            "Fit the mean of the five fold models and submit it to Kaggle, "
            "recording the submission at send time with a null public score."
        ),
    )
    parser.add_argument("run_id", nargs="?", help="the run id to submit")
    parser.add_argument("-m", "--message", default="", help="the Kaggle submission message")
    parser.add_argument(
        "--floor", action="store_true", help="mark this as the Floor (turn 1 alone)"
    )
    parser.add_argument(
        "--unattended",
        action="store_true",
        help="an agent session: at most one slot/day, only after a Confirmation Run",
    )
    parser.add_argument(
        "--confirmed",
        action="store_true",
        help="a passing Confirmation Run authorises an unattended slot",
    )
    parser.add_argument(
        "--final",
        action="store_true",
        help="a final submission (never permitted unattended)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_submit_parser()
    args = parser.parse_args(argv)
    if not args.run_id:
        parser.error("a run id is required")
    submit(
        args.run_id,
        args.message,
        unattended=args.unattended,
        is_final=args.final,
        is_floor=args.floor,
        confirmation_passed=args.confirmed,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="record-score",
        description="Fill in a submission's public leaderboard score after the fact.",
    )
    parser.add_argument("run_id", nargs="?", help="the submission's run id")
    parser.add_argument("score", nargs="?", type=float, help="the public score")
    return parser


def record_score(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.run_id or args.score is None:
        parser.error("a run id and a public score are required")
    try:
        reading = record_score_for(args.run_id, args.score)
    except ValueError as exc:
        raise SystemExit(str(exc))
    print(f"Recorded public score {args.score:.5f} for {args.run_id}.")
    print(reading.message)
    return 0


if __name__ == "__main__":
    record_score()
