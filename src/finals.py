"""Band (iii): freeze, select the two finals, and close the turn (#20, PRD #12).

Band (iii) is the freeze. **Nothing new enters once this ticket begins**, so its
work is three decisions taken against what the ledger already holds — not new
measurement — and every one of them is pure and lives here:

* **The freeze itself** (:func:`assert_no_new_candidate`): every experiment in
  the runs ledger must be one already declared in :mod:`experiments`. A candidate
  introduced after the freeze begins fails loudly rather than quietly entering
  the selection.

* **The two final submissions = best CV + the Floor** (:func:`select_finals`).
  *Not* best public LB: the 57,314-row public split cannot resolve 0.0002, so
  selecting on it selects the same noise named as the leading explanation for
  rank 1. *Not* the two best by CV: those are typically variants of one model and
  fail together. Best-CV-plus-Floor is the only pair covering a real failure mode
  — a bug the experiment queue introduced that CV did not catch. It is a **risk
  decision, reserved for the driving dev**, mirroring the manual promotion gate
  and the unattended-submission ban on finals in :mod:`submission`.

* **Closing the turn** (:func:`render_turn_summary`): a summary rendered for the
  wayfinder map's *Decisions-so-far*, embedding the Paired-Delta-sorted ledger
  table (:func:`render.render_table`) so turn 2 starts by reading results.
  **No separate turn document** — a third narrative artifact competes with the
  map and goes stale first — so this only renders text; the driving dev appends
  it to the map.

The Confirmation Run of the standing Incumbent across fold seeds 0/1/2 — the
other freeze move — lives in :mod:`runner` (``runner.confirm`` /
``runner.confirmation_summary``), beside the Comparison Run it re-runs.

Everything here is dependency-free; :mod:`render` and :mod:`experiments` are the
only imports, and both import by bare name in any environment.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

import experiments
import render
import runner
import submission


def frozen_candidate_names() -> frozenset[str]:
    """The experiments allowed to appear in the ledger at freeze — the registry.

    The whitelist is the declared registry itself, never a hand-maintained copy,
    so it cannot drift from what :mod:`experiments` actually declares.
    """
    return frozenset(experiments._REGISTRY)


def assert_no_new_candidate(
    records: Sequence[Mapping[str, Any]],
    allowed: frozenset[str] | None = None,
) -> None:
    """Enforce the freeze: every ledger experiment must be one already declared.

    Band (iii) begins and nothing new enters. A Run Record whose experiment is
    not in :func:`frozen_candidate_names` is a candidate introduced after the
    freeze — it fails loudly here rather than sliding into the final selection.
    """
    allowed = allowed if allowed is not None else frozen_candidate_names()
    intruders = sorted(
        {
            str(r.get("experiment"))
            for r in records
            if r.get("experiment") not in allowed
        }
    )
    if intruders:
        raise ValueError(
            "the freeze is broken — these experiments are in the ledger but were "
            f"never declared (a new candidate after band (iii) began): "
            f"{', '.join(intruders)}. Nothing new enters once this ticket begins."
        )


def best_cv_run(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The run with the highest OOF AUC — best CV, the unit the finals select on.

    Runs without an ``oof_auc`` (a malformed or incomplete record) are ignored.
    Ties keep the first-recorded run, so the choice is stable. Raises if there is
    nothing to select — an empty ledger has no best CV.
    """
    scored = [r for r in records if r.get("oof_auc") is not None]
    if not scored:
        raise ValueError("no run in the ledger carries an OOF AUC to select best CV from")
    best = scored[0]
    for r in scored[1:]:
        if float(r["oof_auc"]) > float(best["oof_auc"]):
            best = r
    return dict(best)


def floor_submission(
    submissions: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """The Floor submission — turn 1 alone, flagged ``is_floor`` — or ``None``.

    Last write wins, so a Floor line whose public score was filled in later
    resolves over the null-score line it updated.
    """
    match: dict[str, Any] | None = None
    for line in submissions:
        if line.get("is_floor"):
            match = dict(line)
    return match


def _best_public_lb(
    submissions: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """The submission with the highest public leaderboard score, or ``None``.

    Used only to *name* what the rule deliberately refuses — the finals are best
    CV + Floor precisely because best public LB selects the noise the 57,314-row
    split cannot resolve.
    """
    scored = [s for s in submissions if s.get("public_score") is not None]
    if not scored:
        return None
    return max(scored, key=lambda s: float(s["public_score"]))


@dataclass(frozen=True)
class FinalSelection:
    """The two final submissions: best CV + the Floor, and why."""

    best_cv_run_id: str
    best_cv_oof_auc: float
    floor_run_id: str
    rationale: str


# The rule, fixed now rather than on the last day so it is not chosen under the
# conditions that make it worst (PRD #12).
_RATIONALE = (
    "The two finals are best CV + the Floor, a risk decision reserved for the "
    "driving dev. Not best public LB: the 57,314-row public split cannot resolve "
    "0.0002, so selecting on it selects the same noise named as the leading "
    "explanation for rank 1. Not the two best by CV: those are variants of one "
    "model and fail together. Best-CV-plus-Floor is the only pair covering a bug "
    "the queue introduced that CV did not catch — it cedes ~0.0005 of expected "
    "rank on the Floor slot, and that is the premium."
)


def select_finals(
    records: Sequence[Mapping[str, Any]],
    submissions: Sequence[Mapping[str, Any]],
    *,
    driving_dev: bool,
) -> FinalSelection:
    """Select the two final submissions: best CV + the Floor.

    ``driving_dev`` must be ``True``: the two finals are always the driving dev's,
    never an unattended session's — it is a risk decision, not a measurement, and
    an unattended process may not make it (mirroring the final-submission ban in
    :func:`submission.assert_unattended_allowed`). The Floor must already exist —
    it is submitted by 25/09 so no later decision is taken under fear of having
    nothing to select.
    """
    if not driving_dev:
        raise PermissionError(
            "the two finals are always the driving dev's, never an unattended "
            "session's — final selection is a risk decision, not a measurement"
        )
    floor = floor_submission(submissions)
    if floor is None:
        raise ValueError(
            "no Floor submission in the ledger; the Floor (turn 1 alone) must be "
            "submitted before the two finals are selected (best CV + the Floor)"
        )
    best = best_cv_run(records)
    return FinalSelection(
        best_cv_run_id=str(best["run_id"]),
        best_cv_oof_auc=float(best["oof_auc"]),
        floor_run_id=str(floor["run_id"]),
        rationale=_RATIONALE,
    )


def render_turn_summary(
    records: Sequence[Mapping[str, Any]],
    submissions: Sequence[Mapping[str, Any]],
    selection: FinalSelection,
    *,
    when: datetime | None = None,
) -> str:
    """Render the turn's closing summary for the wayfinder map's Decisions-so-far.

    Embeds the Paired-Delta-sorted ledger table (:func:`render.render_table`) so
    turn 2 starts by reading results rather than parsing them, names the two
    finals and states the freeze. Returned as text — **no separate turn
    document** is written; the driving dev appends this to the map, which is where
    the next batch of tickets lands.
    """
    when = when or datetime.now(timezone.utc)
    stamp = when.date().isoformat()
    lines = [
        f"- **Turn 1 closed ({stamp}) — freeze, finals, hand-off to turn 2.** "
        "Band (iii): nothing new entered once the freeze began. The two final "
        f"submissions are **best CV** (`{selection.best_cv_run_id}`, OOF "
        f"{selection.best_cv_oof_auc:.5f}) **+ the Floor** "
        f"(`{selection.floor_run_id}`, turn 1 alone) — the driving dev's risk "
        "decision, not best public LB. The runs ledger, sorted by Paired Delta "
        "for turn 2 to read:",
        "",
        render.render_table(records),
        "",
        f"  _{selection.rationale}_",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI: select the finals (driving-dev only) and print the turn summary.
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="select-finals",
        description=(
            "Select the two final submissions (best CV + the Floor) and print the "
            "turn's closing summary for the wayfinder map. The finals are a risk "
            "decision reserved for the driving dev."
        ),
    )
    parser.add_argument(
        "--driving-dev",
        action="store_true",
        help=(
            "acknowledge this is the driving dev selecting the finals — required, "
            "because the two finals are never an unattended session's"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records = runner.load_records()
    subs = submission.load_submissions()

    # The freeze is asserted first: a new candidate in the ledger breaks band
    # (iii)'s premise before any final is chosen.
    assert_no_new_candidate(records)

    try:
        selection = select_finals(records, subs, driving_dev=args.driving_dev)
    except PermissionError as exc:
        raise SystemExit(str(exc))
    except ValueError as exc:
        raise SystemExit(str(exc))

    print("Selected the two final submissions (best CV + the Floor):")
    print(
        f"  best CV: {selection.best_cv_run_id} (OOF {selection.best_cv_oof_auc:.5f})"
    )
    print(f"  Floor:   {selection.floor_run_id} (turn 1 alone)")
    print()
    print("Turn summary to append to the wayfinder map's Decisions-so-far "
          "(no separate document):")
    print(render_turn_summary(records, subs, selection))
    return 0


if __name__ == "__main__":
    main()
