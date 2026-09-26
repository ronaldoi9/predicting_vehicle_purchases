"""Band (iii): freeze, select the two finals, and close the turn (#20, PRD #12).

Band (iii) is the freeze. **Nothing new enters once this ticket begins**, so its
work is three decisions taken against what the ledger already holds — not new
measurement — and every one of them is pure and lives here:

* **The freeze itself** (:func:`assert_no_new_candidate`): every experiment in
  the runs ledger must be one already declared in :mod:`experiments`. A candidate
  introduced after the freeze begins fails loudly rather than quietly entering
  the selection.

* **Turn 1's two finals = best CV + the Floor** (:func:`select_finals`), kept
  as the record of that turn's rule; the CLI applies the Proven Final clause
  below.
  *Not* best public LB: the 57,314-row public split cannot resolve 0.0002, so
  selecting on it selects the same noise named as the leading explanation for
  rank 1. *Not* the two best by CV: those are typically variants of one model and
  fail together. Best-CV-plus-Floor is the only pair covering a real failure mode
  — a bug the experiment queue introduced that CV did not catch. It is a **risk
  decision, reserved for the driving dev**, mirroring the manual promotion gate
  and the unattended-submission ban on finals in :mod:`submission`.

* **The Proven Final clause** (:func:`select_proven_finals`, #38, ADR-0006
  §5): ADR-0005 replaced the Floor with the best single-family model, and turn
  3 adds that at least one final must be **Proven** — its family and Frame
  configuration already produced a scored submission. The first slot is best
  CV; the second is the best Proven single-family model, so a bug turn-3 code
  introduced cannot sink both. Only a candidate with a passing Confirmation Run
  competes. This is the rule the ``select-finals`` CLI applies.

* **Closing turn 1** (:func:`render_turn_summary`): a summary rendered for the
  wayfinder map's *Decisions-so-far*, embedding the Paired-Delta-sorted ledger
  table (:func:`render.render_table`) so turn 2 starts by reading results.
  **No separate turn document** — a third narrative artifact competes with the
  map and goes stale first — so this only renders text; the driving dev appends
  it to the map.

The Confirmation Run of the standing Incumbent across fold seeds 0/1/2 — the
other freeze move — lives in :mod:`runner` (``runner.confirm`` /
``runner.confirmation_summary``), beside the Comparison Run it re-runs.

Everything here is dependency-free; every import is a bare-name module that
imports in any environment.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import MISSING, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import experiments
import render
import runner
import submission
import verdict


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


# --------------------------------------------------------------------------- #
# The Proven Final clause (#38, ADR-0006 §5).
# --------------------------------------------------------------------------- #
# The second slot when no confirmed model is Proven: the hp-search LightGBM,
# named by ADR-0006 §5 as the final a turn-3 best CV pairs with.
DEFAULT_PROVEN_FINAL = "hpsearch_lightgbm_best_confirm"

# Config fields that tune a code path rather than choose one: a run differing
# from a submitted one only here is on the same family and Frame configuration.
_TUNING_FIELDS = frozenset({"name", "fold_seed", "params", "num_boost_round"})


def _config(record: Mapping[str, Any]) -> Mapping[str, Any]:
    return record.get("config") or {}


def passes_confirmation(records: Sequence[Mapping[str, Any]], experiment: str) -> bool:
    """True iff ``experiment`` has a passing Confirmation Run in the ledger.

    Reads its latest Run Record on each of fold seeds 0/1/2: each must keep a
    positive Paired Delta or, for an Arena family's Member, pass its declared
    Member gate — the same reading ``runner.confirm`` prints.
    """
    latest: dict[int, Mapping[str, Any]] = {}
    for r in records:
        if r.get("experiment") == experiment:
            latest[int(_config(r).get("fold_seed", 0))] = r
    for seed in verdict.CONFIRMATION_SEEDS:
        r = latest.get(seed)
        if r is None:
            return False
        gate = r.get("kill_criterion") or {}
        if gate.get("member_gate"):
            if not gate.get("passed"):
                return False
        elif r.get("paired_delta") is None or float(r["paired_delta"]) <= 0:
            return False
    return True


def _code_path(record: Mapping[str, Any]) -> str:
    """The family and Frame configuration a Run Record ran on, as a stable key.

    Fields absent from an older record take the declaration's default, so a
    record written before a field existed still matches one written after.
    """
    config = {
        f.name: f.default
        for f in fields(experiments.Experiment)
        if f.default is not MISSING and f.name not in _TUNING_FIELDS
    }
    config.update((k, v) for k, v in _config(record).items() if k not in _TUNING_FIELDS)
    return json.dumps(config, sort_keys=True, default=list)


def proven_code_paths(
    records: Sequence[Mapping[str, Any]],
    submissions: Sequence[Mapping[str, Any]],
) -> frozenset[str]:
    """The code paths that have already produced a scored submission."""
    scored = {s.get("run_id") for s in submissions if s.get("public_score") is not None}
    return frozenset(_code_path(r) for r in records if r.get("run_id") in scored)


@dataclass(frozen=True)
class Final:
    """One final: the experiment, the Run Record it is read at, and its standing."""

    experiment: str
    run_id: str
    oof_auc: float
    proven: bool


@dataclass(frozen=True)
class ProvenFinalSelection:
    """The two finals under the Proven Final clause, and what was refused."""

    best_cv: Final
    single_family: Final
    refused: tuple[str, ...]
    rationale: str


_PROVEN_RATIONALE = (
    "The two finals are best CV + the best single-family model by CV (ADR-0005), "
    "and at least one of them is a Proven Final (ADR-0006 §5): its family and "
    "Frame configuration have already produced a scored submission, so a bug the "
    "current turn introduced — which CV cannot see — cannot sink both. Only a "
    "candidate with a passing Confirmation Run competes."
)


def select_proven_finals(
    records: Sequence[Mapping[str, Any]],
    submissions: Sequence[Mapping[str, Any]],
    *,
    driving_dev: bool,
) -> ProvenFinalSelection:
    """Select the two finals: best CV + the best Proven single-family model.

    Each experiment is read at its canonical-seed Run Record. Only one with a
    passing Confirmation Run (:func:`passes_confirmation`) competes; an
    unconfirmed one that would have ranked above the best-CV pick is named in
    ``refused``. The first slot is the best CV, a Blend included. The second is
    the best-CV single-family model among the rest that is Proven, so at least
    one final always is. When none is, the second slot is
    :data:`DEFAULT_PROVEN_FINAL`; and when that is not confirmed either but the
    best CV is itself Proven, the clause already holds and the second slot is
    the best-CV single-family model among the rest.
    ``driving_dev`` must be ``True``, as in :func:`select_finals`.
    """
    if not driving_dev:
        raise PermissionError(
            "the two finals are always the driving dev's, never an unattended "
            "session's — final selection is a risk decision, not a measurement"
        )
    canonical: dict[str, dict[str, Any]] = {}
    for r in records:
        if r.get("oof_auc") is not None and _config(r).get("fold_seed", 0) == 0:
            canonical[str(r.get("experiment"))] = dict(r)
    ranked = sorted(canonical.values(), key=lambda r: -float(r["oof_auc"]))
    confirmed = [r for r in ranked if passes_confirmation(records, r["experiment"])]
    if not confirmed:
        raise ValueError("no run in the ledger has a passing Confirmation Run to select")
    best, rest = confirmed[0], confirmed[1:]
    confirmed_names = {r["experiment"] for r in confirmed}
    refused = tuple(
        r["experiment"]
        for r in ranked
        if float(r["oof_auc"]) > float(best["oof_auc"])
        and r["experiment"] not in confirmed_names
    )

    proven = proven_code_paths(records, submissions)
    single_families = [r for r in rest if _config(r).get("model")]
    single_family = next((r for r in single_families if _code_path(r) in proven), None)
    if single_family is None:
        default = next((r for r in rest if r["experiment"] == DEFAULT_PROVEN_FINAL), None)
        if default is not None:
            single_family = default
        elif _code_path(best) in proven and single_families:
            single_family = single_families[0]
        else:
            raise ValueError(
                "no confirmed single-family model is a Proven Final, and "
                f"{DEFAULT_PROVEN_FINAL!r} has no passing Confirmation Run to take "
                "the second slot — at least one final must be Proven (ADR-0006 §5)"
            )

    def final(r: Mapping[str, Any]) -> Final:
        return Final(
            experiment=str(r["experiment"]),
            run_id=str(r["run_id"]),
            oof_auc=float(r["oof_auc"]),
            proven=_code_path(r) in proven,
        )

    return ProvenFinalSelection(
        best_cv=final(best),
        single_family=final(single_family),
        refused=refused,
        rationale=_PROVEN_RATIONALE,
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
# CLI: select the finals (driving-dev only) under the Proven Final clause.
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="select-finals",
        description=(
            "Select the two final submissions: best CV + the best single-family "
            "model, at least one of them a Proven Final, both with a passing "
            "Confirmation Run. The finals are a risk decision reserved for the "
            "driving dev."
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
    parser.add_argument(
        "--runs-ledger",
        type=Path,
        help="read Run Records from this file instead of the runs ledger (a replay)",
    )
    parser.add_argument(
        "--submissions-ledger",
        type=Path,
        help="read submissions from this file instead of the submissions ledger",
    )
    return parser


def _describe(final: Final) -> str:
    return (
        f"{final.experiment} ({final.run_id}, OOF {final.oof_auc:.5f}, "
        f"{'Proven' if final.proven else 'not Proven'})"
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records = runner.load_records(args.runs_ledger)
    subs = submission.load_submissions(args.submissions_ledger)

    # The freeze is asserted first: a new candidate in the ledger breaks band
    # (iii)'s premise before any final is chosen.
    assert_no_new_candidate(records)

    try:
        selection = select_proven_finals(records, subs, driving_dev=args.driving_dev)
    except (PermissionError, ValueError) as exc:
        raise SystemExit(str(exc))

    print("Selected the two final submissions (best CV + best single-family, "
          "at least one Proven):")
    print("  best CV:       " + _describe(selection.best_cv))
    print("  single-family: " + _describe(selection.single_family))
    if selection.refused:
        print("  refused, no passing Confirmation Run: " + ", ".join(selection.refused))
    print()
    print(f"_{selection.rationale}_")
    return 0


if __name__ == "__main__":
    main()
