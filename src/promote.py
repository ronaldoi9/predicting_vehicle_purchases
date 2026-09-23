"""Promoting an Incumbent: the one ledger write that is a judgement.

Promotion is **manual by explicit command**. The runner evaluates the rule,
prints the verdict and stops; it never advances the Incumbent itself. This
command re-evaluates the same rule (from :mod:`verdict`) against the run's
recorded Paired Delta and, only when the rule authorises it, writes one
promotion line recording the run, the delta, and **which of the rules
authorised it**.

The promotion line goes to ``ledger/promotions.jsonl`` — its own append-only
ledger, separate from the runs ledger because a promotion is a judgement, not
an observed fact, and because ``runner`` is the sole writer of the runs ledger.
``promote`` is the sole writer of the promotions ledger, and the path literal
lives here and nowhere else that writes.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import runner
import verdict as verdict_mod

# promote is the sole writer of the promotions ledger.
PROMOTIONS_LEDGER = Path(__file__).resolve().parent.parent / "ledger" / "promotions.jsonl"


def authorise(
    record: Mapping[str, Any],
    confirmation: Sequence[float] | None = None,
) -> verdict_mod.Verdict:
    """Apply the promotion rule to a Run Record.

    A baseline run (no Paired Delta, so no Incumbent it was measured against)
    cannot be promoted; that is a usage error, distinct from a rule rejection.
    """
    mean_delta = record.get("paired_delta")
    if mean_delta is None:
        raise ValueError(
            f"run {record.get('run_id')!r} has no Paired Delta; it is a "
            "baseline, not a candidate, and cannot be promoted"
        )
    folds_positive = record.get("folds_positive")
    if folds_positive is None:
        raise ValueError(
            f"run {record.get('run_id')!r} records a Paired Delta but no "
            "folds_positive; the four-of-five rule cannot be evaluated"
        )
    return verdict_mod.classify(mean_delta, folds_positive, confirmation=confirmation)


def build_promotion_line(
    record: Mapping[str, Any],
    v: verdict_mod.Verdict,
    when: datetime | None = None,
) -> dict[str, Any]:
    """The promotion line: the run, the delta, and the authorising rule."""
    when = when or datetime.now(timezone.utc)
    return {
        "promoted_run_id": record.get("run_id"),
        "incumbent_run_id": record.get("incumbent_run_id"),
        "paired_delta": record.get("paired_delta"),
        "folds_positive": record.get("folds_positive"),
        "rule": v.rule,
        "reason": v.reason,
        "timestamp": when.isoformat(),
    }


def append_promotion(line: Mapping[str, Any], path: Path | None = None) -> None:
    """Append one promotion line as a single JSON line."""
    path = Path(path) if path is not None else PROMOTIONS_LEDGER
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(line) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="promote",
        description="Promote a run to Incumbent, recording the authorising rule.",
    )
    parser.add_argument("run_id", nargs="?", help="the run id to promote")
    parser.add_argument(
        "--confirm",
        nargs=3,
        type=float,
        metavar=("SEED0", "SEED1", "SEED2"),
        help=(
            "the Confirmation Run's mean paired delta on fold seeds 0/1/2; "
            "required to promote a delta in [0.0001, 0.0003)"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.run_id:
        parser.error("a run id is required")

    record = runner.find_run(args.run_id)
    if record is None:
        parser.error(f"no run with id {args.run_id!r} in the ledger")

    try:
        v = authorise(record, confirmation=args.confirm)
    except ValueError as exc:
        raise SystemExit(str(exc))

    if not v.accepted:
        # The rule refuses. Nothing is written: promotion is a judgement, and
        # this one did not pass.
        raise SystemExit(
            f"NOT PROMOTED — {args.run_id}: {v.reason}\n"
            "The Incumbent is unchanged."
        )

    line = build_promotion_line(record, v)
    append_promotion(line)
    print(
        f"PROMOTED {args.run_id} to Incumbent.\n"
        f"  paired delta: {line['paired_delta']:+.5f}\n"
        f"  authorised by: {v.rule}\n"
        f"  recorded in: {PROMOTIONS_LEDGER}"
    )
    return 0


if __name__ == "__main__":
    main()
