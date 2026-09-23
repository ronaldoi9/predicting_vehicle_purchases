"""The Submission Fit, the CSV, the Kaggle call and the submissions ledger.

Stub for #21: the ``record-score`` entry point is registered and responds to
``--help`` so the seam resolves; the bodies are wired up in a later ticket
(#12). Submissions are recorded in ``ledger/submissions.jsonl``, separate from
the runs ledger because a submission carries a public score and no Paired Delta.
"""

from __future__ import annotations

import argparse


def submit(run_id: str, message: str):
    """Produce the mean-of-folds predictions and send them to Kaggle."""
    raise NotImplementedError("submission.submit is wired up in a later ticket (#12)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="record-score",
        description="Fill in a submission's public leaderboard score after the fact.",
    )
    parser.add_argument("run_id", nargs="?", help="the submission's run id")
    parser.add_argument("score", nargs="?", type=float, help="the public score")
    return parser


def record_score(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    raise SystemExit(
        f"record-score is a stub (#21); run_id={args.run_id!r} score={args.score!r} "
        "is wired up in a later ticket (#12)"
    )


if __name__ == "__main__":
    record_score()
