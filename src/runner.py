"""Executing a Comparison Run and the (only) runs-ledger write.

Stub for #21: the ``run-experiment`` entry point is registered and responds to
``--help`` so the seam resolves; the body is wired up in a later ticket (#12).
``runner`` is the only module allowed to write ``ledger/runs.jsonl``.
"""

from __future__ import annotations

import argparse


def run(config):
    """Execute one Comparison Run and append its Run Record to the ledger."""
    raise NotImplementedError("runner.run is wired up in a later ticket (#12)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run-experiment",
        description="Run a named Experiment as a Comparison Run and record it.",
    )
    parser.add_argument(
        "experiment",
        nargs="?",
        help="name of the Experiment to resolve and run",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    raise SystemExit(
        f"run-experiment is a stub (#21); experiment={args.experiment!r} "
        "is wired up in a later ticket (#12)"
    )


if __name__ == "__main__":
    main()
