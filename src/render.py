"""Rendering the runs ledger as a Markdown table sorted by Paired Delta.

Stub for #21: the ``render-ledger`` entry point is registered and responds to
``--help`` so the seam resolves; the body is wired up in a later ticket (#12).
Turn 2 starts by reading this table rather than by parsing the raw ledger.
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="render-ledger",
        description="Render the runs ledger as a Markdown table sorted by Paired Delta.",
    )
    parser.add_argument(
        "--ledger",
        default="ledger/runs.jsonl",
        help="path to the runs ledger (default: ledger/runs.jsonl)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    raise SystemExit(
        f"render-ledger is a stub (#21); ledger={args.ledger!r} "
        "is wired up in a later ticket (#12)"
    )


if __name__ == "__main__":
    main()
