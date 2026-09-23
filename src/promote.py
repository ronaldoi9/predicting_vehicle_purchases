"""Promoting an Incumbent: the one ledger write that is a judgement.

Stub for #21: the ``promote`` entry point is registered and responds to
``--help`` so the seam resolves; the body is wired up in a later ticket (#12).
Promotion is manual by design: the runner prints a verdict and stops, and this
command records the run, the delta and which rule authorised it.
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="promote",
        description="Promote a run to Incumbent, recording the authorising rule.",
    )
    parser.add_argument("run_id", nargs="?", help="the run id to promote")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    raise SystemExit(
        f"promote is a stub (#21); run_id={args.run_id!r} "
        "is wired up in a later ticket (#12)"
    )


if __name__ == "__main__":
    main()
