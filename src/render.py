"""Rendering the runs ledger as a Markdown table sorted by Paired Delta.

Turn 2 of the CRISP loop starts by reading this table rather than by parsing
the raw ledger, so the ordering that matters — best Paired Delta first — is the
first thing it sees. Runs with no Paired Delta (a baseline with no Incumbent to
beat) sort to the bottom; among them, and among ties, insertion order (the
order the runs were recorded) breaks the tie so the table is stable.

``render`` is the one module other than ``runner`` allowed to name the runs
ledger, because it only reads it. ``runner`` remains the sole writer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

RUNS_LEDGER = Path(__file__).resolve().parent.parent / "ledger" / "runs.jsonl"

# Columns of the rendered table, in order: (header, key).
_COLUMNS: tuple[tuple[str, str], ...] = (
    ("Paired Delta", "paired_delta"),
    ("Run", "run_id"),
    ("Experiment", "experiment"),
    ("OOF AUC", "oof_auc"),
    ("Folds +", "folds_positive"),
    ("Incumbent", "incumbent_run_id"),
    ("Git", "git_sha"),
    ("Dirty", "dirty"),
)


def _fmt(key: str, value: Any) -> str:
    if value is None:
        return "—"
    if key == "paired_delta":
        return f"{value:+.5f}"
    if key == "oof_auc":
        return f"{value:.5f}"
    if key == "git_sha":
        return str(value)[:7]
    if key == "dirty":
        return "yes" if value else "no"
    return str(value)


def sort_key(record: Mapping[str, Any]) -> float:
    """Sort by Paired Delta, descending; null deltas sink to the bottom."""
    delta = record.get("paired_delta")
    return float(delta) if delta is not None else float("-inf")


def sort_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Stable sort by Paired Delta descending (insertion order breaks ties)."""
    return sorted((dict(r) for r in records), key=sort_key, reverse=True)


def render_table(records: Sequence[Mapping[str, Any]]) -> str:
    """Render the Run Records as a Markdown table sorted by Paired Delta."""
    headers = [h for h, _ in _COLUMNS]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for record in sort_records(records):
        cells = [_fmt(key, record.get(key)) for _, key in _COLUMNS]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def load_records(path: Path | None = None) -> list[dict[str, Any]]:
    path = Path(path) if path is not None else RUNS_LEDGER
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="render-ledger",
        description="Render the runs ledger as a Markdown table sorted by Paired Delta.",
    )
    parser.add_argument(
        "--ledger",
        default=str(RUNS_LEDGER),
        help="path to the runs ledger (default: ledger/runs.jsonl)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(render_table(load_records(Path(args.ledger))))
    return 0


if __name__ == "__main__":
    main()
