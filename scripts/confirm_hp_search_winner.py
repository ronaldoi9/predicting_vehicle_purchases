"""Confirmation Run (seeds 0/1/2) of a hyperparameter search's best trial (#30).

Reads the winning trial's params/num_boost_round out of
``runs/hp_search/<family>.json`` (written by ``run_hp_search.py``), rebuilds
it as an Experiment against the family's own Arena baseline, and runs it
through ``runner.confirm`` — the same path every other candidate's
Confirmation Run uses, so the sign-holds gate is not bypassed for a candidate
that happened to come out of a search instead of a single declared config.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import experiments  # noqa: E402
import runner  # noqa: E402

FAMILY_BASELINE = {
    "lightgbm": "income_te_tuned",
    "xgboost": "xgboost_baseline",
    "catboost": "catboost_baseline",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family", choices=sorted(FAMILY_BASELINE))
    args = parser.parse_args(argv)

    log_path = Path(__file__).resolve().parent.parent / "runs" / "hp_search" / f"{args.family}.json"
    trials = json.loads(log_path.read_text())
    trials = [t for t in trials if t["paired_delta"] is not None]
    best = max(trials, key=lambda t: t["paired_delta"])

    baseline_name = FAMILY_BASELINE[args.family]
    base = experiments.resolve(baseline_name)
    candidate = replace(
        base,
        name=f"hpsearch_{args.family}_best_confirm",
        params=best["params"],
        num_boost_round=best["num_boost_round"],
        incumbent=baseline_name,
        kill_delta=None,
    )
    print(f"Confirming {args.family} best trial ({best['run_id']}, seed-0 paired delta {best['paired_delta']:+.5f})")
    runner.confirm(candidate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
