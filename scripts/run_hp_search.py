"""Random hyperparameter search per family, on the frozen Frame (#30).

Time-boxed random search against each family's own Arena baseline
(``income_te_tuned`` / ``xgboost_baseline`` / ``catboost_baseline``). Every
trial is a real Comparison Run — it goes through ``runner.run`` and lands in
the committed ledger like any other run, because there is no separate
measurement path in this instrument.

``max_bin``/``border_count`` stay frozen at 511 (the Resolution control, #7/#8)
and are never sampled. ``num_boost_round`` and the learning rate are sampled
as a joint pair so the frozen fixed-rounds/no-early-stopping protocol keeps
comparing like for like: their product is held close to the family baseline's
own product, clipped to [200, 3000] rounds.

Usage: ``uv run python scripts/run_hp_search.py <family> --seconds 900``
"""

from __future__ import annotations

import argparse
import random
import sys
import time
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


def _sample_round_lr(rng: random.Random, base_rounds: int, base_lr: float) -> tuple[int, float]:
    """A jointly sampled (num_boost_round, learning_rate) pair.

    Learning rate is drawn log-uniform in [0.01, 0.2]; the round count is set
    so ``rounds * lr`` stays close to the baseline's own product, keeping the
    pair coupled rather than letting one vary independently of the other.
    """
    lr = 10 ** rng.uniform(-2, -0.7)
    target_product = base_rounds * base_lr
    rounds = int(round(target_product / lr))
    rounds = max(200, min(3000, rounds))
    return rounds, lr


def _sample_lightgbm(rng: random.Random, base: experiments.Experiment) -> experiments.Experiment:
    rounds, lr = _sample_round_lr(rng, base.num_boost_round, base.params["learning_rate"])
    bagging_freq = rng.randint(0, 10)
    params = {
        **base.params,
        "learning_rate": lr,
        "num_leaves": rng.randint(31, 255),
        "max_depth": rng.choice([-1] + list(range(4, 13))),
        "min_child_samples": rng.randint(20, 500),
        "min_split_gain": rng.uniform(0.0, 0.1),
        "lambda_l1": rng.uniform(0.0, 5.0),
        "lambda_l2": rng.uniform(0.0, 5.0),
        "feature_fraction": rng.uniform(0.5, 1.0),
        "bagging_fraction": rng.uniform(0.5, 1.0),
        "bagging_freq": bagging_freq,
    }
    return replace(base, params=params, num_boost_round=rounds)


def _sample_xgboost(rng: random.Random, base: experiments.Experiment) -> experiments.Experiment:
    rounds, lr = _sample_round_lr(rng, base.num_boost_round, base.params["eta"])
    grow_policy = rng.choice(["depthwise", "lossguide"])
    params = {
        **base.params,
        "eta": lr,
        "max_depth": 0 if grow_policy == "lossguide" else rng.randint(3, 12),
        "min_child_weight": rng.uniform(1.0, 20.0),
        "subsample": rng.uniform(0.5, 1.0),
        "colsample_bytree": rng.uniform(0.5, 1.0),
        "lambda": rng.uniform(0.0, 5.0),
        "alpha": rng.uniform(0.0, 5.0),
        "gamma": rng.uniform(0.0, 5.0),
        "grow_policy": grow_policy,
    }
    if grow_policy == "lossguide":
        params["max_leaves"] = rng.randint(31, 255)
    else:
        params.pop("max_leaves", None)
    return replace(base, params=params, num_boost_round=rounds)


def _sample_catboost(rng: random.Random, base: experiments.Experiment) -> experiments.Experiment:
    rounds, lr = _sample_round_lr(rng, base.num_boost_round, base.params["learning_rate"])
    params = {
        **base.params,
        "learning_rate": lr,
        "depth": rng.randint(4, 10),
        "l2_leaf_reg": rng.uniform(1.0, 10.0),
        "subsample": rng.uniform(0.5, 1.0),
        "rsm": rng.uniform(0.5, 1.0),
        "random_strength": rng.uniform(0.0, 5.0),
    }
    return replace(base, params=params, num_boost_round=rounds)


SAMPLERS = {
    "lightgbm": _sample_lightgbm,
    "xgboost": _sample_xgboost,
    "catboost": _sample_catboost,
}


def search(family: str, seconds: float, seed: int = 0) -> list[dict]:
    baseline_name = FAMILY_BASELINE[family]
    base = experiments.resolve(baseline_name)
    sampler = SAMPLERS[family]
    rng = random.Random(seed)

    results: list[dict] = []
    t0 = time.perf_counter()
    trial = 0
    while time.perf_counter() - t0 < seconds:
        candidate = sampler(rng, base)
        candidate = replace(
            candidate,
            name=f"hpsearch_{family}_{trial:04d}",
            incumbent=baseline_name,
            kill_delta=None,
            kill_min_folds_positive=None,
            kill_value_per_run=None,
            kill_time_budget_s=None,
        )
        record = runner.run(candidate)
        results.append(
            {
                "trial": trial,
                "run_id": record["run_id"],
                "oof_auc": record["oof_auc"],
                "paired_delta": record.get("paired_delta"),
                "folds_positive": record.get("folds_positive"),
                "params": candidate.params,
                "num_boost_round": candidate.num_boost_round,
            }
        )
        trial += 1

    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family", choices=sorted(FAMILY_BASELINE))
    parser.add_argument("--seconds", type=float, default=900.0, help="wall-clock time box")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for the sampler")
    args = parser.parse_args(argv)

    results = search(args.family, args.seconds, args.seed)
    results.sort(key=lambda r: (r["paired_delta"] is not None, r["paired_delta"]), reverse=True)

    print(f"\n{len(results)} trials for {args.family!r} in {args.seconds:.0f}s box.")
    best = results[0] if results else None
    if best:
        print(
            f"Best: {best['run_id']} paired_delta={best['paired_delta']:+.5f} "
            f"folds_positive={best['folds_positive']} oof_auc={best['oof_auc']:.5f}"
        )
        print(f"params: {best['params']}")
        print(f"num_boost_round: {best['num_boost_round']}")

    out_dir = Path(__file__).resolve().parent.parent / "runs" / "hp_search"
    out_dir.mkdir(parents=True, exist_ok=True)
    import json

    out_path = out_dir / f"{args.family}.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"Full trial log: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
