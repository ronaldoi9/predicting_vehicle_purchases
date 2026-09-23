"""Executing a Comparison Run and the (only) runs-ledger write.

`runner` is the **only** module allowed to write ``ledger/runs.jsonl`` — one of
the two hard rules the tracer bullet (#13) establishes, alongside ``adapter``
being the only module that sees ``y`` outside a model's own ``fit``. Keeping the
ledger write in one file is what makes an unrecorded run impossible rather than
merely discouraged.

A Comparison Run is the frozen-protocol fit (fixed round count, early stopping
off) whose OOF AUC is the unit every Paired Delta is measured in. The tracer
runs one Experiment end to end on the raw-13 frame:

    read both CSVs in file order -> build the Canonical Fold Partition
    immediately -> assemble the Baseline Frame -> fit LightGBM across five
    folds -> compute the OOF AUC -> write one Run Record to the ledger and the
    OOF vector to disk.

Heavy imports (pandas/numpy/scikit-learn/lightgbm) are deferred into ``run`` so
the module — and its dependency-free helpers below — import by bare name in any
environment. The Health Gate (OOF 0.94167) only reproduces where those packages
and the gitignored CSVs are present.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


# The runs ledger. This path literal lives here and nowhere else that writes:
# runner is the sole writer of the runs ledger.
RUNS_LEDGER = _repo_root() / "ledger" / "runs.jsonl"

# Per-run OOF vectors live outside git (runs/ is gitignored), one file per run,
# so a Paired Delta stays computable between two configs that never met.
RUNS_DIR = _repo_root() / "runs"


# --------------------------------------------------------------------------- #
# Dependency-free helpers (unit-tested without the ML stack).
# --------------------------------------------------------------------------- #
def run_id(experiment_name: str, when: datetime | None = None) -> str:
    """``<timestamp>-<experiment name>`` — deliberately not a config hash.

    A Confirmation Run executes one config three times; a hash-as-identity would
    collide precisely on the repetitions the protocol needs, so identity is the
    instant of execution. Microseconds are included so two runs in the same
    second still get distinct ids.
    """
    when = when or datetime.now(timezone.utc)
    stamp = when.strftime("%Y%m%dT%H%M%S") + f".{when.microsecond:06d}Z"
    return f"{stamp}-{experiment_name}"


def config_hash(config: Mapping[str, Any]) -> str:
    """sha256 of the config's canonical JSON, so "have I tried this?" is a grep.

    Key order is normalised so two logically identical configs hash the same.
    """
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def git_capture(repo_root: Path | None = None) -> dict[str, Any]:
    """The git sha and dirty-tree flag, so a number traces to the code."""
    repo_root = repo_root or _repo_root()

    def _git(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    try:
        sha = _git("rev-parse", "HEAD")
        dirty = bool(_git("status", "--porcelain"))
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"git_sha": None, "dirty": None}
    return {"git_sha": sha, "dirty": dirty}


def build_run_record(
    *,
    experiment: str,
    config: Mapping[str, Any],
    fold_aucs: Sequence[float],
    oof_auc: float,
    fold_sha256: str,
    oof_path: str,
    wall_time: float,
    seeds: Mapping[str, int] | None = None,
    git_info: Mapping[str, Any] | None = None,
    when: datetime | None = None,
    incumbent_run_id: str | None = None,
    paired_delta: float | None = None,
    folds_positive: int | None = None,
) -> dict[str, Any]:
    """Assemble one Run Record — the unit the next CRISP turn reads.

    Carries everything #13 requires: run id, config hash, timestamp, git sha,
    dirty-tree flag, the full config, the five fold AUCs, the OOF AUC, the
    fold-partition sha256, the seeds, wall time and the OOF vector path. The
    Paired-Delta fields are present but null on the first run (no Incumbent yet).
    """
    when = when or datetime.now(timezone.utc)
    git_info = dict(git_info or git_capture())
    return {
        "run_id": run_id(experiment, when=when),
        "experiment": experiment,
        "config_hash": config_hash(config),
        "timestamp": when.isoformat(),
        "git_sha": git_info.get("git_sha"),
        "dirty": git_info.get("dirty"),
        "config": dict(config),
        "fold_aucs": [float(a) for a in fold_aucs],
        "oof_auc": float(oof_auc),
        "fold_partition_sha256": fold_sha256,
        "fold_sha256": fold_sha256,
        "seeds": dict(seeds) if seeds is not None else None,
        "wall_time": float(wall_time),
        "oof_path": oof_path,
        "incumbent_run_id": incumbent_run_id,
        "paired_delta": paired_delta,
        "folds_positive": folds_positive,
    }


def append_run_record(record: Mapping[str, Any], path: Path | None = None) -> None:
    """Append one Run Record as a single JSON line (jsonl never git-conflicts)."""
    path = Path(path) if path is not None else RUNS_LEDGER
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def load_records(path: Path | None = None) -> list[dict[str, Any]]:
    """Read every Run Record from the ledger, in the order they were written.

    Reading the runs ledger is allowed outside ``runner`` (``render`` does it);
    keeping the parse here too means ``promote`` can look a run up without ever
    naming the ledger file, so the single-writer discipline stays a grep.
    """
    path = Path(path) if path is not None else RUNS_LEDGER
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def find_run(run_id: str, path: Path | None = None) -> dict[str, Any] | None:
    """Return the Run Record with this id, or ``None`` if there is no such run.

    The last write wins, so a re-recorded id resolves to its latest record.
    """
    match: dict[str, Any] | None = None
    for record in load_records(path):
        if record.get("run_id") == run_id:
            match = record
    return match


# --------------------------------------------------------------------------- #
# The Comparison Run itself (needs the ML stack + the gitignored CSVs).
# --------------------------------------------------------------------------- #
def run(config) -> dict[str, Any]:
    """Execute one Comparison Run end to end and record it.

    ``config`` is an :class:`experiments.Experiment`. Returns the Run Record it
    appended to the ledger.
    """
    import numpy as np
    from sklearn.metrics import roc_auc_score

    import adapter as adapter_mod
    import data
    import frame
    import models
    from columns import TARGET_COLUMN

    t0 = time.perf_counter()
    when = datetime.now(timezone.utc)

    # Read in file order, then split immediately — no transform touches the
    # rows the partition depends on before it is built.
    train = data.load_train()
    test = data.load_test()
    y = train[TARGET_COLUMN].to_numpy()

    fold = data.fold_ids_for(y, config.fold_seed)
    fold_sha = data.assert_fold_partition(fold, config.fold_seed)

    X_train, _X_test = frame.build_frame(train, test, config.frame)

    oof = np.zeros(len(y), dtype=np.float64)
    fold_aucs: list[float] = []
    for k in range(data.N_FOLDS):
        tr_idx = np.where(fold != k)[0]
        va_idx = np.where(fold == k)[0]

        adapter = adapter_mod.Adapter(scale=config.scale)
        X_tr = adapter.fit_transform(X_train.iloc[tr_idx], y[tr_idx])
        X_va = adapter.transform(X_train.iloc[va_idx])

        model = models.fit(X_tr, y[tr_idx], config.params, num_boost_round=config.num_boost_round)
        preds = models.predict(model, X_va)
        oof[va_idx] = preds
        fold_aucs.append(float(roc_auc_score(y[va_idx], preds)))

    oof_auc = float(roc_auc_score(y, oof))

    rid = run_id(config.name, when=when)
    oof_path = RUNS_DIR / rid / "oof.npy"
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(oof_path, oof)

    record = build_run_record(
        experiment=config.name,
        config=config.as_config(),
        fold_aucs=fold_aucs,
        oof_auc=oof_auc,
        fold_sha256=fold_sha,
        oof_path=str(oof_path.relative_to(_repo_root())),
        wall_time=time.perf_counter() - t0,
        seeds=config.seeds(),
        when=when,
    )
    # Keep the id and the recorded oof path consistent with the vector on disk.
    record["run_id"] = rid
    append_run_record(record)

    _print_verdict(config, record)
    return record


def _print_verdict(config, record: Mapping[str, Any]) -> None:
    oof_auc = record["oof_auc"]
    gate = config.health_gate
    print(f"[{record['run_id']}] OOF AUC = {oof_auc:.5f}  (fold AUCs: "
          + ", ".join(f"{a:.5f}" for a in record["fold_aucs"]) + ")")
    if oof_auc >= gate:
        print(f"Health Gate PASSED (>= {gate:.4f}); target {config.target_oof:.5f}.")
    else:
        # A miss is a bug report with the per-fold numbers, not a verdict on
        # the model.
        print(
            f"Health Gate FAILED: OOF {oof_auc:.5f} < {gate:.4f}. "
            "This is a bug in how the partition or the Frame was assembled, not "
            "a bad model. First divergence hypothesis: bagging_freq. Per-fold: "
            + ", ".join(f"{a:.5f}" for a in record["fold_aucs"])
        )
    print_promotion_verdict(record)


def print_promotion_verdict(record: Mapping[str, Any]) -> str | None:
    """Print the promotion verdict for a candidate run, then stop.

    Returns the verdict kind (or ``None`` when the run carries no Paired Delta,
    i.e. it is a baseline with no Incumbent to beat). This only *prints* — the
    Incumbent is never advanced here; ``promote`` is the explicit command that
    records a promotion. The rule itself lives in :mod:`verdict`.
    """
    import verdict as verdict_mod

    mean_delta = record.get("paired_delta")
    if mean_delta is None:
        # No Incumbent to compare against (the first run, or a baseline);
        # there is nothing to promote.
        return None

    folds_positive = record.get("folds_positive")
    if folds_positive is None:
        raise ValueError(
            "a run with a Paired Delta must record folds_positive; "
            f"run {record.get('run_id')!r} did not"
        )

    v = verdict_mod.classify(mean_delta, folds_positive)
    banner = {
        verdict_mod.ACCEPT: "VERDICT: ACCEPT",
        verdict_mod.REJECT: "VERDICT: REJECT",
        verdict_mod.NEEDS_CONFIRMATION: "VERDICT: NEEDS CONFIRMATION RUN",
    }[v.kind]
    print(
        f"{banner} — paired delta {mean_delta:+.5f} vs incumbent "
        f"{record.get('incumbent_run_id')!r}; {v.reason}"
    )
    if v.kind == verdict_mod.ACCEPT:
        print(
            "The Incumbent is NOT advanced automatically. To record the "
            f"promotion, run: promote {record.get('run_id')}"
        )
    return v.kind


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
    if not args.experiment:
        build_parser().error("an experiment name is required")

    import experiments

    config = experiments.resolve(args.experiment)
    run(config)
    return 0


if __name__ == "__main__":
    main()
