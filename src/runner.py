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


def fold_deltas(candidate_fold_aucs: Sequence[float], incumbent_fold_aucs: Sequence[float]) -> list[float]:
    """Per-fold Paired Delta: candidate minus Incumbent, fold by fold.

    The two runs must share the Canonical Fold Partition (same ``fold_seed``), so
    each fold's AUCs are paired and the difference is a like-for-like delta.
    """
    if len(candidate_fold_aucs) != len(incumbent_fold_aucs):
        raise ValueError(
            "candidate and incumbent must have the same fold count to pair: "
            f"{len(candidate_fold_aucs)} vs {len(incumbent_fold_aucs)}"
        )
    return [float(c) - float(i) for c, i in zip(candidate_fold_aucs, incumbent_fold_aucs)]


def paired_delta_summary(deltas: Sequence[float]) -> tuple[float, int]:
    """The mean Paired Delta and the count of positive folds (the 4/5 gate input)."""
    deltas = [float(d) for d in deltas]
    mean = sum(deltas) / len(deltas)
    folds_positive = sum(1 for d in deltas if d > 0)
    return mean, folds_positive


def confirmation_summary(per_seed_oof_auc: Mapping[int, float]) -> dict[str, Any]:
    """Summarise a Confirmation Run of the standing Incumbent across seeds 0/1/2.

    Band (iii) freezes: nothing new enters, so this is the freeze-time
    Confirmation Run of the *standing Incumbent itself* — one config re-run on
    each of the three canonical fold seeds — read for how stable its OOF AUC is.
    Unlike a candidate's Confirmation Run there is no Incumbent to pair against
    (:func:`verdict.sign_holds` handles that case), so the summary is the
    per-seed OOF AUCs, their mean, and their spread (max - min). Pure, so it is
    tested without the ML stack; :func:`confirm` produces the inputs live.
    """
    import verdict

    expected = tuple(verdict.CONFIRMATION_SEEDS)
    seeds = tuple(sorted(int(s) for s in per_seed_oof_auc))
    if seeds != expected:
        raise ValueError(
            f"a Confirmation Run has exactly fold seeds {expected}; got {seeds}"
        )
    aucs = [float(per_seed_oof_auc[s]) for s in seeds]
    mean = sum(aucs) / len(aucs)
    return {
        "seeds": list(seeds),
        "oof_aucs": aucs,
        "mean_oof_auc": mean,
        "oof_spread": max(aucs) - min(aucs),
    }


def oof_correlation(a: Sequence[float], b: Sequence[float]) -> float:
    """Pearson correlation between two OOF vectors on the same rows.

    How far a Member's errors move with another's -- the quantity an Arena
    family's gate reads against the Incumbent (ADR-0006 §3). Both vectors must
    come from the same fold seed, so row ``i`` is the same training row in each.
    """
    import numpy as np

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"OOF vectors must align row for row: {a.shape} vs {b.shape}")
    return float(np.corrcoef(a, b)[0, 1])


def kill_criterion_outcome(mean_delta: float, kill_delta: float) -> dict[str, Any]:
    """Apply a candidate's declared kill criterion to its mean Paired Delta.

    The criterion is declared before the run: a mean paired delta *below*
    ``kill_delta`` on the canonical seed kills the candidate. Recorded either
    way, so the outcome is in the ledger whether it lived or died.
    """
    return {"threshold": float(kill_delta), "dead": float(mean_delta) < float(kill_delta)}


def folds_positive_kill_outcome(
    folds_positive: int, min_folds_positive: int, n_folds: int = 5
) -> dict[str, Any]:
    """Apply a folds-positive kill criterion — the Resolution sweep's rule.

    The ``max_bin`` sweep is not judged on a mean-delta threshold: a value
    survives only if it beats the Incumbent in at least ``min_folds_positive`` of
    ``n_folds`` folds, else the Resolution axis freezes at the Incumbent's value.
    Recorded either way, so the outcome is in the ledger whether it lived or died.
    """
    return {
        "min_folds_positive": int(min_folds_positive),
        "n_folds": int(n_folds),
        "folds_positive": int(folds_positive),
        "dead": int(folds_positive) < int(min_folds_positive),
    }


def seed_bag_kill_outcome(
    mean_delta: float, n_seeds: int, value_per_run: float
) -> dict[str, Any]:
    """Apply the seed-averaged bag's cost-versus-gain kill (#19).

    The bag fits the same config once per seed and averages the fold
    predictions, so ``n_seeds - 1`` fits are *extra* over the single Incumbent
    fit. Each extra fit must earn ``value_per_run`` AUC, so the break-even gain
    is ``value_per_run * (n_seeds - 1)``. A mean Paired Delta below that means
    the compute cost exceeds the measured gain and the bag is dead. Recorded
    either way.
    """
    extra_runs = int(n_seeds) - 1
    required_gain = float(value_per_run) * extra_runs
    return {
        "n_seeds": int(n_seeds),
        "extra_runs": extra_runs,
        "value_per_run": float(value_per_run),
        "required_gain": required_gain,
        "dead": float(mean_delta) < required_gain,
    }


def tuning_kill_outcome(
    mean_delta: float, elapsed_s: float, kill_delta: float, time_budget_s: float
) -> dict[str, Any]:
    """Apply the conservative-tuning kill: 2h without +``kill_delta`` -> dead (#19).

    Two ways to die, declared before the run: the mean Paired Delta falls below
    ``kill_delta``, or the machine time spent exceeds ``time_budget_s`` (the 2h
    box) — blowing the budget kills the candidate even if the target was reached.
    Recorded either way.
    """
    over_budget = float(elapsed_s) > float(time_budget_s)
    under_target = float(mean_delta) < float(kill_delta)
    return {
        "threshold": float(kill_delta),
        "time_budget_s": float(time_budget_s),
        "elapsed_s": float(elapsed_s),
        "over_budget": over_budget,
        "dead": under_target or over_budget,
    }


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
        # ``dirty`` answers "was the code that produced this number committed?",
        # so the ledger is excluded from the question. It is append-only and the
        # previous run's own line is normally sitting in it uncommitted, which
        # would otherwise mark every run after the first as dirty and make the
        # flag useless exactly when it matters.
        status = _git("status", "--porcelain", "--", ".", ":(exclude)ledger")
        dirty = bool(status)
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
    fold_deltas: Sequence[float] | None = None,
    kill_criterion: Mapping[str, Any] | None = None,
    fold_wall_times: Sequence[float] | None = None,
    incumbent_oof_corr: float | None = None,
) -> dict[str, Any]:
    """Assemble one Run Record — the unit the next CRISP turn reads.

    Carries everything #13 requires: run id, config hash, timestamp, git sha,
    dirty-tree flag, the full config, the five fold AUCs, the OOF AUC, the
    fold-partition sha256, the seeds, wall time and the OOF vector path. The
    Paired-Delta fields are present but null on the first run (no Incumbent yet).
    #37 adds the wall-clock time of each fold and the OOF correlation with the
    Incumbent's vector, null when there is no Incumbent run to read.
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
        "fold_deltas": [float(d) for d in fold_deltas] if fold_deltas is not None else None,
        "kill_criterion": dict(kill_criterion) if kill_criterion is not None else None,
        "fold_wall_times": (
            [float(t) for t in fold_wall_times] if fold_wall_times is not None else None
        ),
        "incumbent_oof_corr": float(incumbent_oof_corr) if incumbent_oof_corr is not None else None,
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


def _latest_run_for(experiment: str, fold_seed: int, path: Path | None = None) -> dict[str, Any] | None:
    """The most recent Comparison Run of an experiment on a given fold seed.

    Used to resolve a candidate's Incumbent: the pairing is only valid on the
    shared Canonical Fold Partition, so the fold seed must match.
    """
    match: dict[str, Any] | None = None
    for record in load_records(path):
        if record.get("experiment") != experiment:
            continue
        if record.get("config", {}).get("fold_seed") != fold_seed:
            continue
        match = record  # later writes win: the latest run on this seed
    return match


# --------------------------------------------------------------------------- #
# The Comparison Run itself (needs the ML stack + the gitignored CSVs).
# --------------------------------------------------------------------------- #
def fold_adapter(config, outer_fold: int, validation_index):
    """The Model Adapter for one outer fold, configured from ``config``.

    Shared by the Comparison Run and the Submission Fit so the two cannot
    drift. They did drift: the Submission Fit built ``Adapter(scale=...)``
    alone, silently dropping the target encoding, the oversampler and the
    validation index — so a promoted candidate would have submitted
    predictions from a different model than the one that was scored. For the
    Incumbent, whose ``target_encode`` is empty, that difference was invisible.
    """
    import adapter as adapter_mod
    import frame

    oversample = getattr(config, "oversample", None)
    target_encode = getattr(config, "target_encode", ())
    linear_design = getattr(config, "linear_design", False)
    scale_columns = ()
    if config.scale:
        scale_columns = (
            adapter_mod.linear_scale_columns(target_encode)
            if linear_design
            else (frame.INCOME_COLUMN,)
        )
    return adapter_mod.Adapter(
        scale=config.scale,
        target_encode=target_encode,
        outer_fold=outer_fold,
        validation_index=set(validation_index),
        scale_columns=scale_columns,
        scale_exclude=tuple(name for name, _ in frame.INCOME_DIGIT_TRANSFORMS),
        oversample=oversample,
        oversample_continuous_columns=(
            (frame.INCOME_COLUMN,) + tuple(n for n, _ in frame.INCOME_DIGIT_TRANSFORMS)
            if oversample else ()
        ),
        income_column=frame.INCOME_COLUMN if oversample else None,
        income_digit_transforms=frame.INCOME_DIGIT_TRANSFORMS if oversample else (),
        recipe_margin=getattr(config, "recipe_margin", False),
        fitted_margin=getattr(config, "fitted_margin", False),
        fitted_margin_calibrate=getattr(config, "fitted_margin_calibrate", False),
        linear_design=linear_design,
        linear_gate=getattr(config, "linear_gate", True),
    )


def fold_predict(config, X_tr, y_fit, X_out, weight=None):
    """Fit this fold's model(s) on ``X_tr`` and predict ``X_out``.

    Honours a seed bag by averaging one member per seed. Shared with the
    Submission Fit for the same reason as :func:`fold_adapter`. When
    ``config.recipe_margin`` (#28) or ``config.fitted_margin`` (#33) is set,
    the Adapter has carried its margin in :data:`adapter.RECIPE_MARGIN_COLUMN`
    or :data:`adapter.FITTED_MARGIN_COLUMN` — popped out here and passed as
    ``init_score`` rather than left in as an ordinary feature, since it is the
    model's initial prediction, not an input to learn a split on. ``weight``
    (#31) is an optional per-row sample weight — used by pseudo-labelling to
    admit a teacher's test-row labels at less than a real row's implicit 1.0;
    not supported together with a seed bag (nothing declared needs both).
    """
    import numpy as np

    import adapter as adapter_mod
    import experiments as experiments_mod
    import models

    seed_bag = getattr(config, "seed_bag", ()) or ()
    if weight is not None and seed_bag:
        raise ValueError("fold_predict: weight is not supported together with a seed bag")
    cat_features = getattr(config, "cat_features", ()) or ()
    init_score_tr = init_score_out = None
    if getattr(config, "recipe_margin", False):
        init_score_tr = X_tr.pop(adapter_mod.RECIPE_MARGIN_COLUMN).to_numpy()
        init_score_out = X_out.pop(adapter_mod.RECIPE_MARGIN_COLUMN).to_numpy()
    elif getattr(config, "fitted_margin", False):
        init_score_tr = X_tr.pop(adapter_mod.FITTED_MARGIN_COLUMN).to_numpy()
        init_score_out = X_out.pop(adapter_mod.FITTED_MARGIN_COLUMN).to_numpy()
    if seed_bag:
        member_preds = [
            models.predict(
                models.fit(
                    X_tr,
                    y_fit,
                    experiments_mod.seeded_params(config.params, s),
                    num_boost_round=config.num_boost_round,
                    family=config.model,
                    cat_features=cat_features,
                    init_score=init_score_tr,
                ),
                X_out,
                family=config.model,
                cat_features=cat_features,
                init_score=init_score_out,
            )
            for s in seed_bag
        ]
        return np.mean(np.asarray(member_preds), axis=0)
    model = models.fit(
        X_tr,
        y_fit,
        config.params,
        num_boost_round=config.num_boost_round,
        family=config.model,
        cat_features=cat_features,
        init_score=init_score_tr,
        weight=weight,
    )
    return models.predict(model, X_out, family=config.model, cat_features=cat_features, init_score=init_score_out)


def pseudo_label_augment(config, X_tr, y_fit, X_te, teacher_pred):
    """Augment a fold's training rows with the teacher's test-row labels (#31).

    ``teacher_pred`` must come from a model fit on this fold's own ``X_tr``/
    ``y_fit`` only — the per-fold teacher/student split the ticket's kill
    criterion depends on; nothing here re-derives that discipline, it only
    consumes the prediction.

    Two mechanisms, selected by whether ``config.pseudo_label_threshold`` is
    set:

    * **threshold variant** — keep only the test rows the teacher is confident
      on (probability outside ``[1 - threshold, threshold]``), hard-label them
      (round to 0/1), and admit them at ``config.pseudo_label_weight``.
    * **weight variant** (``pseudo_label_threshold`` is ``None``) — keep every
      test row, soft-labelled with the teacher's raw probability (LightGBM's
      binary objective accepts a continuous target as a cross-entropy soft
      label), admitted at ``config.pseudo_label_weight``.

    Real rows keep weight 1.0 either way, so the pseudo-labelled rows are
    additions to the fold, never a replacement of it.
    """
    import numpy as np
    import pandas as pd

    threshold = getattr(config, "pseudo_label_threshold", None)
    weight = float(getattr(config, "pseudo_label_weight", None) or 1.0)
    teacher_pred = np.asarray(teacher_pred, dtype=np.float64)

    if threshold is not None:
        confident = (teacher_pred >= threshold) | (teacher_pred <= 1.0 - threshold)
        X_pl = X_te.iloc[confident].reset_index(drop=True)
        y_pl = np.round(teacher_pred[confident])
    else:
        X_pl = X_te.reset_index(drop=True)
        y_pl = teacher_pred

    X_aug = pd.concat([X_tr.reset_index(drop=True), X_pl], ignore_index=True)
    y_aug = np.concatenate([np.asarray(y_fit, dtype=np.float64), y_pl])
    w_aug = np.concatenate(
        [np.ones(len(y_fit), dtype=np.float64), np.full(len(y_pl), weight, dtype=np.float64)]
    )
    return X_aug, y_aug, w_aug


def run(config) -> dict[str, Any]:
    """Execute one Comparison Run end to end and record it.

    ``config`` is an :class:`experiments.Experiment`. Returns the Run Record it
    appended to the ledger.
    """
    import numpy as np
    from sklearn.metrics import roc_auc_score

    import adapter as adapter_mod
    import data
    import experiments as experiments_mod
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

    X_train, X_test = frame.build_frame(train, test, config.frame)

    # The seed-averaged bag fits the same config once per seed and averages the
    # fold predictions; an empty bag is the single Incumbent fit at its own seed.
    seed_bag = getattr(config, "seed_bag", ()) or ()
    oversample = getattr(config, "oversample", None)
    pseudo_label = getattr(config, "pseudo_label", False)
    if pseudo_label and seed_bag:
        raise ValueError("pseudo_label is not supported together with a seed bag")

    oof = np.zeros(len(y), dtype=np.float64)
    fold_aucs: list[float] = []
    fold_wall_times: list[float] = []
    for k in range(data.N_FOLDS):
        t_fold = time.perf_counter()
        tr_idx = np.where(fold != k)[0]
        va_idx = np.where(fold == k)[0]

        adapter = fold_adapter(config, k, va_idx.tolist())
        X_tr = adapter.fit_transform(X_train.iloc[tr_idx], y[tr_idx])
        X_va = adapter.transform(X_train.iloc[va_idx])
        # Oversampling synthesises training rows, so the model is fitted on the
        # resampled targets, not the original fold slice.
        y_fit = adapter.resampled_y if adapter.resampled_y is not None else y[tr_idx]

        if pseudo_label:
            # The teacher is fit on this fold's own X_tr/y_fit only — never the
            # validation rows, never a fold-crossing view of the training set —
            # and predicts the test rows through the same fold-fitted Adapter
            # the Submission Fit uses (adapter.transform(X_test)). Its
            # predictions become the pseudo-labels the student below is
            # trained on; the student, not the teacher, produces the OOF.
            X_te = adapter.transform(X_test)
            teacher_pred = fold_predict(config, X_tr, y_fit, X_te)
            X_aug, y_aug, w_aug = pseudo_label_augment(config, X_tr, y_fit, X_te, teacher_pred)
            preds = fold_predict(config, X_aug, y_aug, X_va, weight=w_aug)
        else:
            preds = fold_predict(config, X_tr, y_fit, X_va)
        oof[va_idx] = preds
        fold_aucs.append(float(roc_auc_score(y[va_idx], preds)))
        fold_wall_times.append(time.perf_counter() - t_fold)
        print(f"fold {k}: AUC {fold_aucs[-1]:.5f} in {fold_wall_times[-1]:.1f}s", flush=True)

    oof_auc = float(roc_auc_score(y, oof))

    # A candidate is measured as a Paired Delta against its declared Incumbent —
    # per-fold on the shared Canonical Fold Partition — and its declared kill
    # criterion is applied to the result. Recorded either way.
    incumbent_run_id = paired_delta = folds_positive = deltas = kill = None
    incumbent_oof_corr = None
    incumbent_name = getattr(config, "incumbent", None)
    if incumbent_name:
        inc = _latest_run_for(incumbent_name, config.fold_seed)
        if inc is None:
            print(
                f"NOTE: incumbent {incumbent_name!r} has no run on fold_seed "
                f"{config.fold_seed} in the ledger; recording the candidate with "
                "no Paired Delta. Run the Incumbent first to arm the comparison."
            )
        else:
            deltas = fold_deltas(fold_aucs, inc["fold_aucs"])
            paired_delta, folds_positive = paired_delta_summary(deltas)
            incumbent_run_id = inc["run_id"]
            inc_oof_path = _repo_root() / inc["oof_path"]
            if inc_oof_path.exists():
                incumbent_oof_corr = oof_correlation(oof, np.load(inc_oof_path))
            else:
                print(f"NOTE: incumbent OOF vector {inc['oof_path']} is missing; no correlation recorded.")
            elapsed = time.perf_counter() - t0
            if getattr(config, "kill_time_budget_s", None) is not None:
                # Conservative tuning: 2h without +kill_delta -> dead (blowing the
                # machine-time box kills it even if the target was reached).
                kill = tuning_kill_outcome(
                    paired_delta, elapsed, config.kill_delta, config.kill_time_budget_s
                )
            elif getattr(config, "kill_value_per_run", None) is not None:
                # The seed bag's cost-versus-gain kill: each extra fit must earn
                # kill_value_per_run AUC.
                kill = seed_bag_kill_outcome(
                    paired_delta, len(config.seed_bag), config.kill_value_per_run
                )
            elif getattr(config, "kill_delta", None) is not None:
                kill = kill_criterion_outcome(paired_delta, config.kill_delta)
            elif getattr(config, "kill_min_folds_positive", None) is not None:
                kill = folds_positive_kill_outcome(
                    folds_positive, config.kill_min_folds_positive
                )

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
        incumbent_run_id=incumbent_run_id,
        paired_delta=paired_delta,
        folds_positive=folds_positive,
        fold_deltas=deltas,
        kill_criterion=kill,
        fold_wall_times=fold_wall_times,
        incumbent_oof_corr=incumbent_oof_corr,
    )
    # Keep the id and the recorded oof path consistent with the vector on disk.
    record["run_id"] = rid
    append_run_record(record)

    _print_verdict(config, record)
    return record


def confirm(config) -> dict[str, Any]:
    """Run a Confirmation Run across fold seeds 0/1/2 and read whether it passes.

    Two shapes, and issue #6 only gates a submission with the first:

    - **A candidate's** Confirmation Run re-runs *candidate and Incumbent* on
      each seed and requires the paired delta to keep its sign on all three.
      Re-running the candidate alone measures nothing: the seeds move the
      partition, so the candidate's own OOF wanders for reasons that have
      nothing to do with the change under test. The pairing is the point.
    - **The standing Incumbent's** freeze-time run (band (iii)) has nothing to
      pair against, so it is read as the spread of its OOF across the seeds.

    Both are reported. The Incumbent is run first on each seed when it has no
    run there yet, so the pairing is computable; every execution goes through
    the single :func:`run` path, so each seed's ledger write, fold assert and
    git capture happen in one place.

    Heavy: up to six Comparison Runs, so it only executes where the ML stack
    and the gitignored CSVs are present. The summary maths is pure and tested
    separately.
    """
    import verdict
    from dataclasses import replace

    import experiments as experiments_mod

    incumbent_name = getattr(config, "incumbent", None)
    per_seed: dict[int, float] = {}
    per_seed_delta: dict[int, float] = {}
    for seed in verdict.CONFIRMATION_SEEDS:
        if incumbent_name and _latest_run_for(incumbent_name, seed) is None:
            # Arm the pairing on this seed before the candidate runs on it.
            run(replace(experiments_mod.resolve(incumbent_name), fold_seed=seed))
        record = run(replace(config, fold_seed=seed))
        per_seed[seed] = record["oof_auc"]
        if record.get("paired_delta") is not None:
            per_seed_delta[seed] = float(record["paired_delta"])

    summary = confirmation_summary(per_seed)
    print(
        f"Confirmation Run of {config.name!r} across fold seeds "
        f"{'/'.join(map(str, summary['seeds']))}: mean OOF "
        f"{summary['mean_oof_auc']:.5f}, spread {summary['oof_spread']:.5f} "
        "(per-seed: "
        + ", ".join(f"{s}:{a:.5f}" for s, a in zip(summary["seeds"], summary["oof_aucs"]))
        + ")."
    )

    if incumbent_name:
        seeds = tuple(verdict.CONFIRMATION_SEEDS)
        if len(per_seed_delta) != len(seeds):
            missing = [s for s in seeds if s not in per_seed_delta]
            summary["paired_deltas"] = None
            summary["sign_holds"] = None
            print(
                f"CONFIRMATION INCONCLUSIVE — no Paired Delta against "
                f"{incumbent_name!r} on fold seed(s) {missing}. The gate in #6 "
                "requires candidate and Incumbent on all three seeds; nothing "
                "may be submitted on this result."
            )
        else:
            deltas = [per_seed_delta[s] for s in seeds]
            holds = verdict.sign_holds(deltas)
            summary["paired_deltas"] = deltas
            summary["sign_holds"] = holds
            shown = ", ".join(f"{s}:{d:+.5f}" for s, d in zip(seeds, deltas))
            print(
                f"CONFIRMATION {'PASSED' if holds else 'FAILED'} — paired delta vs "
                f"{incumbent_name!r} ({shown}); the sign "
                f"{'holds' if holds else 'does not hold'} on all "
                f"{len(seeds)} seeds."
            )
    return summary


def _print_verdict(config, record: Mapping[str, Any]) -> None:
    oof_auc = record["oof_auc"]
    gate = config.health_gate
    print(f"[{record['run_id']}] OOF AUC = {oof_auc:.5f}  (fold AUCs: "
          + ", ".join(f"{a:.5f}" for a in record["fold_aucs"]) + ")")
    if record.get("fold_wall_times"):
        times = record["fold_wall_times"]
        print("Wall time per fold: " + ", ".join(f"{t:.1f}s" for t in times)
              + f" (mean {sum(times) / len(times):.1f}s)")
    if record.get("incumbent_oof_corr") is not None:
        print(f"OOF correlation with the Incumbent: {record['incumbent_oof_corr']:.5f}")
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
    _print_kill_criterion(record)
    print_promotion_verdict(record)


def _print_kill_criterion(record: Mapping[str, Any]) -> None:
    """Print the declared kill-criterion outcome, if the candidate had one.

    Four shapes, distinguished by the keys their outcome carries:
    the seed bag's cost-versus-gain break-even (``required_gain``), conservative
    tuning's time-boxed threshold (``time_budget_s``), the Resolution sweep's
    folds-positive rule (``min_folds_positive`` — a value must beat the Incumbent
    in at least that many of ``n_folds`` folds, else the axis freezes), and the
    plain mean-delta threshold (income/Age TE).
    """
    kill = record.get("kill_criterion")
    if not kill:
        return
    delta = record.get("paired_delta")
    if "required_gain" in kill:
        # The seed bag's cost-versus-gain kill.
        req = kill["required_gain"]
        extra = kill["extra_runs"]
        if kill["dead"]:
            print(
                f"KILL CRITERION: DEAD — paired delta {delta:+.5f} < break-even "
                f"+{req:.5f} for {extra} extra fit(s); compute cost exceeds the "
                "measured gain and the bag is dead."
            )
        else:
            print(
                f"KILL CRITERION: SURVIVED — paired delta {delta:+.5f} >= break-even "
                f"+{req:.5f} for {extra} extra fit(s); the bag earns its compute."
            )
        return
    if "time_budget_s" in kill:
        # Conservative tuning: 2h without +threshold -> dead.
        thr = kill["threshold"]
        budget = kill["time_budget_s"]
        if kill["dead"]:
            why = (
                f"machine time {kill['elapsed_s']:.0f}s exceeded the {budget:.0f}s box"
                if kill.get("over_budget")
                else f"paired delta {delta:+.5f} < declared +{thr:.4f}"
            )
            print(f"KILL CRITERION: DEAD — {why}; the candidate is killed as declared.")
        else:
            print(
                f"KILL CRITERION: SURVIVED — paired delta {delta:+.5f} >= declared "
                f"+{thr:.4f} within the {budget:.0f}s box; the candidate lives to "
                "the verdict rule."
            )
        return
    if "min_folds_positive" in kill:
        fp = kill["folds_positive"]
        need = kill["min_folds_positive"]
        n = kill["n_folds"]
        if kill["dead"]:
            print(
                f"KILL CRITERION: DEAD — positive in only {fp}/{n} folds < "
                f"declared {need}/{n}; the Resolution axis freezes at 511."
            )
        else:
            print(
                f"KILL CRITERION: SURVIVED — positive in {fp}/{n} folds >= "
                f"declared {need}/{n}; the candidate lives to the verdict rule."
            )
        return
    threshold = kill["threshold"]
    if kill["dead"]:
        print(
            f"KILL CRITERION: DEAD — paired delta {delta:+.5f} < declared "
            f"threshold +{threshold:.4f}; the candidate is killed as declared."
        )
    else:
        print(
            f"KILL CRITERION: SURVIVED — paired delta {delta:+.5f} >= declared "
            f"threshold +{threshold:.4f}; the candidate lives to the verdict rule."
        )


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
    parser.add_argument(
        "--queue",
        action="store_true",
        help="print the band (ii) experiment queue in its declared run order and exit",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="run a Confirmation Run of the experiment across fold seeds 0/1/2 (freeze)",
    )
    return parser


def _print_queue() -> None:
    """Print the band (ii) queue in declared order, so the next run is one read.

    Story 62: the queue is run in its declared order. This surfaces that order —
    the single source of truth in :mod:`experiments` — rather than leaving it to
    be reconstructed from the spec.
    """
    import experiments

    print("Band (ii) experiment queue, in declared run order:")
    for i, name in enumerate(experiments.queue_names(), start=1):
        print(f"  {i}. {name}")
    separate = ", ".join(exp.name for exp in experiments.SEPARATELY_SCHEDULED)
    if separate:
        print(f"Separately scheduled (own slot): {separate}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.queue:
        _print_queue()
        return 0
    if not args.experiment:
        build_parser().error("an experiment name is required")

    import experiments

    config = experiments.resolve(args.experiment)
    if args.confirm:
        confirm(config)
    else:
        run(config)
    return 0


if __name__ == "__main__":
    main()
