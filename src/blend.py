"""The Blend harness: the combination rule, the weight search, and the leak (#23).

Two decisions, declared before any code, because a Blend's whole value is a
number that must be trusted to transfer:

**The rule: weighted rank-average.** Members' OOF vectors are on wildly
different scales once a non-tree family joins the Arena — a linear model's
calibration is not a LightGBM's. ROC AUC only ever asks "is the positive
ranked above the negative", so the scale-free, AUC-metric-native combination
is to average each Member's *rank*, not its raw probability. Logit-average was
considered and rejected: it assumes the members are linear-in-log-odds of one
another, which is exactly the assumption a family as different as a ridge
logistic model breaks. A plain weighted arithmetic mean was rejected for the
same reason logit-average was — both are scale-sensitive where rank-average is
not.

**The weight search: greedy hill-climb with replacement.** Uniform weights
were rejected because they wire in "diversity beats fit" without measuring it,
and non-negative least squares on the logit was rejected because it optimises
squared error in a space that assumes the linear-in-log-odds relationship
:func:`rank_transform` exists to avoid — the wrong objective in the wrong
space. Greedy hill-climb (Caruana's ensemble-selection with replacement)
optimises the actual reported metric, AUC, directly and pairs naturally with a
rank-average combination: repeatedly add whichever Member (with replacement)
most improves the running blend's AUC, stopping when nothing does.

**The leak, and the two numbers.** Weights chosen and scored on the same rows
are exactly ADR-0001's failure shape wearing new clothes — the nested cross-fit
that protects target encoding protects nothing here unless the same discipline
is applied to weight selection. So :func:`honest_blend_oof` chooses fold k's
weights on the *other four* folds' rows only (per-outer-fold nested weight
selection, the direct analogue of the Adapter's inner cross-fit) and applies
them only to fold k — that is the number :func:`run_blend` reports and gates
the kill criterion on. :func:`naive_blend_oof` chooses one weight vector on
every row and scores it on those same rows — the leaked, in-sample number,
produced once per Blend so the size of the trap is a fact on the record
(``record["blend_naive"]``) rather than folklore.

**The Submission Fit.** A Blend's test-set predictions need the same per-family
Submission Fit (mean of the five fold models) submission.py already builds for
a single Experiment, rank-transformed per Member and combined with one weight
vector. The test set carries no target, so there is nothing for a weight
vector fitted on all of the training OOF rows to leak into — the *naive*
weights are the right ones here, not a per-fold set with no test-side fold to
apply them to. This needs no change to ``adapter``: :func:`blend_submission_fit`
composes existing per-Member fold fitting (``runner.fold_adapter`` /
``runner.fold_predict``, exactly as ``submission.submit`` already calls them)
with the pure combination primitives below.

Everything above :func:`run_blend` and :func:`blend_submission_fit` is pure and
dependency-light — numpy when present, a plain-Python fallback otherwise,
exactly like ``submission.mean_of_fold_predictions`` — so the rule, the weight
search and the leak protocol are all tested without the ML stack. The two
heavy functions defer pandas/numpy/the ledger the same way ``runner.run`` and
``submission.submit`` do.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import runner


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# Rank transform and the pure (sklearn-free) AUC the weight search optimises.
# --------------------------------------------------------------------------- #
def _average_ranks_stdlib(values: Sequence[float]) -> list[float]:
    """1-indexed average ranks, ties sharing the mean rank of their group."""
    n = len(values)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def _average_ranks_numpy(values):
    import numpy as np

    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n == 0:
        return arr
    order = np.argsort(arr, kind="mergesort")
    sorted_vals = arr[order]
    pos = np.arange(1, n + 1, dtype=float)
    is_new_group = np.empty(n, dtype=bool)
    is_new_group[0] = True
    is_new_group[1:] = sorted_vals[1:] != sorted_vals[:-1]
    group_id = np.cumsum(is_new_group) - 1
    group_sum = np.bincount(group_id, weights=pos)
    group_count = np.bincount(group_id)
    avg_rank_per_group = group_sum / group_count
    ranks_sorted = avg_rank_per_group[group_id]
    ranks = np.empty(n, dtype=float)
    ranks[order] = ranks_sorted
    return ranks


def rank_transform(values: Sequence[float]) -> list[float]:
    """Fractional rank in ``[0, 1]``, average rank for ties.

    The combination happens on this transform, never on the raw probability
    (see the module docstring): it is what makes the rule scale-free.
    """
    n = len(values)
    if n <= 1:
        return [0.0] * n
    try:
        ranks = _average_ranks_numpy(values)
        return ((ranks - 1.0) / (n - 1.0)).tolist()
    except ImportError:
        ranks = _average_ranks_stdlib(values)
        return [(r - 1.0) / (n - 1.0) for r in ranks]


def pure_auc(y: Sequence[int], scores: Sequence[float]) -> float:
    """ROC AUC via the Mann-Whitney U rank-sum identity — no sklearn needed.

    ``AUC = (sum of ranks of the positive class - n_pos*(n_pos+1)/2) /
    (n_pos * n_neg)``, with ties sharing the average rank of their group (the
    standard tie correction for this statistic). Dependency-free so the weight
    search — called thousands of times inside :func:`greedy_hillclimb_weights`
    — and its tests never need the ML stack.
    """
    n = len(scores)
    if n != len(y):
        raise ValueError(f"y and scores must have the same length: {len(y)} vs {n}")
    y = [float(v) for v in y]
    n_pos = sum(y)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC is undefined with only one class present")
    try:
        ranks = _average_ranks_numpy(scores)
        import numpy as np

        sum_ranks_pos = float(ranks[np.asarray(y) == 1.0].sum())
    except ImportError:
        ranks = _average_ranks_stdlib(scores)
        sum_ranks_pos = sum(r for r, label in zip(ranks, y) if label == 1.0)
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


# --------------------------------------------------------------------------- #
# Weighted combination and the greedy weight search.
# --------------------------------------------------------------------------- #
def weighted_combine(
    rank_vectors: Sequence[Sequence[float]], weights: Sequence[float]
) -> list[float]:
    """The weighted arithmetic mean of already rank-transformed vectors.

    ``weights`` need not sum to 1 — they are normalised here, so a raw vote
    count from :func:`greedy_hillclimb_weights` can be passed directly.
    """
    if len(rank_vectors) != len(weights):
        raise ValueError(
            f"{len(rank_vectors)} rank vector(s) but {len(weights)} weight(s)"
        )
    total = sum(weights)
    if total <= 0:
        raise ValueError("blend weights must sum to a positive total")
    norm = [w / total for w in weights]
    try:
        import numpy as np

        stacked = np.stack([np.asarray(r, dtype=float) for r in rank_vectors])
        return (np.asarray(norm, dtype=float) @ stacked).tolist()
    except ImportError:
        n = len(rank_vectors[0])
        return [sum(w * r[i] for w, r in zip(norm, rank_vectors)) for i in range(n)]


def greedy_hillclimb_weights(
    rank_vectors: Sequence[Sequence[float]],
    y: Sequence[int],
    *,
    max_rounds: int = 40,
    min_gain: float = 1e-9,
) -> list[float]:
    """Caruana-style ensemble selection with replacement, optimising AUC.

    Starting from an empty ensemble, repeatedly add whichever Member (with
    replacement — a Member may be picked more than once) most improves the
    running blend's AUC on ``y``/``rank_vectors``, stopping after
    ``max_rounds`` or once no Member improves it by more than ``min_gain``.
    Ties keep the first (lowest-index) Member, so the search is deterministic.
    Returns normalised weights (a Member never picked gets weight 0); an
    ensemble where nothing ever improved over doing nothing falls back to
    uniform weights rather than returning an all-zero vector.
    """
    n_members = len(rank_vectors)
    if n_members == 0:
        raise ValueError("greedy_hillclimb_weights needs at least one member")
    n = len(y)
    counts = [0.0] * n_members
    running_sum = [0.0] * n
    current_auc = 0.5  # the AUC of no information; the first pick must beat it
    for _ in range(max_rounds):
        total_next = sum(counts) + 1.0
        best_idx = None
        best_auc = current_auc
        for m in range(n_members):
            trial = [(s + r) / total_next for s, r in zip(running_sum, rank_vectors[m])]
            auc = pure_auc(y, trial)
            if auc > best_auc + min_gain:
                best_auc = auc
                best_idx = m
        if best_idx is None:
            break
        counts[best_idx] += 1.0
        running_sum = [s + r for s, r in zip(running_sum, rank_vectors[best_idx])]
        current_auc = best_auc
    total = sum(counts)
    if total == 0:
        return [1.0 / n_members] * n_members
    return [c / total for c in counts]


# --------------------------------------------------------------------------- #
# The two numbers: honest (per-outer-fold nested) and naive (in-sample).
# --------------------------------------------------------------------------- #
def honest_blend_oof(
    rank_vectors: Sequence[Sequence[float]],
    y: Sequence[int],
    fold_ids: Sequence[int],
    *,
    n_folds: int,
    max_rounds: int = 40,
):
    """Per-outer-fold nested weight selection: the honest Paired-Delta number.

    Fold k's weights are chosen with :func:`greedy_hillclimb_weights` on the
    *other* ``n_folds - 1`` folds' rows only, then applied to fold k — the
    direct analogue of the Model Adapter's inner cross-fit (ADR-0001), so a
    weight is never scored on a row that chose it. Returns the combined OOF
    vector and the per-fold weight list (kept for inspection; the Submission
    Fit uses the naive weights instead, see the module docstring).
    """
    import numpy as np

    y_arr = np.asarray(y, dtype=float)
    fold_arr = np.asarray(fold_ids)
    n = len(y_arr)
    oof = np.zeros(n, dtype=float)
    weights_per_fold: list[list[float]] = []
    for k in range(n_folds):
        train_mask = fold_arr != k
        val_mask = fold_arr == k
        train_ranks = [list(np.asarray(r)[train_mask]) for r in rank_vectors]
        weights = greedy_hillclimb_weights(
            train_ranks, y_arr[train_mask].tolist(), max_rounds=max_rounds
        )
        weights_per_fold.append(weights)
        val_ranks = [list(np.asarray(r)[val_mask]) for r in rank_vectors]
        oof[val_mask] = weighted_combine(val_ranks, weights)
    return oof, weights_per_fold


def naive_blend_oof(
    rank_vectors: Sequence[Sequence[float]],
    y: Sequence[int],
    *,
    max_rounds: int = 40,
):
    """Weights chosen and scored on every row — the in-sample, leaked number.

    Produced once per Blend precisely so the gap between this and
    :func:`honest_blend_oof` is a measured quantity (``blend_naive`` on the
    Run Record), not an assumption about how big the trap is.
    """
    weights = greedy_hillclimb_weights(rank_vectors, list(y), max_rounds=max_rounds)
    combined = weighted_combine(rank_vectors, weights)
    return combined, weights


# --------------------------------------------------------------------------- #
# The kill criterion: beats its best Member by +0.0003, positive in >=4/5 folds.
# --------------------------------------------------------------------------- #
def blend_kill_outcome(
    mean_delta: float,
    folds_positive: int,
    *,
    kill_delta: float,
    min_folds_positive: int,
    n_folds: int = 5,
) -> dict[str, Any]:
    """Apply the Blend's declared kill criterion — both conditions, not either.

    A Blend that clears the magnitude bar on one lucky fold but not the other
    four has bought nothing, so both conditions gate independently and the
    outcome is dead unless it clears both. Recorded either way.
    """
    magnitude_ok = float(mean_delta) >= float(kill_delta)
    folds_ok = int(folds_positive) >= int(min_folds_positive)
    return {
        "kill_delta": float(kill_delta),
        "min_folds_positive": int(min_folds_positive),
        "n_folds": int(n_folds),
        "folds_positive": int(folds_positive),
        "dead": not (magnitude_ok and folds_ok),
    }


def _print_kill_criterion(record: Mapping[str, Any]) -> None:
    kill = record.get("kill_criterion")
    if not kill:
        return
    delta = record.get("paired_delta")
    thr = kill["kill_delta"]
    fp = kill["folds_positive"]
    need = kill["min_folds_positive"]
    n = kill["n_folds"]
    if kill["dead"]:
        print(
            f"KILL CRITERION: DEAD — paired delta vs best Member {delta:+.5f} "
            f"(need >= +{thr:.4f}), positive in {fp}/{n} folds (need >= "
            f"{need}/{n}); the Blend does not enter the finals conversation on "
            "its own."
        )
    else:
        print(
            f"KILL CRITERION: SURVIVED — paired delta vs best Member {delta:+.5f} "
            f">= +{thr:.4f}, positive in {fp}/{n} folds >= {need}/{n}."
        )


# --------------------------------------------------------------------------- #
# Declaring a Blend and resolving its Members.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BlendConfig:
    """A declared Blend: which Members, and the rule fixed by this ticket.

    ``members`` names Experiments, not frozen run ids — resolved to each
    name's latest Run Record on ``fold_seed`` at the moment the Blend runs, the
    same "declaration, not a frozen snapshot" idiom :mod:`experiments` uses
    for the Incumbent chain. The resolved run ids are what actually lands in
    the ledger's config hash, so "have I tried this exact combination?" stays
    answerable even as Members' own latest runs change underneath the name.
    """

    name: str
    hypothesis: str
    members: tuple[str, ...]
    fold_seed: int = 0
    kill_delta: float = 0.0003
    kill_min_folds_positive: int = 4
    max_rounds: int = 40


def resolve_members(config: BlendConfig) -> list[dict[str, Any]]:
    """Each declared Member's latest Run Record on ``config.fold_seed``.

    Raises if a declared Member has never been run on that seed — a Blend
    cannot combine a vector that does not exist.
    """
    records = []
    missing = []
    for name in config.members:
        record = runner._latest_run_for(name, config.fold_seed)
        if record is None:
            missing.append(name)
        else:
            records.append(record)
    if missing:
        raise ValueError(
            f"Blend {config.name!r} declares Member(s) with no Run Record on "
            f"fold_seed {config.fold_seed}: {', '.join(missing)}. Run each "
            "Member first — a Blend cannot combine a vector that does not exist."
        )
    return records


def load_member_oof(record: Mapping[str, Any]):
    """Load one Member's OOF vector from the path its Run Record points at."""
    import numpy as np

    return np.load(_repo_root() / record["oof_path"])


def best_member_record(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The declared Member with the highest OOF AUC — what the Blend is scored
    against (CONTEXT.md, "Blend": reported against its best Member, never the
    standing Incumbent)."""
    if not records:
        raise ValueError("cannot select a best Member from an empty set")
    best = records[0]
    for r in records[1:]:
        if float(r["oof_auc"]) > float(best["oof_auc"]):
            best = r
    return dict(best)


# --------------------------------------------------------------------------- #
# The Comparison Run itself (needs the ML stack + the gitignored CSVs).
# --------------------------------------------------------------------------- #
def run_blend(config: BlendConfig) -> dict[str, Any]:
    """Combine the declared Members and record the result like any other run.

    Reads every Member's OOF vector, produces the honest (per-outer-fold
    nested) and naive (in-sample) combined vectors, reports the Paired Delta
    of the *honest* number against the best Member, applies the kill
    criterion to it, and writes one Run Record through :mod:`runner` — the
    ``blend_naive`` field carries the naive number for comparison, never for
    the verdict.
    """
    import numpy as np

    import data
    from columns import TARGET_COLUMN

    t0 = time.perf_counter()
    when = datetime.now(timezone.utc)

    train = data.load_train()
    y = train[TARGET_COLUMN].to_numpy()
    fold = data.fold_ids_for(y, config.fold_seed)
    fold_sha = data.assert_fold_partition(fold, config.fold_seed)

    member_records = resolve_members(config)
    member_oofs = [load_member_oof(r) for r in member_records]
    for r, vec in zip(member_records, member_oofs):
        if len(vec) != len(y):
            raise ValueError(
                f"Member {r['experiment']!r} (run {r['run_id']}) has an OOF "
                f"vector of length {len(vec)}, but the training set has "
                f"{len(y)} rows — it was not run on this fold_seed's partition"
            )
    rank_vectors = [rank_transform(v) for v in member_oofs]

    honest_oof, weights_per_fold = honest_blend_oof(
        rank_vectors, y, fold, n_folds=data.N_FOLDS, max_rounds=config.max_rounds
    )
    fold_aucs = [
        pure_auc(y[fold == k], honest_oof[fold == k]) for k in range(data.N_FOLDS)
    ]
    oof_auc = pure_auc(y, honest_oof)

    naive_oof, naive_weights = naive_blend_oof(
        rank_vectors, y, max_rounds=config.max_rounds
    )
    naive_fold_aucs = [
        pure_auc(y[fold == k], np.asarray(naive_oof)[fold == k])
        for k in range(data.N_FOLDS)
    ]
    naive_oof_auc = pure_auc(y, naive_oof)

    best = best_member_record(member_records)
    deltas = runner.fold_deltas(fold_aucs, best["fold_aucs"])
    paired_delta, folds_positive = runner.paired_delta_summary(deltas)
    naive_deltas = runner.fold_deltas(naive_fold_aucs, best["fold_aucs"])
    naive_paired_delta, naive_folds_positive = runner.paired_delta_summary(naive_deltas)

    kill = blend_kill_outcome(
        paired_delta,
        folds_positive,
        kill_delta=config.kill_delta,
        min_folds_positive=config.kill_min_folds_positive,
    )

    rid = runner.run_id(config.name, when=when)
    oof_path = runner.RUNS_DIR / rid / "oof.npy"
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(oof_path, honest_oof)

    ledger_config = {
        "name": config.name,
        "fold_seed": config.fold_seed,
        "rule": "rank-average",
        "weight_search": "greedy-hillclimb-with-replacement",
        "leak_protocol": "per-outer-fold nested weight selection",
        "max_rounds": config.max_rounds,
        "members": [
            {"experiment": r["experiment"], "run_id": r["run_id"]} for r in member_records
        ],
    }

    record = runner.build_run_record(
        experiment=config.name,
        config=ledger_config,
        fold_aucs=fold_aucs,
        oof_auc=oof_auc,
        fold_sha256=fold_sha,
        oof_path=str(oof_path.relative_to(_repo_root())),
        wall_time=time.perf_counter() - t0,
        seeds=None,
        when=when,
        incumbent_run_id=best["run_id"],
        paired_delta=paired_delta,
        folds_positive=folds_positive,
        fold_deltas=deltas,
        kill_criterion=kill,
    )
    record["run_id"] = rid
    record["blend_naive"] = {
        "oof_auc": naive_oof_auc,
        "fold_aucs": naive_fold_aucs,
        "paired_delta": naive_paired_delta,
        "folds_positive": naive_folds_positive,
        # "the difference is the size of the trap" (#23) — positive means the
        # leak inflates the score, as ADR-0001 predicts.
        "leak_size": naive_oof_auc - oof_auc,
        "weights": naive_weights,
    }
    record["weights_per_fold"] = weights_per_fold
    runner.append_run_record(record)

    _print_blend_verdict(config, record, best)
    return record


def _print_blend_verdict(
    config: BlendConfig, record: Mapping[str, Any], best: Mapping[str, Any]
) -> None:
    print(
        f"[{record['run_id']}] Blend of {len(config.members)} Member(s) "
        f"({', '.join(config.members)}) — honest OOF AUC = "
        f"{record['oof_auc']:.5f} (fold AUCs: "
        + ", ".join(f"{a:.5f}" for a in record["fold_aucs"]) + ")"
    )
    naive = record["blend_naive"]
    print(
        f"  naive (in-sample) OOF AUC = {naive['oof_auc']:.5f} — leak size "
        f"{naive['leak_size']:+.5f} vs the honest number."
    )
    print(
        f"  Paired Delta vs best Member ({best['experiment']!r}, run "
        f"{best['run_id']}, OOF {best['oof_auc']:.5f}): "
        f"{record['paired_delta']:+.5f}, positive in "
        f"{record['folds_positive']}/{len(record['fold_aucs'])} folds."
    )
    _print_kill_criterion(record)
    runner.print_promotion_verdict(record)


# --------------------------------------------------------------------------- #
# The Blend's Submission Fit (needs the ML stack + the gitignored CSVs).
# --------------------------------------------------------------------------- #
def _member_test_predictions(experiment_name: str):
    """One Member's Submission Fit: the mean of its five fold models' test
    predictions, built exactly as ``submission.submit`` builds it — no change
    to ``adapter`` (see the module docstring). Reuses ``runner.fold_adapter``
    and ``runner.fold_predict`` so a Member's Submission Fit here is the same
    computation ``submission.submit`` would produce for it standalone.
    """
    import numpy as np

    import data
    import experiments as experiments_mod
    import frame
    import submission
    from columns import TARGET_COLUMN

    config = experiments_mod.resolve(experiment_name)
    train = data.load_train()
    test = data.load_test()
    y = train[TARGET_COLUMN].to_numpy()
    fold = data.fold_ids_for(y, config.fold_seed)
    data.assert_fold_partition(fold, config.fold_seed)
    X_train, X_test = frame.build_frame(train, test, config.frame)

    fold_preds = []
    for k in range(data.N_FOLDS):
        tr_idx = np.where(fold != k)[0]
        va_idx = np.where(fold == k)[0]
        adapter = runner.fold_adapter(config, k, va_idx.tolist())
        X_tr = adapter.fit_transform(X_train.iloc[tr_idx], y[tr_idx])
        X_te = adapter.transform(X_test)
        y_fit = adapter.resampled_y if adapter.resampled_y is not None else y[tr_idx]
        fold_preds.append(runner.fold_predict(config, X_tr, y_fit, X_te))
    return submission.mean_of_fold_predictions(fold_preds)


def blend_submission_fit(
    member_records: Sequence[Mapping[str, Any]], weights: Sequence[float]
) -> list[float]:
    """The Blend's Submission Fit: each Member's own Submission Fit,
    rank-transformed on the test set and combined with ``weights``.

    ``weights`` should be the *naive* (whole-OOF, in-sample) weights a
    :func:`run_blend` call recorded in ``record["blend_naive"]["weights"]`` —
    the right ones here, not the per-outer-fold set :func:`honest_blend_oof`
    produces: the test set carries no target, so there is no row for a weight
    to leak into, and no per-fold structure on the test side to apply five
    different weight vectors to. Composes existing per-Member fold fitting
    (:func:`_member_test_predictions`) with the pure combination primitives
    above — no change to ``adapter`` (see the module docstring).
    """
    test_preds = [_member_test_predictions(r["experiment"]) for r in member_records]
    test_ranks = [rank_transform(list(p)) for p in test_preds]
    return weighted_combine(test_ranks, weights)


# --------------------------------------------------------------------------- #
# CLI entry point.
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run-blend",
        description="Combine a declared Blend's Members and record it as a Run Record.",
    )
    parser.add_argument("blend", nargs="?", help="name of the declared Blend to run")
    parser.add_argument(
        "--list", action="store_true", help="print the declared Blends and exit"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    import blends as declared_blends

    args = build_parser().parse_args(argv)
    if args.list:
        for name in sorted(declared_blends.REGISTRY):
            print(name)
        return 0
    if not args.blend:
        build_parser().error("a Blend name is required")
    run_blend(declared_blends.resolve(args.blend))
    return 0


if __name__ == "__main__":
    main()
