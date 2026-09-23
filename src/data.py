"""Reading the CSVs in file order and the Canonical Fold Partition.

Owns ``load_train()`` / ``load_test()``, ``fold_ids(seed)``, and the sha256
assert on the fold-id vector. Two rules of the tracer bullet (#13) live here:

* The CSVs are read **in file order** — no sort, filter or row drop — because
  ``StratifiedKFold(shuffle=True)`` assigns folds by input row order, so the
  partition every banked measurement was taken on depends on that order.
* The partition is reconstructed in code from two constants (``N_FOLDS`` and
  ``FOLD_SEED``) and **never materialised to a file**: ``data/`` is gitignored,
  so such an artifact would not travel with the repo. A committed sha256 of the
  fold-id vector is asserted on every run instead.

pandas / numpy / scikit-learn are imported lazily inside the functions so the
module imports by bare name even where the ML stack is absent.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from columns import CATEGORICAL_COLUMNS, TARGET_COLUMN

N_FOLDS = 5
FOLD_SEED = 0


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


DATA_DIR = _repo_root() / "data"
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"

# sha256 of the canonical fold-id vector (int8, row order) for FOLD_SEED. The
# data/ directory is gitignored and absent in the agent environment, so this is
# recorded on the first run where the CSVs are present: assert_fold_partition
# returns the computed value and prints it for committing here. Once set, every
# run asserts against it and a mismatch — reordered rows, dropped rows, or a
# scikit-learn upgrade changing the split algorithm — fails loudly.
CANONICAL_FOLD_SHA256: str | None = None


def _load_csv(path: Path):
    import pandas as pd

    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. The CSVs live in the gitignored data/ dir; "
            "provision them before running a Comparison Run."
        )
    df = pd.read_csv(path)  # file order preserved: no sort/filter/row-drop.
    missing = df.isna().sum()
    offenders = missing[missing > 0]
    if len(offenders):
        raise AssertionError(
            f"{path.name} has missing values, but the no-imputation decision "
            f"requires none: {offenders.to_dict()}"
        )
    return df


def load_train():
    """Read ``train.csv`` in file order, asserting zero missing values."""
    return _load_csv(TRAIN_CSV)


def load_test():
    """Read ``test.csv`` in file order, asserting zero missing values."""
    return _load_csv(TEST_CSV)


def fold_ids_for(y, seed: int = FOLD_SEED):
    """Build the Canonical Fold Partition from an already-read target vector.

    ``StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)`` on the
    target; assignment is by input row order, so ``y`` must be in file order.
    """
    import numpy as np
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    fold = np.empty(len(y), dtype=np.int8)
    for k, (_, val_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        fold[val_idx] = k
    return fold


def fold_ids(seed: int = FOLD_SEED):
    """The Canonical Fold Partition: a fold id per training row, in file order.

    Reads ``train.csv`` and builds the partition immediately after — the #9
    seam signature. The runner, which already holds the target, uses
    :func:`fold_ids_for` to avoid re-reading.
    """
    train = load_train()
    return fold_ids_for(train[TARGET_COLUMN].to_numpy(), seed)


def sha256_of_folds(fold) -> str:
    """sha256 of the int8 fold-id vector in row order (platform-independent)."""
    return hashlib.sha256(fold.astype("int8").tobytes()).hexdigest()


def assert_fold_partition(fold, seed: int = FOLD_SEED) -> str:
    """Assert the fold partition matches the committed sha256; return the sha.

    When ``CANONICAL_FOLD_SHA256`` is still ``None`` (first run on a machine
    that has the CSVs), the computed value is printed for committing and no hard
    failure is raised. Once committed, a mismatch fails loudly.
    """
    sha = sha256_of_folds(fold)
    if seed != FOLD_SEED:
        # Only the canonical seed has a committed checksum; other seeds (used by
        # Confirmation Runs) are computed fresh.
        return sha

    if CANONICAL_FOLD_SHA256 is None:
        print(
            "CANONICAL_FOLD_SHA256 is unset; record this in src/data.py to arm "
            f"the assert on every future run:\n    CANONICAL_FOLD_SHA256 = {sha!r}"
        )
        return sha
    if sha != CANONICAL_FOLD_SHA256:
        raise AssertionError(
            "Canonical Fold Partition checksum mismatch: the fold-id vector no "
            f"longer matches the committed sha256.\n  expected: {CANONICAL_FOLD_SHA256}\n"
            f"  got:      {sha}\n"
            "Rows were reordered/dropped before the split, or scikit-learn's "
            "split algorithm changed. Every banked measurement was taken on the "
            "committed partition — this is a hard stop."
        )
    return sha


# Kept importable for the frame/adapter which select categoricals from the
# committed list, never by dtype.
__all__ = [
    "N_FOLDS",
    "FOLD_SEED",
    "CANONICAL_FOLD_SHA256",
    "load_train",
    "load_test",
    "fold_ids",
    "fold_ids_for",
    "sha256_of_folds",
    "assert_fold_partition",
    "CATEGORICAL_COLUMNS",
]
