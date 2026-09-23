"""Reading the CSVs in file order and the Canonical Fold Partition.

Stub for #21: the interface is fixed here so the seam resolves; the body is
wired up in a later ticket (#12). Owns ``load_train()`` and ``fold_ids(seed)``
and the sha256 assert on the fold-id vector.
"""

from __future__ import annotations


def load_train():
    """Read ``train.csv`` in file order, no sort/filter/row-drop before split."""
    raise NotImplementedError("data.load_train is wired up in a later ticket (#12)")


def fold_ids(seed: int = 0):
    """Reconstruct the Canonical Fold Partition from two constants."""
    raise NotImplementedError("data.fold_ids is wired up in a later ticket (#12)")
