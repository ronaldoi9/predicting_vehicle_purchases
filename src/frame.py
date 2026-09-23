"""The Baseline Frame: one-hot, ordinals, digit decomposition, count encoding.

Stub for #21: the interface is fixed here so the seam resolves; the body is
wired up in a later ticket (#12). Selects categoricals from the committed
:mod:`columns` lists, never by dtype.
"""

from __future__ import annotations


def build_frame(train, test):
    """Assemble the frozen, versioned Baseline Frame from train and test."""
    raise NotImplementedError("frame.build_frame is wired up in a later ticket (#12)")
