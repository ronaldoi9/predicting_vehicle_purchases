"""Model families behind one signature. Turn 1 is LightGBM only.

Stub for #21: the interface is fixed here so the seam resolves; the body is
wired up in a later ticket (#12).
"""

from __future__ import annotations


def fit(X, y, params):
    """Fit one model family on ``X``/``y`` under ``params``."""
    raise NotImplementedError("models.fit is wired up in a later ticket (#12)")
