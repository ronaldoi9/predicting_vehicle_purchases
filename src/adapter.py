"""The Model Adapter: scaling, nested cross-fit target encoding.

Stub for #21: the interface is fixed here so the seam resolves; the body is
wired up in a later ticket (#12). The only module allowed to see ``y`` outside
a model's own ``fit``; refuses a fit that receives validation rows.
"""

from __future__ import annotations


class Adapter:
    """Everything fitted strictly inside the training fold lives here."""

    def fit_transform(self, X_tr, y_tr):
        raise NotImplementedError("adapter.fit_transform is wired up in a later ticket (#12)")

    def transform(self, X_va):
        raise NotImplementedError("adapter.transform is wired up in a later ticket (#12)")
