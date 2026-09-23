"""The Model Adapter: family-specific transforms fitted inside the training fold.

The Adapter is the **only** module that sees ``y`` outside a model's own ``fit``
— the second hard rule the tracer bullet (#13) establishes — so the one place a
target-reading leak can enter is this file. Everything family-specific and every
transform that reads the target lives here, fitted strictly inside the training
fold, never on the rows it is applied to.

For the raw-13 tracer the Adapter is a near-passthrough: scaling is **off** for
a tree family (a no-op scaler must not sit between the CSV and the model on turn
1), and there is no target encoding yet. Nested cross-fit target encoding and
its "refuse a fit that receives validation rows" contract land with the
target-encoding ticket, which widens this file rather than replacing it. The
``scale`` path is wired so a linear family can turn it on without touching the
Frame.

scikit-learn is imported lazily so the module imports by bare name anywhere.
"""

from __future__ import annotations


class Adapter:
    """Holds any transform a particular model family needs and no other does."""

    def __init__(self, scale: bool = False) -> None:
        self.scale = scale
        self._scaler = None
        self._fitted = False
        self._columns = None

    def fit_transform(self, X_tr, y_tr):
        """Fit inside the training fold and return the transformed training rows.

        ``y_tr`` is accepted so target-reading transforms have their single
        legitimate window here; the raw-13 tracer does not read it.
        """
        self._columns = list(X_tr.columns)
        if self.scale:
            from sklearn.preprocessing import StandardScaler

            self._scaler = StandardScaler().fit(X_tr)
        self._fitted = True
        return self._apply(X_tr)

    def transform(self, X_va):
        """Transform validation/test rows with the fold-fitted state only."""
        if not self._fitted:
            raise RuntimeError("Adapter.transform called before fit_transform")
        if list(X_va.columns) != self._columns:
            raise AssertionError("Adapter.transform got a different column layout than fit")
        return self._apply(X_va)

    def _apply(self, X):
        if not self.scale:
            return X
        import pandas as pd

        scaled = self._scaler.transform(X)
        return pd.DataFrame(scaled, columns=self._columns, index=X.index)
