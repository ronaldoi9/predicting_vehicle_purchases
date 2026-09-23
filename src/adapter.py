"""The Model Adapter: family-specific transforms fitted inside the training fold.

The Adapter is the **only** module that sees ``y`` outside a model's own ``fit``
— the second hard rule the tracer bullet (#13) establishes — so the one place a
target-reading leak can enter is this file. Everything family-specific and every
transform that reads the target lives here, fitted strictly inside the training
fold, never on the rows it is applied to.

Two properties this file exists to hold (ADR-0001, PRD #12):

* **Nested cross-fit target encoding.** A target encoding fitted on the rows it
  is applied to measured **-0.00187** in the leak hunt against a published
  **+0.00129** — a 0.003 swing from fold discipline alone, and invisible from the
  Health Gate because the leaked score is *higher*. So a training row's encoding
  is produced by an inner K=5 ``StratifiedKFold`` that puts the row in the inner
  *validation* fold: it is encoded from stats over the other inner-training rows,
  never its own. The inner ``random_state`` is ``100 + outer_fold`` so the inner
  partition can never align with the outer one. Additive smoothing at prior
  weight 20, the fold prior as the fallback for unseen keys. Test-time encoding
  is fitted **once on the full training set, outside the fold loop**.

* **The validation-row refusal.** ``fit_transform`` is told the outer fold's
  validation-row index and asserts that not one of those rows reached it — the
  fold-boundary violation caught at the moment it is attempted, rather than left
  to inflate the score.

Scaling also lives here and is a **family-conditional hook, off for a tree
family** (a GBDT picks splits by gain and is invariant to any monotone transform,
so min-max is literally the same model). When on for a linear family it touches
only the continuous column, **never its digit children** — ``(scaled_income) //
1000`` is meaningless. pandas / numpy / scikit-learn are imported lazily so the
module imports by bare name anywhere.
"""

from __future__ import annotations

from typing import Sequence

# The nested cross-fit contract — the exact parameters behind the banked
# ablation, so its +0.00129 stays comparable rather than silently re-measured.
INNER_SPLITS = 5
INNER_SEED_BASE = 100  # inner random_state = INNER_SEED_BASE + outer_fold
PRIOR_WEIGHT = 20.0  # additive-smoothing prior weight

# The target-encoding column suffix. TE adds a column rather than replacing the
# raw value, so income stays individually addressable alongside its encoding.
TE_SUFFIX = "_te"

# Oversampling (#19). Only the categorical-aware SMOTENC is permitted, and it is
# fitted strictly inside the training fold. Plain ``SMOTE`` is excluded outright:
# interpolating Age to 43.7 and producing income digits that no longer derive
# from their own income destroys Resolution, the one axis that pays.
OVERSAMPLE_SMOTENC = "smotenc"
SMOTENC_SAMPLING_STRATEGY = 0.5  # lift the minority share to 0.5 inside the fold
_VALID_OVERSAMPLE = (None, OVERSAMPLE_SMOTENC)


def inner_seed(outer_fold: int) -> int:
    """The inner split's ``random_state`` for a given outer fold.

    ``INNER_SEED_BASE + outer_fold`` keeps the inner partition off the outer one
    (outer seeds are 0..4), so the cross-fit is real rather than decorative.
    """
    return INNER_SEED_BASE + int(outer_fold)


def _smoothed_mean(count: float, total: float, prior: float, weight: float = PRIOR_WEIGHT) -> float:
    """Additive-smoothing target mean for one key.

    ``(sum + weight*prior) / (count + weight)`` — a key seen many times keeps its
    empirical rate; a thin key is pulled back toward the fold prior. ``weight=0``
    is the raw empirical mean.
    """
    return (total + weight * prior) / (count + weight)


def _key_stats(keys, y) -> tuple[dict, dict]:
    """Per-key row count and target sum over the given rows.

    The shared accumulation behind both the full-training encoding and each
    inner cross-fit fold: ``keys`` and ``y`` are numpy arrays aligned row for row.
    """
    counts: dict = {}
    totals: dict = {}
    for k, t in zip(keys.tolist(), y.tolist()):
        counts[k] = counts.get(k, 0) + 1
        totals[k] = totals.get(k, 0.0) + t
    return counts, totals


def _assert_no_validation_rows(train_index, validation_index) -> None:
    """Refuse a ``fit`` that received any row of the outer validation fold.

    The most expensive available mistake is fitting a target encoding on the
    rows it is applied to. The Adapter is fitted strictly inside the training
    fold, so ``validation_index`` (the outer fold's validation rows) must be
    declared — ``None`` means the caller cannot prove it is inside the fold — and
    not one of those rows may appear among the fitting rows.
    """
    if validation_index is None:
        raise AssertionError(
            "Adapter.fit_transform needs the outer validation index to prove it "
            "is fitting inside the training fold; none was declared (pass an "
            "empty set for the outside-the-loop full-training fit)"
        )
    leaked = set(train_index) & set(validation_index)
    if leaked:
        raise AssertionError(
            f"Adapter.fit_transform received {len(leaked)} validation-fold rows "
            f"(e.g. {sorted(leaked)[:5]}): fitting on rows it will be applied to "
            "leaks the target across the Canonical Fold Partition"
        )


class Adapter:
    """Holds any transform a particular model family needs and no other does.

    Constructed once per outer fold: ``outer_fold`` seeds the inner cross-fit and
    ``validation_index`` is that fold's validation rows (an empty set for the
    outside-the-loop full-training fit that encodes the test set once).
    """

    def __init__(
        self,
        *,
        scale: bool = False,
        target_encode: Sequence[str] = (),
        outer_fold: int | None = None,
        validation_index=None,
        scale_columns: Sequence[str] = (),
        scale_exclude: Sequence[str] = (),
        prior_weight: float = PRIOR_WEIGHT,
        inner_splits: int = INNER_SPLITS,
        oversample: str | None = None,
        oversample_continuous_columns: Sequence[str] = (),
        income_column: str | None = None,
        income_digit_transforms: Sequence = (),
    ) -> None:
        # Plain SMOTE is refused the moment the Adapter is constructed — before
        # any import, so the exclusion holds even where imbalanced-learn is
        # absent. Only the categorical-aware SMOTENC (or no oversampling) is
        # permitted.
        assert oversample in _VALID_OVERSAMPLE, (
            f"oversample must be None or {OVERSAMPLE_SMOTENC!r}; plain SMOTE is "
            "excluded outright because interpolating Age and producing income "
            "digits that no longer derive from their own income destroys "
            f"Resolution — use SMOTENC. Got {oversample!r}."
        )
        self.scale = scale
        self.target_encode = tuple(target_encode)
        self.outer_fold = outer_fold
        self.validation_index = None if validation_index is None else set(validation_index)
        self.scale_columns = tuple(scale_columns)
        self.scale_exclude = tuple(scale_exclude)
        self.prior_weight = prior_weight
        self.inner_splits = inner_splits
        self.oversample = oversample
        self.oversample_continuous_columns = tuple(oversample_continuous_columns)
        self.income_column = income_column
        self.income_digit_transforms = tuple(income_digit_transforms)

        self.resampled_y = None  # the oversampled training targets, once fitted
        self._fitted = False
        self._columns = None  # input column layout captured at fit
        self._scaler = None
        self._scale_cols = None
        self._te_maps: dict[str, dict] = {}  # col -> {key: full-training encoding}
        self._te_priors: dict[str, float] = {}  # col -> full-training fold prior

    # ---- scaling target selection (pure; the digit-child exclusion lives here) #
    def _scale_targets(self) -> list[str]:
        """The continuous columns scaling touches — its digit children removed."""
        return [c for c in self.scale_columns if c not in self.scale_exclude]

    # ---- the fold boundary ------------------------------------------------- #
    def fit_transform(self, X_tr, y_tr):
        """Fit inside the training fold and return the transformed training rows.

        ``y_tr`` is accepted so target-reading transforms have their single
        legitimate window here. Refuses the fit if a validation-fold row reached
        it. Target-encoded columns on the *training* rows are the nested
        cross-fit (out-of-inner-fold) values, so no row sees its own target.
        """
        _assert_no_validation_rows(list(X_tr.index), self.validation_index)

        self._columns = list(X_tr.columns)

        out = X_tr
        y_work = y_tr
        # Oversampling comes first and strictly inside the fold: it synthesises
        # new *training* rows, so the target encoding and scaling that follow see
        # the resampled frame. resampled_y is what the model must be fitted on.
        if self.oversample:
            out, y_work = self._oversample(out, y_work)
            self.resampled_y = y_work

        if self.target_encode:
            out = self._encode_training_rows(out, y_work)

        if self.scale:
            self._fit_scaler(out)

        self._fitted = True
        return self._apply_scale(out)

    # ---- oversampling (fitted inside the training fold only) --------------- #
    def _oversample(self, X, y):
        """Resample the training rows with SMOTENC, then regenerate income digits.

        SMOTENC synthesises minority rows by interpolating the continuous columns
        and picking the most frequent category among neighbours for the rest — so
        Age, being categorical to it, is never interpolated to 43.7 and keeps its
        45 addressable values. The income digit columns *are* continuous, so
        SMOTENC interpolates them into values that no longer derive from their own
        income; we discard those and regenerate each digit column from the
        (possibly synthetic) income, so the Frame stays self-consistent.
        """
        import numpy as np
        from imblearn.over_sampling import SMOTENC

        continuous = set(self.oversample_continuous_columns)
        cat_features = [i for i, c in enumerate(X.columns) if c not in continuous]
        sampler = SMOTENC(
            categorical_features=cat_features,
            sampling_strategy=SMOTENC_SAMPLING_STRATEGY,
            random_state=inner_seed(self.outer_fold or 0),
        )
        X_res, y_res = sampler.fit_resample(X, y)

        if self.income_column is not None and self.income_digit_transforms:
            income = X_res[self.income_column]
            for name, op in self.income_digit_transforms:
                X_res[name] = op(income).astype(X[name].dtype)

        return X_res, np.asarray(y_res)

    def transform(self, X_va):
        """Transform validation/test rows with the fold-fitted state only.

        Target-encoded columns use the encoding fitted on the full training rows
        of the fold (or, for the test set, the full training set): every unseen
        key falls back to that set's fold prior.
        """
        if not self._fitted:
            raise RuntimeError("Adapter.transform called before fit_transform")
        if list(X_va.columns) != self._columns:
            raise AssertionError("Adapter.transform got a different column layout than fit")

        out = X_va
        if self.target_encode:
            out = self._apply_full_encoding(X_va)
        return self._apply_scale(out)

    # ---- target encoding --------------------------------------------------- #
    def _encode_training_rows(self, X_tr, y_tr):
        """Add each TE column to the training rows via nested cross-fit, and
        fit the full-training encoding (used for the validation/test rows)."""
        import numpy as np

        y = np.asarray(y_tr, dtype=np.float64)
        out = X_tr.copy()
        for col in self.target_encode:
            keys = X_tr[col].to_numpy()
            # The full-training encoding, for transform() of held-out rows.
            self._te_maps[col], self._te_priors[col] = self._full_encoding(keys, y)
            # The training rows themselves get the out-of-inner-fold values.
            out[col + TE_SUFFIX] = self._nested_cross_fit(keys, y)
        return out

    def _full_encoding(self, keys, y):
        """The smoothed per-key encoding over all fitting rows, plus the prior."""
        prior = float(y.mean())
        counts, totals = _key_stats(keys, y)
        mapping = {
            k: _smoothed_mean(counts[k], totals[k], prior, self.prior_weight)
            for k in counts
        }
        return mapping, prior

    def _nested_cross_fit(self, keys, y):
        """Out-of-inner-fold encoding for every training row.

        An inner ``StratifiedKFold`` puts each row in the inner *validation* fold;
        it is encoded from stats over the inner-training rows only, so it never
        sees its own target. Unseen-in-inner-training keys fall back to the inner
        fold prior.
        """
        import numpy as np
        from sklearn.model_selection import StratifiedKFold

        enc = np.empty(len(keys), dtype=np.float64)
        inner = StratifiedKFold(
            n_splits=self.inner_splits,
            shuffle=True,
            random_state=inner_seed(self.outer_fold),
        )
        for in_tr, in_va in inner.split(np.zeros(len(keys)), y):
            prior = float(y[in_tr].mean())
            counts, totals = _key_stats(keys[in_tr], y[in_tr])
            for pos in in_va:
                k = keys[pos]
                if k in counts:
                    enc[pos] = _smoothed_mean(counts[k], totals[k], prior, self.prior_weight)
                else:
                    enc[pos] = prior
        return enc

    def _apply_full_encoding(self, X):
        out = X.copy()
        for col in self.target_encode:
            mapping = self._te_maps[col]
            prior = self._te_priors[col]
            out[col + TE_SUFFIX] = X[col].map(mapping).fillna(prior)
        return out

    # ---- scaling ----------------------------------------------------------- #
    def _fit_scaler(self, X):
        from sklearn.preprocessing import StandardScaler

        self._scale_cols = self._scale_targets()
        if self._scale_cols:
            self._scaler = StandardScaler().fit(X[self._scale_cols])

    def _apply_scale(self, X):
        if not self.scale or not self._scale_cols:
            return X
        out = X.copy()
        out[self._scale_cols] = self._scaler.transform(X[self._scale_cols])
        return out
