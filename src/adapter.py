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

# The Recipe (#28, docs/research/generator-recipe.md on branch
# research/generator-recipe): the publicly reverse-engineered generating rule,
# ``buy_score = 1.2*(income/1e5) + 0.6*concern + 2*subsidy - 1*(anxiety==Medium)
# - 3*(anxiety==High)``. Column names match the Baseline Frame's post-one-hot,
# post-ordinal-encoding layout (``frame.build_frame``), not the raw CSV: the
# ordinals already carry ``frame.ORDINAL_ENCODINGS``' numbering (Low=0,
# Medium=1, High=2) and Subsidy is one-hot with no dropped level.
RECIPE_INCOME_COLUMN = "Annual_Income_USD"
RECIPE_CONCERN_COLUMN = "Environmental_Concern_Level"
RECIPE_SUBSIDY_YES_COLUMN = "Subsidy_Available_Yes"
RECIPE_ANXIETY_COLUMN = "Range_Anxiety_Level"
RECIPE_INCOME_COEF = 1.2
RECIPE_CONCERN_COEF = 0.6
RECIPE_SUBSIDY_COEF = 2.0
RECIPE_ANXIETY_MEDIUM_COEF = -1.0
RECIPE_ANXIETY_HIGH_COEF = -3.0

# The column the calibrated Recipe margin is carried in until ``runner`` pops it
# back out to pass as ``init_score`` — it is not a model feature, it is the
# model's initial prediction, so it must never reach ``models.fit`` as a column.
RECIPE_MARGIN_COLUMN = "_recipe_init_score"

# The fitted additive-logistic margin (#33): a saturated gate over
# concern x subsidy x anxiety plus a per-exact-income-value basis, fitted
# strictly inside the training fold and carried the same way as the Recipe
# margin above -- popped by ``runner`` and passed as ``init_score``, never a
# Frame feature a tree could split on.
FITTED_MARGIN_COLUMN = "_fitted_init_score"

# The gate's saturated interaction: concern*6 + subsidy*3 + anxiety, from
# #26's reading of the published notebook (kps6e09-xgb-sample). One-hot
# encoded whole, not marginal dummies per column, so every combination gets
# its own free coefficient (a "31-level saturated gate").
FITTED_MARGIN_GATE_COLUMNS = (
    RECIPE_CONCERN_COLUMN,
    RECIPE_SUBSIDY_YES_COLUMN,
    RECIPE_ANXIETY_COLUMN,
)

# The income basis's box-smoothing half-width, in ranks over the sorted
# distinct training-fold income values (~13,214 of them, matching #26's
# "~13k-coefficient block" almost exactly at box_width=1 -- one column per
# unique value). A half-width of 2 gives every row a 5-wide boxcar of
# adjacent-value columns rather than a single exact-value indicator, so
# neighbouring income values share support ("box smoothing across adjacent
# income values") without a custom penalised-GLM solver: an ordinary L2
# (ridge) logistic fit over the boxcar design already pulls neighbours
# toward each other because their columns overlap.
FITTED_MARGIN_INCOME_HALF_WIDTH = 2

# The ridge strength handed to sklearn's LogisticRegression (its inverse, as
# the library parameterises it). Left at sklearn's own default rather than
# tuned -- tuning the margin model is out of scope for this ticket's
# cheapest-decisive-test question, and belongs to #34 if this axis survives.
FITTED_MARGIN_RIDGE_C = 1.0

# --------------------------------------------------------------------------- #
# The linear Frame (#34): a second representation behind the Model Adapter,
# not a new Baseline Frame spec (docs/research/linear-model-representation.md
# section 5) -- the Frame stays the raw-value baseline and this module turns
# it into the six-change design a linear family needs. Reuses #33's gate-code
# helper (``_gate_code``) for the same saturated interaction ADR-0001 now
# scopes to tree families only (see the amendment).
# --------------------------------------------------------------------------- #

# income_mod1000/_mod100 measured AUC 0.50001 on non-floor rows (#4) -- pure
# Resolution, meaningless to a slope. _div1000 is income rescaled and floored,
# collinear with raw income. All three are dropped rather than kept "just in
# case": a redundant near-duplicate column only hurts the design's conditioning.
LINEAR_MOD_COLUMNS = (
    f"{RECIPE_INCOME_COLUMN}_mod1000",
    f"{RECIPE_INCOME_COLUMN}_mod100",
    f"{RECIPE_INCOME_COLUMN}_div1000",
)

# Raw integer frequencies have a long tail; a linear term on them asserts a
# monotone, unit-per-count effect, which is the wrong scale. log1p is the same
# fix Elefante's published notebook applies to its own count features.
LINEAR_LOG1P_COLUMNS = (f"{RECIPE_INCOME_COLUMN}_count", "Daily_Commute_km_count")

# The two hard edges in the income distribution (#4's ablation): a GBDT finds
# these for free with one split each (+0.00002 there); a linear model has no
# splitter, so they are not redundant with the raw column or its target
# encoding here.
LINEAR_INCOME_FLOOR_VALUE = 30000
LINEAR_INCOME_CEIL_VALUE = 170537
LINEAR_INCOME_FLOOR_COLUMN = f"{RECIPE_INCOME_COLUMN}_eq_floor"
LINEAR_INCOME_CEIL_COLUMN = f"{RECIPE_INCOME_COLUMN}_ge_ceil"

# Age's full, contiguous value domain (25..69, 45 values -- confirmed against
# the training CSV, and the same count the Frame's own build assert protects).
# Fixed here rather than fitted from a training fold: it is a fact about the
# data dictionary, not something a fold could leak by observing it.
LINEAR_AGE_COLUMN = "Age"
LINEAR_AGE_VALUES: tuple[int, ...] = tuple(range(25, 70))

# The gate's full domain -- concern in {1..5} x subsidy in {0,1} x anxiety in
# {0,1,2}, 30 cells -- likewise fixed rather than fitted, so every fold's gate
# design has the same 30 columns even if a rare cell is briefly absent from one
# outer fold.
LINEAR_GATE_CODES: tuple[int, ...] = tuple(
    sorted(c * 6 + s * 3 + a for c in range(1, 6) for s in (0, 1) for a in range(3))
)

# The continuous columns the linear Frame scales -- Age, the anxiety dummies
# and the gate block are already 0/1 and need no scaling. A `_te` column per
# Adapter target encoding is appended by :func:`linear_scale_columns` below,
# since which columns are target-encoded is a per-Experiment choice.
LINEAR_SCALE_BASE_COLUMNS = (
    RECIPE_INCOME_COLUMN,
    "Daily_Commute_km",
    "Number_of_Cars_Owned",
    "Charging_Stations_Near_Home",
    "Charging_Stations_Near_Work",
    RECIPE_CONCERN_COLUMN,
    f"{RECIPE_INCOME_COLUMN}_count",
    "Daily_Commute_km_count",
)


def linear_scale_columns(target_encode: Sequence[str] = ()) -> tuple[str, ...]:
    """The linear Frame's scale targets: the base continuous columns plus a
    ``_te`` column for each column the Adapter target-encodes.

    An L2 penalty is not scale-invariant (ADR-0001's amendment for #34), so
    every continuous column the linear design carries needs to be on the same
    footing, not just income.
    """
    return LINEAR_SCALE_BASE_COLUMNS + tuple(f"{c}{TE_SUFFIX}" for c in target_encode)


def recipe_buy_score(X):
    """The Recipe's raw ``buy_score`` (issue #5), row for row, from a built Frame.

    Pure arithmetic on already-present Frame columns — no target, no fold state
    — so it is the same before or after the fold boundary. Not itself a usable
    margin: it is on an arbitrary scale with the Recipe's own threshold at 5.5,
    which is exactly the ticket #28 question (a raw score handed where a
    log-odds margin is expected depresses the score).
    """
    income = X[RECIPE_INCOME_COLUMN].astype("float64")
    concern = X[RECIPE_CONCERN_COLUMN].astype("float64")
    subsidy = X[RECIPE_SUBSIDY_YES_COLUMN].astype("float64")
    anxiety = X[RECIPE_ANXIETY_COLUMN].astype("float64")
    return (
        RECIPE_INCOME_COEF * (income / 1e5)
        + RECIPE_CONCERN_COEF * concern
        + RECIPE_SUBSIDY_COEF * subsidy
        + RECIPE_ANXIETY_MEDIUM_COEF * (anxiety == 1).astype("float64")
        + RECIPE_ANXIETY_HIGH_COEF * (anxiety == 2).astype("float64")
    )


def _gate_code(X):
    """The saturated gate's integer key: concern*6 + subsidy*3 + anxiety.

    Purely a categorical key for one-hot encoding (the arithmetic carries no
    ordering meaning) -- see :data:`FITTED_MARGIN_GATE_COLUMNS`.
    """
    concern = X[RECIPE_CONCERN_COLUMN].to_numpy().astype("int64")
    subsidy = X[RECIPE_SUBSIDY_YES_COLUMN].to_numpy().astype("int64")
    anxiety = X[RECIPE_ANXIETY_COLUMN].to_numpy().astype("int64")
    return concern * 6 + subsidy * 3 + anxiety


def _income_box_design(income, sorted_uniques, half_width):
    """A boxcar (box-smoothing) design over the sorted distinct income values.

    Row ``i`` activates every column within ``half_width`` ranks of its own
    income value's rank in ``sorted_uniques`` -- a uniform-weight moving
    window, so a value's fitted effect is pulled toward its neighbours by the
    shared columns rather than fitted as an isolated per-value indicator.
    Values absent from ``sorted_uniques`` (a held-out row) fall to the
    neighbouring rank via ``searchsorted``. Boundary ranks near either end
    can hit the same clipped column twice across offsets; ``csr_matrix``
    sums those duplicates, adding at most 1.0 extra weight to the outermost
    handful of the ~13k columns -- negligible next to the ridge penalty.
    """
    import numpy as np
    from scipy import sparse

    n = len(income)
    u = len(sorted_uniques)
    ranks = np.clip(np.searchsorted(sorted_uniques, income), 0, u - 1)
    rows_parts = []
    cols_parts = []
    for offset in range(-half_width, half_width + 1):
        cols_parts.append(np.clip(ranks + offset, 0, u - 1))
        rows_parts.append(np.arange(n))
    rows = np.concatenate(rows_parts)
    cols = np.concatenate(cols_parts)
    data = np.ones(len(rows), dtype=np.float64)
    return sparse.csr_matrix((data, (rows, cols)), shape=(n, u))


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
        recipe_margin: bool = False,
        fitted_margin: bool = False,
        fitted_margin_calibrate: bool = False,
        fitted_margin_income_half_width: int = FITTED_MARGIN_INCOME_HALF_WIDTH,
        linear_design: bool = False,
        linear_gate: bool = True,
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
        self.recipe_margin = recipe_margin
        self.fitted_margin = fitted_margin
        self.fitted_margin_calibrate = fitted_margin_calibrate
        self.fitted_margin_income_half_width = fitted_margin_income_half_width
        self.linear_design = linear_design
        self.linear_gate = linear_gate

        self.resampled_y = None  # the oversampled training targets, once fitted
        self._fitted = False
        self._columns = None  # input column layout captured at fit
        self._scaler = None
        self._scale_cols = None
        self._te_maps: dict[str, dict] = {}  # col -> {key: full-training encoding}
        self._te_priors: dict[str, float] = {}  # col -> full-training fold prior
        self._recipe_calibration: tuple[float, float] | None = None  # (intercept, coef)
        self._fitted_margin_state: dict | None = None

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

        if self.recipe_margin:
            out = self._fit_recipe_margin(out, y_work)

        if self.fitted_margin:
            out = self._fit_fitted_margin(out, y_work)

        if self.linear_design:
            out = self._apply_linear_design(out)

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
        if self.recipe_margin:
            out = self._apply_recipe_margin(out)
        if self.fitted_margin:
            out = self._apply_fitted_margin(out)
        if self.linear_design:
            out = self._apply_linear_design(out)
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

    # ---- the Recipe margin (#28) -------------------------------------------- #
    def _fit_recipe_margin(self, X, y):
        """Calibrate the Recipe's raw ``buy_score`` to a log-odds margin.

        A one-parameter-per-term logistic fit of ``y`` on ``buy_score`` (two
        free numbers: intercept and slope) fitted strictly inside the training
        fold — it reads ``y``, so it belongs here, the one place that may. Fit
        on the whole training fold rather than nested cross-fit like the target
        encoding: two degrees of freedom over hundreds of thousands of rows
        cannot memorise a row's own label the way a per-key lookup can. Its
        ``decision_function`` (``intercept + coef*buy_score``) is by
        construction a valid log-odds margin, unlike the raw ``buy_score`` or an
        assumed-sigma logit of it.
        """
        from sklearn.linear_model import LogisticRegression

        score = recipe_buy_score(X).to_numpy().reshape(-1, 1)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(score, y)
        self._recipe_calibration = (float(clf.intercept_[0]), float(clf.coef_[0][0]))
        return self._apply_recipe_margin(X)

    def _apply_recipe_margin(self, X):
        if self._recipe_calibration is None:
            raise RuntimeError("recipe margin applied before it was fitted")
        intercept, coef = self._recipe_calibration
        out = X.copy()
        out[RECIPE_MARGIN_COLUMN] = intercept + coef * recipe_buy_score(X)
        return out

    # ---- the fitted additive-logistic margin (#33) -------------------------- #
    def _fitted_margin_design(self, X, *, fit: bool):
        """The gate + income-box design matrix, fitting or reusing the encoders.

        Shared by fit and transform so the two paths cannot drift: the gate's
        one-hot vocabulary and the income basis's sorted training values are
        learned once, here, on the training fold only.
        """
        import numpy as np
        from sklearn.preprocessing import OneHotEncoder

        gate = _gate_code(X).reshape(-1, 1)
        income = X[RECIPE_INCOME_COLUMN].to_numpy()

        if fit:
            gate_encoder = OneHotEncoder(handle_unknown="ignore")
            gate_design = gate_encoder.fit_transform(gate)
            sorted_uniques = np.unique(income)
            self._fitted_margin_state = {
                "gate_encoder": gate_encoder,
                "income_sorted_uniques": sorted_uniques,
            }
        else:
            state = self._fitted_margin_state
            gate_design = state["gate_encoder"].transform(gate)
            sorted_uniques = state["income_sorted_uniques"]

        income_design = _income_box_design(
            income, sorted_uniques, self.fitted_margin_income_half_width
        )
        from scipy import sparse

        return sparse.hstack([gate_design, income_design], format="csr")

    def _fit_fitted_margin(self, X, y):
        """Fit the additive-logistic margin (#33) strictly inside the training
        fold: a saturated gate over concern x subsidy x anxiety plus a
        box-smoothed per-income-value basis, ridge-penalised by an ordinary
        L2 logistic fit. Its own ``decision_function`` is by construction a
        valid log-odds margin. When ``fitted_margin_calibrate`` is set, that
        raw margin is further recalibrated by a 2-parameter logistic fit
        (intercept + slope), the same pattern #28 uses for the Recipe margin
        -- fitted on the whole training fold, not nested cross-fit, because a
        handful of coefficients over hundreds of thousands of rows cannot
        memorise a row's own label the way a per-key lookup can.
        """
        from sklearn.linear_model import LogisticRegression

        design = self._fitted_margin_design(X, fit=True)
        # penalty defaults to L2 (ridge); passing it explicitly is deprecated
        # in sklearn>=1.8 in favour of l1_ratio, so C alone selects the ridge
        # strength. max_iter=300: the ~13k-column sparse design does not fully
        # converge at sklearn's default 100 (a bug report about runtime, not
        # about the finding -- the kill criterion already reads decisively).
        clf = LogisticRegression(C=FITTED_MARGIN_RIDGE_C, solver="lbfgs", max_iter=300)
        clf.fit(design, y)
        self._fitted_margin_state["model"] = clf
        self._fitted_margin_state["calibration"] = None

        raw_margin = clf.decision_function(design)
        if self.fitted_margin_calibrate:
            calib = LogisticRegression(max_iter=1000)
            calib.fit(raw_margin.reshape(-1, 1), y)
            self._fitted_margin_state["calibration"] = (
                float(calib.intercept_[0]),
                float(calib.coef_[0][0]),
            )
            margin = self._fitted_margin_state["calibration"][0] + (
                self._fitted_margin_state["calibration"][1] * raw_margin
            )
        else:
            margin = raw_margin

        out = X.copy()
        out[FITTED_MARGIN_COLUMN] = margin
        return out

    def _apply_fitted_margin(self, X):
        if self._fitted_margin_state is None or "model" not in self._fitted_margin_state:
            raise RuntimeError("fitted margin applied before it was fitted")
        design = self._fitted_margin_design(X, fit=False)
        state = self._fitted_margin_state
        raw_margin = state["model"].decision_function(design)
        if state["calibration"] is not None:
            intercept, coef = state["calibration"]
            margin = intercept + coef * raw_margin
        else:
            margin = raw_margin
        out = X.copy()
        out[FITTED_MARGIN_COLUMN] = margin
        return out

    # ---- the linear Frame (#34) ---------------------------------------------#
    def _apply_linear_design(self, X):
        """Turn the Baseline Frame's raw-value layout into the linear design.

        Purely deterministic given ``X`` -- every category domain (Age's 45
        contiguous values, anxiety's three levels, the gate's 30 cells) is
        fixed from the data dictionary rather than fitted from data, so this
        needs no fold-fitted state and reads no ``y``. Order: drop the mod/div
        income digits, log1p the two count columns, add the two income
        threshold flags, one-hot Age and Range_Anxiety_Level (a linear model
        has no splitter, so both need their own column per level rather than
        one slope), then -- when ``linear_gate`` is set -- the saturated
        Concern x Subsidy x Anxiety gate block. Scaling runs after this, in
        the caller, over :func:`linear_scale_columns`.
        """
        import numpy as np
        import pandas as pd

        out = X.drop(columns=[c for c in LINEAR_MOD_COLUMNS if c in X.columns])

        for col in LINEAR_LOG1P_COLUMNS:
            if col in out.columns:
                out[col] = np.log1p(out[col].astype("float64"))

        income = out[RECIPE_INCOME_COLUMN]
        out[LINEAR_INCOME_FLOOR_COLUMN] = (income == LINEAR_INCOME_FLOOR_VALUE).astype("int8")
        out[LINEAR_INCOME_CEIL_COLUMN] = (income >= LINEAR_INCOME_CEIL_VALUE).astype("int8")

        age_cat = pd.Categorical(out[LINEAR_AGE_COLUMN], categories=LINEAR_AGE_VALUES)
        age_dummies = pd.get_dummies(age_cat, prefix=LINEAR_AGE_COLUMN).astype("int8")
        age_dummies.index = out.index

        anxiety_cat = pd.Categorical(out[RECIPE_ANXIETY_COLUMN], categories=(0, 1, 2))
        anxiety_dummies = pd.get_dummies(anxiety_cat, prefix=RECIPE_ANXIETY_COLUMN).astype("int8")
        anxiety_dummies.index = out.index

        out = out.drop(columns=[LINEAR_AGE_COLUMN, RECIPE_ANXIETY_COLUMN])
        parts = [out, age_dummies, anxiety_dummies]

        if self.linear_gate:
            gate_code = pd.Series(_gate_code(X), index=X.index)
            gate_cat = pd.Categorical(gate_code, categories=LINEAR_GATE_CODES)
            gate_dummies = pd.get_dummies(gate_cat, prefix="gate").astype("int8")
            gate_dummies.index = X.index
            parts.append(gate_dummies)

        return pd.concat(parts, axis=1)

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
