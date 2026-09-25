"""heuljax's ``kps6e09-xgb-sample`` pipeline as an Arena family (#37, turn 3).

Ported from the Kaggle notebook "KPS6E09 XGB Sample" by heuljax:
https://www.kaggle.com/code/heuljax/kps6e09-xgb-sample

    Copyright heuljax.
    Licensed under the Apache License, Version 2.0 (the "License"); you may not
    use this file except in compliance with the License. You may obtain a copy
    of the License at http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
    License for the specific language governing permissions and limitations
    under the License.

Modified for this repo (the Apache-2.0 §4(b) notice): the tracer-bullet slice
carries only the 13 raw columns and the 7 exact-value target encodings of the
notebook's 173 features (``RAW_COLS`` + the ``TE_INC_A*``/``TE_CMT_A*``/
``SUPPORT_*_LOG`` block of ``DonorState.transform``); the outer 10-fold split is
replaced by the Canonical Fold Partition (the runner owns it); early stopping on
the scored fold is replaced by a fixed round count; XGBoost runs on CPU
``hist``; the GAM ``base_margin``, the original-data priors and the other
~150 donor-fitted columns are not ported yet; the inner donor split's seed
is a declared parameter held across outer folds instead of ``42 + 3000 +
fold``, and the booster seed is the instrument's 0 instead of ``42 + fold``,
because :func:`models.fit` does not know which outer fold it is fitting; and
the booster reads a plain ``DMatrix`` instead of a ``QuantileDMatrix``.

This is the first family that consumes the **raw columns** (the
``raw_columns`` Frame spec) and builds its own representation inside each
outer fold, so the representation belongs to the family and the Baseline Frame
is untouched (ADR-0001, ADR-0006 §3). The notebook's **donor state** is its
Nested Cross-Fit: every row's encoding comes from target statistics fitted on
other rows only:

* a training row is encoded by the donor state of the inner fold it is held
  out of (inner ``StratifiedKFold(5)`` over the outer training rows), with an
  assert that the inner folds cover every row exactly once;
* validation and test rows are encoded by the donor state fitted on the whole
  outer training fold.

:func:`fit_donor_state` refuses donor rows that overlap the rows they encode.
The outer boundary is the Model Adapter's: the runner hands this family only
rows the Adapter has already proven are outside the validation fold.

pandas / numpy / scikit-learn / xgboost are imported lazily, like every other
family's library, so the module imports by bare name anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

# The notebook's raw columns and category vocabularies (cell 1), in its order:
# the float and integer numerics as floats, the six categoricals as integer
# codes declared categorical to XGBoost. Environmental_Concern_Level is a float
# numeric here and Range_Anxiety_Level a categorical -- the notebook's split,
# not the Baseline Frame's nominal/ordinal one.
FLOAT_COLS = ("Annual_Income_USD", "Daily_Commute_km", "Environmental_Concern_Level")
INT_COLS = (
    "Age",
    "Number_of_Cars_Owned",
    "Charging_Stations_Near_Home",
    "Charging_Stations_Near_Work",
)
CAT_COLS = (
    "Gender",
    "City_Type",
    "Current_Car_Type",
    "Home_Charging_Possible",
    "Subsidy_Available",
    "Range_Anxiety_Level",
)
NUM_COLS = FLOAT_COLS + INT_COLS
RAW_COLS = NUM_COLS + CAT_COLS
CATEGORY_VOCAB: dict[str, tuple[str, ...]] = {
    "Gender": ("Male", "Female", "Other"),
    "City_Type": ("Suburban", "Rural", "Urban"),
    "Current_Car_Type": ("Sedan", "SUV", "Hatchback", "Truck"),
    "Home_Charging_Possible": ("Yes", "No"),
    "Subsidy_Available": ("No", "Yes"),
    "Range_Anxiety_Level": ("Low", "Medium", "High"),
}

INCOME_COLUMN = "Annual_Income_USD"
COMMUTE_COLUMN = "Daily_Commute_km"

# The exact-value target encodings: (S + a*mu) / (n + a) at three income priors
# and two commute priors, plus log1p(n) support for each key (cell 1/5).
INCOME_ALPHAS = (5.0, 20.0, 50.0)
COMMUTE_ALPHAS = (20.0, 100.0)
TE_COLS = (
    tuple(f"TE_INC_A{a:g}" for a in INCOME_ALPHAS)
    + tuple(f"TE_CMT_A{a:g}" for a in COMMUTE_ALPHAS)
    + ("SUPPORT_INC_LOG", "SUPPORT_CMT_LOG")
)
FEATURE_COLUMNS: list[str] = list(RAW_COLS + TE_COLS)
FEATURE_TYPES: list[str] = ["c" if c in CAT_COLS else "q" for c in FEATURE_COLUMNS]

# Positive monotonicity on the target-rate columns (the notebook's
# USE_RATE_MONOTONICITY, restricted to the rate columns this slice carries).
RATE_COLS = tuple(c for c in FEATURE_COLUMNS if c.startswith(("TE_INC_", "TE_CMT_")))
MONOTONE_CONSTRAINTS = "(" + ",".join("1" if c in RATE_COLS else "0" for c in FEATURE_COLUMNS) + ")"

INNER_FOLDS = 5

# Keys in the Experiment's params this module reads itself; the rest go to
# xgboost.train unchanged.
DONOR_SEED_KEY = "donor_seed"
_FAMILY_KEYS = (DONOR_SEED_KEY,)


def _pack_raw(X) -> dict:
    """The raw columns as numpy arrays: numerics as float64, categoricals as codes.

    An unknown category maps to ``len(vocab)``, as in the notebook's
    ``pack_raw``. The loader already refuses missing values, so the notebook's
    median imputation has nothing to do here and is not ported.
    """
    import numpy as np
    import pandas as pd

    missing = [c for c in RAW_COLS if c not in X.columns]
    if missing:
        raise AssertionError(f"heuljax needs the raw columns; missing {missing}")
    out = {c: X[c].to_numpy(dtype=np.float64) for c in NUM_COLS}
    for c in CAT_COLS:
        vocab = CATEGORY_VOCAB[c]
        codes = pd.Categorical(X[c].astype("string"), categories=list(vocab)).codes
        out[c] = np.where(codes < 0, len(vocab), codes).astype(np.int16)
    return out


def _take(raw: dict, rows) -> dict:
    return {k: v[rows] for k, v in raw.items()}


class _TargetTable:
    """Per-exact-value row count and target sum over the donor rows."""

    def __init__(self, values, targets) -> None:
        import numpy as np

        self.values, codes = np.unique(values, return_inverse=True)
        self.count = np.bincount(codes, minlength=len(self.values)).astype(np.float64)
        self.total = np.bincount(codes, weights=targets, minlength=len(self.values))

    def lookup(self, query):
        """``(count, total)`` at each query's exact value; zero for an unseen one."""
        import numpy as np

        pos = np.searchsorted(self.values, query)
        hi = np.minimum(pos, len(self.values) - 1)
        exact = (pos < len(self.values)) & (self.values[hi] == query)
        return self.count[hi] * exact, self.total[hi] * exact


class DonorState:
    """The target statistics of one set of donor rows, applied to other rows."""

    def __init__(self, raw: dict, targets) -> None:
        import numpy as np

        targets = np.asarray(targets, dtype=np.float64)
        if len(raw[INCOME_COLUMN]) != len(targets) or len(np.unique(targets)) != 2:
            raise ValueError("Donor must contain aligned data and both target classes")
        self.prior = float(targets.mean())
        self.income = _TargetTable(raw[INCOME_COLUMN], targets)
        self.commute = _TargetTable(raw[COMMUTE_COLUMN], targets)

    def transform(self, raw: dict):
        """The feature matrix (``FEATURE_COLUMNS`` order, float32) for ``raw``."""
        import numpy as np

        n = len(raw[INCOME_COLUMN])
        col = {name: i for i, name in enumerate(FEATURE_COLUMNS)}
        result = np.empty((n, len(FEATURE_COLUMNS)), dtype=np.float32)
        for c in RAW_COLS:
            result[:, col[c]] = raw[c]
        count, total = self.income.lookup(raw[INCOME_COLUMN])
        ccount, ctotal = self.commute.lookup(raw[COMMUTE_COLUMN])
        for alpha in INCOME_ALPHAS:
            result[:, col[f"TE_INC_A{alpha:g}"]] = (total + alpha * self.prior) / (count + alpha)
        for alpha in COMMUTE_ALPHAS:
            result[:, col[f"TE_CMT_A{alpha:g}"]] = (ctotal + alpha * self.prior) / (ccount + alpha)
        result[:, col["SUPPORT_INC_LOG"]] = np.log1p(count)
        result[:, col["SUPPORT_CMT_LOG"]] = np.log1p(ccount)
        return result


def _fit_donor_state_raw(raw: dict, y, donor, receiving) -> DonorState:
    import numpy as np

    if np.intersect1d(donor, receiving).size:
        raise AssertionError(
            "Donor partition overlaps the rows it encodes: a row's own label "
            "would reach its target encoding"
        )
    return DonorState(_take(raw, donor), np.asarray(y)[donor])


def fit_donor_state(X, y, donor, receiving) -> DonorState:
    """Fit a donor state on rows ``donor`` of ``X``/``y`` to encode rows ``receiving``.

    Positions, not labels. Refuses the fit when the two overlap -- the
    notebook's ``fit_donor_state`` assert, kept as the family's own runtime
    guard (ADR-0003).
    """
    return _fit_donor_state_raw(_pack_raw(X), y, donor, receiving)


def _donor_features_raw(raw: dict, y, seed: int):
    import numpy as np
    from sklearn.model_selection import StratifiedKFold

    y = np.asarray(y)
    n = len(y)
    features = np.empty((n, len(FEATURE_COLUMNS)), dtype=np.float32)
    visits = np.zeros(n, dtype=np.uint8)
    inner = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=seed)
    for donor, receiving in inner.split(np.zeros(n), y):
        state = _fit_donor_state_raw(raw, y, donor, receiving)
        features[receiving] = state.transform(_take(raw, receiving))
        visits[receiving] += 1
    if not np.all(visits == 1):
        raise AssertionError("Incomplete or duplicated inner donor coverage")
    if not np.isfinite(features).all():
        raise AssertionError("Non-finite donor feature")
    return features


def donor_features(X, y, seed: int):
    """The training rows' features: each row encoded by the inner donor fold it
    is held out of, so no row's own label reaches its encoding."""
    return _donor_features_raw(_pack_raw(X), y, seed)


@dataclass
class HeuljaxModel:
    """The full-training donor state and the booster fitted on donor features."""

    state: DonorState
    booster: Any


def _xgb_params(params: Mapping[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in params.items() if k not in _FAMILY_KEYS}
    out["monotone_constraints"] = MONOTONE_CONSTRAINTS
    return out


def _dmatrix(features, label=None):
    import xgboost as xgb

    return xgb.DMatrix(
        features,
        label=label,
        feature_names=FEATURE_COLUMNS,
        feature_types=FEATURE_TYPES,
        enable_categorical=True,
    )


def fit(X, y, params: Mapping[str, Any], num_boost_round: int) -> HeuljaxModel:
    """Build the family's representation on the training rows and fit XGBoost.

    Fixed ``num_boost_round``, no validation set, no early stopping: the
    Comparison Run protocol.
    """
    import numpy as np
    import xgboost as xgb

    y = np.asarray(y)
    raw = _pack_raw(X)
    train_x = _donor_features_raw(raw, y, int(params[DONOR_SEED_KEY]))
    state = DonorState(raw, y)
    booster = xgb.train(_xgb_params(params), _dmatrix(train_x, label=y), num_boost_round=num_boost_round)
    return HeuljaxModel(state=state, booster=booster)


def predict(model: HeuljaxModel, X):
    """Positive-class probabilities for rows encoded by the full-training state."""
    import numpy as np

    features = model.state.transform(_pack_raw(X))
    pred = model.booster.predict(_dmatrix(features))
    if pred.shape != (len(X),) or not np.isfinite(pred).all():
        raise AssertionError("Invalid predicted probabilities")
    return pred
