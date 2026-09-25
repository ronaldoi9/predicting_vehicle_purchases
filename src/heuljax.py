"""heuljax's ``kps6e09-xgb-sample`` pipeline as an Arena family (#37, #40, turn 3).

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

Modified for this repo (the Apache-2.0 §4(b) notice): the notebook's 173
features and its GAM ``base_margin`` are ported from cells 1-6 (#40); the
tracer-bullet slice of #37 -- the 13 raw columns and the 7 exact-value target
encodings, no margin -- stays selectable as the ``tracer`` feature set, and is
the default so ``heuljax_tracer`` still reproduces. The outer 10-fold split is
replaced by the Canonical Fold Partition (the runner owns it); early stopping
on the scored fold is replaced by a fixed round count; XGBoost runs on CPU
``hist`` and reads a plain ``DMatrix`` instead of a ``QuantileDMatrix``; the
numba kernels (``gam_kernel``, ``mixture_kernel``, ``composition_curve_kernel``,
``invert_curve_kernel``) are rewritten as vectorised numpy, same arithmetic in
a different summation order; the median imputation is dropped because the
loader already refuses missing values; the inner donor split's seed is a
declared parameter held across outer folds instead of ``42 + 3000 + fold``,
the income-group model's seeds are ``group_seed + inner_fold`` and
``group_seed + 1000`` instead of ``7000 + 10*fold + inner_fold`` and ``8000 +
fold``, and the booster seed is the instrument's 0 instead of ``42 + fold``,
because :func:`models.fit` does not know which outer fold it is fitting.

This family consumes the **raw columns** (the ``raw_columns`` Frame spec) and
builds its own representation inside each outer fold, so the representation
belongs to the family and the Baseline Frame is untouched (ADR-0001, ADR-0006
§3). The notebook's **donor state** is its Nested Cross-Fit: every group that
reads a label -- the TEs, the neighbour rates, the income-group model, the
gate mixtures, the composition inversion and the ridge-Newton GAM -- is fitted
by one :class:`DonorState` on donor rows only:

* a training row is encoded by the donor state of the inner fold it is held
  out of (inner ``StratifiedKFold(5)`` over the outer training rows), with an
  assert that the inner folds cover every row exactly once;
* validation and test rows are encoded by the donor state fitted on the whole
  outer training fold.

:func:`fit_donor_state` refuses donor rows that overlap the rows they encode.
The outer boundary is the Model Adapter's: the runner hands this family only
rows the Adapter has already proven are outside the validation fold.

The four original-data income priors read the labels of the 10,000-row
original dataset (``itzzomkar/ev-adoption-behavior-and-range-anxiety``), never
a competition label, after dropping every original row whose raw features
equal a train or test row (:func:`dedup_original`).

pandas / numpy / scipy / scikit-learn / xgboost are imported lazily, like every
other family's library, so the module imports by bare name anywhere.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Any, Mapping

TARGET = "Will_Buy_EV"

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

# The notebook's constants (cell 1).
INNER_FOLDS = 5
GROUP_ESTIMATORS = 600
GAM_ITERS = 12
GAM_STEP = 0.5
CHUNK_ROWS = 32768
INCOME_ALPHAS, COMMUTE_ALPHAS = (5.0, 20.0, 50.0), (20.0, 100.0)
INCOME_WIDTHS, COMMUTE_WIDTHS = (1, 2, 5, 10, 25, 50, 100, 250), (1, 2, 5, 10, 25, 50)
HIER_WIDTHS, HIER_ALPHAS = (5, 25, 100), (2.0, 5.0, 10.0, 20.0, 50.0)
DETAIL_PAIRS = ((1, 5), (5, 25), (25, 100), (100, 250))
LATENT_ALPHAS = (2.0, 5.0, 10.0, 20.0, 50.0, 100.0)
MSTE_ALPHAS = (1.0, 2.0, 5.0, 10.0, 50.0, 200.0)
GXP_ALPHAS = (2.0, 5.0, 10.0, 20.0, 50.0)
MIX_WIDTHS = (16, 64, 256)
SHIFT_GRID_SPEC = (-16.0, 16.0, 129)
OTHER_COLS = (
    "Age",
    "Number_of_Cars_Owned",
    "Charging_Stations_Near_Home",
    "Charging_Stations_Near_Work",
    "Gender",
    "City_Type",
    "Current_Car_Type",
    "Home_Charging_Possible",
)
GROUP_PARAMS: dict[str, Any] = dict(
    objective="reg:logistic",
    tree_method="hist",
    device="cpu",
    learning_rate=0.03,
    max_depth=3,
    min_child_weight=20.0,
    subsample=0.8,
    colsample_bynode=0.8,
    reg_alpha=2.0,
    reg_lambda=100.0,
    n_estimators=GROUP_ESTIMATORS,
)

# The 173 columns (cell 2), grouped as the research note recounts them
# (docs/research/income-te-keys-and-published-pipelines.md §2.1).
SOURCE_COLS = ("ORIG_INC_Y", "ORIG_INC_COUNT_LOG", "ORIG_INC_SEEN", "ORIG_INC_LOCAL_Y")
EXACT_TE_COLS = (
    tuple(f"TE_INC_A{a:g}" for a in INCOME_ALPHAS)
    + tuple(f"TE_CMT_A{a:g}" for a in COMMUTE_ALPHAS)
    + ("SUPPORT_INC_LOG", "SUPPORT_CMT_LOG")
)
NEIGHBOUR_COLS = (
    tuple(f"NBR_INC_W{w}" for w in INCOME_WIDTHS)
    + tuple(f"NBR_DETAIL_W{a}_{b}" for a, b in DETAIL_PAIRS)
    + tuple(f"HIER_INC_W{w}_A{a:g}" for w in HIER_WIDTHS for a in HIER_ALPHAS)
    + tuple(f"INC_SURPRISE_W{w}" for w in HIER_WIDTHS)
    + tuple(f"NBR_CMT_W{w}" for w in COMMUTE_WIDTHS)
    + ("DCMT_LOCAL_W5", "P_DCMT_LOCAL_W5")
)
GXP_COLS = ("GXP_D3_R100",) + tuple(f"GXP_D3_R100_A{a:g}" for a in GXP_ALPHAS)
UNCERTAINTY_COLS = ("GXP_POST_SD_A20", "GXP_POST_LO_A20", "GXP_POST_HI_A20")
LATENT_NAMES = ("CONCERN", "SUBSIDY", "ANXIETY_PENALTY", "BUY_SCORE")
LAT_COLS = tuple(f"LAT_INC_{n}_A{a:g}" for n in LATENT_NAMES for a in LATENT_ALPHAS) + (
    "LAT_INC_CONCERN_DEV",
    "LAT_INC_SUBSIDY_DEV",
    "LAT_INC_ANXIETY_DEV",
    "LAT_INC_SCORE_DEV",
)
DIGIT_COLS = tuple(f"DIG_INC_10E{k}" for k in range(6)) + tuple(f"DIG_CMT10_10E{k}" for k in range(4))
WORRY_COLS = ("WTE05_A5", "WHIER05_W25_A20")
MIX_SUFFIXES = ("PRIOR", "POST20", "POST100", "ROW_PROB", "INFO", "GAIN")
MIX_COLS = tuple(f"MIX_W{w}_{s}" for w in MIX_WIDTHS for s in MIX_SUFFIXES)
CHANNELS = ("GXP_D3_R100", "GXP_D3_R100_A20", "GXP_D3_R100_A50") + tuple(
    f"MIX_W{w}_POST100" for w in MIX_WIDTHS
)
COMPOSITION_COLS = tuple(f"CSHIFT_{c}" for c in CHANNELS) + tuple(f"CCORR_{c}" for c in CHANNELS)
COORD_COLS = ("GAM_GATE_COORD", "GAM_INCOME_COORD", "GAM_COMMUTE_COORD", "GAM_OTHER_COORD")
MSTE_KEYS = ("INC10", "INC100", "INC1K", "CMTINT")
MSTE_COLS = tuple(
    c for key in MSTE_KEYS for c in (tuple(f"MSTE_{key}_A{a:g}" for a in MSTE_ALPHAS) + (f"MSTE_{key}_LOGN",))
)

FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "raw": RAW_COLS,
    "original_priors": SOURCE_COLS,
    "exact_te": EXACT_TE_COLS,
    "neighbour_hierarchical": NEIGHBOUR_COLS,
    "group_model_priors": GXP_COLS + UNCERTAINTY_COLS,
    "latent_means": LAT_COLS,
    "digits": DIGIT_COLS,
    "worry_key": WORRY_COLS,
    "gate_mixtures": MIX_COLS,
    "composition": COMPOSITION_COLS,
    "gam_coordinates": COORD_COLS,
    "multi_scale_te": MSTE_COLS,
}

# The notebook's FEATURE_COLS order, which interleaves the groups above.
FEATURE_COLUMNS: list[str] = list(
    RAW_COLS
    + SOURCE_COLS
    + EXACT_TE_COLS
    + NEIGHBOUR_COLS
    + GXP_COLS
    + LAT_COLS
    + DIGIT_COLS
    + WORRY_COLS
    + UNCERTAINTY_COLS
    + MIX_COLS
    + COMPOSITION_COLS
    + COORD_COLS
    + MSTE_COLS
)
if len(FEATURE_COLUMNS) != len(set(FEATURE_COLUMNS)) or len(FEATURE_COLUMNS) != 173:
    raise AssertionError("heuljax's feature list must be 173 unique columns")
COL = {name: i for i, name in enumerate(FEATURE_COLUMNS)}
FEATURE_TYPES: list[str] = ["c" if c in CAT_COLS + DIGIT_COLS else "q" for c in FEATURE_COLUMNS]


def _is_rate(c: str) -> bool:
    """The notebook's RATE_COLS predicate (cell 2): +1 monotone target rates."""
    return (
        c in ("ORIG_INC_Y", "ORIG_INC_LOCAL_Y", "WTE05_A5", "WHIER05_W25_A20", "GXP_POST_LO_A20", "GXP_POST_HI_A20")
        or c.startswith(("TE_INC_", "TE_CMT_", "NBR_INC_", "NBR_CMT_", "HIER_INC_", "GXP_D3_R100_A"))
        or (c.startswith("MIX_") and any(s in c for s in ("_PRIOR", "_POST")))
        or (c.startswith("MSTE_") and "_A" in c)
    )


RATE_COLS = tuple(c for c in FEATURE_COLUMNS if _is_rate(c))


def _monotone(columns) -> str:
    return "(" + ",".join("1" if _is_rate(c) else "0" for c in columns) + ")"


MONOTONE_CONSTRAINTS = _monotone(FEATURE_COLUMNS)

# Keys in the Experiment's params this module reads itself; the rest go to
# xgboost.train unchanged.
DONOR_SEED_KEY = "donor_seed"
GROUP_SEED_KEY = "group_seed"
FEATURE_SET_KEY = "feature_set"
_FAMILY_KEYS = (DONOR_SEED_KEY, GROUP_SEED_KEY, FEATURE_SET_KEY)


@dataclass(frozen=True)
class FeatureSet:
    """The columns handed to the booster, and whether it starts at the GAM margin."""

    name: str
    columns: list[str]
    base_margin: bool

    @property
    def full(self) -> bool:
        return self.name == "full"

    @property
    def feature_types(self) -> list[str]:
        return [FEATURE_TYPES[COL[c]] for c in self.columns]

    @property
    def monotone_constraints(self) -> str:
        return _monotone(self.columns)


FEATURE_SETS: dict[str, FeatureSet] = {
    # #37's slice. The default, so heuljax_tracer's config (which never named a
    # feature set) keeps both its hash and its numbers.
    "tracer": FeatureSet("tracer", list(RAW_COLS + EXACT_TE_COLS), base_margin=False),
    # #40: the notebook's 173 columns, boosted on top of its GAM margin.
    "full": FeatureSet("full", FEATURE_COLUMNS, base_margin=True),
}


def feature_set(params: Mapping[str, Any]) -> FeatureSet:
    name = params.get(FEATURE_SET_KEY, "tracer")
    try:
        return FEATURE_SETS[name]
    except KeyError:
        raise ValueError(f"unknown heuljax feature set {name!r}; known: {sorted(FEATURE_SETS)}") from None


def shift_grid():
    """The composition inversion's grid of shifts: 129 points on [-16, 16]."""
    import numpy as np

    return np.linspace(*SHIFT_GRID_SPEC)




# --------------------------------------------------------------------------- #
# Kernels (cell 3), numpy in place of numba.
# --------------------------------------------------------------------------- #
def box_sum(values, width):
    """Sum over the ``±width`` window on axis 0, clipped at both ends."""
    import numpy as np

    values = np.asarray(values, dtype=np.float64)
    prefix = np.concatenate([np.zeros((1,) + values.shape[1:]), np.cumsum(values, axis=0)], axis=0)
    center = np.arange(len(values))
    return prefix[np.minimum(center + width + 1, len(values))] - prefix[np.maximum(center - width, 0)]


GAUSSIAN_BACKENDS = {16: "direct", 64: "fft", 256: "fft"}


def gaussian_smooth(values, width, backend=None):
    import numpy as np
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import fftconvolve

    backend = GAUSSIAN_BACKENDS[int(width)] if backend is None else backend
    if backend == "direct":
        return gaussian_filter1d(values, width, axis=0, mode="constant")
    radius = int(4.0 * width + 0.5)
    positions = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (positions / width) ** 2)
    kernel /= kernel.sum()
    return fftconvolve(values, kernel[:, None], mode="same", axes=0)


def gam_kernel(codes, sizes, ridges, widths, targets, iterations, step):
    """Backfitted ridge-Newton steps of an additive logistic model.

    One Newton step per term per epoch on the per-code gradient and Hessian,
    box-summed over ``±width`` neighbouring codes for the smoothed terms.
    """
    import numpy as np

    codes = np.asarray(codes)
    n, nt = codes.shape
    targets = np.asarray(targets, dtype=np.float64)
    prior = min(max(float(np.mean(targets)), 1e-6), 1.0 - 1e-6)
    intercept = math.log(prior / (1.0 - prior))
    eta = np.full(n, intercept, dtype=np.float64)
    tables = np.zeros((nt, int(np.max(sizes))), dtype=np.float64)
    for _ in range(iterations):
        for term in range(nt):
            length = int(sizes[term])
            code = codes[:, term]
            p = 1.0 / (1.0 + np.exp(-np.clip(eta, -700.0, 700.0)))
            grad = np.bincount(code, weights=targets - p, minlength=length)
            hess = np.bincount(code, weights=p * (1.0 - p), minlength=length)
            if widths[term]:
                grad = box_sum(grad, int(widths[term]))
                hess = box_sum(hess, int(widths[term]))
            delta = step * grad / (hess + ridges[term])
            tables[term, :length] += delta
            eta += delta[code]
    return intercept, tables, eta


def mixture_kernel(counts, q0, q1, mu):
    """Per income value: the mixture weight maximising a penalised likelihood.

    24 clamped Newton steps on ``p`` for ``sum_j n_j log(q0_j + p (q1_j - q0_j))``
    plus a Beta-like pull toward ``mu``; then its Fisher information and the
    mean log-likelihood gain over ``p = mu``.
    """
    import numpy as np

    d = q1 - q0
    p = np.asarray(mu, dtype=np.float64).copy()
    for _ in range(24):
        g = 10.0 * (mu / p - (1.0 - mu) / (1.0 - p))
        h = 10.0 * (mu / (p * p) + (1.0 - mu) / ((1.0 - p) * (1.0 - p)))
        r = d / np.maximum(q0 + p[:, None] * d, 1e-300)
        g = g + (counts * r).sum(axis=1)
        h = h + (counts * r * r).sum(axis=1)
        p = np.clip(p + np.clip(g / np.maximum(h, 1e-8), -0.15, 0.15), 1e-4, 1.0 - 1e-4)
    m = np.maximum(q0 + p[:, None] * d, 1e-300)
    old = np.maximum(q0 + mu[:, None] * d, 1e-300)
    information = np.log1p((counts * (d / m) ** 2).sum(axis=1))
    gain = (counts * np.log(m / old)).sum(axis=1) / np.maximum(counts.sum(axis=1), 1.0)
    return p, information, gain


def composition_curve_kernel(nuisance_sorted, boundaries, grid):
    """Per group of rows (sorted by group), the mean of ``sigmoid(nuisance + s)`` over the grid."""
    import numpy as np

    ng = len(boundaries) - 1
    curve = np.zeros((ng, len(grid)), dtype=np.float64)
    group = np.repeat(np.arange(ng), np.diff(boundaries))
    for start in range(0, len(nuisance_sorted), CHUNK_ROWS):
        stop = min(start + CHUNK_ROWS, len(nuisance_sorted))
        z = np.clip(nuisance_sorted[start:stop, None] + grid[None, :], -700.0, 700.0)
        values = 1.0 / (1.0 + np.exp(-z))
        g = group[start:stop]
        firsts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
        curve[g[firsts]] += np.add.reduceat(values, firsts, axis=0)
    return curve / np.maximum(np.diff(boundaries), 1)[:, None]


def invert_curve_kernel(curves, indices, probabilities, grid):
    """For each row and channel, the shift ``s`` at which its group's curve reaches ``p``.

    Linear interpolation between grid points, clamped to the grid's ends.
    """
    import numpy as np

    n, nc = probabilities.shape
    out = np.empty((n, nc), dtype=np.float64)
    first, last = curves[indices, 0], curves[indices, -1]
    for j in range(nc):
        p = probabilities[:, j]
        lo = np.zeros(n, dtype=np.int64)
        hi = np.full(n, len(grid) - 1, dtype=np.int64)
        while True:
            open_ = hi - lo > 1
            if not open_.any():
                break
            mid = (lo + hi) // 2
            below = curves[indices, mid] < p
            lo = np.where(open_ & below, mid, lo)
            hi = np.where(open_ & ~below, mid, hi)
        c_lo, c_hi = curves[indices, lo], curves[indices, hi]
        denom = c_hi - c_lo
        safe = denom > 1e-14
        frac = np.where(safe, (p - c_lo) / np.where(safe, denom, 1.0), 0.0)
        value = grid[lo] + frac * (grid[hi] - grid[lo])
        out[:, j] = np.where(p <= first, grid[0], np.where(p >= last, grid[-1], value))
    return out


class ValueAxis:
    """The sorted distinct values of a donor column, and a nearest-value locator."""

    def __init__(self, values) -> None:
        import numpy as np

        self.values, codes = np.unique(np.asarray(values), return_inverse=True)
        self.codes = codes.astype(np.int32)
        self.size = len(self.values)
        if self.size == 0:
            raise ValueError("Empty donor value axis")

    def locate(self, query):
        """The index of each query's exact value, else its nearest; and the exact flag."""
        import numpy as np

        query = np.asarray(query)
        pos = np.searchsorted(self.values, query)
        hi = np.minimum(pos, self.size - 1)
        lo = np.maximum(pos - 1, 0)
        exact = (pos < self.size) & (self.values[hi] == query)
        nearest = np.where(np.abs(query - self.values[lo]) <= np.abs(query - self.values[hi]), lo, hi)
        return np.where(exact, hi, nearest).astype(np.int32), exact


class TargetTable:
    """Per-exact-value row count and target sum over the donor rows."""

    def __init__(self, values, targets) -> None:
        import numpy as np

        self.axis = ValueAxis(values)
        self.count = np.bincount(self.axis.codes, minlength=self.axis.size).astype(np.float64)
        self.total = np.bincount(self.axis.codes, weights=targets, minlength=self.axis.size)
        self.prior = float(np.mean(targets))
        self._local: dict[tuple[int, float], Any] = {}

    def lookup(self, query):
        """``(index, exact, count, total)``; count and total are zero off an exact value."""
        idx, exact = self.axis.locate(query)
        return idx, exact, self.count[idx] * exact, self.total[idx] * exact

    def local(self, width, alpha=20.0):
        """The rate of the ``±width`` neighbouring distinct values, own value left out."""
        import numpy as np

        key = (int(width), float(alpha))
        if key not in self._local:
            count = np.maximum(box_sum(self.count, width) - self.count, 0.0)
            total = box_sum(self.total, width) - self.total
            self._local[key] = (total + alpha * self.prior) / (count + alpha)
        return self._local[key]


# --------------------------------------------------------------------------- #
# Raw columns and the label-free derivations (cell 4).
# --------------------------------------------------------------------------- #
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
    out = {c: pd.to_numeric(X[c]).to_numpy(dtype=np.float64) for c in NUM_COLS}
    for c in CAT_COLS:
        vocab = CATEGORY_VOCAB[c]
        codes = pd.Categorical(X[c].astype("string"), categories=list(vocab)).codes
        out[c] = np.where(codes < 0, len(vocab), codes).astype(np.int16)
    return out


def _take(raw: dict, rows) -> dict:
    import numpy as np

    return {k: np.ascontiguousarray(v[rows]) for k, v in raw.items()}


def latent_values(raw):
    """The Recipe's inputs per row: concern, subsidy, anxiety penalty, buy score."""
    import numpy as np

    income = raw[INCOME_COLUMN]
    concern = raw["Environmental_Concern_Level"]
    subsidy = (raw["Subsidy_Available"] == CATEGORY_VOCAB["Subsidy_Available"].index("Yes")).astype(np.float64)
    anxiety = raw["Range_Anxiety_Level"]
    penalty = np.where(anxiety == 1, 1.0, np.where(anxiety == 2, 3.0, 0.0))
    score = 1.2 * income / 100000.0 + 0.6 * concern + 2.0 * subsidy - penalty
    return np.column_stack([concern, subsidy, penalty, score])


def gate_codes(raw):
    """The concern x subsidy x anxiety cell (0-29), 30 for an unrecognised row."""
    import numpy as np

    concern = np.rint(raw["Environmental_Concern_Level"]).astype(np.int32) - 1
    anxiety = raw["Range_Anxiety_Level"].astype(np.int32)
    subsidy = (raw["Subsidy_Available"] == 1).astype(np.int32)
    valid = (concern >= 0) & (concern < 5) & (anxiety < 3) & (raw["Subsidy_Available"] < 2)
    return np.where(valid, concern * 6 + subsidy * 3 + anxiety, 30).astype(np.int32)


def worry_key(raw):
    """``round((commute - 5 home - 5 work - 150 [no home charging]) / 0.5)``."""
    import numpy as np

    home = (raw["Home_Charging_Possible"] == 0).astype(np.float64)
    value = (
        raw[COMMUTE_COLUMN]
        - 5.0 * raw["Charging_Stations_Near_Home"]
        - 5.0 * raw["Charging_Stations_Near_Work"]
        - 150.0 * home
    )
    return np.rint(value / 0.5).astype(np.int64)


def mste_keys(raw):
    import numpy as np

    income, commute = raw[INCOME_COLUMN], raw[COMMUTE_COLUMN]
    return dict(
        INC10=np.floor(income / 10),
        INC100=np.floor(income / 100),
        INC1K=np.floor(income / 1000),
        CMTINT=np.floor(commute),
    )


def digit_features(raw):
    """Single decimal digits of income (10^0..10^5) and of ``round(10 commute)`` (10^0..10^3)."""
    import numpy as np

    income = np.rint(raw[INCOME_COLUMN]).astype(np.int64)
    commute = np.rint(10.0 * raw[COMMUTE_COLUMN]).astype(np.int64)
    return np.column_stack(
        [(income // 10**k) % 10 for k in range(6)] + [(commute // 10**k) % 10 for k in range(4)]
    )


def _feature_hashes(frame):
    import numpy as np
    import pandas as pd

    normalized = frame[list(RAW_COLS)].copy()
    for c in NUM_COLS:
        normalized[c] = pd.to_numeric(normalized[c], errors="coerce").astype(np.float64)
    for c in CAT_COLS:
        normalized[c] = normalized[c].astype("string").fillna("__MISSING__")
    return pd.util.hash_pandas_object(normalized, index=False).to_numpy(np.uint64)


def dedup_original(original, *competition_frames):
    """The original rows whose raw features match no competition row, labelled Yes/No.

    Label-free with respect to the competition: only the train and test
    *features* are hashed.
    """
    import numpy as np

    if not set(RAW_COLS + (TARGET,)).issubset(original.columns):
        raise ValueError("Original dataset schema mismatch")
    original_hash = _feature_hashes(original)
    keep = original[TARGET].isin(["No", "Yes"]).to_numpy()
    for frame in competition_frames:
        keep &= ~np.isin(original_hash, _feature_hashes(frame))
    kept = original.loc[keep].reset_index(drop=True)
    if not len(kept):
        raise ValueError("No eligible original rows remain")
    return kept


class SourceIncome:
    """The original dataset's income priors: its mean label, support, seen flag, local rate."""

    def __init__(self, source) -> None:
        import numpy as np
        import pandas as pd

        yy = source[TARGET].eq("Yes").to_numpy(np.float64)
        self.prior = float(yy.mean())
        vals = pd.to_numeric(source[INCOME_COLUMN], errors="coerce").to_numpy(np.float64)
        good = np.isfinite(vals)
        if not good.any():
            raise ValueError("Original income is entirely missing")
        self.table = TargetTable(vals[good], yy[good])
        self.mean = self.table.total / self.table.count
        self.local = (box_sum(self.table.total, 25) + 20.0 * self.prior) / (box_sum(self.table.count, 25) + 20.0)

    def transform(self, income):
        import numpy as np

        idx, exact, count, _ = self.table.lookup(income)
        mean = np.where(exact, self.mean[idx], self.prior)
        return np.column_stack([mean, np.log1p(count), exact.astype(np.float64), self.local[idx]])


@functools.lru_cache(maxsize=1)
def load_source() -> SourceIncome:
    """The original-data income priors, from ``data/`` (read once per process)."""
    import data

    return SourceIncome(dedup_original(data.load_original(), data.load_train(), data.load_test()))


# --------------------------------------------------------------------------- #
# The fold-fitted models (cell 5).
# --------------------------------------------------------------------------- #
class AdditiveState:
    """The ridge-Newton GAM: a gate term, income and commute terms, the other columns."""

    def __init__(self, raw, targets, axes, main_only=False) -> None:
        import numpy as np

        self.axes = axes
        self.terms = [("gate", 0, 5.0, 0)]
        if not main_only:
            self.terms += [("income", w, 25.0, 1) for w in (0, 2, 8, 32, 128)]
            self.terms += [("commute", w, 25.0, 2) for w in (0, 5)]
        self.terms += [(c, 0, 5.0, 3) for c in OTHER_COLS]
        arrays, sizes = [], []
        for key, _, _, _ in self.terms:
            if key == "gate":
                arrays.append(gate_codes(raw))
                sizes.append(31)
            else:
                arrays.append(axes[key].codes)
                sizes.append(axes[key].size)
        codes = np.ascontiguousarray(np.column_stack(arrays), dtype=np.int32)
        self.intercept, self.tables, self.fitted_eta = gam_kernel(
            codes,
            np.asarray(sizes, np.int32),
            np.asarray([t[2] for t in self.terms], np.float64),
            np.asarray([t[1] for t in self.terms], np.int32),
            np.asarray(targets, np.float64),
            GAM_ITERS,
            GAM_STEP,
        )

    def components(self, raw):
        """Per row: the gate, income, commute and other (with intercept) components."""
        import numpy as np

        n = len(raw["Age"])
        result = np.zeros((n, 4), dtype=np.float64)
        result[:, 3] = self.intercept
        located: dict[str, Any] = {}
        for t, (key, width, _, component) in enumerate(self.terms):
            if key not in located:
                if key == "gate":
                    located[key] = (gate_codes(raw), np.ones(n, dtype=bool))
                else:
                    column = {"income": INCOME_COLUMN, "commute": COMMUTE_COLUMN}.get(key, key)
                    located[key] = self.axes[key].locate(raw[column])
            idx, exact = located[key]
            effect = self.tables[t, idx]
            result[:, component] += effect if width else effect * exact
        return result


class IncomeGroupPrior:
    """An XGBRegressor at the income-value level: per-value covariate means -> per-value rate."""

    def __init__(self, raw, income_table, source, seed, nthread) -> None:
        import numpy as np
        import xgboost as xgb
        from scipy.special import expit

        axis, counts, total = income_table.axis, income_table.count, income_table.total
        latent = latent_values(raw)
        score = latent[:, 3]
        commute = raw[COMMUTE_COLUMN]
        values = [
            latent[:, 0], latent[:, 1], latent[:, 2], score, score**2,
            expit(score - 5.5), expit(2.0 * (score - 5.5)), expit(3.0 * (score - 5.5)),
            (raw["Home_Charging_Possible"] == 0).astype(np.float64), commute, commute**2,
            raw["Charging_Stations_Near_Home"], raw["Charging_Stations_Near_Work"],
            raw["Age"], raw["Number_of_Cars_Owned"],
        ]  # fmt: skip
        values += [(score >= threshold).astype(np.float64) for threshold in (4.5, 5.0, 5.5, 6.0, 6.5)]
        for column in ("City_Type", "Current_Car_Type", "Gender"):
            values += [
                (raw[column] == CATEGORY_VOCAB[column].index(category)).astype(np.float64)
                for category in sorted(CATEGORY_VOCAB[column])
            ]
        base = [axis.values / 100000.0, np.log1p(counts)]
        base += [np.bincount(axis.codes, weights=v, minlength=axis.size) / counts for v in values]
        base += list(source.transform(axis.values).T)
        matrix = np.column_stack(base).astype(np.float64)
        self.template = matrix.mean(axis=0)
        self.mean = self.template.copy()
        self.scale = matrix.std(axis=0)
        self.scale[self.scale < 1e-8] = 1.0
        matrix = np.ascontiguousarray((matrix - self.mean) / self.scale)
        self.model = xgb.XGBRegressor(**GROUP_PARAMS, n_jobs=nthread, random_state=seed)
        self.model.fit(matrix, total / counts, sample_weight=np.sqrt(counts), verbose=False)
        self.prediction = np.clip(self.model.predict(matrix), 1e-5, 1.0 - 1e-5).astype(np.float64)
        self.source = source

    def transform(self, income, idx, exact):
        import numpy as np

        prior = self.prediction[idx].copy()
        if not exact.all():
            unseen, reverse = np.unique(income[~exact], return_inverse=True)
            matrix = np.tile(self.template, (len(unseen), 1))
            matrix[:, 0] = unseen / 100000.0
            matrix[:, 1] = 0.0
            matrix[:, -4:] = self.source.transform(unseen)
            scaled = np.ascontiguousarray((matrix - self.mean) / self.scale)
            prior[~exact] = np.clip(self.model.predict(scaled), 1e-5, 1.0 - 1e-5)[reverse]
        return prior


class DonorState:
    """The target statistics of one set of donor rows, applied to other rows.

    ``full=False`` fits only the income and commute tables the tracer slice
    reads; ``full=True`` fits every group of the 173 and the GAM.
    """

    def __init__(self, raw: dict, targets, *, full: bool = False, source=None, seed: int = 0, nthread: int = 1):
        import numpy as np

        targets = np.asarray(targets, dtype=np.float64)
        if len(raw[INCOME_COLUMN]) != len(targets) or len(np.unique(targets)) != 2:
            raise ValueError("Donor must contain aligned data and both target classes")
        self.full = full
        self.prior = float(targets.mean())
        self.income = TargetTable(raw[INCOME_COLUMN], targets)
        self.commute = TargetTable(raw[COMMUTE_COLUMN], targets)
        if full:
            if source is None:
                raise ValueError("The full donor state needs the original-data income priors")
            self._fit_full(raw, targets, source, seed, nthread)

    def _fit_full(self, raw, targets, source, seed, nthread) -> None:
        import numpy as np
        from scipy.special import expit

        self.source = source
        self.worry = TargetTable(worry_key(raw), targets)
        self.mste = {key: TargetTable(value, targets) for key, value in mste_keys(raw).items()}
        axes = {"income": self.income.axis, "commute": self.commute.axis}
        axes.update({c: ValueAxis(raw[c]) for c in OTHER_COLS})
        self.gam = AdditiveState(raw, targets, axes)
        main = AdditiveState(raw, targets, axes, main_only=True)
        main_probability = expit(main.fitted_eta)
        cg = self.commute.axis.codes
        size = self.commute.axis.size
        gradient = np.bincount(cg, weights=targets - main_probability, minlength=size)
        hessian = np.bincount(cg, weights=main_probability * (1.0 - main_probability), minlength=size)
        self.commute_effect = box_sum(gradient, 5) / (box_sum(hessian, 5) + 20.0)
        self.main = main
        self.gxp = IncomeGroupPrior(raw, self.income, source, seed, nthread)
        latent = latent_values(raw)
        self.latent_prior = latent.mean(axis=0)
        income_code = self.income.axis.codes
        self.latent_total = np.column_stack(
            [np.bincount(income_code, weights=v, minlength=self.income.axis.size) for v in latent.T]
        )
        gate = gate_codes(raw)
        self.n_gate = max(30, int(gate.max()) + 1)
        joint = income_code.astype(np.int64) * self.n_gate + gate
        cells = self.income.axis.size * self.n_gate
        c1 = np.bincount(joint, weights=targets, minlength=cells).reshape(-1, self.n_gate)
        allcounts = np.bincount(joint, minlength=cells).reshape(-1, self.n_gate).astype(np.float64)
        c0 = allcounts - c1
        self.marg0 = (c0.sum(axis=0) + 0.5) / (c0.sum() + 0.5 * self.n_gate)
        self.marg1 = (c1.sum(axis=0) + 0.5) / (c1.sum() + 0.5 * self.n_gate)
        self.mixtures = {}
        combined = np.column_stack([c0, c1])
        for width in MIX_WIDTHS:
            smooth = gaussian_smooth(combined, width) * np.sqrt(2.0 * np.pi) * width
            s0 = np.maximum(smooth[:, : self.n_gate] - c0, 0.0)
            s1 = np.maximum(smooth[:, self.n_gate :] - c1, 0.0)
            q0 = (s0 + 30.0 * self.marg0) / (s0.sum(axis=1, keepdims=True) + 30.0)
            q1 = (s1 + 30.0 * self.marg1) / (s1.sum(axis=1, keepdims=True) + 30.0)
            mu = np.clip(
                (s1.sum(axis=1) + 20.0 * self.prior) / (s0.sum(axis=1) + s1.sum(axis=1) + 20.0), 1e-4, 1.0 - 1e-4
            )
            pi, information, gain = mixture_kernel(allcounts, q0, q1, mu)
            row_probability = pi[:, None] * q1 / np.maximum(q0 + pi[:, None] * (q1 - q0), 1e-300)
            self.mixtures[width] = dict(pi=pi, row_probability=row_probability, information=information, gain=gain)
        donor_components = self.gam.components(raw)
        nuisance = donor_components[:, 0] + donor_components[:, 3] + 0.25 * donor_components[:, 2]
        order = np.argsort(income_code, kind="stable")
        boundaries = np.r_[0, np.cumsum(self.income.count.astype(np.int64))]
        curve = composition_curve_kernel(nuisance[order], boundaries, shift_grid())
        global_curve = np.average(curve, weights=self.income.count, axis=0)
        self.curves = np.vstack([curve, global_curve])
        if not np.isfinite(self.curves).all() or np.any(np.diff(self.curves, axis=1) < -1e-12):
            raise AssertionError("Invalid composition curves")
        # Drop the donor-row arrays; only the fitted tables are applied.
        self.gam.fitted_eta = None
        self.main.fitted_eta = None
        for axis in list(axes.values()) + [self.worry.axis] + [t.axis for t in self.mste.values()]:
            axis.codes = None

    def transform(self, raw: dict):
        """``(features, margin)`` for ``raw``: all 173 columns (float32) and the GAM margin.

        The tracer state fills only the raw and exact-TE columns; its margin is
        ``None``.
        """
        import numpy as np

        n = len(raw[INCOME_COLUMN])
        result = np.zeros((n, len(FEATURE_COLUMNS)), dtype=np.float32)
        for c in RAW_COLS:
            result[:, COL[c]] = raw[c]
        income, commute = raw[INCOME_COLUMN], raw[COMMUTE_COLUMN]
        ii, ie, count, total = self.income.lookup(income)
        ci, _, ccount, ctotal = self.commute.lookup(commute)
        for alpha in INCOME_ALPHAS:
            result[:, COL[f"TE_INC_A{alpha:g}"]] = (total + alpha * self.prior) / (count + alpha)
        for alpha in COMMUTE_ALPHAS:
            result[:, COL[f"TE_CMT_A{alpha:g}"]] = (ctotal + alpha * self.prior) / (ccount + alpha)
        result[:, COL["SUPPORT_INC_LOG"]] = np.log1p(count)
        result[:, COL["SUPPORT_CMT_LOG"]] = np.log1p(ccount)
        if not self.full:
            if not np.isfinite(result).all():
                raise AssertionError("Non-finite donor feature")
            return result, None
        margin = self._transform_full(raw, result, ii, ie, ci, count, total)
        if not np.isfinite(result).all() or not np.isfinite(margin).all():
            raise AssertionError("Nonfinite features or initial margin")
        return result, margin

    def _transform_full(self, raw, result, ii, ie, ci, count, total):
        import numpy as np
        from scipy.special import expit

        income = raw[INCOME_COLUMN]
        result[:, [COL[c] for c in SOURCE_COLS]] = self.source.transform(income)
        local = {w: self.income.local(w)[ii] for w in INCOME_WIDTHS}
        for width in INCOME_WIDTHS:
            result[:, COL[f"NBR_INC_W{width}"]] = local[width]
        for a, b in DETAIL_PAIRS:
            result[:, COL[f"NBR_DETAIL_W{a}_{b}"]] = local[a] - local[b]
        for width in HIER_WIDTHS:
            for alpha in HIER_ALPHAS:
                result[:, COL[f"HIER_INC_W{width}_A{alpha:g}"]] = (total + alpha * local[width]) / (count + alpha)
            result[:, COL[f"INC_SURPRISE_W{width}"]] = result[:, COL["TE_INC_A5"]] - local[width]
        for width in COMMUTE_WIDTHS:
            result[:, COL[f"NBR_CMT_W{width}"]] = self.commute.local(width)[ci]
        main = self.main.components(raw).sum(axis=1)
        effect = self.commute_effect[ci]
        result[:, COL["DCMT_LOCAL_W5"]] = effect
        result[:, COL["P_DCMT_LOCAL_W5"]] = expit(main + effect)
        prior = self.gxp.transform(income, ii, ie)
        result[:, COL["GXP_D3_R100"]] = prior
        for alpha in GXP_ALPHAS:
            result[:, COL[f"GXP_D3_R100_A{alpha:g}"]] = (total + alpha * prior) / (count + alpha)
        latent = latent_values(raw)
        grouped = self.latent_total[ii] * ie[:, None]
        for j, name in enumerate(LATENT_NAMES):
            for alpha in LATENT_ALPHAS:
                result[:, COL[f"LAT_INC_{name}_A{alpha:g}"]] = (
                    (grouped[:, j] + alpha * self.latent_prior[j]) / (count + alpha)
                )
        for j, name in enumerate(("CONCERN", "SUBSIDY", "ANXIETY", "SCORE")):
            expectation = (grouped[:, j] + 20.0 * self.latent_prior[j]) / (count + 20.0)
            result[:, COL[f"LAT_INC_{name}_DEV"]] = latent[:, j] - expectation
        result[:, [COL[c] for c in DIGIT_COLS]] = digit_features(raw)
        wi, _, wc, wt = self.worry.lookup(worry_key(raw))
        result[:, COL["WTE05_A5"]] = (wt + 5.0 * self.prior) / (wc + 5.0)
        result[:, COL["WHIER05_W25_A20"]] = (wt + 20.0 * self.worry.local(25)[wi]) / (wc + 20.0)
        p20 = (total + 20.0 * prior) / (count + 20.0)
        sd = np.sqrt(np.maximum(p20 * (1.0 - p20) / (count + 21.0), 1e-12))
        for c, value in zip(UNCERTAINTY_COLS, (sd, p20 - sd, p20 + sd)):
            result[:, COL[c]] = value
        gg = gate_codes(raw)
        safe_gate = np.minimum(gg, self.n_gate - 1)
        known_gate = gg < self.n_gate
        unknown_row_probability = (
            self.prior * self.marg1[safe_gate]
            / np.maximum(self.marg0[safe_gate] + self.prior * (self.marg1[safe_gate] - self.marg0[safe_gate]), 1e-300)
        )  # fmt: skip
        for width in MIX_WIDTHS:
            state = self.mixtures[width]
            pi = np.where(ie, state["pi"][ii], self.prior)
            row_p = np.where(ie, state["row_probability"][ii, safe_gate], unknown_row_probability)
            row_p = np.where(known_gate, row_p, pi)
            values = (
                pi,
                (total + 20.0 * pi) / (count + 20.0),
                (total + 100.0 * pi) / (count + 100.0),
                row_p,
                np.where(ie, state["information"][ii], 0.0),
                np.where(ie, state["gain"][ii], 0.0),
            )
            for suffix, value in zip(MIX_SUFFIXES, values):
                result[:, COL[f"MIX_W{width}_{suffix}"]] = value
        components = self.gam.components(raw)
        result[:, [COL[c] for c in COORD_COLS]] = components
        channel_probabilities = np.asarray(result[:, [COL[c] for c in CHANNELS]], dtype=np.float64)
        curve_index = np.where(ie, ii, self.income.axis.size).astype(np.int32)
        shift = invert_curve_kernel(self.curves, curve_index, channel_probabilities, shift_grid())
        result[:, [COL[c] for c in COMPOSITION_COLS[:6]]] = shift
        result[:, [COL[c] for c in COMPOSITION_COLS[6:]]] = shift - 0.25 * components[:, 1, None]
        for key, values in mste_keys(raw).items():
            _, _, cn, sy = self.mste[key].lookup(values)
            for alpha in MSTE_ALPHAS:
                result[:, COL[f"MSTE_{key}_A{alpha:g}"]] = (sy + alpha * self.prior) / (cn + alpha)
            result[:, COL[f"MSTE_{key}_LOGN"]] = np.log1p(cn)
        return (components[:, 0] + components[:, 3] + 0.25 * (components[:, 1] + components[:, 2])).astype(
            np.float32
        )

    def transform_chunked(self, raw: dict):
        """:meth:`transform` over ``CHUNK_ROWS``-row slices, as the notebook's ``transform_into``."""
        import numpy as np

        n = len(raw[INCOME_COLUMN])
        features = np.empty((n, len(FEATURE_COLUMNS)), dtype=np.float32)
        margin = np.empty(n, dtype=np.float32) if self.full else None
        for start in range(0, n, CHUNK_ROWS):
            rows = np.arange(start, min(start + CHUNK_ROWS, n))
            f, m = self.transform(_take(raw, rows))
            features[rows] = f
            if margin is not None:
                margin[rows] = m
        return features, margin


# --------------------------------------------------------------------------- #
# The Nested Cross-Fit (cell 6).
# --------------------------------------------------------------------------- #
def _fit_donor_state_raw(raw: dict, y, donor, receiving, *, full, group_seed, nthread=1) -> DonorState:
    import numpy as np

    if np.intersect1d(donor, receiving).size:
        raise AssertionError(
            "Donor partition overlaps the rows it encodes: a row's own label "
            "would reach its target encoding"
        )
    return DonorState(
        _take(raw, donor),
        np.asarray(y)[donor],
        full=full,
        source=load_source() if full else None,
        seed=group_seed,
        nthread=nthread,
    )


def fit_donor_state(X, y, donor, receiving, *, full: bool = False, group_seed: int = 0) -> DonorState:
    """Fit a donor state on rows ``donor`` of ``X``/``y`` to encode rows ``receiving``.

    Positions, not labels. Every group of the 173 -- the GAM included -- is
    fitted inside this one state, so this one refusal covers them all: the
    notebook's ``fit_donor_state`` assert, kept as the family's own runtime
    guard (ADR-0003).
    """
    return _fit_donor_state_raw(_pack_raw(X), y, donor, receiving, full=full, group_seed=group_seed)


def _donor_features_raw(raw: dict, y, seed: int, *, full, group_seed, nthread=1):
    import numpy as np
    from sklearn.model_selection import StratifiedKFold

    y = np.asarray(y)
    n = len(y)
    features = np.empty((n, len(FEATURE_COLUMNS)), dtype=np.float32)
    margin = np.empty(n, dtype=np.float32)
    visits = np.zeros(n, dtype=np.uint8)
    inner = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=seed)
    for inner_fold, (donor, receiving) in enumerate(inner.split(np.zeros(n), y)):
        state = _fit_donor_state_raw(
            raw, y, donor, receiving, full=full, group_seed=group_seed + inner_fold, nthread=nthread
        )
        f, m = state.transform_chunked(_take(raw, receiving))
        features[receiving] = f
        if m is not None:
            margin[receiving] = m
        visits[receiving] += 1
        del state
    if not np.all(visits == 1):
        raise AssertionError("Incomplete or duplicated inner donor coverage")
    return features, (margin if full else None)


def donor_features(X, y, seed: int, *, full: bool = False, group_seed: int = 0):
    """The training rows' ``(features, margin)``: each row encoded by the inner
    donor fold it is held out of, so no row's own label reaches any column."""
    return _donor_features_raw(_pack_raw(X), y, seed, full=full, group_seed=group_seed)


# --------------------------------------------------------------------------- #
# The booster (cell 8, at fixed rounds).
# --------------------------------------------------------------------------- #
@dataclass
class HeuljaxModel:
    """The full-training donor state, the feature set, and the booster fitted on donor features."""

    state: DonorState
    booster: Any
    spec: FeatureSet


def _xgb_params(params: Mapping[str, Any], spec: FeatureSet) -> dict[str, Any]:
    out = {k: v for k, v in params.items() if k not in _FAMILY_KEYS}
    out["monotone_constraints"] = spec.monotone_constraints
    return out


def _dmatrix(features, spec: FeatureSet, label=None, base_margin=None):
    import xgboost as xgb

    return xgb.DMatrix(
        features[:, [COL[c] for c in spec.columns]],
        label=label,
        base_margin=base_margin if spec.base_margin else None,
        feature_names=spec.columns,
        feature_types=spec.feature_types,
        enable_categorical=True,
    )


# The group model's seed for the whole-training-fold state, offset from the
# inner states' group_seed + inner_fold (the notebook's 8000 + fold vs 7000 + ...).
FULL_STATE_SEED_OFFSET = 1000


def fit(X, y, params: Mapping[str, Any], num_boost_round: int) -> HeuljaxModel:
    """Build the family's representation on the training rows and fit XGBoost.

    Fixed ``num_boost_round``, no validation set, no early stopping: the
    Comparison Run protocol.
    """
    import numpy as np
    import xgboost as xgb

    spec = feature_set(params)
    group_seed = int(params.get(GROUP_SEED_KEY, 0))
    nthread = int(params.get("nthread", 1))
    y = np.asarray(y)
    raw = _pack_raw(X)
    train_x, train_margin = _donor_features_raw(
        raw, y, int(params[DONOR_SEED_KEY]), full=spec.full, group_seed=group_seed, nthread=nthread
    )
    state = DonorState(
        raw,
        y,
        full=spec.full,
        source=load_source() if spec.full else None,
        seed=group_seed + FULL_STATE_SEED_OFFSET,
        nthread=nthread,
    )
    booster = xgb.train(
        _xgb_params(params, spec),
        _dmatrix(train_x, spec, label=y, base_margin=train_margin),
        num_boost_round=num_boost_round,
    )
    return HeuljaxModel(state=state, booster=booster, spec=spec)


def predict(model: HeuljaxModel, X):
    """Positive-class probabilities for rows encoded by the full-training state."""
    import numpy as np

    features, margin = model.state.transform_chunked(_pack_raw(X))
    pred = model.booster.predict(_dmatrix(features, model.spec, base_margin=margin))
    if pred.shape != (len(X),) or not np.isfinite(pred).all():
        raise AssertionError("Invalid predicted probabilities")
    return pred
