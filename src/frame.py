"""The Baseline Frame — the frozen feature representation the turn-1 Incumbent
is fit on (#14), and the raw-13 subset the tracer bullet (#13) measured.

Two frame specs share this one builder:

* ``raw13`` — the tracer's narrow path: ``id`` dropped, the five nominals
  one-hot encoded (no dropped level, vocabulary fixed over train+test), the two
  ordinals as single numeric columns, the six numeric columns raw.
* ``baseline`` — the complete Baseline Frame: raw13 plus the two measured
  Resolution steps. **Digit decomposition** of income (``//1000``, ``%1000``,
  ``%100``) carried alongside raw income and, per ADR-0001, computed from the
  original integer — ``(scaled_income) // 1000`` is meaningless. **Count
  encoding** of income and commute over train+test combined, computed once and
  frozen. No hand-engineered interactions: a GBDT on the Recipe's own inputs
  beats its closed form by more than twice what all other columns are worth.

A third group of specs (#29) is purely *additive* on top of ``baseline`` — each
adds exactly the columns its name says and nothing else, so a candidate stays a
single-field change (the frame spec) against the standing Incumbent. They exist
to measure the frequency-encoding breadth #27 found in the published stack (we
count 2 columns; the field counts ~79) and one composite-key target-encoding
hypothesis #27 measured negative and this project had not run itself:

* ``count_composites`` — count-encodes the five composite keys the published
  stack counts (income×subsidy, income×concern, income-per-car, charging
  total, commute-per-concern), on top of ``baseline``.
* ``count_all13`` — count-encodes all 13 raw columns, not just income and
  commute.
* ``count_digits`` — count-encodes the three income digit columns.
* ``count_all13_composites`` — ``count_all13`` plus ``count_composites``
  together; only meant to run if both land individually.
* ``digits7`` — units/tens digit decomposition of the six numeric columns
  income digit decomposition does not already cover (Age, commute, cars
  owned, both charging-station counts, and concern), to see whether the
  Resolution mechanism pays anywhere it has not been tried. Expected close to
  a no-op: unlike income, these columns are already low-cardinality and fully
  addressable raw, so a tree gets their individual values for free.
* ``composite_te_keys`` — adds two composite categorical keys
  (income×subsidy, income×city) as plain columns, *uncounted* and
  *untarget-encoded* by this module; a candidate target-encodes one of them
  via the Model Adapter's existing ``target_encode`` field, so no Adapter
  change is needed to test a composite-key target encoding.

Categorical columns are selected from the committed :mod:`columns` lists, never
by dtype. Two guards live on the execution path and read as bug reports, not
score verdicts: ``Age`` must keep all 45 distinct values individually
addressable (a numerics cleanup that smooths or bins it destroys a 9-sigma
non-monotone residual invisible to single-feature AUC), and the assembled Frame
must carry zero ``NaN`` (a silently failed lookup or join). pandas is imported
lazily so the module imports by bare name anywhere.
"""

from __future__ import annotations

import collections

from columns import (
    ID_COLUMN,
    NOMINAL_COLUMNS,
    NUMERIC_COLUMNS,
    ORDINAL_COLUMNS,
    TARGET_COLUMN,
)

# The two ordinals do NOT share an encoding, and assuming they did was a bug the
# build assert caught: Environmental_Concern_Level ships as float64 holding only
# the integers 1.0-5.0 (#2), while Range_Anxiety_Level is a Low/Medium/High
# string. Each carries its own map, and concern keeps its own numbering so a
# level's value means what the column says it means. Both land as a single
# numeric column, so the model gets the ordering without one-hot width.
ORDINAL_ENCODINGS: dict[str, dict] = {
    "Environmental_Concern_Level": {float(v): v for v in range(1, 6)},
    "Range_Anxiety_Level": {"Low": 0, "Medium": 1, "High": 2},
}

# Income digit decomposition. The lambdas are plain integer ops, so they mean
# the same on a Python int (testable without the ML stack) and on a pandas
# Series (vectorised). Per ADR-0001 they run on the original integer income,
# before any scaling the Adapter might apply — digits first, scale last.
INCOME_COLUMN = "Annual_Income_USD"
INCOME_DIGIT_TRANSFORMS = (
    (f"{INCOME_COLUMN}_div1000", lambda v: v // 1000),
    (f"{INCOME_COLUMN}_mod1000", lambda v: v % 1000),
    (f"{INCOME_COLUMN}_mod100", lambda v: v % 100),
)

# Count encoding over train+test combined (the 955,236-row transductive artifact
# the leak hunt measured), for income and commute. No target is read, so this is
# a Frame transform, not an Adapter one.
COUNT_ENCODED_COLUMNS = ("Annual_Income_USD", "Daily_Commute_km")

# Age must keep every one of its 45 values individually addressable. A hard
# equality here fails the build if a refactor smooths, bins or clips the column.
AGE_COLUMN = "Age"
AGE_DISTINCT_VALUES = 45

# All 13 raw columns, for the ``count_all13`` / ``count_all13_composites``
# specs (#29 A2/A4) — nominals then ordinals then numerics, matching how the
# columns module orders CATEGORICAL_COLUMNS.
ALL_RAW_COLUMNS: tuple[str, ...] = NOMINAL_COLUMNS + ORDINAL_COLUMNS + NUMERIC_COLUMNS

# The five composite keys the published stack (#27) counts. Each is a function
# of the *combined* raw frame (before one-hot), computed once, never reading
# the target — a Frame transform, not an Adapter one.
_SUBSIDY_COLUMN = "Subsidy_Available"
_CONCERN_COLUMN = "Environmental_Concern_Level"
_CITY_COLUMN = "City_Type"
_CARS_COLUMN = "Number_of_Cars_Owned"
_COMMUTE_COLUMN = "Daily_Commute_km"
_CHARGING_HOME_COLUMN = "Charging_Stations_Near_Home"
_CHARGING_WORK_COLUMN = "Charging_Stations_Near_Work"


def _composite_income_x_subsidy(c):
    return c[INCOME_COLUMN].astype(str) + "_" + c[_SUBSIDY_COLUMN].astype(str)


def _composite_income_x_concern(c):
    return c[INCOME_COLUMN].astype(str) + "_" + c[_CONCERN_COLUMN].astype(str)


def _composite_income_x_city(c):
    return c[INCOME_COLUMN].astype(str) + "_" + c[_CITY_COLUMN].astype(str)


def _composite_income_per_car(c):
    return c[INCOME_COLUMN] // (c[_CARS_COLUMN] + 1)


def _composite_charging_total(c):
    return c[_CHARGING_HOME_COLUMN] + c[_CHARGING_WORK_COLUMN]


def _composite_commute_per_concern(c):
    return (c[_COMMUTE_COLUMN] // (c[_CONCERN_COLUMN] + 1)).astype("int32")


# name -> builder, for the five keys ``count_composites`` counts (#29 A1).
COMPOSITE_COUNT_KEYS: dict[str, "Callable"] = {
    "income_x_subsidy": _composite_income_x_subsidy,
    "income_x_concern": _composite_income_x_concern,
    "income_per_car": _composite_income_per_car,
    "charging_total": _composite_charging_total,
    "commute_per_concern": _composite_commute_per_concern,
}

# The two composite keys ``composite_te_keys`` adds as plain (uncounted)
# columns for a candidate to target-encode via the Adapter (#29 C1/C2).
COMPOSITE_TE_KEYS: dict[str, "Callable"] = {
    "income_x_subsidy": _composite_income_x_subsidy,
    "income_x_city": _composite_income_x_city,
}

# The six numeric columns income's own digit decomposition does not cover,
# for the ``digits7`` spec (#29 D1).
DIGIT7_COLUMNS: tuple[str, ...] = (
    AGE_COLUMN,
    _COMMUTE_COLUMN,
    _CARS_COLUMN,
    _CHARGING_HOME_COLUMN,
    _CHARGING_WORK_COLUMN,
    _CONCERN_COLUMN,
)

# The frame specs this builder knows. ``raw13`` and ``baseline`` are frozen;
# the rest are #29's additive frequency/digit/composite-key candidates, each
# built strictly on top of ``baseline``.
FRAME_SPECS = (
    "raw13",
    "baseline",
    "count_composites",
    "count_all13",
    "count_digits",
    "count_all13_composites",
    "digits7",
    "composite_te_keys",
)


def _require_known_spec(spec: str) -> None:
    """Reject a frame spec this builder does not know, naming the ones it does."""
    if spec not in FRAME_SPECS:
        known = ", ".join(repr(s) for s in FRAME_SPECS)
        raise ValueError(f"unknown frame spec {spec!r}; known: {known}")


def _baseline_extra_columns() -> list[str]:
    return [name for name, _ in INCOME_DIGIT_TRANSFORMS] + [
        f"{col}_count" for col in COUNT_ENCODED_COLUMNS
    ]


def extra_columns(spec: str) -> list[str]:
    """The Frame columns a spec adds on top of the raw-13 layout.

    Pure metadata (no pandas), so a spec's output shape is checkable without
    the ML stack. ``raw13`` adds nothing; every other spec adds the baseline
    extras plus whatever that spec's own docstring (module-level) says it adds.
    """
    _require_known_spec(spec)
    if spec == "raw13":
        return []
    base = _baseline_extra_columns()
    if spec == "baseline":
        return base
    if spec == "count_composites":
        return base + [f"{k}_count" for k in COMPOSITE_COUNT_KEYS]
    if spec == "count_all13":
        return base + [
            f"{c}_count" for c in ALL_RAW_COLUMNS if c not in COUNT_ENCODED_COLUMNS
        ]
    if spec == "count_digits":
        return base + [f"{name}_count" for name, _ in INCOME_DIGIT_TRANSFORMS]
    if spec == "count_all13_composites":
        return (
            base
            + [f"{c}_count" for c in ALL_RAW_COLUMNS if c not in COUNT_ENCODED_COLUMNS]
            + [f"{k}_count" for k in COMPOSITE_COUNT_KEYS]
        )
    if spec == "digits7":
        cols = []
        for col in DIGIT7_COLUMNS:
            cols += [f"{col}_mod10", f"{col}_div10"]
        return base + cols
    if spec == "composite_te_keys":
        return base + list(COMPOSITE_TE_KEYS)
    raise AssertionError(f"extra_columns has no case for known spec {spec!r}")


def _count_lookup(values):
    """Map each value to its frequency over the (combined) values it is given."""
    return dict(collections.Counter(values))


def _drop_id_and_target(df):
    """Drop ``id`` and the target from a frame if present (test has no target)."""
    return df.drop(columns=[c for c in (ID_COLUMN, TARGET_COLUMN) if c in df.columns])


def _encode_ordinals(df):
    out = {}
    for col in ORDINAL_COLUMNS:
        levels = ORDINAL_ENCODINGS[col]
        mapped = df[col].map(levels)
        if mapped.isna().any():
            unknown = sorted(set(df[col]) - set(levels))
            raise AssertionError(f"{col} has levels outside {sorted(levels)}: {unknown}")
        out[col] = mapped.astype("int8")
    return out


def _income_digits(combined):
    """The three income digit columns, computed from the original integer."""
    import pandas as pd

    income = combined[INCOME_COLUMN]
    return pd.DataFrame(
        {name: op(income).astype("int32") for name, op in INCOME_DIGIT_TRANSFORMS},
        index=combined.index,
    )


def _count_encodings(combined):
    """Count encoding of income and commute over the combined frame, once."""
    import pandas as pd

    out = {}
    for col in COUNT_ENCODED_COLUMNS:
        lookup = _count_lookup(combined[col].tolist())
        out[f"{col}_count"] = combined[col].map(lookup).astype("int32")
    return pd.DataFrame(out, index=combined.index)


def _count_encode_columns(combined, columns):
    """Count encoding of the given raw ``combined`` columns, by name."""
    import pandas as pd

    out = {}
    for col in columns:
        lookup = _count_lookup(combined[col].tolist())
        out[f"{col}_count"] = combined[col].map(lookup).astype("int32")
    return pd.DataFrame(out, index=combined.index)


def _count_encode_keys(combined, key_builders):
    """Count encoding of derived keys (``name -> f(combined) -> Series``)."""
    import pandas as pd

    out = {}
    for name, build in key_builders.items():
        key = build(combined)
        lookup = _count_lookup(key.tolist())
        out[f"{name}_count"] = key.map(lookup).astype("int32")
    return pd.DataFrame(out, index=combined.index)


def _composite_te_key_columns(combined):
    """The composite categorical keys #29's C1/C2 target-encode, uncounted.

    Factorised to int32 codes, not left as strings: the raw key stays in the
    Frame (the Adapter adds a ``_te`` column alongside it, never replacing it),
    and an object-dtype column reaching LightGBM is the exact shape of bug #2
    (the target handed over as strings). Factorising is computed once over the
    combined frame, so train and test share one code space, and it changes
    nothing about what the Adapter's target encoding sees — it keys on
    whatever hashable value the column holds, string or int alike.
    """
    import pandas as pd

    out = {}
    for name, build in COMPOSITE_TE_KEYS.items():
        codes, _ = pd.factorize(build(combined))
        out[name] = codes.astype("int32")
    return pd.DataFrame(out, index=combined.index)


def _digit7(combined):
    """Units/tens digit decomposition of the six non-income numeric columns."""
    import pandas as pd

    out = {}
    for col in DIGIT7_COLUMNS:
        values = combined[col]
        out[f"{col}_mod10"] = (values % 10).astype("int32")
        out[f"{col}_div10"] = (values // 10).astype("int32")
    return pd.DataFrame(out, index=combined.index)


def build_frame(train, test, spec: str = "baseline"):
    """Assemble the Baseline Frame from train and test.

    ``spec`` selects ``raw13`` (the tracer's narrow path) or ``baseline`` (the
    complete Frame with income digit decomposition and count encoding). Returns
    ``(X_train, X_test)`` with an identical column layout. The one-hot vocabulary
    is fixed over train+test combined and asserted identical across both, so a
    category present in one file and absent from the other cannot shift the
    layout between fit and predict. Count encoding is likewise computed over the
    combined frame so it is frozen and consistent across the two files.
    """
    import pandas as pd

    _require_known_spec(spec)

    n_train = len(train)

    # One-hot the nominals over the combined frame so the vocabulary is shared.
    combined = pd.concat(
        [_drop_id_and_target(train), _drop_id_and_target(test)],
        axis=0,
        ignore_index=True,
    )
    dummies = pd.get_dummies(
        combined[list(NOMINAL_COLUMNS)],
        columns=list(NOMINAL_COLUMNS),
        drop_first=False,
    ).astype("int8")  # 0/1, not bool, so LightGBM sees plain indicators

    ordinals = pd.DataFrame(_encode_ordinals(combined), index=combined.index)
    numerics = combined[list(NUMERIC_COLUMNS)].reset_index(drop=True)

    parts = [numerics, ordinals.reset_index(drop=True)]
    if spec != "raw13":
        # Every non-raw13 spec carries the frozen baseline extras first, then
        # (for #29's specs) its own additive columns on top — strictly
        # additive, so `baseline` itself is built by exactly the same two
        # calls it always was.
        parts.append(_income_digits(combined).reset_index(drop=True))
        parts.append(_count_encodings(combined).reset_index(drop=True))
        if spec == "count_composites":
            parts.append(_count_encode_keys(combined, COMPOSITE_COUNT_KEYS).reset_index(drop=True))
        elif spec == "count_all13":
            cols = [c for c in ALL_RAW_COLUMNS if c not in COUNT_ENCODED_COLUMNS]
            parts.append(_count_encode_columns(combined, cols).reset_index(drop=True))
        elif spec == "count_digits":
            digit_cols = [name for name, _ in INCOME_DIGIT_TRANSFORMS]
            digits_frame = _income_digits(combined)
            parts.append(_count_encode_columns(digits_frame, digit_cols).reset_index(drop=True))
        elif spec == "count_all13_composites":
            cols = [c for c in ALL_RAW_COLUMNS if c not in COUNT_ENCODED_COLUMNS]
            parts.append(_count_encode_columns(combined, cols).reset_index(drop=True))
            parts.append(_count_encode_keys(combined, COMPOSITE_COUNT_KEYS).reset_index(drop=True))
        elif spec == "digits7":
            parts.append(_digit7(combined).reset_index(drop=True))
        elif spec == "composite_te_keys":
            parts.append(_composite_te_key_columns(combined).reset_index(drop=True))
    parts.append(dummies.reset_index(drop=True))

    frame = pd.concat(parts, axis=1)

    X_train = frame.iloc[:n_train].reset_index(drop=True)
    X_test = frame.iloc[n_train:].reset_index(drop=True)

    # Structural asserts on the execution path.
    if list(X_train.columns) != list(X_test.columns):
        raise AssertionError("one-hot column layout differs between train and test")
    for name, part in (("train", X_train), ("test", X_test)):
        nan_cols = part.columns[part.isna().any()].tolist()
        if nan_cols:
            raise AssertionError(f"{name} Frame has NaN in {nan_cols}; a lookup/join failed silently")

    # Age keeps every one of its 45 values individually addressable. Measured on
    # the training rows, where the 9-sigma non-monotone residual lives. A miss is
    # a bug report — a numerics cleanup smoothed, binned or clipped the column —
    # not a score verdict.
    age_distinct = int(X_train[AGE_COLUMN].nunique())
    if age_distinct != AGE_DISTINCT_VALUES:
        raise AssertionError(
            f"Age has {age_distinct} distinct values, expected {AGE_DISTINCT_VALUES}: "
            "a transform smoothed, binned or clipped Age and destroyed the 9-sigma "
            "non-monotone residual it carries (invisible to single-feature AUC). "
            "This is a bug in how the Frame was assembled, not a model verdict."
        )

    return X_train, X_test
