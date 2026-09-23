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

# The frame specs this builder knows: the tracer's narrow path and the complete
# Baseline Frame. One source of truth for both the validity check and its error.
FRAME_SPECS = ("raw13", "baseline")


def _require_known_spec(spec: str) -> None:
    """Reject a frame spec this builder does not know, naming the ones it does."""
    if spec not in FRAME_SPECS:
        known = ", ".join(repr(s) for s in FRAME_SPECS)
        raise ValueError(f"unknown frame spec {spec!r}; known: {known}")


def extra_columns(spec: str) -> list[str]:
    """The Frame columns a spec adds on top of the raw-13 layout.

    Pure metadata (no pandas), so the two measured steps' output shape is
    checkable without the ML stack. ``raw13`` adds nothing; ``baseline`` adds
    the three income digits and the two count encodings.
    """
    _require_known_spec(spec)
    if spec == "raw13":
        return []
    return [name for name, _ in INCOME_DIGIT_TRANSFORMS] + [
        f"{col}_count" for col in COUNT_ENCODED_COLUMNS
    ]


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
    if spec == "baseline":
        # Digits from the original integer, then the frozen combined counts.
        parts.append(_income_digits(combined).reset_index(drop=True))
        parts.append(_count_encodings(combined).reset_index(drop=True))
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
