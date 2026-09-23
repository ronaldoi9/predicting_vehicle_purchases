"""The raw-13 Baseline Frame for the tracer bullet (#13).

The narrow path: ``id`` dropped, the five nominals one-hot encoded with no
dropped level and a vocabulary fixed over train+test combined, the two ordinals
carried as single numeric columns, and the six numeric columns carried raw. No
digit decomposition, no count encoding, no interactions — later tickets widen
the Frame; the tracer measures whether the partition and read order are right
before any feature engineering can obscure the question.

Categorical columns are selected from the committed :mod:`columns` lists, never
by dtype. pandas is imported lazily so the module imports by bare name anywhere.
"""

from __future__ import annotations

from columns import (
    ID_COLUMN,
    NOMINAL_COLUMNS,
    NUMERIC_COLUMNS,
    ORDINAL_COLUMNS,
    TARGET_COLUMN,
)

# The two ordinals share this ordering; carried as a single numeric column so
# the model gets the ordering without spending one-hot width on it.
ORDINAL_LEVELS = {"Low": 0, "Medium": 1, "High": 2}


def _encode_ordinals(df):
    out = {}
    for col in ORDINAL_COLUMNS:
        mapped = df[col].map(ORDINAL_LEVELS)
        if mapped.isna().any():
            unknown = sorted(set(df[col]) - set(ORDINAL_LEVELS))
            raise AssertionError(f"{col} has levels outside {ORDINAL_LEVELS}: {unknown}")
        out[col] = mapped.astype("int8")
    return out


def build_frame(train, test):
    """Assemble the raw-13 Baseline Frame from train and test.

    Returns ``(X_train, X_test)`` with an identical column layout. The one-hot
    vocabulary is fixed over train+test combined and asserted identical across
    both, so a category present in one file and absent from the other cannot
    shift the layout between fit and predict.
    """
    import pandas as pd

    n_train = len(train)
    drop = [c for c in (ID_COLUMN, TARGET_COLUMN) if c in train.columns or c in test.columns]

    # One-hot the nominals over the combined frame so the vocabulary is shared.
    combined = pd.concat(
        [train.drop(columns=[c for c in drop if c in train.columns]),
         test.drop(columns=[c for c in drop if c in test.columns])],
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

    frame = pd.concat(
        [numerics, ordinals.reset_index(drop=True), dummies.reset_index(drop=True)],
        axis=1,
    )

    X_train = frame.iloc[:n_train].reset_index(drop=True)
    X_test = frame.iloc[n_train:].reset_index(drop=True)

    # Structural asserts on the execution path.
    if list(X_train.columns) != list(X_test.columns):
        raise AssertionError("one-hot column layout differs between train and test")
    for name, part in (("train", X_train), ("test", X_test)):
        nan_cols = part.columns[part.isna().any()].tolist()
        if nan_cols:
            raise AssertionError(f"{name} Frame has NaN in {nan_cols}; a lookup/join failed silently")

    return X_train, X_test
