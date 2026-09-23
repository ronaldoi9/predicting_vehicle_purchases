"""The explicit, committed column schema of the raw Kaggle files.

Every column the pipeline touches is named here by hand. Nothing in the
codebase is allowed to select columns by dtype: under ``pandas>=3`` a string
column no longer satisfies ``dtype == object`` (measured while resolving #8),
so a dtype-driven selection silently drops every categorical column instead of
raising. The committed lists below are correct under either pandas major
version, which is the whole point of writing them down.

These lists are a schema declaration, not pipeline logic. The Baseline Frame,
the loader and the Model Adapter are built in later tickets (#12); this module
only fixes the vocabulary they will share.
"""

from __future__ import annotations

# The pure row index. Dropped from the Baseline Frame so it cannot be mistaken
# for signal.
ID_COLUMN = "id"

# The binary prediction target, ROC AUC scored.
TARGET_COLUMN = "Will_Buy_EV"

# The five nominal (unordered) categoricals. One-hot encoded with no dropped
# level, vocabulary fixed over train+test combined. This is *the* explicit
# categorical list the pipeline selects on — never a dtype inspection.
NOMINAL_COLUMNS: tuple[str, ...] = (
    "Gender",
    "City_Type",
    "Current_Car_Type",
    "Home_Charging_Possible",
    "Subsidy_Available",
)

# The two ordinal categoricals, carried as single numeric columns so their
# ordering is available to the model without spending one-hot width on it.
ORDINAL_COLUMNS: tuple[str, ...] = (
    "Environmental_Concern_Level",
    "Range_Anxiety_Level",
)

# The continuous / integer columns carried into the Frame. ``Age`` stays a raw
# integer with all 45 values individually addressable; income is carried raw
# alongside its digit decomposition.
NUMERIC_COLUMNS: tuple[str, ...] = (
    "Age",
    "Annual_Income_USD",
    "Daily_Commute_km",
    "Number_of_Cars_Owned",
    "Charging_Stations_Near_Home",
    "Charging_Stations_Near_Work",
)

# Every categorical column, ordered nominals-then-ordinals. The single list any
# categorical-aware step reads instead of asking pandas for the dtypes.
CATEGORICAL_COLUMNS: tuple[str, ...] = NOMINAL_COLUMNS + ORDINAL_COLUMNS
