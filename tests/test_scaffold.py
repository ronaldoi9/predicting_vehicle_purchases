"""Infrastructure smoke test for #21.

Proves the acceptance criteria the environment ticket is responsible for: the
flat src/ import root resolves by bare name, the four entry points respond to
--help, the categorical schema is committed, and nothing selects columns by
dtype. This is not the general pipeline suite ADR-0003/#9 rejected — it asserts
only that the scaffold the later tickets build on actually holds together.
"""

from __future__ import annotations

import ast
import importlib
import subprocess
import sys
from pathlib import Path

import pytest

BARE_MODULES = [
    "columns",
    "data",
    "frame",
    "adapter",
    "models",
    "runner",
    "submission",
    "promote",
    "render",
]

ENTRY_POINTS = ["run-experiment", "promote", "render-ledger", "submit", "record-score"]

SRC = Path(__file__).resolve().parent.parent / "src"


@pytest.mark.parametrize("name", BARE_MODULES)
def test_module_imports_by_bare_name(name: str) -> None:
    assert importlib.import_module(name) is not None


def test_pandas_below_3() -> None:
    import pandas as pd

    major = int(pd.__version__.split(".")[0])
    assert major < 3, f"pandas must be <3, got {pd.__version__}"


def test_interpreter_is_312() -> None:
    assert sys.version_info[:2] == (3, 12), f"expected 3.12, got {sys.version_info}"


@pytest.mark.parametrize("script", ENTRY_POINTS)
def test_entry_point_responds_to_help(script: str) -> None:
    result = subprocess.run([script, "--help"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert script in result.stdout


def test_categorical_list_is_committed() -> None:
    import columns

    assert columns.NOMINAL_COLUMNS
    assert columns.CATEGORICAL_COLUMNS == columns.NOMINAL_COLUMNS + columns.ORDINAL_COLUMNS
    # No overlap and the target/id are never in the categorical selection.
    assert columns.TARGET_COLUMN not in columns.CATEGORICAL_COLUMNS
    assert columns.ID_COLUMN not in columns.CATEGORICAL_COLUMNS


def test_no_column_selection_by_dtype() -> None:
    """The whole reason columns.py exists: nothing inspects dtypes to select."""
    # Both the `.select_dtypes(...)` selector and bare `.dtypes` comparisons
    # (`df.dtypes == 'object'`) count as dtype-driven column selection.
    banned_attrs = {"select_dtypes", "dtypes"}
    offenders: list[str] = []

    for path in SRC.glob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in banned_attrs:
                offenders.append(f"{path.name}: .{node.attr}")

    assert not offenders, f"dtype-based column selection found: {offenders}"
