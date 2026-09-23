"""Minimal stdlib harness to run the dependency-free tests.

pytest is unavailable in the agent environment (no pip/uv/network), so this
runs the ``test_*`` functions of the dependency-free test modules directly,
supplying a ``tmp_path`` where a test asks for one. The real suite still runs
under pytest on a provisioned machine.
"""

from __future__ import annotations

import importlib
import inspect
import tempfile
import traceback
from pathlib import Path

# The dependency-free modules: structural tracer seams (#13) and the Baseline
# Frame's pure helpers and declaration (#14). test_scaffold needs pandas/pytest
# and only runs under the provisioned env.
TEST_MODULES = ("test_tracer", "test_baseline")


def main() -> int:
    failures = 0
    for modname in TEST_MODULES:
        mod = importlib.import_module(modname)
        print(f"== {modname} ==")
        for name in sorted(vars(mod)):
            if not name.startswith("test_"):
                continue
            fn = getattr(mod, name)
            kwargs = {}
            if "tmp_path" in inspect.signature(fn).parameters:
                kwargs["tmp_path"] = Path(tempfile.mkdtemp())
            try:
                fn(**kwargs)
                print(f"PASS {name}")
            except Exception:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}")
                traceback.print_exc()
    print(f"\n{'-'*40}\n{failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
