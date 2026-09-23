"""Minimal stdlib harness for the band (ii) queue declaration tests (#12).

pytest is unavailable in the agent environment (no pip/uv/network), so this runs
the dependency-free ``test_*`` functions of test_queue directly. The real suite
still runs under pytest on a provisioned machine.
"""

from __future__ import annotations

import importlib
import inspect
import tempfile
import traceback
from pathlib import Path

TEST_MODULES = ("test_queue",)


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
