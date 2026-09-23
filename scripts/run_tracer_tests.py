"""Minimal stdlib harness to run the dependency-free tracer tests.

pytest is unavailable in the agent environment (no pip/uv/network), so this
runs the ``test_*`` functions of tests/test_tracer.py directly, supplying a
``tmp_path`` where a test asks for one. The real suite still runs under pytest
on a provisioned machine.
"""

from __future__ import annotations

import inspect
import tempfile
import traceback
from pathlib import Path

import test_tracer as mod


def main() -> int:
    failures = 0
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
