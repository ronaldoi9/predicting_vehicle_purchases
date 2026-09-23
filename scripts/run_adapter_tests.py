"""Minimal stdlib harness for the dependency-free Adapter tests (#15).

Mirrors scripts/run_tracer_tests.py and scripts/run_promotion_tests.py: pytest
is unavailable in the agent environment (no pip/uv/network), so this runs the
``test_*`` functions of tests/test_adapter.py directly. The one property test
that proves a target encoding never sees its own row needs pandas / numpy /
scikit-learn; where those are absent it is reported SKIP rather than FAIL, and
it runs for real under pytest on a provisioned machine.
"""

from __future__ import annotations

import importlib
import traceback

import test_adapter as mod


def main() -> int:
    failures = 0
    skipped = 0
    for name in sorted(vars(mod)):
        if not name.startswith("test_"):
            continue
        fn = getattr(mod, name)
        try:
            fn()
            print(f"PASS {name}")
        except ModuleNotFoundError as exc:
            skipped += 1
            print(f"SKIP {name} (needs {exc.name}: run under pytest on a provisioned machine)")
        except Exception:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"\n{'-'*40}\n{failures} failure(s), {skipped} skipped")
    return 1 if failures else 0


if __name__ == "__main__":
    importlib.import_module("test_adapter")
    raise SystemExit(main())
