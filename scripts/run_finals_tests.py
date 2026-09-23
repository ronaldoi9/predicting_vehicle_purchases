"""Minimal stdlib harness for the band (iii) freeze/select/close tests (#20).

Mirrors scripts/run_queue_tests.py and its siblings: pytest is unavailable in the
agent environment (no pip/uv/network), so this runs the ``test_*`` functions of
tests/test_finals.py directly. Band (iii) is decisions taken against the ledger —
all dependency-free; a test that reached for the ML stack would be reported SKIP
and run for real under pytest on a provisioned machine.
"""

from __future__ import annotations

import traceback

import test_finals as mod


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
    raise SystemExit(main())
