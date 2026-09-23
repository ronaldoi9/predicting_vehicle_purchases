"""Minimal stdlib harness for the dependency-free tail tests (#19).

Mirrors scripts/run_resolution_tests.py and its siblings: pytest is unavailable
in the agent environment (no pip/uv/network), so this runs the ``test_*``
functions of tests/test_tail.py directly. The seed-bag, SMOTENC and
conservative-tuning candidates are declarations plus kill bookkeeping, all
dependency-free; the one SMOTENC property test reaches for imbalanced-learn /
pandas / numpy and is reported SKIP where absent, running for real under pytest
on a provisioned machine.
"""

from __future__ import annotations

import traceback

import test_tail as mod


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
