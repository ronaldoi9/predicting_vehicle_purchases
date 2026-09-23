"""The band (ii) experiment queue as a declared, ordered artifact (#12, PRD #12).

Issue 12's title is "turn 1 pipeline **and the band (ii) experiment queue**", and
its spec (user story 62) requires the queue to be "run in its declared order, so
that the candidate with a published gain is measured before the ones that merely
sound promising". Each of the closed child tickets (#15/#18/#19) added its own
candidates to the registry, but none declared the *order* — it lived only as
prose in the spec table and scattered across three modules. This ties them into
one ordered source of truth so turn 2 (and any queue runner) reads the order
rather than reconstructing it.

The declared order, from the spec's queue table:

    1. income_te                            (published gain — measured first)
    2. max_bin_255, max_bin_1023, max_bin_2047   (the Resolution sweep)
    3. age_te
    4. seed_bag
    5. conservative_tuning

``smotenc`` is a *separately scheduled* candidate — it "keeps its own slot" in
the spec — so it is declared but not on the main queue's ordered track. The
multi-family blend (candidate 6) is gated behind the queue exhausting early and
is deliberately NOT built, so it must not resolve.

Everything here is dependency-free (the queue is Experiment declarations), so the
stdlib harness (scripts/run_queue_tests.py) runs it where the ML stack is absent.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout

import experiments
import runner


# --------------------------------------------------------------------------- #
# The queue is declared, and it is in the spec's order.
# --------------------------------------------------------------------------- #
def test_band_ii_queue_is_declared_in_the_spec_order() -> None:
    assert experiments.queue_names() == (
        "income_te",
        "max_bin_255",
        "max_bin_1023",
        "max_bin_2047",
        "age_te",
        "seed_bag",
        "conservative_tuning",
    )


def test_income_te_is_measured_first() -> None:
    # The candidate with a published gain is measured before the ones that merely
    # sound promising (spec story 62/63).
    assert experiments.queue_names()[0] == "income_te"


def test_resolution_sweep_sits_in_the_queue_in_ascending_max_bin_order() -> None:
    names = experiments.queue_names()
    sweep = [n for n in names if n.startswith("max_bin_")]
    assert sweep == [f"max_bin_{mb}" for mb in experiments.MAX_BIN_SWEEP]


# --------------------------------------------------------------------------- #
# Every queue member is a real, resolvable, single-field-change candidate.
# --------------------------------------------------------------------------- #
def test_every_queue_member_resolves_and_is_a_candidate_against_the_incumbent() -> None:
    for exp in experiments.BAND_II_QUEUE:
        # It resolves by name through the single entry point.
        assert experiments.resolve(exp.name) is exp
        # It is a Paired Delta against the standing Incumbent...
        assert exp.incumbent == "baseline"
        # ...and it carries a kill criterion declared before it runs (one of the
        # four shapes the runner dispatches on).
        has_kill = (
            exp.kill_delta is not None
            or exp.kill_min_folds_positive is not None
            or exp.kill_value_per_run is not None
            or exp.kill_time_budget_s is not None
        )
        assert has_kill, f"{exp.name} must declare a kill criterion before it runs"


def test_queue_contains_no_duplicate_names() -> None:
    names = experiments.queue_names()
    assert len(names) == len(set(names))


# --------------------------------------------------------------------------- #
# The separately-scheduled and the gated candidates.
# --------------------------------------------------------------------------- #
def test_smotenc_is_separately_scheduled_not_on_the_main_queue() -> None:
    # It keeps its own slot (spec), so it is declared and resolvable...
    assert experiments.resolve("smotenc").oversample == experiments.OVERSAMPLE_SMOTENC
    # ...but it is not on the ordered main-track queue.
    assert "smotenc" not in experiments.queue_names()


def test_run_experiment_queue_flag_lists_the_declared_order() -> None:
    # ``run-experiment --queue`` surfaces the queue without touching the ML stack
    # (the heavy imports live inside run(), which --queue never calls).
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = runner.main(["--queue"])
    assert rc == 0
    out = buf.getvalue()
    for i, name in enumerate(experiments.queue_names(), start=1):
        assert f"{i}. {name}" in out
    assert "smotenc" in out  # the separately-scheduled candidate is surfaced


def test_multi_family_blend_is_gated_and_not_built() -> None:
    # Candidate 6 enters only if candidates 1-5 exhaust before 29/09; it is
    # deliberately not built here, so it neither resolves nor sits on the queue.
    assert "blend" not in experiments.queue_names()
    try:
        experiments.resolve("blend")
    except KeyError:
        pass
    else:
        raise AssertionError("the multi-family blend must not be built (it is gated)")
