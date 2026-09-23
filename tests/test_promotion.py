"""The judgement layer for #16: the verdict rule, promotion, the ledger table.

Dependency-free by design — the promotion rule is arithmetic on a mean paired
delta and a per-fold-positive count, the promotion write is a JSON line, and
the ledger table is string formatting. None of it needs the ML stack or the
gitignored CSVs, so all of it is pinned here (and runs under the stdlib harness
as well as pytest).
"""

from __future__ import annotations

import json
from pathlib import Path

import promote
import render
import runner
import verdict as verdict_mod

SRC = Path(__file__).resolve().parent.parent / "src"


def _candidate_record(**overrides) -> dict:
    record = {
        "run_id": "20260923T120000.000000Z-candidate",
        "experiment": "candidate",
        "oof_auc": 0.94372,
        "fold_aucs": [0.9436, 0.9438, 0.9435, 0.9439, 0.9437],
        "incumbent_run_id": "20260923T010000.000000Z-tracer_raw13",
        "paired_delta": 0.0005,
        "folds_positive": 5,
        "git_sha": "abcdef1234567890",
        "dirty": False,
    }
    record.update(overrides)
    return record


# --------------------------------------------------------------------------- #
# The staggered verdict rule.
# --------------------------------------------------------------------------- #
def test_delta_at_or_above_0_0003_is_accepted_on_canonical_seed() -> None:
    v = verdict_mod.classify(0.0003, folds_positive=5)
    assert v.kind == verdict_mod.ACCEPT
    assert v.rule == verdict_mod.RULE_MAGNITUDE
    # And comfortably above.
    assert verdict_mod.classify(0.0009, folds_positive=4).kind == verdict_mod.ACCEPT


def test_delta_below_0_0001_is_rejected_whatever_its_sign() -> None:
    assert verdict_mod.classify(0.00009, folds_positive=5).kind == verdict_mod.REJECT
    assert verdict_mod.classify(0.0, folds_positive=5).kind == verdict_mod.REJECT
    assert verdict_mod.classify(-0.01, folds_positive=5).kind == verdict_mod.REJECT


def test_mid_band_delta_needs_a_confirmation_run() -> None:
    v = verdict_mod.classify(0.0002, folds_positive=5)
    assert v.kind == verdict_mod.NEEDS_CONFIRMATION
    assert v.rule is None


def test_mid_band_accepted_only_if_confirmation_holds_the_sign() -> None:
    held = verdict_mod.classify(0.0002, folds_positive=5, confirmation=[0.0002, 0.0001, 0.0003])
    assert held.kind == verdict_mod.ACCEPT
    assert held.rule == verdict_mod.RULE_CONFIRMATION

    # A sign flip on any one of the three seeds fails the confirmation.
    flipped = verdict_mod.classify(0.0002, folds_positive=5, confirmation=[0.0002, -0.0001, 0.0003])
    assert flipped.kind == verdict_mod.REJECT


def test_four_of_five_folds_is_required_independently_of_magnitude() -> None:
    # A huge mean delta carried by only three positive folds is still rejected.
    v = verdict_mod.classify(0.01, folds_positive=3)
    assert v.kind == verdict_mod.REJECT
    assert "3/5" in v.reason
    # Exactly four is enough.
    assert verdict_mod.classify(0.01, folds_positive=4).kind == verdict_mod.ACCEPT


def test_confirmation_run_spans_exactly_fold_seeds_0_1_2() -> None:
    assert verdict_mod.CONFIRMATION_SEEDS == (0, 1, 2)
    assert verdict_mod.sign_holds([0.1, 0.2, 0.3]) is True
    assert verdict_mod.sign_holds([0.1, -0.2, 0.3]) is False
    # A confirmation with the wrong number of seeds is a programming error.
    try:
        verdict_mod.sign_holds([0.1, 0.2])
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("a Confirmation Run must carry three seed deltas")


# --------------------------------------------------------------------------- #
# The runner prints the verdict and never advances the Incumbent.
# --------------------------------------------------------------------------- #
def test_runner_prints_verdict_for_a_candidate_and_stays_read_only(capsys=None) -> None:
    kind = runner.print_promotion_verdict(_candidate_record())
    assert kind == verdict_mod.ACCEPT
    # A baseline with no Paired Delta has nothing to promote.
    assert runner.print_promotion_verdict(_candidate_record(paired_delta=None)) is None


def test_runner_is_still_the_only_writer_of_the_runs_ledger() -> None:
    """promote must look runs up without ever naming the runs ledger file."""
    writers = set()
    for path in SRC.glob("*.py"):
        if "runs.jsonl" in path.read_text():
            writers.add(path.name)
    assert writers <= {"runner.py", "render.py"}, f"unexpected mentions: {writers}"
    # promote is the sole writer of the promotions ledger.
    promo_writers = {
        path.name for path in SRC.glob("*.py") if "promotions.jsonl" in path.read_text()
    }
    assert promo_writers == {"promote.py"}, promo_writers


# --------------------------------------------------------------------------- #
# The explicit promotion write.
# --------------------------------------------------------------------------- #
def test_promotion_line_records_run_delta_and_authorising_rule(tmp_path) -> None:
    record = _candidate_record()
    v = promote.authorise(record)
    line = promote.build_promotion_line(record, v)
    assert line["promoted_run_id"] == record["run_id"]
    assert line["incumbent_run_id"] == record["incumbent_run_id"]
    assert line["paired_delta"] == record["paired_delta"]
    assert line["rule"] == verdict_mod.RULE_MAGNITUDE

    ledger = tmp_path / "promotions.jsonl"
    promote.append_promotion(line, path=ledger)
    written = json.loads(ledger.read_text().splitlines()[0])
    assert written["promoted_run_id"] == record["run_id"]
    assert written["rule"] == verdict_mod.RULE_MAGNITUDE


def test_authorise_refuses_a_baseline_run() -> None:
    try:
        promote.authorise(_candidate_record(paired_delta=None))
    except ValueError as exc:
        assert "baseline" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("a baseline run must not be promotable")


def test_authorise_mid_band_needs_confirmation_then_accepts() -> None:
    record = _candidate_record(paired_delta=0.0002)
    assert promote.authorise(record).kind == verdict_mod.NEEDS_CONFIRMATION
    accepted = promote.authorise(record, confirmation=[0.0002, 0.0001, 0.0003])
    assert accepted.kind == verdict_mod.ACCEPT
    assert accepted.rule == verdict_mod.RULE_CONFIRMATION


# --------------------------------------------------------------------------- #
# The ledger table.
# --------------------------------------------------------------------------- #
def test_render_table_is_markdown_sorted_by_paired_delta() -> None:
    records = [
        _candidate_record(run_id="r-small", paired_delta=0.0001),
        _candidate_record(run_id="r-baseline", paired_delta=None),
        _candidate_record(run_id="r-big", paired_delta=0.0009),
    ]
    table = render.render_table(records)
    lines = table.splitlines()
    # Header + separator + one row per record.
    assert lines[0].startswith("| Paired Delta |")
    assert set(lines[1].replace("|", "").split()) == {"---"}
    assert len(lines) == 2 + len(records)

    order = [ln for ln in lines[2:]]
    assert "r-big" in order[0]
    assert "r-small" in order[1]
    # A run with no Paired Delta sinks to the bottom and renders as a dash.
    assert "r-baseline" in order[2]
    assert "| — |" in order[2]


def test_render_table_reads_the_committed_ledger(tmp_path) -> None:
    ledger = tmp_path / "runs.jsonl"
    runner.append_run_record(_candidate_record(run_id="r1", paired_delta=0.0004), path=ledger)
    runner.append_run_record(_candidate_record(run_id="r2", paired_delta=0.0008), path=ledger)
    records = render.load_records(ledger)
    table = render.render_table(records)
    assert "r2" in table.splitlines()[2]  # the larger delta is first
