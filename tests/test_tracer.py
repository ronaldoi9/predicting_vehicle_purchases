"""Tracer-bullet seams for #13 that need no ML stack to verify.

The Health Gate itself (reproducing OOF 0.94167 on the raw-13 frame) can only
run where pandas/scikit-learn/lightgbm and the gitignored CSVs are present, so
it is exercised by ``run-experiment`` on a provisioned machine, not here. What
*is* checkable without the stack are the structural rules the tracer bullet
establishes: the run-id shape, config hashing, the Run Record schema, and the
single-writer discipline on the runs ledger. Those are what this file pins.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import experiments
import runner

SRC = Path(__file__).resolve().parent.parent / "src"


def _sample_config() -> dict:
    return {
        "name": "tracer_raw13",
        "frame": "raw13",
        "fold_seed": 0,
        "model": "lightgbm",
        "params": {"objective": "binary", "num_leaves": 127, "seed": 0},
    }


def test_run_id_is_timestamp_then_experiment_name() -> None:
    when = datetime(2026, 9, 23, 1, 47, 0, 123456, tzinfo=timezone.utc)
    rid = runner.run_id("tracer_raw13", when=when)
    assert rid.endswith("-tracer_raw13")
    stamp = rid[: -len("-tracer_raw13")]
    # The timestamp half parses back to the instant it was built from.
    assert stamp.startswith("20260923T014700")


def test_run_id_is_not_a_config_hash() -> None:
    """A Confirmation Run repeats one config; its runs must not collide."""
    config = _sample_config()
    a = runner.run_id("tracer_raw13", when=datetime(2026, 9, 23, 1, 47, 0, 1, tzinfo=timezone.utc))
    b = runner.run_id("tracer_raw13", when=datetime(2026, 9, 23, 1, 47, 0, 2, tzinfo=timezone.utc))
    assert a != b, "two runs of one config must get distinct ids"
    assert runner.config_hash(config) not in (a, b)


def test_config_hash_is_stable_and_key_order_independent() -> None:
    a = {"a": 1, "b": {"x": 2, "y": 3}}
    b = {"b": {"y": 3, "x": 2}, "a": 1}
    assert runner.config_hash(a) == runner.config_hash(b)
    assert runner.config_hash(a) != runner.config_hash({"a": 2, "b": {"x": 2, "y": 3}})
    assert len(runner.config_hash(a)) == 64  # sha256 hex


def test_run_record_carries_every_required_field() -> None:
    record = runner.build_run_record(
        experiment="tracer_raw13",
        config=_sample_config(),
        fold_aucs=[0.941, 0.942, 0.940, 0.943, 0.9415],
        oof_auc=0.94167,
        fold_sha256="deadbeef",
        oof_path="runs/some-run/oof.npy",
        wall_time=12.5,
        seeds={"seed": 0, "bagging_seed": 0, "feature_fraction_seed": 0, "data_random_seed": 0},
        git_info={"git_sha": "abc123", "dirty": False},
        when=datetime(2026, 9, 23, 1, 47, 0, tzinfo=timezone.utc),
    )
    required = {
        "run_id",
        "config_hash",
        "timestamp",
        "git_sha",
        "dirty",
        "config",
        "fold_aucs",
        "oof_auc",
        "fold_sha256",
        "seeds",
        "wall_time",
        "oof_path",
    }
    assert required <= set(record), f"missing: {required - set(record)}"
    assert len(record["fold_aucs"]) == 5
    assert record["run_id"].endswith("-tracer_raw13")
    assert record["config_hash"] == runner.config_hash(_sample_config())
    # The whole record must round-trip as one JSON line (the ledger is jsonl).
    line = json.dumps(record)
    assert "\n" not in line
    assert json.loads(line)["oof_auc"] == 0.94167


def test_runner_is_the_only_module_that_knows_the_runs_ledger() -> None:
    """`runner` is the sole writer of ledger/runs.jsonl (a hard rule of #13).

    Reading is allowed (render), but no other seam module may even name the
    runs ledger, so an unrecorded run or a second writer is impossible.
    """
    mentions = set()
    for path in SRC.glob("*.py"):
        if "runs.jsonl" in path.read_text():
            mentions.add(path.name)
    assert mentions <= {"runner.py", "render.py"}, f"unexpected runs-ledger mentions: {mentions}"


def test_tracer_experiment_declares_the_verbatim_turn1_config() -> None:
    exp = experiments.resolve("tracer_raw13")
    assert exp.frame == "raw13"
    assert exp.model == "lightgbm"
    assert exp.fold_seed == 0
    assert exp.num_boost_round == 700  # fixed rounds, early stopping disabled
    assert exp.scale is False  # scaling off for a tree family
    assert exp.target_oof == 0.94167  # the raw-13 ablation row this reproduces
    # The instrument's determinism knobs, taken verbatim from the ablation.
    p = exp.params
    assert p["num_threads"] == 10
    assert p["deterministic"] is True
    assert p["force_row_wise"] is True
    assert p["bagging_freq"] == 0  # bagging_fraction stays inert
    assert set(exp.seeds()) == {"seed", "bagging_seed", "feature_fraction_seed", "data_random_seed"}
    # max_bin comfortably clears the 45-distinct-Age representation floor.
    assert p["max_bin"] >= 64


def test_unknown_experiment_is_rejected() -> None:
    try:
        experiments.resolve("does_not_exist")
    except KeyError as e:
        assert "does_not_exist" in str(e)
    else:  # pragma: no cover
        raise AssertionError("resolve must reject an unknown experiment name")


def test_append_run_record_writes_one_json_line(tmp_path) -> None:
    ledger = tmp_path / "runs.jsonl"
    rec1 = {"run_id": "r1", "oof_auc": 0.9}
    rec2 = {"run_id": "r2", "oof_auc": 0.94167}
    runner.append_run_record(rec1, path=ledger)
    runner.append_run_record(rec2, path=ledger)
    lines = ledger.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["run_id"] == "r1"
    assert json.loads(lines[1])["oof_auc"] == 0.94167
