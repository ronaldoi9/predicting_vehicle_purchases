"""Declared Blends — the harness's registry, mirroring ``experiments._REGISTRY``.

The map (#22) opens the axis with "there are currently 22 Run Records, so the
harness has real vectors to exercise on immediately." ``BLEND_ALL_MEMBERS``
exercises it on every declared Experiment that has actually been run on the
canonical fold seed — the whole ledger, not a hand-picked subset, because a
hand-picked subset is exactly the kind of choice that should be a measured
comparison later, not a default baked into the first Blend. It is expected to
struggle against its kill criterion: every current Member is LightGBM, and
ADR-0005 names the Blend's real input as the *diversity* the XGBoost/CatBoost
axes have not yet supplied — this Blend is infrastructure the harness proves
works, not a claim that combining near-identical vectors pays.
"""

from __future__ import annotations

from typing import Mapping

import experiments
from blend import BlendConfig

# Every declared Experiment EXCEPT two that would make ``resolve_members`` fail
# on today's ledger: ``tracer_raw13`` is the tracer bullet proving the pipeline
# wiring (#13), never a Frame candidate, and never run at fold_seed=0 in the
# same sense the queue's candidates were; ``count_all13_composites`` is #29's
# stacked follow-up, explicitly gated behind both of its parents landing
# individually — they did not, so it was never run and stays unrun.
# ``heuljax_tracer`` (#37) is a tracer bullet for a family whose full port is
# still to come, and ``heuljax_full`` (#40) is that port; turn 3's Blend (T3)
# declares its own Members instead. The same holds for #39's two stacked-TE
# configurations: turn-3 candidates, not retroactive Members of a Blend turn 2
# already ran.
_EXCLUDED_FROM_ALL_MEMBERS = frozenset(
    {
        "tracer_raw13",
        "count_all13_composites",
        "heuljax_tracer",
        "heuljax_full",
        *(exp.name for exp in experiments.TE_KEYS_AXIS),
    }
)

BLEND_ALL_MEMBERS = BlendConfig(
    name="blend_all_members",
    hypothesis=(
        "Rank-averaging every measured candidate's latest fold_seed=0 Run "
        "Record, with weights chosen by greedy hill-climb under per-outer-fold "
        "nested weight selection, beats the best of those Members. Expected to "
        "struggle: every current Member is LightGBM (no diversity yet — the "
        "map names the XGBoost/CatBoost axes as the Blend's real input), so "
        "this run exercises the harness end to end (both the honest and the "
        "naive number) rather than claiming a win. Kill criterion, declared "
        "before running: paired delta vs best Member < +0.0003, or positive in "
        "fewer than 4/5 folds -> dead; the harness still stands either way."
    ),
    members=tuple(sorted(set(experiments._REGISTRY) - _EXCLUDED_FROM_ALL_MEMBERS)),
    fold_seed=0,
    kill_delta=0.0003,
    kill_min_folds_positive=4,
)

# Turn 3's Blend (#41). The Incumbent is te_keys_prior5 (#39's stacked TE,
# promoted), so it is both the Incumbent and #39's survivor; heuljax_full (#40)
# passed its Member gate on seeds 0/1/2. Nothing else: the older Arena chains
# already had their Blend in turn 2, and ADR-0006 scopes turn 3 to its two
# published anchors. The Incumbent is listed first so a per-round hill-climb
# tie picks it.
BLEND_TURN3 = BlendConfig(
    name="blend_turn3",
    hypothesis=(
        "Rank-averaging the Incumbent te_keys_prior5 with heuljax_full, the "
        "first Member with a genuinely different representation (173 "
        "features, XGBoost), with weights chosen by greedy hill-climb under "
        "per-outer-fold nested weight selection, beats the better of the two. "
        "The OOF correlation of the pair is 0.998, so the room is small. Kill "
        "criterion, declared before running and unchanged from the turn-2 "
        "Blend: paired delta vs best Member < +0.0003, or positive in fewer "
        "than 4/5 folds -> dead."
    ),
    members=("te_keys_prior5", "heuljax_full"),
    fold_seed=0,
    kill_delta=BLEND_ALL_MEMBERS.kill_delta,
    kill_min_folds_positive=BLEND_ALL_MEMBERS.kill_min_folds_positive,
)

REGISTRY: Mapping[str, BlendConfig] = {
    BLEND_ALL_MEMBERS.name: BLEND_ALL_MEMBERS,
    BLEND_TURN3.name: BLEND_TURN3,
}


def resolve(name: str) -> BlendConfig:
    """Look a declared Blend up by name for the single entry point."""
    try:
        return REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(REGISTRY)) or "(none)"
        raise KeyError(f"unknown blend {name!r}; known: {known}") from None
