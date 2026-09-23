"""The promotion verdict rule: the one judgement the instrument encodes.

Promotion of the Incumbent is manual by explicit command. The runner *prints*
this verdict and stops; it never advances the Incumbent itself. The rule is the
only place a number becomes a judgement rather than an observed fact, and it is
applied under deadline pressure — exactly the condition in which an automatic
rule turns a noise-sized delta into the reference every later measurement is
taken against. So the rule is written down here, dependency-free and tested,
and evaluated identically wherever it is needed (the runner's printout and the
``promote`` command's authorisation).

The rule, staggered by magnitude and anchored on the measured Paired-Delta
noise floor (PRD #12):

* a mean paired delta ``>= 0.0003`` is **accepted** on the canonical seed;
* a mean paired delta in ``[0.0001, 0.0003)`` is **accepted only** if a
  Confirmation Run holds the delta's sign across fold seeds 0, 1 and 2;
* a mean paired delta ``< 0.0001`` is **rejected** whatever its sign;
* independently of magnitude, the delta must be positive in **at least four of
  five folds** — one lucky fold cannot carry a verdict.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

# Anchored on the measured Paired-Delta noise floor, not the leaderboard's.
NOISE_FLOOR = 0.0001
ACCEPT_THRESHOLD = 0.0003
MIN_FOLDS_POSITIVE = 4
N_FOLDS = 5
# A Confirmation Run re-runs candidate and Incumbent across these fold seeds.
CONFIRMATION_SEEDS = (0, 1, 2)

# Verdict kinds.
ACCEPT = "accept"
REJECT = "reject"
NEEDS_CONFIRMATION = "needs-confirmation"

# The two rules that can authorise a promotion (named so the promotion line
# records which of them did). The third band never authorises: it rejects.
RULE_MAGNITUDE = "mean paired delta >= 0.0003 on the canonical seed"
RULE_CONFIRMATION = (
    "mean paired delta in [0.0001, 0.0003) confirmed across fold seeds 0/1/2"
)


@dataclass(frozen=True)
class Verdict:
    """The outcome of the rule: what to do, and which rule said so."""

    kind: str  # ACCEPT | REJECT | NEEDS_CONFIRMATION
    reason: str
    rule: str | None = None  # the authorising rule, set only when kind == ACCEPT

    @property
    def accepted(self) -> bool:
        return self.kind == ACCEPT


def sign_holds(confirmation_deltas: Sequence[float]) -> bool:
    """True iff a Confirmation Run keeps the (positive) sign on every seed.

    The Confirmation Run re-runs candidate and Incumbent across fold seeds
    0/1/2; passing means the mean paired delta stays positive on all three.
    """
    deltas = list(confirmation_deltas)
    if len(deltas) != len(CONFIRMATION_SEEDS):
        raise ValueError(
            f"a Confirmation Run has {len(CONFIRMATION_SEEDS)} fold seeds "
            f"{CONFIRMATION_SEEDS}; got {len(deltas)} deltas"
        )
    return all(d > 0 for d in deltas)


def classify(
    mean_delta: float,
    folds_positive: int,
    *,
    confirmation: Sequence[float] | None = None,
    n_folds: int = N_FOLDS,
) -> Verdict:
    """Apply the staggered rule to one candidate's Comparison Run.

    ``mean_delta`` is the mean paired delta on the canonical seed;
    ``folds_positive`` is how many of the ``n_folds`` per-fold deltas were
    positive. ``confirmation``, when given, is the mean paired delta on each of
    fold seeds 0/1/2 from a Confirmation Run.
    """
    # The four-of-five gate is independent of magnitude and applies first:
    # a huge mean delta carried by a single lucky fold is still rejected.
    if folds_positive < MIN_FOLDS_POSITIVE:
        return Verdict(
            REJECT,
            f"positive in only {folds_positive}/{n_folds} folds; "
            f"the rule requires at least {MIN_FOLDS_POSITIVE}/{n_folds}",
        )

    if mean_delta < NOISE_FLOOR:
        return Verdict(
            REJECT,
            f"mean paired delta {mean_delta:+.5f} is below the {NOISE_FLOOR} "
            "Paired-Delta noise floor; rejected whatever its sign",
        )

    if mean_delta >= ACCEPT_THRESHOLD:
        return Verdict(
            ACCEPT,
            f"mean paired delta {mean_delta:+.5f} >= {ACCEPT_THRESHOLD} "
            "on the canonical seed",
            rule=RULE_MAGNITUDE,
        )

    # NOISE_FLOOR <= mean_delta < ACCEPT_THRESHOLD: the Confirmation band.
    if confirmation is None:
        return Verdict(
            NEEDS_CONFIRMATION,
            f"mean paired delta {mean_delta:+.5f} is in "
            f"[{NOISE_FLOOR}, {ACCEPT_THRESHOLD}); a Confirmation Run across "
            f"fold seeds {'/'.join(map(str, CONFIRMATION_SEEDS))} is required",
        )

    if sign_holds(confirmation):
        return Verdict(
            ACCEPT,
            f"mean paired delta {mean_delta:+.5f} in "
            f"[{NOISE_FLOOR}, {ACCEPT_THRESHOLD}); Confirmation Run held the "
            f"sign across fold seeds {'/'.join(map(str, CONFIRMATION_SEEDS))}",
            rule=RULE_CONFIRMATION,
        )

    return Verdict(
        REJECT,
        f"mean paired delta {mean_delta:+.5f} in [{NOISE_FLOOR}, "
        f"{ACCEPT_THRESHOLD}); Confirmation Run did not hold the sign across "
        f"fold seeds {'/'.join(map(str, CONFIRMATION_SEEDS))} "
        f"(deltas: {', '.join(f'{d:+.5f}' for d in confirmation)})",
    )
