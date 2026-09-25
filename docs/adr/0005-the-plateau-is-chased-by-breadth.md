# The remaining gap below the Plateau is chased by breadth, not by one family

> **Amended by [ADR-0006](0006-turn-3-reopens-on-published-anchors.md)**: the finals rule gains a Proven Final clause, and promotion may proceed by recorded override. Everything else stands.

Turn 1 ended at OOF 0.94528 / LB 0.94560 — roughly **0.0011 short of the Plateau** and of the 0.9466–0.94672 band where ranks 3–20 sit. Turn 2 spends the remaining budget closing that distance by **widening the Arena**: more model families, a measured Blend, the published representation gains we never implemented, and a real hyperparameter search. The destination is a **Confirmation Run at OOF ≥ 0.9464** with the CV→LB Offset still stable, which the measured Offset of +0.00032 projects to LB ~0.9467.

Turn 1's single-family, single-field discipline was the right instrument for building a trustworthy measurement and the wrong one for finding score: six of the seven axes with published gains were never touched, and one of them — a plain logistic regression another team reports at CV 0.94640 — already sits above our Incumbent.

## Status

Amends **ADR-0004** on two consequences, and leaves its core intact.

| ADR-0004 | now |
| --- | --- |
| §2 finals = best CV **+ the Floor** | finals = best CV **+ the best single-family model by CV**. The Floor was insurance against a bug CV cannot see; the suspect is now the Blend, and a single-family model whose code path has been exercised on every run covers that class for ~0.0002 instead of the ~0.0012 the turn-1 Floor now costs. |
| §4 pseudo-labelling **killed unrun** | revived as a ticket with a kill criterion declared before it runs, last in axis order. Under an exploration mandate "no mechanism published" ranks a hypothesis; it no longer excludes it. |

**Unchanged: the Rank-1 Gap stays out of scope.** The Winner's Curse arithmetic and the absence of a mechanism are untouched by this decision — and nothing here is aimed above the Plateau. Newly ruled out on the same grounds: **reading the public split** (the megayak identities). It buys public rank only, and every slot spent on it is a slot not spent measuring an Offset, which is the only thing connecting CV to private rank.

## What is deliberately *not* relaxed

The instrument. The Canonical Fold Partition, the sha256 assert, the Paired Delta as the only reported unit, the 4-of-5 fold gate, the staggered magnitude thresholds and the Confirmation Run all stand exactly as ADR-0003 and issue #6 left them. The `income_te` encoding inverting from −0.00187 to +0.00114 on fold discipline alone is the measurement this project exists to be able to trust; breadth is bought with machine time, never by loosening a gate.

Two additions make breadth expressible without touching any of that:

- **The Arena.** A new family is not a single-field change and cannot be one. Each family carries its own Incumbent chain on the same partition, so adding one never invalidates another's banked deltas. Cross-family comparison happens through OOF vectors, which the ledger already keeps for exactly this reason.
- **An Axis dies only by a criterion that names its configuration count.** Turn 1 came close to recording "the Resolution axis is saturated" from three `max_bin` values. One configuration failing is a configuration failing.

## Consequences

1. **Execution happens inside the wayfinder map.** "Does XGBoost pay?" is not decidable without running it, and a full `/to-spec` → `/to-tickets` cycle per axis does not fit in six days. `task` tickets therefore build the harness and run the measurements directly. The runtime asserts, `adapter` as the only module that sees `y`, and `runner` as the only ledger writer are what make that safe — and **changes to the Adapter or the Canonical Fold Partition are exempt**: they become decisions, because that is precisely where turn 1's most expensive bug lived (a Submission Fit built without the target encoding).

2. **The Blend harness is built before the families it will combine.** Built after, someone has to reprocess vectors and reopen verdicts already recorded; built first, every run from then on is a Member the moment it lands.

3. **Blend weights are part of the configuration.** Choosing them on the rows the Blend is scored on is a leak of exactly the shape ADR-0001 exists to prevent, and it will make the score go *up*. The weight-selection protocol is a declared decision, not an implementation detail.

4. **Unattended sessions get more rope on compute, slightly more on slots.** Multi-hour sweeps run AFK without consultation; an agent may spend up to 3 slots a day after a passing Confirmation Run. **The two finals remain the driving dev's**, unchanged — the only irreversible action in the project.

Terms in capitals are defined in `CONTEXT.md`. The turn-1 numbers come from `ledger/runs.jsonl`, the published gains and the ceiling from issue #3, and this decision from the turn-2 wayfinder map.
