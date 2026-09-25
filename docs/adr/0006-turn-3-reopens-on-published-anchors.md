# Turn 3 reopens the closed turn on published anchors

> Amends **ADR-0005** on the finals rule, and reverses the "no turn 3" decision recorded when the turn-2 map (#22) closed.

#22 closed turn 2 on a saturation argument: six of seven axes reported negative or marginal, and #29 read the representation axis as closed. That argument rested on an untested gap. #29 only ever target-encoded income at its *exact* value, while two published pipelines sit above our OOF 0.94557 / LB 0.94567 with mechanisms we never measured: Maldonado's multi-key income TE (pooled OOF **0.94603**, log-verified, near-honest protocol) and heuljax's 173-feature XGBoost (OOF 0.946309, of which protocol optimism explains only ~0.00017). A saturation claim that never measured the nearest published result is not saturation. Turn 3 reopens to measure exactly those two, and nothing else, before the 2026-09-30 deadline. The primary-source reading behind every number here is `docs/research/income-te-keys-and-published-pipelines.md`.

## Considered options

- **(a) An ADR plus a thin turn-3 map issue that only points at tickets.** Chosen.
- **(b) Reopen #22 and continue inside it** — mixes two turns' decisions in one map.
- **(c) A new `/wayfinder` map** — there is no fog left to chart: three experiments, each with a published anchor.

## Decisions

1. **The Maldonado TE is measured as one stacked candidate, not a chain.** Its published parts — TE prior weight 20 → 1 (+0.00025), `//100` and `//1000` keys (+0.00012), exact and floored commute keys — each land in the rule's Confirmation band [0.0001, 0.0003), so each would need its own three-seed Confirmation Run at the Incumbent's 2,341 rounds. A chain of them does not fit before the freeze; one stack needs one Confirmation Run. The gate is unchanged — the reason is time, not the threshold. Two configurations are declared (prior 1 and prior 5), and the Axis dies only if both fail. No ablation if it wins: the stack enters whole.
2. **Dropped, deliberately:** najiama's LightGBM parameter point (+0.00014 — tuning is its own Axis, already searched in #30) and the nearest-original key (+0.00005, needs the external original dataset, touches 2.1% of rows).
3. **heuljax enters as an Arena family, not as ported features.** Porting ~160 features in three days is not realistic; running its pipeline unchanged except for the outer split (the Canonical Fold Partition), fixed rounds (1000, mid-plateau of its own AUC curves) instead of early stopping on the scored fold, and CPU `hist` yields an out-of-fold vector on our rows — a legitimate **Member**, and the first with a genuinely different representation. Its gate is OOF ≥ 0.9455 *or* OOF correlation < 0.985 with the Incumbent. This replaces a raw public-leaderboard submission of the notebook, which the public split's SE (0.0016) could not have read anyway.
4. **`hpsearch_lightgbm_best_confirm` is promoted to Incumbent under the existing rule.** Its canonical-seed Paired Delta is +0.00029, inside the Confirmation band, and its Confirmation Run held the sign on all three seeds (+0.00029 / +0.00036 / +0.00029, 5/5 folds each) — which the rule already accepts. It was never promoted only because no one ran the command. Turn-3 candidates pair against it, so a Frame change is measured on top of the best model rather than double-counting tuning. The promotion is the driving dev's, like every promotion (ADR-0003).
5. **The finals rule gains a Proven Final clause.** ADR-0005's "best CV + best single-family model" stands, but at least one final must be a **Proven Final**. If the best CV depends on turn-3 code, the second final is `hpsearch_lightgbm_best_confirm`. ADR-0005 moved the suspect from the Floor to the Blend; the suspect now is any code path that has never produced a scored submission, and a new TE that leaks would sink a Blend and its own single-family model together.

## Consequences

- **Calendar.** 26–28/09 build and measure; 29/09 Blend and Confirmation Runs, **frozen at 23:59 BRT**; 30/09 finals only. A candidate without a passing Confirmation Run by the freeze does not compete, however good it looks.
- **Unchanged:** the instrument (partition, sha256 assert, Paired Delta, 4-of-5 folds, Confirmation Run), the Blend's declared kill criterion, ADR-0005's slot rules, the Rank-1 Gap out of scope. The two finals stay the driving dev's.
- **The anchors may shrink to zero here.** No published gain was measured on a Frame that already carries income digits; #29's anchors did exactly that. A dead Axis is an acceptable outcome of this turn, not a reason to extend it past the freeze.

Terms in capitals are defined in `CONTEXT.md`. The decisions come from the turn-3 grilling session of 2026-09-25.
