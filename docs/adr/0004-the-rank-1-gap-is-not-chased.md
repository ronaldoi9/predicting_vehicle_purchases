# The Rank-1 Gap is not chased

> **Amended by [ADR-0005](0005-the-plateau-is-chased-by-breadth.md)**: the finals rule (§2) and the pseudo-labelling kill (§4) are superseded. The Rank-1 Gap itself stays out of scope.

No day of the remaining budget is allocated to the 0.0027 separating rank 1 from the **Plateau**, and the two hypotheses that would have attacked it — pseudo-labelling on the 286,571 unlabelled test rows, and any successor to the structural leak hunt — are killed unrun. The budget goes instead to the *other* 0.0027: the distance between turn 1's expected Incumbent (OOF 0.94372) and the Plateau (CV ~0.9464).

Walking away from the only part of the leaderboard that distinguishes a result looks like giving up, which is why it is recorded here rather than left implicit in a closed issue.

## Considered options

- **(a) Rule the Rank-1 Gap out of scope; spend the budget closing the gap below us.** Chosen.
- **(b) Rank it last among live hypotheses,** attacking it only if the queue exhausts early.
- **(c) Allocate a fixed slice — say two of the eight days — to a dedicated gap hunt.**

(b) reads as the prudent middle and is the trap. A hypothesis ranked last is a hypothesis still on the route, and the one moment it becomes tempting is day 6 with slots unspent and the queue dry — precisely the state in which its kill criterion will not be honoured. The point of deciding this before the queue runs is that a hypothesis with no mechanism should not be waiting in reserve for the moment judgement is worst. (c) is (b) with a budget line attached, and buys the same nothing more expensively.

## Why the gap is most likely not there to be won

The published evidence closes it first. Every structural leak hypothesis raised on this competition resolved negative with measurement (issue #4), and the Bayes-optimal re-ranking margin over the **Baseline Frame** is **0.00007** — the representation is exhausted, so there is no better ordering left for a clever method to find. Rank 3 is the corroboration that matters: 0.94672 in **three submissions**. The Plateau is where honest engineering terminates.

The statistics then explain most of what is left. With ROC AUC at 0.9466 on a 17.5%-positive split, the Hanley–McNeil SE is **0.00158** on the 57,314-row public split and 0.00079 on the 229,257-row private one, so a single model's public and private scores differ with SD **0.00177**. Taking the maximum over 2,704 teams, with an independent public-noise fraction of ρ = 0.90–0.95, gives an expected winner's curse of **+0.0012 to +0.0017**.

Two things this calculation corrects, both of which the map had recorded loosely:

- **Rank 1's own 39 submissions explain almost nothing.** Near-identical models have highly correlated public noise, so selecting the best of them is worth roughly 0.0002. "They probed the public leaderboard" is not, by itself, an arithmetic that reaches 0.0027.
- **The winner's curse explains a majority of the gap, not all of it.** Roughly 0.001 remains unaccounted for. It is either a tail draw on a maximum of 2,704 or something real, and **no evidence available to this project can tell the two apart.**

The decision does not depend on resolving that residual. It depends on there being no mechanism to act on, which is a stronger and much cheaper thing to establish.

## Consequences

1. **The Floor exists early and by date, not by quality.** Turn 1 alone — Health Gate ≥ 0.9434 — is submitted by 25/09. The Floor is insurance, so it must be the simplest thing already measured and reproducible; the income target encoding is the *target*, not the floor. 25/09 arriving with nothing submitted is an alarm about execution, not about the model.

2. **The final two submissions are best-CV plus the Floor, and the rule is fixed now rather than on the last day.** Best public LB is excluded on its own merits: the public split cannot resolve 0.0002, so selecting on it selects noise — the very mechanism named above as the leading explanation for rank 1, which it would be incoherent to walk into deliberately. Two best-by-CV is excluded because those are typically variants of one model and fail together. The pair costs ~0.0005 of expected rank on the Floor slot and covers a real failure mode: a bug the experiment queue introduced and CV did not catch.

3. **Submission slots stop being scarce, and attention becomes the budget.** ~89 slots remain against ~2h/day of driving-dev time. A routine submission exists to check the **CV→LB Offset**'s stability — a measurement, not a risk decision — so an agent session may spend one unattended after a passing **Confirmation Run**, at most one per day, never a final. The final two are always the driving dev's, mirroring the manual Incumbent promotion gate of ADR-0003.

4. **Every remaining candidate carries a kill criterion declared before it runs.** Income TE dies below +0.0005 on one seed; the `max_bin` sweep freezes at 511 unless a value wins ≥4/5 folds; `Age`-as-lookup dies below +0.0001; seed-averaging dies if compute exceeds measured gain; tuning gets 2h of machine time to find +0.0003; a multi-family blend only enters if the queue exhausts before 29/09. These are the discipline that makes (a) safe — without them, band (ii) decays into the same open-ended hunt this ADR refuses.

5. **If a new gap hypothesis appears, it competes rather than pre-empts.** It enters the queue with a mechanism and a kill criterion, and loses to candidates holding published gains unless it is very good. Out of scope here means off the route, not forbidden to think about.

Terms in capitals are defined in `CONTEXT.md`. The leak-hunt measurements come from issue #4, the competition mechanics from #3, the Health Gate from #8, the promotion gate from #9, and this decision from #11.
