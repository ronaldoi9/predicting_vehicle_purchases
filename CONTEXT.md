# Predicting Vehicle Purchases

A Kaggle Playground Series (S6E9) effort to predict `Will_Buy_EV`, scored by ROC AUC. The work is run as CRISP-DS turns: a turn is a cycle of decisions, one measured experiment sequence, and a leaderboard reading.

## Language

### Measurement

**Canonical Fold Partition**:
The single `StratifiedKFold` split of the training set that every experiment is scored on, fixed by a constant seed and verified by checksum. There is exactly one.
_Avoid_: the CV split, the folds, our validation set

**Paired Delta**:
The difference in OOF AUC between a candidate and the incumbent, measured on the Canonical Fold Partition so both see identical rows. The only unit in which an experiment's result is reported.
_Avoid_: improvement, gain, score difference

**Incumbent**:
The configuration a candidate must beat — the current best by Paired Delta. Distinct from whatever was last submitted.
_Avoid_: baseline, best model, current model

**Confirmation Run**:
A re-run of candidate and Incumbent across the three fold seeds, required before a candidate may consume a submission slot. Passing means the Paired Delta keeps its sign on all three.
_Avoid_: repeat CV, seed averaging, validation run

**Comparison Run**:
An experiment fit under the frozen protocol — fixed round count, no early stopping — so that only the change under test moves the Paired Delta. Contrasted with the **Submission Fit**, the final full-data fit that produces predictions and may use early stopping.
_Avoid_: training run, experiment (ambiguous between the two)

**Noise Floor**:
The magnitude below which a difference is unreadable on the instrument in question. Distinct floors exist for the public leaderboard and for a Paired Delta; naming which one is meant is required.
_Avoid_: error bar, margin, significance

**CV→LB Offset**:
The gap between a configuration's OOF AUC and its public leaderboard score. Its *stability over time* is what the public leaderboard is read for; its absolute value ranks nothing.
_Avoid_: leaderboard correlation, CV/LB gap

**Experiment Ledger**:
The append-only record of every Comparison Run and every submission, versioned with the repo. The source of truth for what has been tried.
_Avoid_: results table, experiment log, tracking sheet

**Nested Cross-Fit**:
Fitting a target encoding on an inner split of each training fold, never on the rows it is applied to. Required for every target encoding here; its absence inverts the measured gain.
_Avoid_: out-of-fold encoding, cross-validated encoding

**Health Gate**:
The OOF AUC a turn's build must reproduce to prove the Canonical Fold Partition and the Baseline Frame were assembled correctly. Falling short is a bug report, not a verdict on the model — it is deliberately not an ambition about score.
_Avoid_: target score, acceptance threshold, sanity check

### Competition landscape

**Plateau**:
The dense band of leaderboard scores that commodity engineering reaches (~0.9459-0.9466). Reaching it is expected; it distinguishes nothing.
_Avoid_: the pack, baseline score

**Rank-1 Gap**:
The distance between the plateau and the top score, and the only part of the leaderboard that represents an open question.
_Avoid_: the lead, the top score

**Recipe**:
The publicly reverse-engineered generating rule behind the synthetic data — a linear `buy_score` with a threshold. A fact about the data's origin, not a usable model; it scores well under the Plateau.
_Avoid_: the formula, the generator, ground truth

**Residual Positive**:
A positive-labelled row the Recipe would not have called positive — a row carried over the threshold by its noise draw. Real signal, but a negligible share of pairwise AUC weight.
_Avoid_: outlier, mislabelled row, hard positive

### Representation

**Baseline Frame**:
The frozen feature representation the turn-1 Incumbent is fit on. A named, versioned artifact — a candidate changes exactly one thing about it and is reported as a Paired Delta against it.
_Avoid_: the features, the feature set, the input matrix

**Resolution**:
The degree to which a representation lets the model address individual values of a high-cardinality column rather than a merged range of them. The mechanism behind both digit decomposition and exact-value target encoding, and the only axis on which this dataset's representation has been shown to pay.
_Avoid_: granularity, digit signal, binning

**Model Adapter**:
The layer between the Baseline Frame and a model family, holding any transform a particular family needs and no other family does — scaling above all. Keeps family-specific preprocessing out of the Frame, so every family is scored on the same rows.
_Avoid_: preprocessor, transformer, pipeline step
