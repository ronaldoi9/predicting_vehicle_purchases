# Transforms belong to the Model Adapter, fitted inside the training fold

The Baseline Frame stores raw values and buys **Resolution**; every transform that is specific to one model family, or that reads the target, lives in the **Model Adapter** and is fitted inside the training fold — never on the full training set, and never before the transforms it depends on. This looks like an omission from the outside (there is no scaler in the pipeline, and `Age` is left as a bare integer), so it is recorded here to stop it being "fixed".

## Considered options

Scaling was the decision that surfaced this, with three options on the table:

- **(a) The Frame holds raw values; a scale-sensitive family switches its scaler on in the Model Adapter.** Chosen.
- **(b) Scale always,** accepting it is a no-op on turn 1.
- **(c) Scale always,** to pre-empt a non-tree family arriving later.

The motivation originally offered for scaling — putting features "on the same level of importance" — does not hold for a gradient-boosted tree. A GBDT chooses splits by gain and is invariant to any monotone transform of a feature: min-max on `Age` yields the same 45 distinct values, the same bins, the same splits, and literally the same model. Importance comes from gain, not magnitude, so there is nothing to equalise.

That makes scaling *conditional* rather than wrong — a no-op if the model family is a GBDT, mandatory if it is linear, an SVM, or a neural net. Option (a) keeps that conditionality where it belongs (with the family) and keeps one fewer transform between the CSV and the model on turn 1, which matters because the validation protocol requires that only the change under test moves a Paired Delta.

## Consequences

Four rules follow, and they are the same risk wearing different clothes — **a transform applied at the wrong point in the fold, or in the wrong order in the pipeline.** None of them fails loudly.

1. **Anything fitted on data is fitted inside the training fold.** Scalers and target encodings alike. A scaler fitted on the full training set leaks across the Canonical Fold Partition; a target encoding fitted on the training folds and applied to those same rows measured **-0.00187** in the leak hunt, against a published **+0.00129**. That is a measurement failure, not a refutation — and it is the single most likely way the implementation silently loses that gain. Target encodings are **nested cross-fit**: inner K=5, `random_state = 100 + outer_fold`, prior weight 20, test encoded on full train.

2. **Digit decomposition is computed from the original integer, before any scaling.** `(scaled_income) // 1000` is meaningless. Digits first, scale last, and scaling touches only the continuous column, never its digit children.

3. **`Age` stays individually addressable — all 45 values, never smoothed, never monotone-constrained.** It carries a 9-sigma non-monotone residual (chi2 1271 on 44 df, p = 1.5e-237; 13.02% positive at age 68 against 21.77% at age 27, uncorrelated with the Recipe score) that is **invisible to single-feature AUC** (0.5055) precisely because it is non-monotone. A tree gets it for free and a well-meant "clean up the numerics" refactor destroys it without breaking a single test. The model family decides `max_bin`; it does not get to violate this.

4. **The Frame carries no hand-engineered interactions.** A GBDT given nothing but the Recipe's own four inputs beats the Recipe's closed form by **+0.00269** — more than twice what all nine remaining columns are worth together (+0.00120). The non-linearity inside the driving variables is what the tree finds unaided; the axis that actually paid in the ablation was Resolution (digit decomposition +0.00175, count encoding +0.00030), not interaction. Domain plausibility alone does not earn a cross a place in the Frame.

Terms in capitals are defined in `CONTEXT.md`. The measurements come from issues #2, #4 and #5; the fold protocol from #6; the Baseline Frame from #7.
