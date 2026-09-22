# The turn-1 Incumbent is a reproduction, not a fresh start

Turn 1 trains LightGBM alone, on the exact untuned configuration the leak hunt's ablation already ran — `binary`, `lr=0.05`, 700 fixed rounds, `num_leaves=127`, `max_bin=511`, `feature_fraction=0.8`, no early stopping — so that the first number the pipeline produces is a **reproduction of a known 0.94372**, not a new measurement. This looks like an omission from the outside (nothing is tuned, no second family is tried, bagging is off), so it is recorded here to stop it being "fixed".

## Considered options

- **(a) Reproduce the ablation's family and parameters verbatim, and let every later change be a Paired Delta against it.** Chosen.
- **(b) Start from a tuned parameter set published for this competition,** reaching the plateau sooner.
- **(c) Train two or three families in turn 1** and pick the best.

(b) and (c) both buy a higher first score and both cost the same thing: **measurement continuity**. Every result this project holds — the +0.00175 from digit decomposition, the +0.00030 from count encoding, the −0.00187 from a mis-fitted target encoding, the −0.0029 from the Recipe as `init_score` — was produced by the configuration in (a) on the Canonical Fold Partition. Change the family or the parameters and the Incumbent resets: those deltas stop being comparable to anything measured afterwards, and the ablation stops being usable as a **Health Gate**. A higher first score is worth less than a first score that can be *checked*, because the competition's open question is a 0.0027 gap and the instrument's own noise floor is 0.0003.

The corollary is that turn 1 is not the tuning turn. `max_bin` is the exception worth naming: it is not an ordinary knob but the **Resolution** control, the only axis this dataset has been shown to pay on, so the sweep `{255, 1023, 2047}` is a measured experiment in its own right rather than tuning.

## Consequences

1. **The instrument includes the model's execution configuration, not just the fold split.** The committed sha256 of the fold-id vector pins the rows; it does not pin LightGBM. Thread count changes histogram reduction order and moves the last digits, so the run is fixed at `num_threads=10` — the performance cores only, never the mixed 15 — with `deterministic=true`, `force_row_wise=true`, and `seed`, `bagging_seed`, `feature_fraction_seed` and `data_random_seed` all set explicitly. Without this, a Paired Delta at the 0.0003 threshold the protocol calls "real" is indistinguishable from execution noise.

2. **`bagging_fraction=0.8` is inert and stays inert.** LightGBM ignores it while `bagging_freq=0`, which is the default, so the ablation's bagging almost certainly never ran. Turn 1 keeps `bagging_freq=0` deliberately. If the reproduction misses 0.94372, `bagging_freq=1` is the first divergence hypothesis to test — and turning bagging on is thereafter a measured candidate, not a default anybody restores quietly.

3. **The `Age` invariant binds `max_bin` from below, and is asserted rather than assumed.** `Age` has 45 distinct values, so any `max_bin >= 64` satisfies ADR-0001's rule; the binding constraint is asserted in code (`nunique(Age) == 45 and max_bin >= 64`). `min_data_in_bin` at its default of 3 does not merge anything here — roughly 14,859 rows per age and 45 rows per distinct income value.

4. **Test predictions are the mean of the five fold models, not a full-data refit.** The models already exist once the Comparison Run has finished, ROC AUC is a rank metric so averaging probabilities is legitimate, and — the reason that matters — the **CV→LB Offset** then compares the same set of models on both sides of the gap. A full-data refit changes the round count's meaning at a different training size and is a candidate for a later turn.

5. **Class imbalance gets no default treatment, and the exception is a measurement, not a policy.** At 17.5% positive there are ~117,000 positives; ROC AUC is rank-based and invariant to monotone transforms of the score, so reweighting changes the fitted trees rather than correcting anything. `is_unbalance=False`, no `scale_pos_weight`. A single `SMOTENC` candidate is scheduled against the Incumbent because it was asked for explicitly — `sampling_strategy=0.5`, fitted **inside the training fold only**, with income digits regenerated from the synthetic income so the Frame stays self-consistent. Plain `SMOTE` is excluded on the grounds of ADR-0001 rule 3: interpolating `Age` to 43.7 and producing digits that no longer derive from their own income destroys Resolution, the one axis that pays. Oversampling before the split is the failure mode that would make this test appear to succeed.

Terms in capitals are defined in `CONTEXT.md`. The ablation and its parameters come from issue #4, the fold protocol from #6, the Baseline Frame from #7, and this decision from #8.
