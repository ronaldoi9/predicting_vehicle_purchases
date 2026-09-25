"""The stacked income/commute target encoding (#39, ADR-0006 §1).

An Experiment can declare the TE prior weight and a set of derived TE keys
(a source column plus ``div100`` / ``div1000`` / ``floor``). Each derived key
gets the same Nested Cross-Fit and smoothed mean as the exact-value encoding,
computed from the original, unscaled values. Both fields default so that every
existing declaration and the frozen baseline path stay byte-identical — that is
pinned below against a hash taken from the Adapter before these fields existed.
"""

from __future__ import annotations

import hashlib

import pytest

import adapter
import experiments
import runner

INCOME = "Annual_Income_USD"
COMMUTE = "Daily_Commute_km"

ALL_DERIVED = ((INCOME, "div100"), (INCOME, "div1000"), (COMMUTE, "floor"))


def _synthetic():
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(7)
    n = 400
    X = pd.DataFrame(
        {
            INCOME: rng.integers(30000, 40000, n),
            COMMUTE: np.round(rng.uniform(0, 30, n), 1),
            "Age": rng.integers(25, 70, n),
        }
    )
    y = (rng.uniform(size=n) < 0.3).astype(int)
    return X, y


def _frame_hash(*frames) -> str:
    import pandas as pd

    payload = b"".join(pd.util.hash_pandas_object(f).values.tobytes() for f in frames)
    return hashlib.sha256(payload).hexdigest()


# Taken from the Adapter at 89232c5, before te_prior_weight / te_derived_keys
# existed: exact income + commute TE on the synthetic frame, outer fold 2.
PINNED_DEFAULT_HASH = "6f0ab17d60c025109a04e8c2e464e1c6142eda8d134803cf205b3f0f913e2e90"


# --------------------------------------------------------------------------- #
# Byte-identical defaults.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "extra",
    [{}, {"derived_keys": (), "prior_weight": adapter.PRIOR_WEIGHT}],
    ids=["omitted", "explicit-defaults"],
)
def test_default_fields_give_todays_output_byte_for_byte(extra) -> None:
    import numpy as np

    X, y = _synthetic()
    tr, va = np.arange(0, 320), np.arange(320, len(y))
    ad = adapter.Adapter(
        target_encode=(INCOME, COMMUTE),
        outer_fold=2,
        validation_index=set(va.tolist()),
        **extra,
    )
    out_tr = ad.fit_transform(X.iloc[tr], y[tr])
    out_va = ad.transform(X.iloc[va])
    assert list(out_tr.columns) == ["Annual_Income_USD", "Daily_Commute_km", "Age",
                                    "Annual_Income_USD_te", "Daily_Commute_km_te"]
    assert _frame_hash(out_tr, out_va) == PINNED_DEFAULT_HASH


def test_default_declarations_keep_their_config_and_adapter() -> None:
    """Every Experiment that does not set the new fields serialises exactly as
    before (no new keys, so its config hash does not move) and gets the
    Adapter it got before: prior weight 20 and no derived keys."""
    new = {"te_prior_weight", "te_derived_keys"}
    for name in experiments._REGISTRY:
        exp = experiments.resolve(name)
        if exp in experiments.TE_KEYS_AXIS:
            continue
        assert not new & set(exp.as_config()), name
        ad = runner.fold_adapter(exp, 0, [])
        assert ad.prior_weight == adapter.PRIOR_WEIGHT, name
        assert ad.derived_keys == (), name


# --------------------------------------------------------------------------- #
# Derived keys, encoded with the requested prior.
# --------------------------------------------------------------------------- #
def test_derivations_operate_on_the_original_values() -> None:
    import numpy as np

    income = np.array([123456, 742, 30099])
    commute = np.array([12.9, 0.4, 7.0])
    assert adapter.derive_key(income, "div100").tolist() == [1234, 7, 300]
    assert adapter.derive_key(income, "div1000").tolist() == [123, 0, 30]
    assert adapter.derive_key(commute, "floor").tolist() == [12.0, 0.0, 7.0]


def test_an_unknown_derivation_is_refused() -> None:
    with pytest.raises(AssertionError, match="derivation"):
        adapter.Adapter(derived_keys=((INCOME, "sqrt"),))


def test_derived_te_column_names() -> None:
    assert adapter.derived_te_column(INCOME, "div100") == "Annual_Income_USD_div100_te"
    assert adapter.derived_te_column(COMMUTE, "floor") == "Daily_Commute_km_floor_te"


@pytest.mark.parametrize("prior_weight", [1.0, 5.0])
@pytest.mark.parametrize("source,derivation", ALL_DERIVED)
def test_derived_key_is_encoded_with_the_requested_prior(prior_weight, source, derivation) -> None:
    """Held-out rows get the smoothed mean of their *derived* key over the
    fitting rows, at the declared prior weight; the training rows get the
    nested cross-fit value under the same weight."""
    import numpy as np

    X, y = _synthetic()
    tr, va = np.arange(0, 320), np.arange(320, len(y))
    ad = adapter.Adapter(
        derived_keys=((source, derivation),),
        prior_weight=prior_weight,
        outer_fold=0,
        validation_index=set(va.tolist()),
    )
    out_tr = ad.fit_transform(X.iloc[tr], y[tr])
    out_va = ad.transform(X.iloc[va])
    col = adapter.derived_te_column(source, derivation)
    assert col in out_tr.columns and col in out_va.columns
    # The source column is kept raw; only the encoding is added.
    assert (out_va[source] == X.iloc[va][source]).all()

    keys_tr = adapter.derive_key(X.iloc[tr][source].to_numpy(), derivation)
    keys_va = adapter.derive_key(X.iloc[va][source].to_numpy(), derivation)
    y_tr = y[tr].astype(float)
    prior = y_tr.mean()
    for key, got in zip(keys_va, out_va[col].to_numpy()):
        mask = keys_tr == key
        if mask.any():
            expected = (y_tr[mask].sum() + prior_weight * prior) / (mask.sum() + prior_weight)
        else:
            expected = prior
        assert got == pytest.approx(expected)

    # The same key under a different prior encodes differently: the weight is
    # actually threaded through, not only stored.
    other = adapter.Adapter(
        derived_keys=((source, derivation),),
        prior_weight=prior_weight + 10.0,
        outer_fold=0,
        validation_index=set(va.tolist()),
    )
    other_tr = other.fit_transform(X.iloc[tr], y[tr])
    assert not np.allclose(other_tr[col].to_numpy(), out_tr[col].to_numpy())


# --------------------------------------------------------------------------- #
# No validation row reaches the fit, for every derived key.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("source,derivation", ALL_DERIVED)
def test_a_validation_row_is_refused_for_every_derived_key(source, derivation) -> None:
    X, y = _synthetic()
    ad = adapter.Adapter(
        derived_keys=((source, derivation),),
        prior_weight=1.0,
        outer_fold=0,
        validation_index={5, 399},
    )
    with pytest.raises(AssertionError, match="validation-fold rows"):
        ad.fit_transform(X, y)


@pytest.mark.parametrize("source,derivation", ALL_DERIVED)
def test_a_derived_key_never_sees_its_own_row(source, derivation) -> None:
    """Every row's derived key is unique, so the nested cross-fit must fall
    back to the fold prior on every training row. With smoothing off, a
    leaked encoder would return each row's own target instead."""
    import numpy as np
    import pandas as pd

    n = 20
    y = np.array([i % 2 for i in range(n)])
    step = {"div100": 100, "div1000": 1000, "floor": 1}[derivation]
    base = np.arange(n) * step + (0.5 if derivation == "floor" else 7)
    X = pd.DataFrame({INCOME: base, COMMUTE: base})

    ad = adapter.Adapter(
        derived_keys=((source, derivation),),
        prior_weight=0.0,
        outer_fold=0,
        validation_index=set(),
    )
    te = ad.fit_transform(X, y)[adapter.derived_te_column(source, derivation)].to_numpy()
    assert len(set(adapter.derive_key(X[source].to_numpy(), derivation).tolist())) == n
    assert np.allclose(te, y.mean())
    assert not np.allclose(te, y)


# --------------------------------------------------------------------------- #
# The two declarations.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,prior", [("te_keys_prior1", 1.0), ("te_keys_prior5", 5.0)])
def test_the_two_configurations_are_declared_on_the_incumbent(name, prior) -> None:
    exp = experiments.resolve(name)
    inc = experiments.resolve("hpsearch_lightgbm_best_confirm")
    assert exp in experiments.TE_KEYS_AXIS
    assert exp.incumbent == inc.name
    # On the Incumbent's frozen LightGBM params and 2,341 rounds.
    assert exp.params == inc.params and exp.num_boost_round == 2341
    assert exp.frame == inc.frame and exp.model == "lightgbm"
    assert exp.target_encode == (INCOME, COMMUTE)
    assert exp.te_derived_keys == ALL_DERIVED
    assert exp.te_prior_weight == prior
    # The staggered rule's reject band, declared before the run.
    assert exp.kill_delta == 0.0001
    assert "4 of 5" in exp.hypothesis and "+0.0003" in exp.hypothesis
    cfg = exp.as_config()
    assert cfg["te_prior_weight"] == prior
    assert cfg["te_derived_keys"] == [list(k) for k in ALL_DERIVED]
    ad = runner.fold_adapter(exp, 3, [])
    assert ad.prior_weight == prior and ad.derived_keys == ALL_DERIVED
