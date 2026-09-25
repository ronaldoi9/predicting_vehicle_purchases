"""heuljax completed to its 173 features, and the Member gate (#40, turn 3).

The declaration and group-count tests are dependency-free. The family tests
run the full donor state on a small synthetic frame shaped like the raw CSVs,
with a synthetic original dataset standing in for the Kaggle one (monkeypatched
over :func:`heuljax.load_source`), so no CSV is read. They check that the 173
columns match the notebook's groups, that exactly the notebook's 78 rate
columns carry the +1 constraint, that every group comes out of the one donor
state whose overlap assert fires when forced, that no receiving row's label
reaches any of its 173 columns or its GAM margin, and that the numpy ports of
the notebook's numba kernels agree with the loops they replace.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

import experiments
import heuljax
import models
import runner
from test_heuljax_arena import _synthetic_raw

# The research note's recount (docs/research/income-te-keys-and-published-
# pipelines.md §2.1): 13 + 4 + 7 + 38 + 9 + 28 + 10 + 2 + 18 + 12 + 4 + 28.
GROUP_COUNTS = {
    "raw": 13,
    "original_priors": 4,
    "exact_te": 7,
    "neighbour_hierarchical": 38,
    "group_model_priors": 9,
    "latent_means": 28,
    "digits": 10,
    "worry_key": 2,
    "gate_mixtures": 18,
    "composition": 12,
    "gam_coordinates": 4,
    "multi_scale_te": 28,
}


def _synthetic_original(n: int = 300, seed: int = 5):
    """A frame with the original dataset's schema: Buyer_ID, raw 13, Yes/No target."""
    import numpy as np

    df = _synthetic_raw(n, seed=seed).drop(columns=["id"])
    df.insert(0, "Buyer_ID", [f"EV{i:05d}" for i in range(n)])
    df["Will_Buy_EV"] = np.where(df["Will_Buy_EV"] == 1, "Yes", "No")
    # The Kaggle original carries missing incomes; the source prior skips them.
    df.loc[:4, "Annual_Income_USD"] = np.nan
    return df


@pytest.fixture
def source(monkeypatch):
    state = heuljax.SourceIncome(_synthetic_original())
    monkeypatch.setattr(heuljax, "load_source", lambda: state)
    return state


def _xy(n: int = 600, seed: int = 0):
    df = _synthetic_raw(n, seed=seed)
    return df.drop(columns=["id", "Will_Buy_EV"]), df["Will_Buy_EV"].to_numpy()


def _small_full(num_boost_round: int = 20):
    exp = experiments.resolve("heuljax_full")
    return replace(exp, num_boost_round=num_boost_round, params={**exp.params, "nthread": 1})


# --------------------------------------------------------------------------- #
# The 173 columns, group by group.
# --------------------------------------------------------------------------- #
def test_feature_count_per_group_matches_the_notebooks_173() -> None:
    counts = {name: len(cols) for name, cols in heuljax.FEATURE_GROUPS.items()}
    assert counts == GROUP_COUNTS
    assert sum(counts.values()) == 173
    assert len(heuljax.FEATURE_COLUMNS) == len(set(heuljax.FEATURE_COLUMNS)) == 173
    grouped = {c for cols in heuljax.FEATURE_GROUPS.values() for c in cols}
    assert grouped == set(heuljax.FEATURE_COLUMNS)


def test_feature_columns_keep_the_notebooks_order() -> None:
    cols = heuljax.FEATURE_COLUMNS
    assert cols[:13] == list(heuljax.RAW_COLS)
    assert cols[13:17] == ["ORIG_INC_Y", "ORIG_INC_COUNT_LOG", "ORIG_INC_SEEN", "ORIG_INC_LOCAL_Y"]
    assert cols[17] == "TE_INC_A5"
    assert cols[-1] == "MSTE_CMTINT_LOGN"
    assert cols.index("GAM_GATE_COORD") == 173 - 28 - 4


def test_exactly_the_notebooks_78_rate_columns_are_monotone() -> None:
    flags = heuljax.MONOTONE_CONSTRAINTS.strip("()").split(",")
    assert len(flags) == 173
    constrained = [c for c, f in zip(heuljax.FEATURE_COLUMNS, flags) if f == "1"]
    assert set(flags) == {"0", "1"}
    assert len(constrained) == 78
    assert constrained == list(heuljax.RATE_COLS)
    # Rates, not supports, counts, deviations or digits.
    assert not any(c.endswith(("_LOG", "_LOGN", "_DEV")) for c in constrained)
    assert "GXP_D3_R100" not in constrained  # the bare group prediction is not
    assert "GXP_POST_SD_A20" not in constrained


def test_raw_categoricals_and_digits_are_declared_categorical() -> None:
    categorical = [c for c, t in zip(heuljax.FEATURE_COLUMNS, heuljax.FEATURE_TYPES) if t == "c"]
    assert categorical == list(heuljax.CAT_COLS) + list(heuljax.FEATURE_GROUPS["digits"])


def test_the_tracer_slice_stays_the_default_feature_set() -> None:
    # heuljax_tracer's params never named a feature set; its Run Record must
    # stay reproducible, so the absent key still means the 20-column slice.
    tracer = heuljax.feature_set({})
    assert tracer.columns == list(heuljax.RAW_COLS + heuljax.FEATURE_GROUPS["exact_te"])
    assert tracer.base_margin is False
    full = heuljax.feature_set({"feature_set": "full"})
    assert full.columns == heuljax.FEATURE_COLUMNS
    assert full.base_margin is True
    with pytest.raises(ValueError, match="feature set"):
        heuljax.feature_set({"feature_set": "most"})


# --------------------------------------------------------------------------- #
# The declaration and its gate.
# --------------------------------------------------------------------------- #
def test_heuljax_full_is_the_tracer_completed_on_the_same_protocol() -> None:
    full = experiments.resolve("heuljax_full")
    tracer = experiments.resolve("heuljax_tracer")
    assert full.model == "heuljax" and full.frame == "raw_columns"
    assert full.num_boost_round == 1000
    assert full.fold_seed == 0
    assert full.params["tree_method"] == "hist" and full.params["device"] == "cpu"
    assert full.params["feature_set"] == "full"
    # Only the representation (and the group model's seed it brings) changes.
    changed = {k for k in full.params if full.params.get(k) != tracer.params.get(k)}
    assert changed == {"feature_set", "group_seed"}
    # The gate reads the Incumbent standing when it was declared (#39's
    # promotion), not the one the tracer was paired with.
    assert full.incumbent == "te_keys_prior5"


def test_heuljax_full_declares_the_member_gate_not_a_kill_delta() -> None:
    full = experiments.resolve("heuljax_full")
    assert full.member_gate_oof == 0.9455
    assert full.member_gate_corr == 0.985
    assert full.kill_delta is None
    assert full.kill_min_folds_positive is None
    # The gate is on the Run Record, not the config: it must not move the hash.
    assert "member_gate_oof" not in full.as_config()


def test_a_member_gate_is_declared_whole_and_alone() -> None:
    full = experiments.resolve("heuljax_full")
    with pytest.raises(ValueError, match="both"):
        replace(full, member_gate_corr=None)
    with pytest.raises(ValueError, match="not both"):
        replace(full, kill_delta=0.0001)


def test_member_gate_passes_on_oof_or_on_low_correlation() -> None:
    by_oof = runner.member_gate_outcome(0.94560, 0.998, min_oof=0.9455, max_corr=0.985)
    assert by_oof["passed"] is True and by_oof["dead"] is False
    by_corr = runner.member_gate_outcome(0.94400, 0.980, min_oof=0.9455, max_corr=0.985)
    assert by_corr["passed"] is True
    neither = runner.member_gate_outcome(0.94526, 0.998, min_oof=0.9455, max_corr=0.985)
    assert neither["passed"] is False and neither["dead"] is True
    assert neither == {
        "member_gate": True,
        "min_oof": 0.9455,
        "max_corr": 0.985,
        "oof_auc": 0.94526,
        "incumbent_oof_corr": 0.998,
        "passed": False,
        "dead": True,
    }
    # Boundaries as declared: OOF >= 0.9455, correlation strictly < 0.985.
    assert runner.member_gate_outcome(0.9455, 0.999, min_oof=0.9455, max_corr=0.985)["passed"]
    assert not runner.member_gate_outcome(0.9450, 0.985, min_oof=0.9455, max_corr=0.985)["passed"]
    # Without the Incumbent's vector only the OOF arm can be read.
    assert not runner.member_gate_outcome(0.9450, None, min_oof=0.9455, max_corr=0.985)["passed"]


def test_heuljax_full_stays_out_of_the_turn2_blend() -> None:
    import blends

    assert "heuljax_full" not in blends.BLEND_ALL_MEMBERS.members


# --------------------------------------------------------------------------- #
# The original-data income priors.
# --------------------------------------------------------------------------- #
def test_original_rows_identical_to_a_competition_row_are_dropped() -> None:
    original = _synthetic_original(50)
    train = _synthetic_raw(40, seed=9)
    # Row 10 of the original is copied verbatim into train (features only).
    raw = list(heuljax.RAW_COLS)
    train.loc[3, raw] = original.loc[10, raw].to_numpy()
    kept = heuljax.dedup_original(original, train, _synthetic_raw(10, seed=8, with_target=False))
    assert len(kept) == 49
    assert "EV00010" not in set(kept["Buyer_ID"])


def test_source_prior_reads_the_originals_labels_at_the_exact_income() -> None:
    import numpy as np
    import pandas as pd

    original = pd.DataFrame(
        {c: _synthetic_original(4)[c] for c in heuljax.RAW_COLS}
    )
    original["Annual_Income_USD"] = [100.0, 100.0, 200.0, np.nan]
    original["Will_Buy_EV"] = ["Yes", "No", "Yes", "Yes"]
    src = heuljax.SourceIncome(original)
    out = src.transform(np.array([100.0, 200.0, 150.0]))
    assert out.shape == (3, 4)
    assert out[0, 0] == pytest.approx(0.5)  # mean label at 100
    assert out[1, 0] == pytest.approx(1.0)
    assert out[2, 0] == pytest.approx(src.prior)  # unseen: the original's prior
    assert out[:, 2].tolist() == [1.0, 1.0, 0.0]  # seen flag
    assert out[0, 1] == pytest.approx(math.log1p(2))


# --------------------------------------------------------------------------- #
# The donor state: every group fitted on donor rows only.
# --------------------------------------------------------------------------- #
def test_donor_features_cover_all_173_columns_and_a_finite_margin(source) -> None:
    import numpy as np

    X, y = _xy(500)
    features, margin = heuljax.donor_features(X, y, seed=0, group_seed=7000, full=True)
    assert features.shape == (500, 173)
    assert features.dtype == np.float32
    assert np.isfinite(features).all() and np.isfinite(margin).all()
    assert margin.shape == (500,)
    # Every group actually varies across rows -- none left at a placeholder.
    for name, cols in heuljax.FEATURE_GROUPS.items():
        block = features[:, [heuljax.FEATURE_COLUMNS.index(c) for c in cols]]
        assert np.ptp(block, axis=0).max() > 0, name


def test_donor_overlap_assert_fires_for_the_full_state(source) -> None:
    import numpy as np

    X, y = _xy(200)
    with pytest.raises(AssertionError, match="overlaps"):
        heuljax.fit_donor_state(
            X, y, donor=np.arange(0, 120), receiving=np.arange(100, 200), full=True, group_seed=1
        )


def test_no_receiving_label_reaches_any_group_or_the_gam_margin(source) -> None:
    import numpy as np

    X, y = _xy(400)
    donor, receiving = np.arange(0, 300), np.arange(300, 400)
    state = heuljax.fit_donor_state(X, y, donor, receiving, full=True, group_seed=1)
    flipped = y.copy()
    flipped[receiving] = 1 - flipped[receiving]
    again = heuljax.fit_donor_state(X, flipped, donor, receiving, full=True, group_seed=1)
    raw = heuljax._take(heuljax._pack_raw(X), receiving)
    a, ma = state.transform(raw)
    b, mb = again.transform(raw)
    for name, cols in heuljax.FEATURE_GROUPS.items():
        idx = [heuljax.FEATURE_COLUMNS.index(c) for c in cols]
        assert np.array_equal(a[:, idx], b[:, idx]), name
    assert np.array_equal(ma, mb)


def test_a_donor_row_label_does_move_the_target_groups(source) -> None:
    # The converse, so the invariance above is not vacuous: flipping donor
    # labels moves every target-reading group.
    import numpy as np

    X, y = _xy(400)
    donor, receiving = np.arange(0, 300), np.arange(300, 400)
    flipped = y.copy()
    flipped[donor[:150]] = 1 - flipped[donor[:150]]
    raw = heuljax._take(heuljax._pack_raw(X), receiving)
    a, _ = heuljax.fit_donor_state(X, y, donor, receiving, full=True, group_seed=1).transform(raw)
    b, _ = heuljax.fit_donor_state(X, flipped, donor, receiving, full=True, group_seed=1).transform(raw)
    label_free = {"raw", "original_priors", "latent_means", "digits"}
    for name, cols in heuljax.FEATURE_GROUPS.items():
        idx = [heuljax.FEATURE_COLUMNS.index(c) for c in cols]
        moved = not np.array_equal(a[:, idx], b[:, idx])
        assert moved == (name not in label_free), name


def test_single_digits_of_income_and_tenths_of_commute() -> None:
    import numpy as np

    X, y = _xy(50)
    X.loc[0, "Annual_Income_USD"] = 123456.0
    X.loc[0, "Daily_Commute_km"] = 43.7
    raw = heuljax._pack_raw(X)
    digits = heuljax.digit_features(raw)[0]
    assert digits.tolist() == [6, 5, 4, 3, 2, 1, 7, 3, 4, 0]
    assert np.all((heuljax.digit_features(raw) >= 0) & (heuljax.digit_features(raw) <= 9))


# --------------------------------------------------------------------------- #
# The numpy kernels against the notebook's numba loops, run as plain Python.
# --------------------------------------------------------------------------- #
def _gam_loop(codes, sizes, ridges, widths, targets, iterations, step):
    import numpy as np

    n, nt = codes.shape
    prior = min(max(np.mean(targets), 1e-6), 1.0 - 1e-6)
    intercept = math.log(prior / (1.0 - prior))
    eta = np.full(n, intercept)
    tables = np.zeros((nt, np.max(sizes)))
    for _ in range(iterations):
        for term in range(nt):
            length = sizes[term]
            grad, hess = np.zeros(length), np.zeros(length)
            for i in range(n):
                p = 1.0 / (1.0 + math.exp(-max(min(eta[i], 700.0), -700.0)))
                grad[codes[i, term]] += targets[i] - p
                hess[codes[i, term]] += p * (1.0 - p)
            width = widths[term]
            if width:
                g2, h2 = grad.copy(), hess.copy()
                for j in range(length):
                    lo, hi = max(0, j - width), min(length, j + width + 1)
                    g2[j], h2[j] = grad[lo:hi].sum(), hess[lo:hi].sum()
                grad, hess = g2, h2
            delta = step * grad / (hess + ridges[term])
            tables[term, :length] += delta
            eta += delta[codes[:, term]]
    return intercept, tables, eta


def test_gam_kernel_matches_the_notebook_loop() -> None:
    import numpy as np

    rng = np.random.default_rng(3)
    n = 300
    sizes = np.array([7, 12, 5], np.int32)
    codes = np.column_stack([rng.integers(0, s, n) for s in sizes]).astype(np.int32)
    ridges = np.array([5.0, 25.0, 5.0])
    widths = np.array([0, 2, 0], np.int32)
    targets = (rng.uniform(size=n) < 0.3).astype(float)
    got = heuljax.gam_kernel(codes, sizes, ridges, widths, targets, 4, 0.5)
    want = _gam_loop(codes, sizes, ridges, widths, targets, 4, 0.5)
    assert got[0] == pytest.approx(want[0])
    assert np.allclose(got[1], want[1], atol=1e-10)
    assert np.allclose(got[2], want[2], atol=1e-10)


def test_mixture_kernel_matches_the_notebook_loop() -> None:
    import numpy as np

    rng = np.random.default_rng(4)
    ng, nk = 6, 30
    counts = rng.integers(0, 4, (ng, nk)).astype(float)
    q0 = rng.dirichlet(np.ones(nk), ng)
    q1 = rng.dirichlet(np.ones(nk), ng)
    mu = rng.uniform(0.1, 0.4, ng)
    pi, info, gain = heuljax.mixture_kernel(counts, q0, q1, mu)
    for i in range(ng):
        p = mu[i]
        for _ in range(24):
            g = 10.0 * (mu[i] / p - (1.0 - mu[i]) / (1.0 - p))
            h = 10.0 * (mu[i] / p**2 + (1.0 - mu[i]) / (1.0 - p) ** 2)
            for j in range(nk):
                d = q1[i, j] - q0[i, j]
                r = d / max(q0[i, j] + p * d, 1e-300)
                g += counts[i, j] * r
                h += counts[i, j] * r * r
            p = max(1e-4, min(1.0 - 1e-4, p + max(-0.15, min(0.15, g / max(h, 1e-8)))))
        d = q1[i] - q0[i]
        m = np.maximum(q0[i] + p * d, 1e-300)
        old = np.maximum(q0[i] + mu[i] * d, 1e-300)
        assert pi[i] == pytest.approx(p)
        assert info[i] == pytest.approx(math.log1p((counts[i] * (d / m) ** 2).sum()))
        assert gain[i] == pytest.approx((counts[i] * np.log(m / old)).sum() / max(counts[i].sum(), 1.0))


def test_composition_curves_and_their_inversion_match_the_notebook_loops() -> None:
    import numpy as np

    rng = np.random.default_rng(5)
    sizes = np.array([3, 1, 5, 2])
    boundaries = np.r_[0, np.cumsum(sizes)]
    nuisance = rng.normal(scale=2.0, size=boundaries[-1])
    grid = heuljax.shift_grid()
    curve = heuljax.composition_curve_kernel(nuisance, boundaries, grid)
    for g in range(len(sizes)):
        rows = nuisance[boundaries[g]:boundaries[g + 1]]
        want = [np.mean(1.0 / (1.0 + np.exp(-(rows + s)))) for s in grid]
        assert np.allclose(curve[g], want, atol=1e-12)

    indices = rng.integers(0, len(sizes), 40).astype(np.int32)
    probs = rng.uniform(0.0, 1.0, (40, 3))
    probs[0, 0], probs[1, 0] = 0.0, 1.0  # the clamped ends
    got = heuljax.invert_curve_kernel(curve, indices, probs, grid)
    for i in range(40):
        c = curve[indices[i]]
        for j in range(3):
            p = probs[i, j]
            if p <= c[0]:
                want = grid[0]
            elif p >= c[-1]:
                want = grid[-1]
            else:
                hi = int(np.searchsorted(c, p, side="left"))
                lo = hi - 1
                den = c[hi] - c[lo]
                want = grid[lo] + ((p - c[lo]) / den if den > 1e-14 else 0.0) * (grid[hi] - grid[lo])
            assert got[i, j] == pytest.approx(want, abs=1e-12)


# --------------------------------------------------------------------------- #
# The family end to end.
# --------------------------------------------------------------------------- #
def test_models_fit_trains_the_full_family_for_the_fixed_rounds(source) -> None:
    config = _small_full(num_boost_round=13)
    X, y = _xy(500)
    model = models.fit(X, y, config.params, num_boost_round=13, family="heuljax")
    assert model.booster.num_boosted_rounds() == 13
    assert model.booster.num_features() == 173


def test_full_family_predicts_one_probability_per_row_deterministically(source) -> None:
    import numpy as np

    config = _small_full()
    X, y = _xy(600)
    runs = [
        models.predict(
            models.fit(X.iloc[:450], y[:450], config.params, num_boost_round=20, family="heuljax"),
            X.iloc[450:],
            family="heuljax",
        )
        for _ in range(2)
    ]
    assert runs[0].shape == (150,)
    assert ((runs[0] > 0.0) & (runs[0] < 1.0)).all()
    assert np.array_equal(runs[0], runs[1])


def test_prediction_boosts_on_top_of_the_gam_margin(source) -> None:
    # The notebook's base_margin: gate + other + 0.25 (income + commute), fed to
    # XGBoost on every row it scores, so predict must carry it too.
    import numpy as np

    config = _small_full()
    X, y = _xy(500)
    model = models.fit(X.iloc[:400], y[:400], config.params, num_boost_round=5, family="heuljax")
    features, margin = model.state.transform(heuljax._pack_raw(X.iloc[400:]))
    with_margin = model.booster.predict(heuljax._dmatrix(features, model.spec, base_margin=margin))
    without = model.booster.predict(heuljax._dmatrix(features, model.spec))
    got = models.predict(model, X.iloc[400:], family="heuljax")
    assert np.array_equal(got, with_margin)
    assert not np.allclose(got, without)
