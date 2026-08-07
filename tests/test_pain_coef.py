"""
Tests for analysis/pain_coef.py and the gap-closing / predictor-shuffle additions
to cluster_permutation.

Two of these encode findings rather than behaviour, and are the reason the module
exists at all:

  - `test_closing_the_gap_makes_high_frequency_bins_testable` pins the measured
    fact that with the line-noise bins merely INVALIDATED, 129/144/200 Hz sit in
    runs shorter than min_extent and can never reach significance at any effect
    size. If someone reverts to invalidation, that test says what it costs.
  - `test_subset_slope_matches_full_regression` guards the one piece of arithmetic
    here that is easy to get subtly wrong: for a column with missing epochs the
    predictor must be re-centred over the surviving rows, and re-centring the
    WEIGHTS instead silently drops a scale factor.
"""

import numpy as np
import pandas as pd
import pytest

from ieeg_ehr.analysis import cluster_permutation as cp
from ieeg_ehr.analysis import pain_coef


def _epochs(subject, scores, n_region=2, n_bin=4, seed=0, coef=0.0):
    """A tidy epoch table for one subject, optionally with a planted slope."""
    rng = np.random.default_rng(seed)
    rows = []
    for e, s in enumerate(scores):
        for r in range(n_region):
            for b in range(n_bin):
                rows.append({'subject_id': subject, 'epoch_id': f'{subject}-{e}',
                             'pain_score': float(s), 'region': f'R{r}',
                             'freq_bin_index': b,
                             'value': coef * s + rng.normal(0, 0.01)})
    return pd.DataFrame(rows)


# ============================================================================
# THE REGRESSION
# ============================================================================

def test_weights_identity_matches_linregress():
    """w @ y IS the OLS slope -- the identity the whole module is built on."""
    from scipy import stats
    rng = np.random.default_rng(0)
    x = rng.normal(5, 2, size=30)
    Y = rng.normal(0, 1, size=(30, 7))

    got = pain_coef.coef_from_predictor(x, Y)
    for j in range(Y.shape[1]):
        assert np.isclose(got[j], stats.linregress(x, Y[:, j]).slope)


def test_subset_slope_matches_full_regression():
    """A column with missing epochs must be fitted on its surviving rows, with the
    predictor RE-CENTRED over exactly those rows."""
    from scipy import stats
    rng = np.random.default_rng(1)
    x = rng.normal(4, 3, size=25)
    Y = rng.normal(0, 1, size=(25, 3))
    Y[:8, 1] = np.nan                       # column 1 keeps only 17 epochs

    got = pain_coef.coef_from_predictor(x, Y)
    assert np.isclose(got[0], stats.linregress(x, Y[:, 0]).slope)
    assert np.isclose(got[1], stats.linregress(x[8:], Y[8:, 1]).slope)


def test_zero_variance_predictor_gives_none_not_a_division():
    assert pain_coef.regression_weights(np.full(10, 3.0)) is None
    out = pain_coef.coef_from_predictor(np.full(10, 3.0), np.ones((10, 2)))
    assert np.isnan(out).all()


def test_a_planted_slope_is_recovered():
    scores = list(range(11)) * 3
    df = _epochs('sub-001', scores, seed=2, coef=0.05)
    coef, subjects, _, _ = pain_coef.subject_coef_matrix(
        df, ['R0', 'R1'], [0, 1, 2, 3])
    assert subjects == ['sub-001']
    assert np.allclose(coef[0], 0.05, atol=5e-3)


def test_missing_cells_are_nan_not_zero():
    """0 is a real coefficient meaning 'no relationship'; a missing cell must not
    masquerade as one."""
    df = _epochs('sub-001', list(range(11)) * 2, seed=3, coef=0.02)
    df = df[~((df.region == 'R1') & (df.freq_bin_index == 2))]
    coef, _, _, _ = pain_coef.subject_coef_matrix(df, ['R0', 'R1'], [0, 1, 2, 3])
    assert np.isnan(coef[0, 1, 2])
    assert np.isfinite(coef[0, 0, 2])


# ============================================================================
# ELIGIBILITY
# ============================================================================

def test_each_criterion_excludes_exactly_its_own_subject():
    frames = [
        _epochs('sub-ok', list(range(11)) * 2, seed=4),          # passes all
        _epochs('sub-few', [0, 5, 9], seed=5),                   # <=10 epochs
        _epochs('sub-narrow', [3, 4, 3, 4] * 4, seed=6),         # range 1
        _epochs('sub-modal', [0] * 20 + [9, 9], seed=7),         # 2 non-modal
    ]
    kept, diag = pain_coef.eligible_subjects(pd.concat(frames, ignore_index=True))
    assert kept == ['sub-ok']

    why = dict(zip(diag.subject_id, diag.excluded_because))
    assert 'epochs' in why['sub-few']
    assert 'range' in why['sub-narrow']
    assert 'non-modal' in why['sub-modal']


def test_non_modal_rule_catches_what_range_alone_misses():
    """The rule that earns its place: range 9 from two epochs is an outlier
    statistic, not a trend, and only the non-modal count sees it."""
    df = _epochs('sub-x', [0] * 30 + [9, 9], seed=8)
    _, diag = pain_coef.eligible_subjects(df)
    row = diag.iloc[0]
    assert row.pain_range >= 4                      # range alone would admit it
    assert not row.included
    assert 'non-modal' in row.excluded_because


def test_eligibility_counts_epochs_not_exploded_rows():
    """The epoch table is exploded over region x bin; counting rows would inflate
    every criterion by ~1000x and admit everything."""
    df = _epochs('sub-few', [0, 5, 9], n_region=5, n_bin=50, seed=9)
    assert len(df) > 700
    _, diag = pain_coef.eligible_subjects(df)
    assert diag.iloc[0].n_epochs == 3
    assert not diag.iloc[0].included


# ============================================================================
# CLOSING THE LINE-NOISE GAP
# ============================================================================

def test_compact_and_expand_round_trip():
    x = np.arange(2 * 10, dtype=float).reshape(1, 2, 10)
    compact, kept = cp.compact_bins(x, [3, 7], 10)
    assert compact.shape == (1, 2, 8)
    assert kept.tolist() == [0, 1, 2, 4, 5, 6, 8, 9]

    mask = np.zeros((2, 8), dtype=bool)
    mask[0, 2:5] = True
    full = cp.expand_mask(mask, kept, 10)
    assert full[0].tolist() == [False, False, True, False, True, True,
                                False, False, False, False]
    assert not full[:, [3, 7]].any(), 'removed bins must never be outlined'


def test_spans_removed_gap_flags_a_bridging_cluster():
    kept = np.array([0, 1, 2, 4, 5])
    assert not cp.spans_removed_gap(0, 2, kept)
    assert cp.spans_removed_gap(2, 4, kept)     # bridges the removed bin 3


def test_closing_the_gap_makes_high_frequency_bins_testable():
    """THE measured bias, as a regression test.

    On the real 50-bin axis the line-noise bins are 36, 37, 43, 46, 47, 49. Left
    invalid they leave runs of 2 (bins 44-45, ~129-144 Hz) and 1 (bin 48, ~200 Hz),
    both under min_extent=3 -- so no effect of any size can be found there.
    Removing them and closing the axis makes everything above bin 38 one run.
    """
    noise = [36, 37, 43, 46, 47, 49]
    n_bins, min_extent = 50, 3

    # invalidated: find the runs the clusterer would see
    valid = np.ones((1, n_bins), dtype=bool)
    valid[0, noise] = False
    runs = cp._runs(valid[0])
    untestable = [b for lo, hi in runs if hi - lo + 1 < min_extent
                  for b in range(lo, hi + 1)]
    assert set(untestable) == {44, 45, 48}

    # removed and closed: every surviving bin sits in one long run
    _, kept = cp.compact_bins(np.zeros((1, 1, n_bins)), noise, n_bins)
    compact_runs = cp._runs(np.ones(len(kept), dtype=bool))
    assert compact_runs == [(0, len(kept) - 1)]
    assert set(untestable) <= set(kept.tolist())


# ============================================================================
# THE PREDICTOR-SHUFFLE NULL
# ============================================================================

def _per_subject(n_subj=12, n_ep=20, n_cells=12, coef=0.0, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for s in range(n_subj):
        x = rng.integers(0, 10, size=n_ep).astype(float)
        if x.std() == 0:
            x[0] += 1
        Y = rng.normal(0, 1, size=(n_ep, n_cells)) + coef * x[:, None]
        out[f'sub-{s:03d}'] = (Y, x)
    return out


def test_shuffle_null_is_calibrated_on_noise():
    per_subject = _per_subject(coef=0.0, seed=11)
    valid = np.ones((3, 4), dtype=bool)
    nr, ng = cp.predictor_shuffle_null(
        per_subject, (3, 4), valid, alpha=0.05, min_extent=3, n_perm=60, seed=0,
        coef_fn=pain_coef.coef_from_predictor)
    assert nr.shape == (60, 3) and ng.shape == (60,)
    assert (ng >= 0).all()


def test_shuffle_null_shifts_when_signal_is_planted():
    """A real relationship must exceed its own shuffled null; if it did not, the
    null would be reproducing the signal and the test would be worthless."""
    per_subject = _per_subject(coef=0.6, seed=12)
    valid = np.ones((3, 4), dtype=bool)
    _, ng = cp.predictor_shuffle_null(
        per_subject, (3, 4), valid, alpha=0.05, min_extent=3, n_perm=60, seed=0,
        coef_fn=pain_coef.coef_from_predictor)

    obs = np.array([pain_coef.coef_from_predictor(x, Y)
                    for Y, x in per_subject.values()]).reshape(-1, 3, 4)
    t, n = cp.onesample_t(obs)
    mass, _ = cp.max_mass_per_region(t, valid, cp.critical_t(n, 0.05), 3, 3)
    assert mass.max() > np.percentile(ng, 95)


def test_shuffle_is_one_permutation_per_subject_across_all_cells():
    """An epoch is relabelled as a WHOLE. Shuffling per cell would destroy the
    within-subject correlation across regions and give a null far too narrow."""
    rng = np.random.default_rng(13)
    x = rng.permutation(np.arange(16).astype(float))
    base = rng.normal(0, 1, size=(16, 1))
    Y = np.repeat(base, 6, axis=1)          # six IDENTICAL cells

    seen = []

    def spy(xp, Yp):
        seen.append(xp.copy())
        return pain_coef.coef_from_predictor(xp, Yp)

    cp.predictor_shuffle_null({'sub-a': (Y, x)}, (2, 3),
                              np.ones((2, 3), dtype=bool), 0.05, 3, 5, seed=0,
                              coef_fn=spy)
    assert len(seen) == 5, 'coef_fn must be called ONCE per subject per permutation'
    # Identical cells must therefore give identical coefficients within a permutation.
    coefs = pain_coef.coef_from_predictor(seen[0], Y)
    assert np.allclose(coefs, coefs[0])


def test_shuffle_null_is_reproducible_from_the_seed():
    """Seeded reproducibility, tested on the PERMUTATIONS rather than on the null.

    The obvious version -- compare two null distributions -- is degenerate here: on
    a small map with min_extent=3, shuffled data almost never forms a cluster, so
    every max-mass is 0.0 and two different seeds compare equal for a reason that
    has nothing to do with seeding. Spying on the shuffled predictors tests exactly
    what the seed controls.
    """
    per_subject = _per_subject(coef=0.2, seed=14)
    valid = np.ones((3, 4), dtype=bool)

    def run(seed):
        seen = []

        def spy(xp, Yp):
            seen.append(xp.copy())
            return pain_coef.coef_from_predictor(xp, Yp)

        cp.predictor_shuffle_null(per_subject, (3, 4), valid, alpha=0.05,
                                  min_extent=3, n_perm=5, seed=seed, coef_fn=spy)
        return np.array(seen)

    a, b, c = run(7), run(7), run(8)
    assert np.array_equal(a, b), 'same seed must give the same permutations'
    assert not np.array_equal(a, c), 'different seeds must differ'
