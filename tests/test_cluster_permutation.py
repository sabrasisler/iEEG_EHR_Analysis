"""
Tests for analysis/cluster_permutation.py.

These are the real deliverable of the cluster analysis, so they are tested against
properties rather than against golden numbers: a permutation test that is subtly
wrong still produces plausible-looking clusters, and only calibration and
invariance checks catch that.

The four failure modes each get a dedicated test, because each one silently
inflates significance rather than raising:
  - min_extent applied only to the observed map, not to the null
  - independent sign flips per region instead of one vector across all regions
  - clusters bridging an excluded (line-noise) bin
  - all-NaN slices producing a fabricated statistic
"""

import numpy as np
import pytest

from ieeg_ehr.analysis import cluster_permutation as cp


# ============================================================================
# BIN-LEVEL STATISTICS
# ============================================================================

def test_onesample_t_matches_scipy():
    from scipy import stats
    rng = np.random.default_rng(0)
    x = rng.normal(0.5, 1.0, size=(20, 3, 4))
    t, n = cp.onesample_t(x)
    expected = stats.ttest_1samp(x, 0.0, axis=0).statistic
    assert np.allclose(t, expected)
    assert (n == 20).all()


def test_onesample_t_is_nan_aware_and_reports_per_cell_n():
    from scipy import stats
    rng = np.random.default_rng(1)
    x = rng.normal(0.3, 1.0, size=(12, 2, 2))
    x[:5, 0, 0] = np.nan                      # this cell has n=7, the rest n=12

    t, n = cp.onesample_t(x)
    assert n[0, 0] == 7 and n[1, 1] == 12
    assert np.isclose(t[0, 0], stats.ttest_1samp(x[5:, 0, 0], 0.0).statistic)


def test_all_nan_slice_gives_nan_without_warning():
    """Subject x region combinations with NO coverage are a normal path here --
    4 subjects have no DK labels at all -- so this must not warn or fabricate."""
    x = np.full((10, 2, 3), np.nan)
    x[:, 1, :] = 1.0
    with np.errstate(all='raise'):            # any real numeric fault would raise
        t, n = cp.onesample_t(x)
    assert np.isnan(t[0]).all()
    assert (n[0] == 0).all()
    assert np.isnan(t[1]).all()               # zero variance -> NaN, not inf


def test_single_observation_cell_is_nan_not_infinite():
    """n=1 has no variance estimate. An inf would pass a `> t_crit` test and
    become a cluster of one subject."""
    x = np.full((5, 1, 1), np.nan)
    x[0, 0, 0] = 3.0
    t, n = cp.onesample_t(x)
    assert n[0, 0] == 1
    assert np.isnan(t[0, 0])


def test_critical_t_is_per_cell_from_that_cells_df():
    from scipy import stats
    n = np.array([[21, 51], [8, 1]])
    crit = cp.critical_t(n, alpha=0.05)
    assert np.isclose(crit[0, 0], stats.t.ppf(0.975, 20))
    assert np.isclose(crit[0, 1], stats.t.ppf(0.975, 50))
    assert crit[0, 0] > crit[0, 1]            # fewer subjects -> stricter threshold
    assert np.isnan(crit[1, 1])               # df=0


def test_yuen_matches_hand_computed_value():
    """20% trim on n=10 drops 2 from each tail; the denominator uses the
    WINSORIZED sum of squares, which is what makes it the trimmed mean's own
    standard error rather than the raw one."""
    x = np.array([-8.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 20.0])
    t, n = cp.yuen_onesample_t(x.reshape(10, 1, 1), trim=0.2)

    col = np.sort(x)
    g = 2
    trimmed = col[g:len(col) - g]
    wins = np.clip(col, col[g], col[len(col) - 1 - g])
    ssq = ((wins - wins.mean()) ** 2).sum()
    expected = trimmed.mean() / np.sqrt(ssq / ((10 - 2 * g) * (10 - 2 * g - 1)))

    assert n[0, 0] == 10
    assert np.isclose(t[0, 0], expected)


def test_yuen_is_less_moved_by_one_outlier_than_t():
    """The whole reason --robust exists: sign-flipping does not protect against a
    single subject driving a cluster, because t is still mean/SD."""
    base = np.full((20, 1, 1), 0.1)
    base += np.random.default_rng(3).normal(0, 0.05, base.shape)
    spiked = base.copy()
    spiked[0, 0, 0] = 8.0

    t_base, _ = cp.onesample_t(base)
    t_spiked, _ = cp.onesample_t(spiked)
    y_base, _ = cp.yuen_onesample_t(base)
    y_spiked, _ = cp.yuen_onesample_t(spiked)

    assert abs(y_spiked[0, 0] - y_base[0, 0]) < abs(t_spiked[0, 0] - t_base[0, 0])


# ============================================================================
# CLUSTERING / ADJACENCY
# ============================================================================

def test_runs_finds_maximal_runs():
    assert cp._runs(np.array([0, 1, 1, 0, 1, 0, 0], dtype=bool)) == [(1, 2), (4, 4)]
    assert cp._runs(np.zeros(5, dtype=bool)) == []
    assert cp._runs(np.ones(3, dtype=bool)) == [(0, 2)]


def test_cluster_never_bridges_an_invalid_bin():
    """An excluded line-noise bin TERMINATES a run. Without this, a cluster would
    span the 60 Hz notch and claim frequencies that were never tested."""
    t = np.array([[5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]])
    t_crit = np.full_like(t, 2.0)
    valid = np.ones_like(t, dtype=bool)
    valid[0, 3] = False                       # the notch

    clusters = cp.find_clusters(t, valid, t_crit, min_extent=3)
    spans = sorted((c['bin_lo'], c['bin_hi']) for c in clusters)
    assert spans == [(0, 2), (4, 6)]
    assert all(not (c['bin_lo'] <= 3 <= c['bin_hi']) for c in clusters)


def test_clusters_do_not_span_regions():
    """Region rows are not neighbours. Two identical rows must give two clusters,
    never one of double the extent."""
    t = np.full((2, 4), 5.0)
    clusters = cp.find_clusters(t, np.ones_like(t, dtype=bool),
                                np.full_like(t, 2.0), min_extent=3)
    assert len(clusters) == 2
    assert {c['region_idx'] for c in clusters} == {0, 1}
    assert all(c['n_bins'] == 4 for c in clusters)


def test_positive_and_negative_clusters_form_separately():
    """A sign change must break the run, or the masses would cancel and a real
    bidirectional effect would disappear."""
    t = np.array([[5.0, 5.0, 5.0, -5.0, -5.0, -5.0]])
    clusters = cp.find_clusters(t, np.ones_like(t, dtype=bool),
                                np.full_like(t, 2.0), min_extent=3)
    assert len(clusters) == 2
    assert sorted(c['sign'] for c in clusters) == [-1, 1]
    assert sorted(round(c['mass']) for c in clusters) == [-15, 15]


def test_min_extent_filters_short_runs():
    t = np.array([[5.0, 5.0, 0.0, 5.0, 5.0, 5.0]])
    args = (np.ones_like(t, dtype=bool), np.full_like(t, 2.0))
    assert len(cp.find_clusters(t, *args, min_extent=3)) == 1
    assert len(cp.find_clusters(t, *args, min_extent=2)) == 2


def test_mass_and_peak_are_signed_and_from_the_run_only():
    t = np.array([[0.0, 3.0, 9.0, 4.0, 0.0]])
    c, = cp.find_clusters(t, np.ones_like(t, dtype=bool),
                          np.full_like(t, 2.0), min_extent=3)
    assert c['bin_lo'] == 1 and c['bin_hi'] == 3
    assert np.isclose(c['mass'], 16.0)
    assert np.isclose(c['peak_t'], 9.0)


# ============================================================================
# P-VALUES AND CORRECTION
# ============================================================================

def test_permutation_p_is_never_zero():
    """(1 + k) / (n + 1). A finite randomisation cannot support 'p = 0'."""
    assert cp.permutation_p(1e9, np.zeros(999)) == pytest.approx(1 / 1000)
    assert cp.permutation_p(0.0, np.ones(9)) == pytest.approx(10 / 10)


def test_bh_fdr_matches_step_up_by_hand():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.9])
    rejected, adj = cp.bh_fdr(p, q=0.05)
    assert np.allclose(adj, [0.005, 0.02, 0.05125, 0.05125, 0.9])
    assert rejected.tolist() == [True, True, False, False, False]


def test_bh_fdr_is_monotone():
    rng = np.random.default_rng(7)
    p = rng.uniform(size=15)
    _, adj = cp.bh_fdr(p)
    order = np.argsort(p)
    assert np.all(np.diff(adj[order]) >= -1e-12)


def test_bh_family_includes_regions_with_no_cluster():
    """A region that produced nothing is still a region that was looked at. If it
    were dropped from the family, the correction would be over however many
    regions happened to fire -- which is choosing the denominator after the fact."""
    rng = np.random.default_rng(11)
    x = rng.normal(0, 1, size=(24, 6, 12))
    x[:, 0, 3:8] += 2.0                       # only region 0 carries signal
    res = cp.cluster_test(x, n_perm=200, seed=0, min_extent=3)

    assert res['region_p'].size == 6
    assert res['region_p_adj'].size == 6
    silent = [r for r in range(6) if not any(c['region_idx'] == r
                                             for c in res['clusters'])]
    assert silent, 'expected at least one region with no cluster'
    assert all(res['region_p'][r] == 1.0 for r in silent)


# ============================================================================
# THE TEST AS A WHOLE
# ============================================================================

def test_recovers_a_planted_band_in_the_right_region():
    rng = np.random.default_rng(23)
    x = rng.normal(0, 1, size=(30, 5, 20))
    x[:, 2, 8:14] += 1.5

    res = cp.cluster_test(x, n_perm=500, seed=0, min_extent=3)
    sig = [c for c in res['clusters'] if c['sig_two_stage']]
    assert sig, 'planted signal was not detected'
    assert all(c['region_idx'] == 2 for c in sig)
    hit = max(sig, key=lambda c: abs(c['mass']))
    assert hit['bin_lo'] >= 6 and hit['bin_hi'] <= 15
    assert hit['sign'] == 1


def test_null_data_yields_few_significant_clusters():
    """Calibration. Pure noise over several independent realisations should
    produce a significant cluster only rarely -- this is the check that would fail
    if the null were built wrongly (e.g. min_extent skipped in the loop)."""
    n_runs, hits = 12, 0
    for s in range(n_runs):
        x = np.random.default_rng(100 + s).normal(0, 1, size=(25, 5, 20))
        res = cp.cluster_test(x, n_perm=200, seed=s, min_extent=3)
        if any(c['sig_two_stage'] for c in res['clusters']):
            hits += 1
    assert hits <= 3, f'{hits}/{n_runs} null runs significant -- test is anticonservative'


def test_p_global_is_at_least_p_within_region():
    """A global max-stat null is stochastically larger than any one region's, so
    the global p can never be the smaller of the two. Cheap invariant that catches
    the two nulls being swapped."""
    rng = np.random.default_rng(31)
    x = rng.normal(0, 1, size=(28, 6, 16))
    x[:, 1, 4:10] += 1.2
    res = cp.cluster_test(x, n_perm=300, seed=0, min_extent=3)
    assert res['clusters']
    for c in res['clusters']:
        assert c['p_global'] >= c['p_within_region'] - 1e-12


def test_min_extent_is_enforced_inside_the_null():
    """THE critical test. If min_extent were applied only to the observed map, the
    null would contain masses from runs shorter than min_extent, which cannot occur
    in the observation -- an unfair comparison that inflates significance. So the
    null distributions under two different min_extent values must DIFFER."""
    rng = np.random.default_rng(41)
    x = rng.normal(0, 1, size=(20, 4, 24))
    a = cp.cluster_test(x, n_perm=200, seed=0, min_extent=1)
    b = cp.cluster_test(x, n_perm=200, seed=0, min_extent=6)

    # Same seed, same data: the only difference is the extent filter. A null built
    # without the filter would be identical here.
    assert not np.allclose(a['null_global'], b['null_global'])
    assert a['null_global'].mean() > b['null_global'].mean()


def test_one_sign_vector_is_shared_across_regions():
    """With regions perfectly correlated, a shared sign vector gives every region
    the same flip, so per-region max masses move together across permutations.
    Independent per-region flips would decorrelate them -- and would understate the
    null, inflating significance."""
    rng = np.random.default_rng(53)
    base = rng.normal(0, 1, size=(20, 1, 18))
    x = np.repeat(base, 4, axis=1)             # 4 identical regions

    res = cp.cluster_test(x, n_perm=150, seed=0, min_extent=3)
    # Identical regions under one shared flip must produce identical nulls.
    for r in range(1, 4):
        assert np.allclose(res['region_p'][r], res['region_p'][0])


def test_seed_makes_the_result_reproducible():
    x = np.random.default_rng(61).normal(0.4, 1, size=(22, 3, 14))
    a = cp.cluster_test(x, n_perm=150, seed=7, min_extent=3)
    b = cp.cluster_test(x, n_perm=150, seed=7, min_extent=3)
    c = cp.cluster_test(x, n_perm=150, seed=8, min_extent=3)
    assert np.allclose(a['null_global'], b['null_global'])
    assert not np.allclose(a['null_global'], c['null_global'])


def test_valid_mask_keeps_excluded_bins_out_of_every_cluster():
    rng = np.random.default_rng(71)
    x = rng.normal(0, 1, size=(25, 2, 15))
    x[:, 0, :] += 2.0                          # whole row is signal
    valid = np.ones((2, 15), dtype=bool)
    valid[:, 7] = False

    res = cp.cluster_test(x, valid=valid, n_perm=200, seed=0, min_extent=3)
    assert res['clusters']
    for c in res['clusters']:
        assert not (c['bin_lo'] <= 7 <= c['bin_hi'])
    assert not res['valid_map'][:, 7].any()


def test_min_subjects_floor_invalidates_low_coverage_cells():
    rng = np.random.default_rng(83)
    x = rng.normal(1.0, 1, size=(20, 2, 10))
    x[4:, 1, :] = np.nan                       # region 1 keeps only 4 subjects

    res = cp.cluster_test(x, n_perm=100, seed=0, min_extent=3, min_subjects=8)
    assert res['valid_map'][0].all()
    assert not res['valid_map'][1].any()
    assert all(c['region_idx'] == 0 for c in res['clusters'])


def test_significant_mask_covers_exactly_the_significant_clusters():
    rng = np.random.default_rng(97)
    x = rng.normal(0, 1, size=(30, 4, 18))
    x[:, 3, 5:12] += 1.6
    res = cp.cluster_test(x, n_perm=300, seed=0, min_extent=3)
    mask = cp.significant_mask(res, 4, 18, correction='two_stage')

    expected = np.zeros((4, 18), dtype=bool)
    for c in res['clusters']:
        if c['sig_two_stage']:
            expected[c['region_idx'], c['bin_lo']:c['bin_hi'] + 1] = True
    assert np.array_equal(mask, expected)
    assert mask.any()


# ============================================================================
# DETREND
# ============================================================================

def test_detrend_removes_the_per_subject_region_offset():
    rng = np.random.default_rng(101)
    shape = (15, 3, 12)
    offsets = rng.normal(0, 3, size=(15, 3))
    x = rng.normal(0, 0.1, size=shape) + offsets[:, :, None]

    d = cp.detrend_over_frequency(x)
    assert np.allclose(np.nanmean(d, axis=2), 0.0, atol=1e-12)
    assert not np.allclose(np.nanmean(x, axis=2), 0.0)


def test_detrend_ignores_invalid_bins_when_centering():
    """An excluded line-noise bin must not shift the mean that gets subtracted."""
    x = np.zeros((4, 1, 5))
    x[:, 0, :] = [1.0, 1.0, 1.0, 1.0, 100.0]
    valid = np.ones((1, 5), dtype=bool)
    valid[0, 4] = False

    d = cp.detrend_over_frequency(x, valid=valid)
    assert np.allclose(d[:, 0, :4], 0.0)
    assert np.isnan(d[:, 0, 4]).all()


# ============================================================================
# OUTLINE GEOMETRY
# ============================================================================

def test_outline_never_merges_vertically_adjacent_rows():
    """Rows are ROIs and clusters are one row tall, so two stacked significant
    cells must be drawn as TWO boxes. Merging them into one staircase reads as a
    cluster spanning regions -- which the frequency-only adjacency cannot produce."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from ieeg_ehr.features import common

    mask = np.zeros((3, 4), dtype=bool)
    mask[0, 0:3] = True
    mask[1, 0:3] = True                        # directly below, same columns

    fig, ax = plt.subplots()
    n_default = common.draw_mask_outline(ax, mask)
    n_connected = common.draw_mask_outline(ax, mask, connect_rows=True)
    plt.close(fig)

    # Default draws both rows' top AND bottom edges (2 rows x 3 cols x 2) plus the
    # 2 side edges per row; connect_rows drops the shared interior boundary.
    assert n_default > n_connected
    assert n_default == 2 * (3 + 3 + 1 + 1)


def test_outline_lands_on_cell_edges_not_centres():
    """The whole reason this is not contour(): a cell spans +-0.5 around its index,
    so a single True cell at (1, 2) must be bounded by 1.5/2.5 and 0.5/1.5."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from ieeg_ehr.features import common

    mask = np.zeros((3, 4), dtype=bool)
    mask[1, 2] = True
    fig, ax = plt.subplots()
    common.draw_mask_outline(ax, mask)
    seg = ax.collections[-1].get_segments()
    plt.close(fig)

    xs = {round(p[0], 6) for s in seg for p in s}
    ys = {round(p[1], 6) for s in seg for p in s}
    assert xs == {1.5, 2.5}
    assert ys == {0.5, 1.5}


def test_boundary_caveat_is_present_and_says_the_key_thing():
    """This string travels with every table and figure; it is load-bearing prose,
    so a rewrite that drops the point should fail a test."""
    assert 'AS A WHOLE' in cp.BOUNDARY_CAVEAT
    assert 'boundaries' in cp.BOUNDARY_CAVEAT


# ============================================================================
# BH FAMILY MEMBERSHIP
# ============================================================================

def test_untested_regions_are_not_counted_in_the_bh_family():
    """A region with data but no cluster IS a test that came back negative and
    stays in the family. A region with NO VALID CELL was never tested, and counting
    it inflates m -- which weakens every region that did fire. Both look like
    p = 1.0, so the distinction has to come from the validity map."""
    rng = np.random.default_rng(1234)
    x = rng.normal(0, 1, size=(24, 6, 14))
    x[:, 0, 4:10] += 2.0
    x[:, 4:, :] = np.nan                       # regions 4 and 5 have no data at all

    res = cp.cluster_test(x, n_perm=200, seed=0, min_extent=3)
    assert res['n_regions_in_bh_family'] == 4
    assert res['region_tested'].tolist() == [True, True, True, True, False, False]
    # An untested region can never be rejected, whatever its nominal p.
    assert not res['region_rejected'][4] and not res['region_rejected'][5]


def test_shrinking_the_bh_family_cannot_weaken_a_real_region():
    """Same data, but padded with empty regions. Under the fix the adjusted p of the
    real region must not get worse, because the empty rows are not tests."""
    rng = np.random.default_rng(4321)
    core = rng.normal(0, 1, size=(26, 3, 14))
    core[:, 1, 3:9] += 1.8
    padded = np.concatenate([core, np.full((26, 5, 14), np.nan)], axis=1)

    a = cp.cluster_test(core, n_perm=200, seed=0, min_extent=3)
    b = cp.cluster_test(padded, n_perm=200, seed=0, min_extent=3)
    assert b['n_regions_in_bh_family'] == 3
    assert np.isclose(a['region_p_adj'][1], b['region_p_adj'][1])
