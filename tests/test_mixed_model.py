"""
Tests for analysis/mixed_model.py.

Three of these encode decisions rather than behaviour, and are the reason to read
this file before changing the module:

  - `test_lrt_refuses_mismatched_fixed_effects` guards the ONE condition that
    makes a REML likelihood-ratio test legal here. Add a covariate to the full
    formula and forget the reduced one and every heterogeneity p-value silently
    becomes meaningless; this makes it an exception instead.
  - `test_mixture_p_is_half_the_chi2_tail` pins the 50:50 boundary mixture. A
    plain chi2(1) reference is conservative, which sounds safe and here means
    under-detecting exactly the heterogeneity the analysis exists to find.
  - `test_permutation_preserves_submean_when_balanced` pins the claim the module
    docstring makes about what a within-subject shuffle does and does not move.

The fitted-model tests use small synthetic cells so the whole file stays inside a
few seconds. They assert RECOVERY (the planted value is inside a wide interval),
not precision -- a mixed model on 10 subjects is not a precise instrument, and a
tight tolerance here would be a flaky test rather than a strong one.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from ieeg_ehr.analysis import mixed_model as mm


def synth_cell(n_subj=12, n_chan=4, n_epoch=25, beta=0.010, sd_slope=0.0,
               sd_subj=0.5, sd_chan=0.3, sd_noise=0.10, seed=0):
    """One cell with a KNOWN fixed slope and a known between-subject slope SD."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_subj):
        subject = f'sub-{s:03d}'
        nrs = rng.integers(0, 11, n_epoch).astype(float)
        b_s = beta + rng.normal(0, sd_slope)
        a_s = rng.normal(0, sd_subj)
        for c in range(n_chan):
            a_c = rng.normal(0, sd_chan)
            for e in range(n_epoch):
                rows.append({
                    'subject_id': subject, 'channel': f'ch{c:02d}',
                    'epoch_id': f'{subject}-{e}', 'pain_score': nrs[e],
                    'value': a_s + a_c + b_s * (nrs[e] - nrs.mean())
                             + rng.normal(0, sd_noise),
                })
    return mm.build_cell_frame(pd.DataFrame(rows), region='R', freq_bin_index=0)


# ============================================================================
# FRAME CONSTRUCTION
# ============================================================================

def test_channel_uid_is_globally_unique():
    """The same channel NAME in two subjects must never be pooled."""
    df = synth_cell(n_subj=3, n_chan=2, n_epoch=5)
    per_subject = df.groupby('channel_uid')['subject'].nunique()
    assert (per_subject == 1).all()
    assert df['channel_uid'].nunique() == 3 * 2


def test_nrs_components_decompose_exactly():
    df = synth_cell(n_subj=5, n_chan=2, n_epoch=8)
    assert np.allclose(df['NRS'], df['NRS_within'] + df['NRS_submean'])
    # Row-weighted centring: within-subject, NRS_within sums to zero.
    sums = df.groupby('subject')['NRS_within'].sum()
    assert np.allclose(sums.to_numpy(), 0.0, atol=1e-10)


def test_nrs_within_is_orthogonal_to_subject_dummies():
    """The property the centring exists to produce -- no between-subject contrast
    leaks into the within-subject slope."""
    df = synth_cell(n_subj=6, n_chan=3, n_epoch=10)
    dummies = pd.get_dummies(df['subject']).to_numpy(dtype=float)
    proj = dummies.T @ df['NRS_within'].to_numpy()
    assert np.allclose(proj, 0.0, atol=1e-9)


def test_frame_upcasts_to_float64():
    """The cache is float32; every reduction downstream must accumulate wider."""
    raw = pd.DataFrame({
        'subject_id': ['sub-001'] * 4, 'channel': ['a', 'a', 'b', 'b'],
        'epoch_id': [0, 1, 0, 1], 'pain_score': np.float32([0, 5, 0, 5]),
        'value': np.float32([1.0, 1.1, 2.0, 2.1]),
    })
    df = mm.build_cell_frame(raw)
    assert df['log10_power'].dtype == np.float64
    assert df['NRS'].dtype == np.float64


def test_non_finite_rows_are_dropped():
    raw = pd.DataFrame({
        'subject_id': ['sub-001'] * 3, 'channel': ['a'] * 3, 'epoch_id': [0, 1, 2],
        'pain_score': [0.0, 5.0, 7.0], 'value': [1.0, np.nan, 2.0],
    })
    assert len(mm.build_cell_frame(raw)) == 2


# ============================================================================
# STRUCTURAL REFUSALS
# ============================================================================

def test_refuses_too_few_subjects():
    df = synth_cell(n_subj=3, n_chan=2, n_epoch=10)
    ok, reason = mm.cell_is_fittable(df, min_subjects=8)
    assert not ok and 'subjects' in reason


def test_refuses_no_within_subject_variance():
    """A subject who reported the same score every time carries no within-subject
    information, and a cell where that is true of EVERYONE is not estimable."""
    rows = []
    for s in range(10):
        for e in range(6):
            rows.append({'subject_id': f'sub-{s}', 'channel': 'a',
                         'epoch_id': e, 'pain_score': float(s),
                         'value': float(s) + 0.01 * e})
    df = mm.build_cell_frame(pd.DataFrame(rows))
    ok, reason = mm.cell_is_fittable(df)
    assert not ok and 'variance' in reason


# ============================================================================
# RECOVERY
# ============================================================================

@pytest.mark.slow
def test_recovers_a_known_fixed_slope():
    beta = 0.012
    df = synth_cell(n_subj=14, n_chan=4, n_epoch=25, beta=beta, sd_slope=0.0, seed=1)
    res, _ = mm.fit_cell(df)
    assert res.converged
    est = float(res.fe_params['NRS_within'])
    se = float(res.bse['NRS_within'])
    assert abs(est - beta) < 4 * se, f'{est} vs planted {beta} (se {se})'
    assert res.pvalues['NRS_within'] < 0.05


@pytest.mark.slow
def test_recovers_a_known_null():
    """No planted relationship -> the fixed effect is not significant and the
    slope variance sits at the boundary."""
    df = synth_cell(n_subj=14, n_chan=4, n_epoch=25, beta=0.0, sd_slope=0.0, seed=2)
    res, _ = mm.fit_cell(df)
    assert res.pvalues['NRS_within'] > 0.05
    at_boundary, ratios = mm.boundary_report(res, df)
    assert ratios['subj_slope'] < ratios['channel']


@pytest.mark.slow
def test_detects_planted_heterogeneity():
    """Heterogeneous subjects -> the LRT fires; homogeneous ones -> it does not."""
    hetero = synth_cell(n_subj=16, n_chan=3, n_epoch=25, beta=0.0, sd_slope=0.030,
                        sd_noise=0.05, seed=3)
    homo = synth_cell(n_subj=16, n_chan=3, n_epoch=25, beta=0.0, sd_slope=0.0,
                      sd_noise=0.05, seed=3)

    def lrt_p(df):
        full, _ = mm.fit_cell(df, mm.VC_FULL)
        red, _ = mm.fit_cell(df, mm.VC_REDUCED)
        return mm.lrt(full, red)[1]

    p_hetero, p_homo = lrt_p(hetero), lrt_p(homo)
    assert p_hetero < 0.05, f'planted heterogeneity missed (p={p_hetero})'
    assert p_homo > p_hetero


@pytest.mark.slow
def test_channel_random_intercept_absorbs_channel_offsets():
    """The reason no normalization is needed: a large multiplicative per-contact
    gain is an additive shift in log space and must not move the slope.

    Tolerance is a fraction of the STANDARD ERROR, not an absolute epsilon. The
    estimator is invariant to these offsets in exact arithmetic, but the fit is an
    iterative optimum and shifting the data by SD 2.0 moves where the optimizer
    lands by ~1e-5. Against a standard error of ~2.5e-3 that is a 0.4% wobble;
    demanding 1e-6 absolute would be testing the optimizer's tolerance rather than
    the invariance the model is claimed to have.
    """
    df = synth_cell(n_subj=12, n_chan=4, n_epoch=25, beta=0.012, sd_chan=0.0, seed=4)
    shifted = df.copy()
    rng = np.random.default_rng(0)
    offsets = {uid: rng.normal(0, 2.0) for uid in shifted['channel_uid'].unique()}
    shifted['log10_power'] += shifted['channel_uid'].map(offsets)

    a, _ = mm.fit_cell(df)
    b, _ = mm.fit_cell(shifted)
    delta = abs(float(a.fe_params['NRS_within']) - float(b.fe_params['NRS_within']))
    assert delta < 0.02 * float(a.bse['NRS_within']), (
        f'offsets of SD 2.0 moved the slope by {delta:.2e}, '
        f'{delta / float(a.bse["NRS_within"]):.1%} of its SE')


# ============================================================================
# THE LRT
# ============================================================================

def test_mixture_p_is_half_the_chi2_tail():
    """The boundary correction, pinned. Not chi2(1)."""
    class FakeRes:
        def __init__(self, llf, exog):
            self.llf = llf
            self.model = type('M', (), {'exog': exog})()

    exog = np.arange(12.0).reshape(4, 3)
    stat, p = mm.lrt(FakeRes(100.0, exog), FakeRes(97.0, exog))
    assert stat == pytest.approx(6.0)
    assert p == pytest.approx(0.5 * stats.chi2.sf(6.0, 1))
    assert p < stats.chi2.sf(6.0, 1)          # strictly less conservative


def test_lrt_returns_p_one_for_a_non_positive_statistic():
    class FakeRes:
        def __init__(self, llf, exog):
            self.llf = llf
            self.model = type('M', (), {'exog': exog})()

    exog = np.eye(3)
    stat, p = mm.lrt(FakeRes(10.0, exog), FakeRes(10.5, exog))
    assert stat < 0 and p == 1.0


def test_lrt_refuses_mismatched_fixed_effects():
    """A REML comparison is legal ONLY with identical fixed effects."""
    class FakeRes:
        def __init__(self, llf, exog):
            self.llf = llf
            self.model = type('M', (), {'exog': exog})()

    with pytest.raises(mm.CellFitError, match='different fixed-effects'):
        mm.lrt(FakeRes(10.0, np.eye(3)), FakeRes(9.0, np.eye(4)))


# ============================================================================
# THE PERMUTATION
# ============================================================================

def test_permutation_relabels_a_whole_epoch():
    """Every channel row of an epoch must get the SAME new score -- the
    exchangeability rule the cluster test already uses."""
    df = synth_cell(n_subj=4, n_chan=3, n_epoch=8)
    out = mm.permute_within_subject(df, np.random.default_rng(0))
    per_epoch = out.groupby(['subject', 'epoch_id'])['NRS'].nunique()
    assert (per_epoch == 1).all()


def test_permutation_preserves_the_score_multiset_within_subject():
    df = synth_cell(n_subj=4, n_chan=3, n_epoch=8)
    out = mm.permute_within_subject(df, np.random.default_rng(1))
    for subject in df['subject'].unique():
        before = sorted(df[df['subject'] == subject]
                        .drop_duplicates('epoch_id')['NRS'])
        after = sorted(out[out['subject'] == subject]
                       .drop_duplicates('epoch_id')['NRS'])
        assert before == after


def test_permutation_preserves_submean_when_balanced():
    """The claim the docstring makes: with every channel present in every epoch,
    a within-subject shuffle leaves NRS_submean exactly where it was."""
    df = synth_cell(n_subj=5, n_chan=3, n_epoch=10)
    out = mm.permute_within_subject(df, np.random.default_rng(2))
    a = df.groupby('subject')['NRS_submean'].first()
    b = out.groupby('subject')['NRS_submean'].first()
    assert np.allclose(a.to_numpy(), b.to_numpy())


def test_permutation_actually_changes_the_within_component():
    df = synth_cell(n_subj=5, n_chan=2, n_epoch=12)
    out = mm.permute_within_subject(df, np.random.default_rng(3))
    assert not np.allclose(df['NRS_within'].to_numpy(), out['NRS_within'].to_numpy())


def test_permutation_is_reproducible_from_the_seed():
    df = synth_cell(n_subj=4, n_chan=2, n_epoch=10)
    a = mm.permute_within_subject(df, np.random.default_rng(7))['NRS'].to_numpy()
    b = mm.permute_within_subject(df, np.random.default_rng(7))['NRS'].to_numpy()
    assert np.array_equal(a, b)


def test_permutation_p_is_never_zero():
    """Same estimator as cluster_permutation.permutation_p, so the two analyses'
    p-values mean the same thing."""
    p, n = mm.permutation_p(10.0, np.zeros(99))
    assert p == pytest.approx(1 / 100)
    assert n == 99


def test_permutation_p_is_two_sided_by_magnitude():
    null = np.array([-5.0, -1.0, 0.0, 1.0, 5.0])
    p_pos, _ = mm.permutation_p(3.0, null)
    p_neg, _ = mm.permutation_p(-3.0, null)
    assert p_pos == p_neg


def test_permutation_p_ignores_failed_fits():
    """A shuffle whose fit failed is NaN and must not count in the denominator."""
    null = np.array([0.0, 1.0, np.nan, np.nan])
    p, n = mm.permutation_p(0.5, null)
    assert n == 2


# ============================================================================
# EXTRACTION
# ============================================================================

@pytest.mark.slow
def test_vcomp_is_read_by_name_not_position():
    """statsmodels sorts the variance component names; positional indexing into
    res.vcomp would silently swap `channel` and `subj_slope`."""
    df = synth_cell(n_subj=10, n_chan=3, n_epoch=15, seed=5)
    res, _ = mm.fit_cell(df)
    names = list(res.model.exog_vc.names)
    assert names == sorted(names), 'statsmodels no longer sorts; revisit vcomp_by_name'
    by_name = mm.vcomp_by_name(res)
    assert set(by_name) == {'channel', 'subj_int', 'subj_slope'}
    assert by_name['channel'] == pytest.approx(res.vcomp[names.index('channel')])


@pytest.mark.slow
def test_blups_are_one_row_per_subject_with_counts():
    df = synth_cell(n_subj=9, n_chan=3, n_epoch=12, beta=0.01, sd_slope=0.01, seed=6)
    res, _ = mm.fit_cell(df)
    rows = pd.DataFrame(mm.blup_rows(res, df, region='R', freq_bin_index=7))
    assert len(rows) == df['subject'].nunique()
    assert set(rows['subject']) == set(df['subject'])
    assert (rows['n_reports'] == 12).all()
    assert (rows['n_channels'] == 3).all()
    assert np.allclose(rows['subject_slope'],
                       float(res.fe_params['NRS_within']) + rows['blup_slope'])
    # BLUPs are shrunk deviations about the fixed effect: they average to ~0.
    assert abs(float(rows['blup_slope'].mean())) < 0.01


@pytest.mark.slow
def test_cell_record_has_the_full_output_schema():
    df = synth_cell(n_subj=10, n_chan=3, n_epoch=15, seed=8)
    full, wf = mm.fit_cell(df, mm.VC_FULL)
    red, wr = mm.fit_cell(df, mm.VC_REDUCED)
    rec = mm.cell_record(full, red, df, region='R', freq_bin_index=3,
                         bin_low_hz=19.7, bin_high_hz=22.0, fit_seconds=1.0,
                         warnings_full=wf, warnings_reduced=wr)
    for field in ('region', 'freq_bin_low', 'freq_bin_high', 'n_subjects',
                  'n_channels', 'n_rows', 'beta_nrs_within', 'se', 'z', 'p',
                  'beta_nrs_submean', 'se_submean', 'p_submean', 'var_subj_int',
                  'var_subj_slope', 'var_channel', 'var_resid', 'lrt_stat',
                  'p_lrt_mixture', 'converged', 'singular_flag',
                  'boundary_components', 'fit_seconds'):
        assert field in rec, field
    assert rec['n_subjects'] == 10
    assert rec['n_channels'] == 30
    assert rec['n_rows'] == len(df)


def test_failed_record_matches_the_success_schema():
    """A cell that could not be fitted becomes a ROW, never a silent drop -- and
    the row must have the same columns or the parquet write will be ragged."""
    df = synth_cell(n_subj=3, n_chan=2, n_epoch=5)
    rec = mm.failed_record('R', 3, 19.7, 22.0, 'too few subjects', df=df)
    assert rec['converged'] is False
    assert rec['error']
    assert np.isnan(rec['beta_nrs_within'])
    assert rec['n_subjects'] == 3


@pytest.mark.slow
def test_boundary_report_scales_slope_components_by_predictor_variance():
    """A raw slope variance looks tiny for arithmetic reasons alone -- NRS spans
    0-10 while log power spans a fraction of a decade."""
    df = synth_cell(n_subj=10, n_chan=3, n_epoch=15, sd_slope=0.02, seed=9)
    res, _ = mm.fit_cell(df)
    _, ratios = mm.boundary_report(res, df)
    vc = mm.vcomp_by_name(res)
    var_within = float(np.var(df['NRS_within'], ddof=0))
    assert ratios['subj_slope'] == pytest.approx(
        vc['subj_slope'] * var_within / res.scale)
    assert ratios['channel'] == pytest.approx(vc['channel'] / res.scale)
