"""The all_pain_epochs z-score view and the pain_change per-channel fit."""

import numpy as np
import pandas as pd
import pytest

from ieeg_ehr.analysis import pain_change
from ieeg_ehr.views import axes
from ieeg_ehr.views.view_config import ViewConfig


def _epochs(rng, n_epochs=6, n_win=40, n_pairs=3, n_freq=4):
    blocks = [rng.normal(loc=-10 + e, scale=1 + 0.1 * e, size=(n_win, n_pairs, n_freq))
              for e in range(n_epochs)]
    blocks[2][5:9, 1, :] = np.nan            # masked windows
    return blocks


def _zscored(blocks):
    acc = axes.BaselineAccumulator(*blocks[0].shape[1:])
    for b in blocks:
        acc.update(b)
    mu, sd = acc.finalize()
    return [axes.normalize(b, mu, sd, 'zscore_vs_baseline') for b in blocks], mu, sd


def test_all_epoch_baseline_standardizes_every_window():
    """Pooled over all windows of all epochs, each channel x freq has mean 0 and SD 1."""
    z, _, _ = _zscored(_epochs(np.random.default_rng(0)))
    pooled = np.concatenate(z, axis=0)
    np.testing.assert_allclose(np.nanmean(pooled, axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.nanstd(pooled, axis=0, ddof=1), 1.0, atol=1e-12)


def test_epoch_mean_of_window_z_is_standardized_epoch_mean():
    blocks = _epochs(np.random.default_rng(1))
    z, mu, sd = _zscored(blocks)
    for b, zb in zip(blocks, z):
        np.testing.assert_allclose(axes.epoch_mean(zb), (axes.epoch_mean(b) - mu) / sd,
                                   atol=1e-12)


def test_baseline_epoch_filter():
    assert axes.baseline_epoch_filter('all_pain_epochs') is None
    assert axes.baseline_epoch_filter('zero_pain_epochs') is axes.is_baseline_epoch
    with pytest.raises(ValueError):
        axes.baseline_epoch_filter('whole_session')


def test_scheme_code_names_a_non_default_baseline_only():
    assert ViewConfig(mask_label='m').scheme_code == 'zscore-relpain'
    assert (ViewConfig(mask_label='m', baseline='all_pain_epochs').scheme_code
            == 'zscoreallep-relpain')


def test_cell_frame_centres_pain_1_within_subject():
    index = pd.DataFrame({'subject': ['a', 'a', 'b', 'b'], 'channel_uid': list('wxyz'),
                          'pair_id': [0, 1, 2, 3], 'region': 'R'})
    pairs = pd.DataFrame({'pair_id': [0, 1, 2, 3], 'd_pain': [1., -1, 2, 0],
                          'pain_1': [2., 4, 7, 9], 'gap_h': 1.0, 'dC': 0.0, 'Cbar': 0.0})
    values = np.array([[0.1], [np.nan], [0.3], [0.4]])
    df = pain_change.cell_frame(index, values, 0, pairs)
    assert list(df['pair_id']) == [0, 2, 3]                 # non-finite d_z dropped
    np.testing.assert_allclose(df['pain_1_within'], [0.0, -1.0, 1.0])


@pytest.mark.filterwarnings('ignore')
def test_fit_cell_record_recovers_d_pain():
    rng = np.random.default_rng(4)
    rows = []
    for s in range(12):
        slope = 0.05 + rng.normal(scale=0.01)
        for p in range(15):
            d_pain = float(rng.integers(-4, 5))
            for c in range(3):
                rows.append({'subject': f's{s}', 'channel_uid': f's{s}|{c}',
                             'pair_id': s * 100 + p, 'd_pain': d_pain,
                             'pain_1_within': rng.normal(), 'gap_h': rng.uniform(0.5, 4),
                             'd_z': slope * d_pain + rng.normal(scale=0.1)})
    rec = pain_change.fit_cell_record(pd.DataFrame(rows), {'region': 'R', 'freq': 'alpha'},
                                      min_subjects=8, model='painchange')
    assert rec['error'] == '' and rec['fit_status'] in ('ok', 'boundary', 'warn')
    assert abs(rec['dpain_beta'] - 0.05) < 0.01 and rec['dpain_p'] < 1e-6


@pytest.mark.filterwarnings('ignore')
def test_opioid_model_recovers_the_interaction():
    rng = np.random.default_rng(5)
    rows = []
    for s in range(12):
        for p in range(20):
            d_pain, cbar = float(rng.integers(-4, 5)), rng.normal()
            dc = rng.normal()
            for c in range(3):
                rows.append({'subject': f's{s}', 'channel_uid': f's{s}|{c}',
                             'pair_id': s * 100 + p, 'd_pain': d_pain, 'dC': dc,
                             'Cbar': cbar, 'pain_1_within': rng.normal(),
                             'gap_h': rng.uniform(0.5, 4),
                             'd_z': (0.05 + 0.03 * cbar) * d_pain + 0.1 * dc
                                    + rng.normal(scale=0.1)})
    rec = pain_change.fit_cell_record(pd.DataFrame(rows), {'region': 'R', 'freq': 'alpha'},
                                      min_subjects=8, model='opioid')
    assert abs(rec['dpain_beta'] - 0.05) < 0.01
    assert abs(rec['dpain_x_cbar_beta'] - 0.03) < 0.01 and rec['dpain_x_cbar_p'] < 1e-6
    assert abs(rec['dc_beta'] - 0.1) < 0.02


def test_below_floor_is_floor_not_failed():
    df = pd.DataFrame({'subject': ['a'] * 5, 'channel_uid': 'a|0', 'pair_id': range(5),
                       'd_pain': [1., 2, 3, 4, 5], 'd_z': 0.0})
    rec = pain_change.fit_cell_record(df, {}, min_subjects=8, model='painchange')
    assert rec['fit_status'] == 'floor' and not rec['converged']


def test_fit_status_order():
    ok = {'subj_int': 0.1, 'subj_dpain': 0.2}
    assert pain_change.fit_status(False, ok, []) == ('failed', '')
    assert (pain_change.fit_status(True, {**ok, 'subj_int': 0.0}, [])
            == ('boundary', 'subj_int'))
    hessian = ['The Hessian matrix at the estimated parameter values is not positive definite.']
    assert pain_change.fit_status(True, ok, hessian) == ('warn', '')
    # statsmodels' absolute-0.01 boundary message alone is not a warning
    assert (pain_change.fit_status(True, ok, ['The MLE may be on the boundary of the '
                                              'parameter space.']) == ('ok', ''))


def test_fit_summary_shares_exclude_the_floor():
    cells = pd.DataFrame({'fit_status': ['ok'] * 6 + ['failed'] * 2 + ['boundary'] * 2
                          + ['floor'] * 5})
    s = pain_change.fit_summary(cells)
    assert list(s.index) == list(pain_change.FIT_STATUSES)
    assert s.at['failed', 'pct_of_fitted'] == 20.0 and s.at['ok', 'n_cells'] == 6
    assert np.isnan(s.at['floor', 'pct_of_fitted'])


def test_tokens():
    vc = ViewConfig(mask_label='m', baseline='all_pain_epochs', freq='fullres',
                    region='none', roi_scheme='roi_v2_ofc_ins')
    assert (pain_change.data_token(vc, 'paper_bands_6_hg200', 24)
            == 'zscoreallep-paperbands6hg200-roiv2ofcins-nofirst24h')
    assert (pain_change.data_token(vc, 'paper_bands_6_hg200', None)
            == 'zscoreallep-paperbands6hg200-roiv2ofcins')
    assert pain_change.run_token('opioid', 'sm') == 'opioid_mme3_sm'
    assert pain_change.run_token('opioid', 'lme4') == 'opioid_mme3_lme4'
    assert pain_change.run_token('painchange', 'sm') == 'painchange_sm'


def test_add_exposure(monkeypatch):
    """C in typical doses, no-MAR sessions dropped, early pairs dropped by first-epoch start."""
    from ieeg_ehr.config import opioid_mme
    start = pd.Timestamp('2000-01-01 08:00')
    admin = pd.DataFrame([{
        'subject': '001', 'session': '01', 'level2': 'Opioids', 'drug': 'MORPHINE',
        'medication': 'MORPHINE 4 MG/ML INJECTION', 'route': 'Intravenous',
        'dose': 4.0, 'dose_unit': 'mg', 'session_start': start,
        'session_end': start + pd.Timedelta(days=3),
        'taken_dt': start + pd.Timedelta(hours=30)}])
    monkeypatch.setattr(pain_change.load, 'load_administrations', lambda: admin)
    h = lambda x: start + pd.Timedelta(hours=x)
    pairs = pd.DataFrame({
        'subject': ['001', '001', '002'], 'session': '01',
        'subject_id': ['sub-001', 'sub-001', 'sub-002'],
        't1': [h(2), h(30), h(30)], 't2': [h(3), h(31), h(31)]})
    out, report = pain_change.add_exposure(pairs, exclude_first_h=24)
    assert report['no_mar_subjects'] == ['sub-002'] and report['n_pairs_early'] == 1
    r = out.iloc[0]
    assert r['C_1'] == 0.0                       # the dose at t1 follows epoch 1
    assert 0 < r['Cbar'] < r['C_2']   # morphine's effect site still rises at t2 (peak 94 min)
    assert r['dC'] == pytest.approx(r['C_2'] - r['C_1'])
    mme = 4.0 * opioid_mme.MME_FACTOR[('MORPHINE', 'IV')].factor
    assert r['Cbar'] * opioid_mme.TYPICAL_DOSE_MME < mme


def test_add_exposure_first_post_moves_c1(monkeypatch):
    """--first-epoch-post: C_1 is the mean over [t1, t1 + 5), so a dose at t1 counts."""
    start = pd.Timestamp('2000-01-01 08:00')
    admin = pd.DataFrame([{
        'subject': '001', 'session': '01', 'level2': 'Opioids', 'drug': 'FENTANYL',
        'medication': 'FENTANYL (PF) 50 MCG/ML INJECTION', 'route': 'Intravenous',
        'dose': 50.0, 'dose_unit': 'mcg', 'session_start': start,
        'session_end': start + pd.Timedelta(days=3),
        'taken_dt': start + pd.Timedelta(hours=30)}])
    monkeypatch.setattr(pain_change.load, 'load_administrations', lambda: admin)
    h = lambda x: start + pd.Timedelta(hours=x)
    pairs = pd.DataFrame({'subject': ['001'], 'session': '01', 'subject_id': ['sub-001'],
                          't1': [h(30)], 't2': [h(31)]})
    pre, _ = pain_change.add_exposure(pairs)
    post, _ = pain_change.add_exposure(pairs, first_post=True)
    assert pre['C_1'].iloc[0] == 0.0
    assert post['C_1'].iloc[0] > 0.0
    assert post['C_2'].iloc[0] == pytest.approx(pre['C_2'].iloc[0])
    assert post['e1_start_h'].iloc[0] == pytest.approx(30.0)


def test_change_matrix_takes_the_earlier_epoch_from_first():
    cols = ['f000']
    table = pd.DataFrame({'channel': ['A', 'A'], 'epoch_id': [1, 2], 'f000': [1.0, 5.0]})
    first = pd.DataFrame({'channel': ['A', 'A'], 'epoch_id': [1, 2], 'f000': [3.0, 9.0]})
    pairs = pd.DataFrame({'e1': [1], 'e2': [2]})
    assert pain_change.change_matrix(table, pairs, cols)['A'][0, 0] == 4.0
    assert pain_change.change_matrix(table, pairs, cols, first=first)['A'][0, 0] == 2.0


def test_post_baseline_matches_numpy():
    from ieeg_ehr.features.build_pain_change_post_z import post_baseline
    rng = np.random.default_rng(0)
    z = rng.normal(0.3, 1.7, size=(400, 2, 3))
    m, s = post_baseline(np.full((2, 3), 400), z.sum(axis=0), (z * z).sum(axis=0), 2)
    assert np.allclose(m, z.mean(axis=0))
    assert np.allclose(s, z.std(axis=0, ddof=1))
    m, s = post_baseline(np.ones((2, 3), dtype=int), z[0], z[0] ** 2, 2)
    assert np.isnan(s).all()


def test_tokens_for_first_post():
    from ieeg_ehr.views import view_config
    vc = view_config.ViewConfig(baseline='post_report_epochs', freq='fullres',
                                region='none', mask_level='bipolar', mask_label='x')
    token = pain_change.data_token(vc, 'paper_bands_6_hg200', None, first_post=True)
    assert token.startswith('zscorepostep-') and token.endswith('-e1post0to5')
    assert pain_change.run_token('opioid-dc-pairint', 'lme4', first_post=True) == \
        'opioid-dc-pairint_mme4_lme4'


def _synthetic_rows(seed=6, n_subjects=12):
    """A Rows of two regions x one band whose d_z follows the opioid model."""
    rng = np.random.default_rng(seed)
    pairs, index, values = [], [], []
    for s in range(n_subjects):
        intercept, slope = rng.normal(scale=0.1), rng.normal(scale=0.03)
        for p in range(20):
            pid = s * 100 + p
            d_pain, cbar, dc = float(rng.integers(-4, 5)), rng.normal(), rng.normal()
            pairs.append({'pair_id': pid, 'd_pain': d_pain, 'pain_1': float(rng.integers(0, 11)),
                          'gap_h': rng.uniform(0.5, 4), 'dC': dc, 'Cbar': cbar})
            for region in ('R1', 'R2'):
                for c in range(2):
                    index.append({'subject': f'sub-{s}', 'channel_uid': f'{s}|{region}|{c}',
                                  'pair_id': pid, 'region': region})
                    values.append([intercept + (0.05 + slope + 0.03 * cbar) * d_pain
                                   + 0.1 * dc + rng.normal(scale=0.1)])
    return pain_change.Rows(pd.DataFrame(index), np.array(values), pd.DataFrame(pairs),
                            ['R1', 'R2'], ['alpha'], {'alpha': (8.0, 12.0)}, None, [],
                            [], {}, {}, 0.0)


@pytest.mark.filterwarnings('ignore')
@pytest.mark.skipif(__import__('shutil').which('Rscript') is None, reason='needs Rscript')
def test_lme4_engine_matches_statsmodels():
    """Same frames, same model: the two engines agree on the fixed effects."""
    rows = _synthetic_rows()
    lme4 = pd.DataFrame(pain_change.lme4_records(rows, 8, 'opioid', 2)).set_index('region')
    sm = pd.DataFrame([pain_change.fit_cell_record(df, meta, 8, 'opioid', with_lrt=False)
                       for meta, df in pain_change.cell_frames(rows)]).set_index('region')
    for col in ('dpain_beta', 'dpain_x_cbar_beta', 'dc_beta', 'dpain_se'):
        np.testing.assert_allclose(lme4[col], sm.loc[lme4.index, col], rtol=1e-3)
    assert lme4['converged'].all() and set(lme4['fit_status']) <= {'ok', 'boundary'}


def test_sm_refuses_a_pair_model():
    with pytest.raises(ValueError, match='lme4'):
        pain_change.fit_cell_record(pd.DataFrame(), {}, 8, 'opioid-pairint')


@pytest.mark.filterwarnings('ignore')
@pytest.mark.skipif(__import__('shutil').which('Rscript') is None, reason='needs Rscript')
def test_pair_intercept_widens_pair_level_errors():
    """Noise shared by a pair's channels inflates a pair-level term's z without
    `(1 | pair_id)`. With it, the pair variance is recovered and the SE widens."""
    rng = np.random.default_rng(7)
    rows = _synthetic_rows()
    shared = {pid: rng.normal(scale=0.3) for pid in rows.pairs['pair_id']}
    rows.values[:, 0] += rows.index['pair_id'].map(shared).to_numpy()
    plain = pd.DataFrame(pain_change.lme4_records(rows, 8, 'opioid', 2)).set_index('region')
    pair = pd.DataFrame(pain_change.lme4_records(rows, 8, 'opioid-pairint', 2)).set_index('region')
    assert (pair['dpain_x_cbar_se'] > 1.3 * plain['dpain_x_cbar_se']).all()
    np.testing.assert_allclose(pair['var_pair'], 0.09, rtol=0.35)
    assert plain['var_pair'].isna().all()


def test_opioid_dc_model_swaps_cbar_for_dc():
    terms = pain_change.model_terms('opioid-dc-pairint')
    assert terms == ['d_pain', 'dC', 'd_pain:dC', 'pain_1_within', 'gap_h']
    assert 'Cbar' not in terms and pain_change.uses_exposure('opioid-dc-pairint')
    assert pain_change.run_token('opioid-dc-pairint', 'lme4') == 'opioid-dc-pairint_mme3_lme4'


def test_epoch_autonomic_mean_hr_and_ln_rmssd():
    t0 = np.datetime64('2000-01-01T00:00:00', 'ns')
    sec = np.timedelta64(1, 's')
    hr = pd.DataFrame({'time': t0 + np.arange(600) * sec,
                       'hr_bpm': np.r_[np.full(300, 60.), np.full(300, 90.)]})
    hr = hr.drop(index=range(300, 450))            # half of the second epoch lost to quality
    # NN alternating 0.8 / 0.9 s: every successive difference is 100 ms, so RMSSD = 100 ms.
    nn = np.tile([0.8, 0.9], 400)
    t = t0 + (np.r_[0, np.cumsum(nn)] * 1e9).astype('timedelta64[ns]')
    beats = pd.DataFrame({'run': 'A', 'time': t, 'quality': 1.0})
    diffs = pain_change.successive_differences(beats)
    np.testing.assert_allclose(diffs['d2_ms2'], 100.0 ** 2)
    starts = np.array([t0, t0 + 300 * sec])
    mean_hr, ln_rmssd = pain_change.epoch_autonomic(hr, diffs, starts, starts + 300 * sec)
    np.testing.assert_allclose(mean_hr, [60, 90])
    np.testing.assert_allclose(ln_rmssd, np.log(100.0))
    early = starts - 900 * sec
    assert np.isnan(pain_change.epoch_autonomic(hr, diffs, early, early + 300 * sec)).all()


def test_successive_differences_skip_bad_beats_and_jumps():
    t0 = np.datetime64('2000-01-01T00:00:00', 'ns')
    nn = [0.8, 0.8, 0.8, 1.2, 0.8, 0.8]            # 0.8 -> 1.2 jumps 50%: skipped
    t = t0 + (np.r_[0, np.cumsum(nn)] * 1e9).astype('timedelta64[ns]')
    q = np.ones(len(t)); q[-1] = 0.1               # the last beat fails quality
    d = pain_change.successive_differences(pd.DataFrame({'run': 'A', 'time': t, 'quality': q}))
    assert len(d) == 2 and (d['d2_ms2'] < 1e-6).all()
