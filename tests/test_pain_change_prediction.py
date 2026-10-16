"""The pain_change prediction: inversion of the d_pain weights, folds, and bins."""

import numpy as np
import pandas as pd
import pytest

from ieeg_ehr.analysis import pain_change_prediction as pcp


def _weights(**extra):
    return pd.DataFrame({'roi': ['A', 'A', 'B'], 'band': ['alpha', 'beta', 'alpha'],
                         'beta': [0.02, -0.01, 0.03], 'var_resid': [0.1, 0.2, 0.05],
                         'n_subjects': 9, 'n_pairs': 50, **extra})


def test_noise_free_cells_recover_delta_pain_under_both_weightings():
    w = _weights()
    dz = pd.DataFrame({'pair_id': ['p1', 'p1'], 't_min': [0, 1],
                       'A|alpha': [0.02 * 3, 0.02 * -2], 'A|beta': [-0.01 * 3, -0.01 * -2],
                       'B|alpha': [0.03 * 3, np.nan]})
    pairs = pd.DataFrame({'pair_id': ['p1'], 'subject': ['sub-001']})
    out = pcp.predict(dz, pairs, w)
    np.testing.assert_allclose(out['pred_raw_uniform'], [3, -2])
    np.testing.assert_allclose(out['pred_raw_ivw'], [3, -2])
    np.testing.assert_array_equal(out['n_cells_used'], [3, 2])
    np.testing.assert_allclose(out['sum_beta2'], [0.0014, 0.0005])


def test_inverse_variance_weights_the_low_variance_cell():
    w = _weights()
    dz = pd.DataFrame({'pair_id': ['p1'], 't_min': [0], 'A|alpha': [0.2],
                       'A|beta': [0.0], 'B|alpha': [0.0]})
    out = pcp.predict(dz, pd.DataFrame({'pair_id': ['p1'], 'subject': ['s']}), w)
    b, v = w['beta'].to_numpy(), w['var_resid'].to_numpy()
    assert out['pred_raw_uniform'].iloc[0] == pytest.approx(0.02 * 0.2 / (b ** 2).sum())
    assert out['pred_raw_ivw'].iloc[0] == pytest.approx(
        0.02 * 0.2 / 0.1 / (b ** 2 / v).sum())


def test_held_out_weights_score_each_subject_with_its_own_fold():
    folds = pd.concat([_weights(heldout_subject='s1'),
                       _weights(heldout_subject='s2').assign(beta=[0.04, -0.02, 0.06])])
    dz = pd.DataFrame({'pair_id': ['p1', 'p2'], 't_min': [0, 0],
                       'A|alpha': [0.02, 0.04], 'A|beta': [-0.01, -0.02],
                       'B|alpha': [0.03, 0.06]})
    pairs = pd.DataFrame({'pair_id': ['p1', 'p2'], 'subject': ['s1', 's2']})
    out = pcp.predict(dz, pairs, folds).set_index('pair_id')
    np.testing.assert_allclose(out['pred_raw_uniform'], [1.0, 1.0])
    with pytest.raises(ValueError, match='holds out'):
        pcp.predict(dz, pairs.assign(subject=['s1', 's3']), folds)


def test_bins_cover_every_score_change_and_count_both_levels():
    pairs = pd.DataFrame({'pair_id': list('abcdef'), 'subject': list('xxyyzz'),
                          'delta_pain': [-5, -1, 0, 0, 2, 7]})
    bins = pcp.assign_bins(pairs['delta_pain'])
    assert list(bins) == ['large decrease', 'small decrease', 'no change', 'no change',
                          'small increase', 'large increase']
    counts = pcp.count_table(pairs).set_index(['level', 'group'])
    assert counts.loc[('bin', 'no change'), 'n_subjects'] == 1
    assert counts.loc[('delta_pain', 0), 'n_pairs'] == 2


def test_loo_export_keeps_folds_and_drops_the_gate_fold(tmp_path, monkeypatch):
    cells = pd.DataFrame({
        'region': ['A'] * 3, 'freq': ['alpha'] * 3, 'dpain_beta': [0.02, 0.03, np.nan],
        'var_resid': [0.1, 0.1, np.nan], 'n_subjects': 9, 'n_pairs': 50,
        'converged': [True, True, False], 'heldout_subject': ['none', 's1', 's2']})
    monkeypatch.setattr(pcp.io, 'read_table', lambda path, on_stale: cells)
    (tmp_path / 'cells_loo.csv').touch()
    w = pcp.export_weights(tmp_path)
    assert list(w['heldout_subject']) == ['s1']
    assert w['beta'].tolist() == [0.03]


def test_dc_bin_edges():
    from ieeg_ehr.analysis import plot_pain_change_prediction as plots
    dc = pd.Series([-0.5, -0.35, -0.2, -0.0999, 0.0, 0.1, 0.35, 0.36])
    assert list(plots.dc_bin(dc)) == ['large fall', 'small fall', 'small fall', 'no change',
                                      'no change', 'small rise', 'small rise', 'large rise']


def test_exposure_steps_matches_the_model_dc(monkeypatch):
    """dc(t2) equals add_exposure's dC; dc(t1) = 0; a session without MAR drops out."""
    from ieeg_ehr.analysis import pain_change
    from ieeg_ehr.analysis import plot_pain_change_prediction as plots
    from ieeg_ehr.med_analysis import mme_curve
    start = pd.Timestamp('2000-01-01 08:00')
    admin = pd.DataFrame([{
        'subject': '001', 'session': '01', 'level2': 'Opioids', 'drug': 'OXYCODONE',
        'medication': 'OXYCODONE 5 MG PO TABS', 'route': 'Oral', 'dose': 10.0,
        'dose_unit': 'mg', 'session_start': start, 'session_end': start + pd.Timedelta(days=3),
        'taken_dt': start + pd.Timedelta(hours=30, minutes=20)}])
    loader = lambda: (admin.groupby(['subject', 'session'])['session_start'].first(),
                      mme_curve.opioid_doses(admin))
    monkeypatch.setattr(plots, 'load_exposure', loader)
    monkeypatch.setattr(pain_change, 'load_exposure', loader)
    h = lambda x: start + pd.Timedelta(hours=x)
    pairs = pd.DataFrame({'pair_id': ['a', 'b'], 'subject': ['sub-001', 'sub-002'],
                          'session': '01', 't1': [h(30), h(30)], 't2': [h(31), h(31)],
                          'gap_min': 60.0})
    steps, per_pair = plots.exposure_steps(pairs)
    assert list(per_pair['pair_id']) == ['a'] and set(steps['pair_id']) == {'a'}
    assert steps['t_min'].tolist() == list(range(61)) and steps['dc'].iloc[0] == 0.0
    model, _ = pain_change.add_exposure(
        pairs[['subject', 't1', 't2']].assign(subject=['001', '002'], session='01',
                                              subject_id=['sub-001', 'sub-002']))
    assert per_pair['dc_interval'].iloc[0] == pytest.approx(model['dC'].iloc[0])
    assert per_pair['e1_start_h'].iloc[0] == pytest.approx(30 - 5 / 60)


def test_adjusted_prediction_inverts_the_full_model():
    """d_z built from the model itself: adj recovers d_pain, raw does not."""
    w = _weights().assign(alpha=[0.05, -0.02, 0.01], gamma=[0.004, 0.002, -0.003],
                          delta=[-0.03, 0.01, 0.02])
    centering = pd.DataFrame({'roi': ['A', 'A', 'B'], 'band': ['alpha', 'beta', 'alpha'],
                              'subject': 's', 'pain_1_mean': [5.0, 5.5, 4.0]})
    pairs = pd.DataFrame({'pair_id': ['p1'], 'subject': ['s'], 'pain1': [7.0]})
    t_min, d_pain = np.array([20, 60, 120]), -3.0
    p1w = 7.0 - centering['pain_1_mean'].to_numpy()
    dz = (w['alpha'].to_numpy() + w['beta'].to_numpy() * d_pain
          + w['gamma'].to_numpy() * p1w + w['delta'].to_numpy() * t_min[:, None] / 60)
    frame = pd.DataFrame(dz, columns=['A|alpha', 'A|beta', 'B|alpha']).assign(
        pair_id='p1', t_min=t_min)
    out = pcp.predict(frame, pairs, w, centering)
    np.testing.assert_allclose(out['pred_adj_uniform'], d_pain)
    np.testing.assert_allclose(out['pred_adj_ivw'], d_pain)
    assert not np.allclose(out['pred_raw_uniform'], d_pain)
    assert list(out['extrapolated']) == [True, False, False]
