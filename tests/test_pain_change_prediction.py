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
    np.testing.assert_allclose(out['pred_uniform'], [3, -2])
    np.testing.assert_allclose(out['pred_ivw'], [3, -2])
    np.testing.assert_array_equal(out['n_cells_used'], [3, 2])
    np.testing.assert_allclose(out['sum_beta2'], [0.0014, 0.0005])


def test_inverse_variance_weights_the_low_variance_cell():
    w = _weights()
    dz = pd.DataFrame({'pair_id': ['p1'], 't_min': [0], 'A|alpha': [0.2],
                       'A|beta': [0.0], 'B|alpha': [0.0]})
    out = pcp.predict(dz, pd.DataFrame({'pair_id': ['p1'], 'subject': ['s']}), w)
    b, v = w['beta'].to_numpy(), w['var_resid'].to_numpy()
    assert out['pred_uniform'].iloc[0] == pytest.approx(0.02 * 0.2 / (b ** 2).sum())
    assert out['pred_ivw'].iloc[0] == pytest.approx(
        0.02 * 0.2 / 0.1 / (b ** 2 / v).sum())


def test_held_out_weights_score_each_subject_with_its_own_fold():
    folds = pd.concat([_weights(heldout_subject='s1'),
                       _weights(heldout_subject='s2').assign(beta=[0.04, -0.02, 0.06])])
    dz = pd.DataFrame({'pair_id': ['p1', 'p2'], 't_min': [0, 0],
                       'A|alpha': [0.02, 0.04], 'A|beta': [-0.01, -0.02],
                       'B|alpha': [0.03, 0.06]})
    pairs = pd.DataFrame({'pair_id': ['p1', 'p2'], 'subject': ['s1', 's2']})
    out = pcp.predict(dz, pairs, folds).set_index('pair_id')
    np.testing.assert_allclose(out['pred_uniform'], [1.0, 1.0])
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
