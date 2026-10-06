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
                          'pain_1': [2., 4, 7, 9], 'gap_h': 1.0})
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
                                      min_subjects=8)
    assert rec['error'] == ''
    assert abs(rec['dpain_beta'] - 0.05) < 0.01 and rec['dpain_p'] < 1e-6
