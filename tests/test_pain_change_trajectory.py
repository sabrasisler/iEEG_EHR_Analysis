"""Minute blocks of the pain_change sliding-window extraction against a direct window mean."""

from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd

from ieeg_ehr import config
from ieeg_ehr.features import build_pain_change_trajectory as traj


def _run_file(tmp_path, data):
    path = tmp_path / 'sub-000_ses-01_run-AAA_bipolar_psd_fullres.nwb'
    with h5py.File(path, 'w') as f:
        f[traj.DATA_PATH] = data
    return path


def test_window_means_match_nanmean_over_window_rows(tmp_path, monkeypatch):
    """Off-minute run start, chunk-crossing reads, NaNs and a channel absent from the run."""
    monkeypatch.setattr(config, 'PSD_FULLRES_HDF5_CHUNK_WINDOWS', 70)
    rng = np.random.default_rng(0)
    n_win, n_pairs, n_freq = 1500, 3, 4
    data = rng.normal(-10, 1, size=(n_win, n_pairs, n_freq)).astype(np.float32)
    data[200:230, 1, 2] = -np.inf
    run = traj.Run('AAA', _run_file(tmp_path, data), pd.Timestamp('2000-01-01 10:00:37'),
                   n_win, ['A', 'B', 'C'])
    session_channels = ['A', 'B', 'C', 'D']
    mean = rng.normal(-10, 0.1, size=(4, n_freq))
    sd = rng.uniform(0.5, 1.5, size=(4, n_freq))
    sess = SimpleNamespace(session_channels=session_channels, channels_by_run={},
                           mask=None)
    channels = ['A', 'B', 'D']
    blocks = traj.MinuteBlocks([run], sess, mean, sd, channels, np.arange(n_freq))

    t1 = pd.Timestamp('2000-01-01 10:08:00')
    s, cnt, v, s2 = blocks.window(t1 - pd.Timedelta(minutes=traj.WINDOW_MIN), 12)
    zmean, frac = traj.window_means(s, cnt, v, max_excluded_frac=0.5)

    dts = run.start + pd.to_timedelta(np.arange(n_win), unit='s')
    z = np.asarray(data, dtype=float)
    z[~np.isfinite(z)] = np.nan
    z = (z[:, [0, 1]] - mean[[0, 1]]) / sd[[0, 1]]
    for k in range(zmean.shape[0]):
        t = t1 + pd.Timedelta(minutes=k)
        rows = (dts >= t - pd.Timedelta(minutes=traj.WINDOW_MIN)) & (dts < t)
        assert rows.sum() == 300
        np.testing.assert_allclose(zmean[k, :2], np.nanmean(z[rows], axis=0), rtol=1e-12)
        np.testing.assert_allclose(s2[k:k + traj.WINDOW_MIN, :2].sum(axis=0),
                                   np.nansum(z[rows] ** 2, axis=0), rtol=1e-10)
        assert np.isnan(zmean[k, 2]).all() and frac[k, 2] == 1.0
        np.testing.assert_array_equal(frac[k, :2], 0.0)


def test_uncovered_minutes_count_as_excluded(tmp_path):
    """A window that runs past the end of the recording loses channels above the threshold."""
    n_win = 600
    data = np.zeros((n_win, 1, 2), dtype=np.float32)
    run = traj.Run('AAA', _run_file(tmp_path, data), pd.Timestamp('2000-01-01 10:00:00'),
                   n_win, ['A'])
    sess = SimpleNamespace(session_channels=['A'], channels_by_run={}, mask=None)
    blocks = traj.MinuteBlocks([run], sess, np.zeros((1, 2)), np.ones((1, 2)), ['A'],
                               np.arange(2))
    s, cnt, v, _ = blocks.window(pd.Timestamp('2000-01-01 10:05:00'), 10)
    _, frac = traj.window_means(s, cnt, v, max_excluded_frac=0.5)
    np.testing.assert_allclose(frac[:, 0], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])


def _nwb(path, start, data, names):
    with h5py.File(path, 'w') as f:
        f['session_start_time'] = start
        f[traj.DATA_PATH] = data
        f[traj.STARTING_TIME_PATH] = 0.0
        f[traj.STARTING_TIME_PATH].attrs['rate'] = 1.0
        f[traj.ELECTRODE_LOCATION_PATH] = names


def test_runs_sharing_boundary_seconds_count_each_second_once(tmp_path, monkeypatch):
    """The earlier run keeps the shared seconds, so no minute holds more than 60 windows."""
    monkeypatch.setattr(config, 'bipolar_psd_fullres_nwb_path',
                        lambda s, ses, run: tmp_path / f'sub-{s}_ses-{ses}_run-{run}_x.nwb')
    _nwb(tmp_path / 'sub-000_ses-01_run-AAA_x.nwb', '2000-01-01T10:00:00-08:00',
         np.zeros((604, 1, 2), dtype=np.float32), ['A'])
    _nwb(tmp_path / 'sub-000_ses-01_run-BBB_x.nwb', '2000-01-01T10:10:00-08:00',
         np.ones((600, 1, 2), dtype=np.float32), ['A'])
    runs = traj.session_runs('000', '01')
    assert [r.first_row for r in runs] == [0, 4]
    sess = SimpleNamespace(session_channels=['A'], channels_by_run={}, mask=None)
    blocks = traj.MinuteBlocks(runs, sess, np.zeros((1, 2)), np.ones((1, 2)), ['A'],
                               np.arange(2))
    s, cnt, v, _ = blocks.window(pd.Timestamp('2000-01-01 10:09:00'), 3)
    np.testing.assert_array_equal(v[:, 0], [60, 60, 60])
    # 10:10 holds four windows of run AAA (z = 0) and 56 of run BBB (z = 1).
    np.testing.assert_array_equal(s[1, 0], [56.0, 56.0])


def test_a_run_inside_earlier_runs_adds_nothing(tmp_path, monkeypatch):
    """A recording-system switch: the short run lies wholly inside an earlier long one."""
    monkeypatch.setattr(config, 'bipolar_psd_fullres_nwb_path',
                        lambda s, ses, run: tmp_path / f'sub-{s}_ses-{ses}_run-{run}_x.nwb')
    _nwb(tmp_path / 'sub-000_ses-01_run-EA1_x.nwb', '2000-01-01T10:00:00-08:00',
         np.zeros((1200, 1, 2), dtype=np.float32), ['A'])
    _nwb(tmp_path / 'sub-000_ses-01_run-SA1_x.nwb', '2000-01-01T10:05:00-08:00',
         np.ones((300, 1, 2), dtype=np.float32), ['A'])
    _nwb(tmp_path / 'sub-000_ses-01_run-EA2_x.nwb', '2000-01-01T10:19:00-08:00',
         np.ones((600, 1, 2), dtype=np.float32), ['A'])
    runs = traj.session_runs('000', '01')
    assert [(r.run_id, r.first_row) for r in runs] == [('EA1', 0), ('SA1', 300), ('EA2', 60)]
    sess = SimpleNamespace(session_channels=['A'], channels_by_run={}, mask=None)
    blocks = traj.MinuteBlocks(runs, sess, np.zeros((1, 2)), np.ones((1, 2)), ['A'],
                               np.arange(2))
    s, cnt, v, _ = blocks.window(pd.Timestamp('2000-01-01 10:04:00'), 18)
    np.testing.assert_array_equal(v[:, 0], 60)
    # SA1 (z = 1) never enters. EA2 (z = 1) supplies 10:20 and 10:21.
    np.testing.assert_array_equal(s[:, 0, 0], [0.0] * 16 + [60.0, 60.0])
