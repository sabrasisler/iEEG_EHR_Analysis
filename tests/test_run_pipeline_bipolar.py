"""Unit tests for the continuous native-grid PSD extractor.

Every failure pinned here would be SILENT on the cluster: a block one window
short, a block read off by a hop, a variance window split across blocks, or a
transposed write all produce plausible spectra with wrong numbers.
"""

from datetime import datetime, timezone
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest
from pynwb import NWBFile, NWBHDF5IO
from pynwb.ecephys import ElectricalSeries
from scipy import signal

from ieeg_ehr import config
from ieeg_ehr.config import paths
from ieeg_ehr.features import build_pain_epoch_fullres_psd as fullres
from ieeg_ehr.preprocessing import bipolar_reref
from ieeg_ehr.preprocessing import run_pipeline_bipolar as rpb

CHUNK = config.PSD_FULLRES_HDF5_CHUNK_WINDOWS
CONTACTS = ['LA1', 'LA2', 'LA3', 'LB1', 'LB2']


def _fake_run(x, sfreq):
    """A RawRun over an in-memory trace, with no NWB behind it."""
    raw = rpb.RawRun.__new__(rpb.RawRun)
    raw.raw_path = SimpleNamespace(name='synthetic')
    raw.series = SimpleNamespace(data=x, conversion=1.0)
    raw.sfreq = sfreq
    raw.elec_indices = np.arange(x.shape[1])
    raw.pairs = [{'anode_idx': i, 'cathode_idx': i + 1, 'location': f'p{i}',
                  'anode_location': f'c{i}', 'cathode_location': f'c{i + 1}'}
                 for i in range(x.shape[1] - 1)]
    raw.nperseg, raw.noverlap, raw.hop = fullres.window_geometry(sfreq)
    raw.n_samples = x.shape[0]
    raw.n_windows = rpb.n_windows(raw.n_samples, raw.nperseg, raw.noverlap)
    raw.freq_sl, raw.freqs = fullres.freq_slice(fullres.analytic_freqs(sfreq, raw.nperseg))
    return raw


def _trace(sfreq, seconds, n_ch=3, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((int(round(seconds * sfreq)), n_ch)) * 1e-5


@pytest.mark.parametrize('sfreq', [500.0, 1000.0])
@pytest.mark.parametrize('seconds', [1.5, 2.0, 2.7, 301.0, 650.3])
def test_n_windows_matches_scipy(sfreq, seconds):
    x = _trace(sfreq, seconds, n_ch=1)[:, 0]
    nperseg, noverlap, _ = fullres.window_geometry(sfreq)
    expected = 0
    if len(x) >= nperseg:
        expected = signal.spectrogram(x, fs=sfreq, nperseg=nperseg, noverlap=noverlap)[2].shape[1]
    assert rpb.n_windows(len(x), nperseg, noverlap) == expected


@pytest.mark.parametrize('sfreq', [500.0, 1000.0])
def test_blocks_reproduce_the_whole_run_spectrogram_bit_exactly(sfreq):
    """The extractor's core claim. Three blocks, the last one short, against one
    whole-run spectrogram per bipolar channel, the retired pipeline's shape."""
    x = _trace(sfreq, 650.3)
    raw = _fake_run(x, sfreq)
    bipolar = bipolar_reref.rereference(x.astype(np.float32), raw.elec_indices, raw.pairs)

    got = []
    for w0 in rpb.block_starts(raw.n_windows):
        w1 = min(w0 + CHUNK, raw.n_windows)
        _, cube = raw.linear_block(raw.bipolar_block(*raw.block_span(w0, w1)), w1 - w0, None)
        got.append(cube)
    got = np.concatenate(got, axis=1)

    assert got.shape[1] == raw.n_windows
    for ch in range(bipolar.shape[1]):
        _, _, whole = signal.spectrogram(
            bipolar[:, ch], fs=sfreq, nperseg=raw.nperseg, noverlap=raw.noverlap,
            window='hann', detrend='constant', scaling='density', mode='psd')
        assert np.array_equal(got[:, :, ch], whole), f'channel {ch} not bit-exact'


@pytest.mark.parametrize('seconds', [650.3, 600.0, 601.0, 899.9])
def test_block_variance_concatenates_to_the_whole_run_metric(seconds):
    sfreq = 500.0
    x = _trace(sfreq, seconds)
    raw = _fake_run(x, sfreq)
    bipolar = bipolar_reref.rereference(x.astype(np.float32), raw.elec_indices, raw.pairs)
    whole = bipolar_reref.compute_variance_windows(
        bipolar, sfreq, window_sec=config.BIPOLAR_VARIANCE_WINDOW_SEC)

    starts, metric = [], []
    for w0 in rpb.block_starts(raw.n_windows):
        w1 = min(w0 + CHUNK, raw.n_windows)
        s, m = rpb.block_variance(raw, raw.bipolar_block(*raw.block_span(w0, w1)), w0, w1)
        starts.append(s)
        metric.append(m)

    assert np.array_equal(np.concatenate(starts), whole['window_start'])
    assert np.array_equal(np.concatenate(metric), whole['metric_value'])


@pytest.fixture
def registry(tmp_path, monkeypatch):
    rows = [('sub-001', 'ses-01', f'run-A{i}', 3000.0) for i in range(5)]
    rows += [('sub-002', 'ses-01', f'run-B{i}', 1000.0) for i in range(3)]
    reg = pd.DataFrame(rows, columns=['sub_id', 'ses_id', 'run_id', 'raw_file_size_mb'])
    reg['raw_file_path'] = [f'/raw/{r}.nwb' for r in reg['run_id']]
    path = tmp_path / 'registry.csv'
    reg.to_csv(path, index=False)
    monkeypatch.setattr(config, 'FILE_REGISTRY_CSV', str(path))
    return reg


def test_plan_covers_every_run_once_and_closes_at_the_budget(registry):
    tasks = rpb.plan(['001', '002'], budget_gb=5.0)
    flat = [t for task in tasks for t in task]
    assert sorted(flat) == sorted(rpb._token(r) for r in registry.itertuples())
    assert [len(t) for t in tasks] == [2, 2, 3, 1]


def test_plan_whole_subjects_never_splits_a_subject(registry):
    tasks = rpb.plan(['001', '002'], budget_gb=5.0, whole_subjects=True)
    assert [{t.split(':')[0] for t in task} for task in tasks] == [{'001'}, {'002'}]


def test_variance_mode_refuses_a_task_that_splits_a_subject(registry):
    with pytest.raises(ValueError, match='--whole-subjects'):
        rpb.run_task(['001:01:A0', '001:01:A1'], variance_metrics=True)


def _write_raw_nwb(path, x, sfreq):
    nwb = NWBFile(session_description='synthetic', identifier='synthetic',
                  session_start_time=datetime(2001, 1, 1, tzinfo=timezone.utc))
    device = nwb.create_device(name='dev')
    group = nwb.create_electrode_group(name='g', description='g', location='brain',
                                       device=device)
    for name in CONTACTS:
        nwb.add_electrode(location=name, group=group)
    region = nwb.create_electrode_table_region(list(range(len(CONTACTS))), 'all')
    nwb.add_acquisition(ElectricalSeries(name='ElectricalSeries_sEEG', data=x,
                                         electrodes=region, rate=sfreq))
    with NWBHDF5IO(str(path), 'w') as io_out:
        io_out.write(nwb)


def test_extract_run_round_trips_through_nwb(tmp_path, monkeypatch):
    """End to end on a synthetic raw NWB: the empty-dataset allocation, the
    chunked h5py fill, the transpose, the log and the sidecar."""
    sfreq = 500.0
    x = _trace(sfreq, 650.3, n_ch=len(CONTACTS))
    raw_path = tmp_path / 'raw.nwb'
    _write_raw_nwb(raw_path, x, sfreq)
    monkeypatch.setattr(paths, 'BIPOLAR_PSD_FULLRES_ROOT', tmp_path / 'bipolar_psd')

    sig = rpb.extract_run('001', '01', 'X0', raw_path)
    out = config.bipolar_psd_fullres_nwb_path('001', '01', 'X0')
    assert out.exists() and not out.with_name(out.stem + '.partial.nwb').exists()
    assert sig == (('LA1', 'LA2'), ('LA2', 'LA3'), ('LB1', 'LB2'))

    with rpb.RawRun(raw_path) as raw, h5py.File(out, 'r') as f:
        stored = f[rpb.DATA_PATH][:]
        names = list(f[rpb.ELECTRODE_LOCATION_PATH].asstr()[:])
        chunks = f[rpb.DATA_PATH].chunks
        bipolar = raw.bipolar_block(0, raw.n_samples)
        _, linear = raw.linear_block(bipolar, raw.n_windows, raw.freq_sl)
    assert names == ['LA1-LA2', 'LA2-LA3', 'LB1-LB2']
    assert chunks == (CHUNK, 1, len(raw.freqs))
    assert stored.shape == (raw.n_windows, 3, len(raw.freqs))
    assert np.array_equal(stored, np.log10(linear).transpose(1, 2, 0))

    # A finished run is skipped, not rewritten.
    assert rpb.extract_run('001', '01', 'X0', raw_path) is None
