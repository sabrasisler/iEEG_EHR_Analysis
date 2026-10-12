#!/usr/bin/env python3
"""
Continuous bipolar PSD on the native FFT grid, one NWB per run (DECISIONS 2026-10-12).

    python -m ieeg_ehr.preprocessing.run_pipeline_bipolar plan \
        --subjects-file SUBJECTS.txt --budget-gb 20 --out TASKS.txt
    python -m ieeg_ehr.preprocessing.run_pipeline_bipolar run \
        --task-file TASKS.txt --task-index 0
    python -m ieeg_ehr.preprocessing.run_pipeline_bipolar manifest \
        --subjects-file SUBJECTS.txt

`plan` packs registry runs into array tasks of about `--budget-gb` of raw NWB
each. `run` extracts one task's runs. `manifest` writes the tree's manifest.json.

WHAT IS STORED. preprocessed/bipolar_psd/sub-XXX/ses-YY/
sub-XXX_ses-YY_run-ZZZ_bipolar_psd_fullres.nwb, with a `.provenance.json`
sidecar beside it. `processing/ecephys/psd_fullres` is a DecompositionSeries of
shape (n_windows, n_pairs, n_freqs), float32 log10(V^2/Hz), on the native grid:
0.5 Hz from 1.0 to 250.0 Hz, 499 frequencies. Row i covers raw samples
[i*hop, i*hop + nperseg), so [i, i + 2) s after the run's session_start_time,
which in V1 is the RUN start (data_sop.md §5.1). No mask, no notch, no binning,
no averaging. Those are views, and the old 50 log bins are one of them.

THE SAME FFT AS EVERY OTHER PSD IN THIS PROJECT. Per channel, on the float32
bipolar trace, through `fullres.epoch_spectrogram`, which matches the retired
50-bin pipeline call for call. That is what lets `bipolar_psd_fullres_audit`
reproduce the retired 50-bin NWBs bit-exactly.

BLOCK READS, NOT WHOLE RUNS. One block = one HDF5 chunk row =
PSD_FULLRES_HDF5_CHUNK_WINDOWS windows, read as 301 s of full-width raw, the
fast direction for V1's time-chunked layout. Spectrogram segments are
independent, so a block's windows equal the whole-run windows exactly. The
pain-epoch extractor proved that with the same 301 s read. Memory stays at one
block (~2 GB at worst, 231 channels at 2000 Hz) for any run length, against
~17 GB for the old whole-run load.

THE VARIANCE METRIC (`--variance-metrics`, off by default). The per-2 s bipolar
variance that feeds qc/build_bipolar_exclusions.py, computed from the same block
reads into qc/bipolar/metrics/per_window/sub-XXX_bipolar_variance.csv. Metric
only, no threshold. The CSV is one file per SUBJECT, rebuilt whole, so a task
that computes it must hold every run of each subject it touches: plan with
`--whole-subjects`, and `run` refuses a task that splits a subject.

READS V1. Every derivative this joins to (QC masks, epoch defs, channel meta)
came from V1, and V1 is the only tree holding all subjects.

RESTARTABLE. A run whose NWB and sidecar both exist is skipped, except under
`--variance-metrics`, which needs every raw read. Each NWB is written as
`.partial.nwb` and renamed when complete, so a killed task never leaves a file
that looks finished.
"""

import argparse
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from hdmf.backends.hdf5.h5_utils import H5DataIO
from pynwb import NWBFile, NWBHDF5IO
from pynwb.file import Subject
from pynwb.misc import DecompositionSeries, FrequencyBandsTable

from ieeg_ehr import config, io
from ieeg_ehr.features import build_pain_epoch_fullres_psd as fullres
from ieeg_ehr.io import nwb as nwb_io
from ieeg_ehr.preprocessing import bipolar_reref

logger = logging.getLogger(__name__)

SCRIPT = 'ieeg_ehr/preprocessing/run_pipeline_bipolar.py'
SERIES_NAME = 'psd_fullres'
DATA_PATH = f'processing/ecephys/{SERIES_NAME}/data'
ELECTRODE_LOCATION_PATH = 'general/extracellular_ephys/electrodes/location'


# ============================================================================
# The run, opened once
# ============================================================================

class RawRun:
    """One raw V1 run's intracranial series, its bipolar pairs and its window grid.

    Holds the NWB open, so use it as a context manager. Shared by the extractor
    and the audit so both read the raw signal the same way.
    """

    def __init__(self, raw_path):
        self.raw_path = Path(raw_path)

    def __enter__(self):
        self._io = NWBHDF5IO(str(self.raw_path), 'r')
        self.nwb = self._io.read()
        self.series_name = nwb_io.pick_intracranial_series(self.nwb)
        self.series = self.nwb.acquisition[self.series_name]
        if self.series.unit != 'volts':
            raise ValueError(f"unexpected unit {self.series.unit!r} in {self.raw_path}")
        self.sfreq = float(self.series.rate)

        self.elec_indices = self.series.electrodes.data[:]
        elec_df = self.nwb.electrodes.to_dataframe().iloc[self.elec_indices]
        self.pairs, self.filtered_elec_df = bipolar_reref.derive_pairs(elec_df)
        if not self.pairs:
            raise RuntimeError(f'{self.raw_path.name}: no bipolar pairs could be derived')

        self.nperseg, self.noverlap, self.hop = fullres.window_geometry(self.sfreq)
        self.n_samples = self.series.data.shape[0]
        self.n_windows = n_windows(self.n_samples, self.nperseg, self.noverlap)
        self.freq_sl, self.freqs = fullres.freq_slice(
            fullres.analytic_freqs(self.sfreq, self.nperseg))
        return self

    def __exit__(self, *exc):
        self._io.close()

    def block_span(self, w0, w1):
        """Raw samples [lo, hi) that windows [w0, w1) cover."""
        return w0 * self.hop, (w1 - 1) * self.hop + self.nperseg

    def bipolar_block(self, lo, hi):
        """(hi - lo, n_pairs) float32 bipolar trace. The same cast
        `nwb_io.load_all_channels_with_electrodes` applies to a whole run, so it is
        elementwise identical to slicing the whole-run trace."""
        data_v = (self.series.data[lo:hi, :].astype(np.float32)
                  * np.float32(self.series.conversion))
        return bipolar_reref.rereference(data_v, self.elec_indices, self.pairs)

    def linear_block(self, bipolar_v, n_expected, freq_sl):
        """(freqs_full, cube): scipy's full frequency grid, and the block's
        (n_freq, n_expected, n_pairs) float32 linear PSD cut to `freq_sl`."""
        freqs_full, cube = fullres.epoch_spectrogram(bipolar_v, self.sfreq, self.nperseg,
                                                     self.noverlap, freq_sl=freq_sl)
        if cube.shape[1] != n_expected:
            raise RuntimeError(f'{self.raw_path.name}: spectrogram gave {cube.shape[1]} '
                               f'windows for a block of {n_expected}')
        return freqs_full, cube


def n_windows(n_samples, nperseg, noverlap):
    """`scipy.signal.spectrogram`'s segment count (boundary=None, padded=False)."""
    if n_samples < nperseg:
        return 0
    return (n_samples - noverlap) // (nperseg - noverlap)


def block_starts(n_win):
    """First window of every block. Blocks are chunk rows, so each HDF5 chunk is
    written, and compressed, exactly once."""
    return range(0, n_win, config.PSD_FULLRES_HDF5_CHUNK_WINDOWS)


def stored_freqs():
    """The stored frequency axis. The same at every sfreq, since df = 1/PSD_WINDOW_SEC."""
    nperseg, _, _ = fullres.window_geometry(1000.0)
    return fullres.freq_slice(fullres.analytic_freqs(1000.0, nperseg))[1]


def unit_params():
    """The config that changes the output, and nothing else (hashed into config_hash)."""
    freqs = stored_freqs()
    return {
        'window_sec': config.PSD_WINDOW_SEC,
        'overlap_frac': config.PSD_OVERLAP_FRAC,
        'window_fn': config.PSD_WINDOW_FN,
        'detrend': 'constant',
        'scaling': 'density',
        'mode': 'psd',
        'df_hz': float(config.PSD_FULLRES_DF_HZ),
        'freq_min_hz': float(freqs[0]),
        'freq_max_hz': float(freqs[-1]),
        'n_freqs': int(len(freqs)),
        'domain': 'log10',
        'dtype': str(config.CACHE_FLOAT_DTYPE),
        'source': 'raw V1',
        'rereferencing_method': 'bipolar_adjacent',
        'series_name': SERIES_NAME,
        'hdf5_chunk_windows': config.PSD_FULLRES_HDF5_CHUNK_WINDOWS,
        'compression': 'gzip-4 + shuffle',
        'masked': False, 'notched': False, 'binned': False, 'averaged': False,
    }


# ============================================================================
# Writing one run
# ============================================================================

def _nwb_skeleton(raw, n_freq):
    """NWBFile with the run's metadata, the bipolar electrode table and an EMPTY
    psd_fullres dataset, allocated at full size and filled later by h5py."""
    nwb_in = raw.nwb
    subject_out = None
    if nwb_in.subject is not None:
        subject_out = Subject(
            subject_id=nwb_in.subject.subject_id,
            age=getattr(nwb_in.subject, 'age', None),
            sex=getattr(nwb_in.subject, 'sex', None),
            species=getattr(nwb_in.subject, 'species', None),
            description=getattr(nwb_in.subject, 'description', None),
        )
    nwb_out = NWBFile(
        session_description=nwb_in.session_description
        + ' - bipolar re-referenced, native-grid PSD',
        identifier=nwb_in.identifier + '_bipolar_psd_fullres',
        session_start_time=nwb_in.session_start_time,
        timestamps_reference_time=nwb_in.timestamps_reference_time,
        file_create_date=datetime.now().astimezone(),
        experimenter=nwb_in.experimenter,
        lab=nwb_in.lab,
        institution=nwb_in.institution,
        subject=subject_out,
    )

    device = nwb_out.create_device(name='NihonKohden', description='Nihon Kohden EEG-1200A')
    elec_group = nwb_out.create_electrode_group(
        name='sEEG_bipolar', description='Bipolar referenced sEEG electrodes (anode-cathode pairs)',
        location='multiple', device=device)
    bipolar_elec_df = bipolar_reref.create_bipolar_electrode_table(raw.filtered_elec_df, raw.pairs)
    for col in bipolar_elec_df.columns:
        if col not in ('location', 'group', 'group_name'):
            nwb_out.add_electrode_column(name=col, description=f'Custom column: {col}')
    for _, row in bipolar_elec_df.iterrows():
        kwargs = {'group': elec_group}
        for col in bipolar_elec_df.columns:
            if col not in ('group', 'group_name'):
                kwargs[col] = row[col]
        nwb_out.add_electrode(**kwargs)
    electrode_region = nwb_out.create_electrode_table_region(
        region=list(range(len(raw.pairs))), description='bipolar electrode pairs')

    bands = FrequencyBandsTable()
    half = config.PSD_FULLRES_DF_HZ / 2.0
    for name, f in zip(fullres.freq_column_names(n_freq), raw.freqs):
        bands.add_band(band_name=name, band_limits=(f - half, f + half), band_mean=float(f))

    hop_sec = config.PSD_WINDOW_SEC * (1.0 - config.PSD_OVERLAP_FRAC)
    shape = (raw.n_windows, len(raw.pairs), n_freq)
    data = H5DataIO(data=None, shape=shape, dtype=np.float32,
                    chunks=(min(config.PSD_FULLRES_HDF5_CHUNK_WINDOWS, raw.n_windows), 1, n_freq),
                    compression='gzip', compression_opts=4, shuffle=True)
    module = nwb_out.create_processing_module(
        name='ecephys', description='Bipolar re-referenced PSD on the native FFT grid')
    module.add(DecompositionSeries(
        name=SERIES_NAME,
        description=(f'Per-window bipolar PSD, native {config.PSD_FULLRES_DF_HZ} Hz grid. '
                     f'Row i covers [i*{hop_sec}, i*{hop_sec} + {config.PSD_WINDOW_SEC}) s '
                     'from session_start_time (the run start in V1). Unmasked, un-notched, '
                     'unbinned. Provenance in the .provenance.json sidecar.'),
        data=data, metric='power', unit='log10(V^2/Hz)', bands=bands,
        rate=1.0 / hop_sec, starting_time=0.0, source_channels=electrode_region,
    ))
    return nwb_out


def block_variance(raw, bipolar_v, w0, w1):
    """(window_start_sec, metric) of the 2 s variance windows a PSD block owns.

    Variance windows tile the run from sample 0 without overlap. A block owns
    those from its first sample to the next block's first sample, or to the last
    whole window for the final block. Concatenated over blocks, this is
    `compute_variance_windows` on the whole run.
    """
    spw = int(round(config.BIPOLAR_VARIANCE_WINDOW_SEC * raw.sfreq))
    lo, hi = raw.block_span(w0, w1)
    if lo % spw:
        raise RuntimeError(f'block at sample {lo} does not start on a {spw}-sample '
                           'variance window')
    v_hi = (raw.n_samples // spw) * spw if w1 == raw.n_windows else w1 * raw.hop
    if v_hi > hi:
        raise RuntimeError(f'variance span ends at sample {v_hi}, past the block read at {hi}')
    var = bipolar_reref.compute_variance_windows(
        bipolar_v[:v_hi - lo], raw.sfreq, window_sec=config.BIPOLAR_VARIANCE_WINDOW_SEC)
    return var['window_start'] + lo / raw.sfreq, var['metric_value']


def _variance_rows(subject, session, run, pairs, window_start, metric):
    n_windows_, n_pairs = metric.shape
    return pd.DataFrame({
        'subject_id': f'sub-{subject}', 'session_id': f'ses-{session}', 'run_id': f'run-{run}',
        'channel': np.repeat([p['location'] for p in pairs], n_windows_),
        'anode_channel': np.repeat([p['anode_location'] for p in pairs], n_windows_),
        'cathode_channel': np.repeat([p['cathode_location'] for p in pairs], n_windows_),
        'window_start_time': np.tile(window_start, n_pairs),
        'window_end_time': np.tile(window_start + config.BIPOLAR_VARIANCE_WINDOW_SEC, n_pairs),
        # pair0's windows, then pair1's, ... matches `channel`'s repeat order
        'metric_value': metric.T.reshape(-1),
        'artifact_type': 'bipolar_variance',
    })


def extract_run(subject, session, run, raw_path, variance_csv=None):
    """Write one run's NWB + sidecar, and append its variance rows if asked.

    Returns the run's pairs signature, or None when it was skipped as finished.
    """
    out_path = config.bipolar_psd_fullres_nwb_path(subject, session, run)
    if variance_csv is None and out_path.exists() and io.sidecar_path(out_path).exists():
        logger.info('sub-%s ses-%s run-%s: exists, skipping', subject, session, run)
        return None
    t0 = time.time()
    partial = out_path.with_name(out_path.stem + '.partial.nwb')
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with RawRun(raw_path) as raw:
        if raw.n_windows == 0:
            raise RuntimeError(f'run-{run}: {raw.n_samples} samples, shorter than one window')
        n_freq = len(raw.freqs)
        with NWBHDF5IO(str(partial), 'w') as io_out:
            io_out.write(_nwb_skeleton(raw, n_freq))

        var_starts, var_metric = [], []

        n_nonfinite = 0
        with h5py.File(partial, 'r+') as f:
            ds = f[DATA_PATH]
            for w0 in block_starts(raw.n_windows):
                w1 = min(w0 + config.PSD_FULLRES_HDF5_CHUNK_WINDOWS, raw.n_windows)
                lo, hi = raw.block_span(w0, w1)
                bipolar_v = raw.bipolar_block(lo, hi)

                if variance_csv is not None:
                    starts, metric = block_variance(raw, bipolar_v, w0, w1)
                    var_starts.append(starts)
                    var_metric.append(metric)

                freqs_full, cube = raw.linear_block(bipolar_v, w1 - w0, raw.freq_sl)
                if not np.allclose(freqs_full[raw.freq_sl], raw.freqs, rtol=0, atol=1e-9):
                    raise RuntimeError(f'run-{run}: the real frequency grid does not '
                                       'match the analytic one the slice came from')
                del bipolar_v
                with np.errstate(divide='ignore'):
                    # -inf for an exactly-zero bin, as every PSD here stores it.
                    np.log10(cube, out=cube)
                n_nonfinite += int((~np.isfinite(cube)).sum())
                ds[w0:w1] = cube.transpose(1, 2, 0)
        os.replace(partial, out_path)

        if variance_csv is not None:
            config.append_table(
                _variance_rows(subject, session, run, raw.pairs,
                               np.concatenate(var_starts), np.concatenate(var_metric)),
                variance_csv)

        io.write_sidecar(
            out_path, kind='artifact', script=SCRIPT, params=unit_params(),
            parents=[raw.raw_path], subjects=[f'sub-{subject}'],
            extra={'n_windows': int(raw.n_windows), 'n_pairs': len(raw.pairs),
                   'sfreq_hz': raw.sfreq, 'series': raw.series_name,
                   'n_nonfinite_values': n_nonfinite,
                   'elapsed_sec': round(time.time() - t0, 1)})
        logger.info('sub-%s ses-%s run-%s: %d windows x %d pairs x %d freqs @ %g Hz -> '
                    '%.2f GB in %.0f s%s', subject, session, run, raw.n_windows,
                    len(raw.pairs), n_freq, raw.sfreq, out_path.stat().st_size / 1e9,
                    time.time() - t0, f'  [{n_nonfinite} non-finite]' if n_nonfinite else '')
        return bipolar_reref.pairs_signature(raw.pairs)


# ============================================================================
# Tasks
# ============================================================================

def _registry(subjects):
    reg = pd.read_csv(config.FILE_REGISTRY_CSV)
    reg = reg[reg['sub_id'].isin([f'sub-{s}' for s in subjects])]
    missing = set(subjects) - set(reg['sub_id'].str.replace('sub-', '', regex=False))
    if missing:
        raise ValueError(f'subjects absent from the registry: {sorted(missing)}')
    return reg.sort_values(['sub_id', 'ses_id', 'run_id'])


def _token(row):
    return ':'.join(str(v).split('-', 1)[1] for v in (row.sub_id, row.ses_id, row.run_id))


def plan(subjects, budget_gb, whole_subjects=False):
    """Pack runs, in registry order, into tasks of about `budget_gb` of raw NWB.

    A task closes once it reaches the budget, so one task can exceed it by up to
    one run (or, with `whole_subjects`, one subject). Returns a list of token lists.
    """
    reg = _registry(subjects)
    units = ([list(g.itertuples()) for _, g in reg.groupby('sub_id', sort=False)]
             if whole_subjects else [[row] for row in reg.itertuples()])
    tasks, current, size_gb = [], [], 0.0
    for unit in units:
        current += [_token(row) for row in unit]
        size_gb += sum(row.raw_file_size_mb for row in unit) / 1000.0
        if size_gb >= budget_gb:
            tasks.append(current)
            current, size_gb = [], 0.0
    if current:
        tasks.append(current)
    return tasks


def run_task(tokens, variance_metrics=False):
    """Extract every run in one task. Returns the list of failed tokens."""
    parsed = [tuple(t.split(':')) for t in tokens]
    subjects = list(dict.fromkeys(s for s, _, _ in parsed))
    reg = _registry(subjects)
    raw_paths = {_token(row): row.raw_file_path for row in reg.itertuples()}

    if variance_metrics:
        split = sorted(set(raw_paths) - set(tokens))
        if split:
            raise ValueError(f'--variance-metrics rebuilds whole subject CSVs, but this task '
                             f'lacks {len(split)} of its subjects\' runs (first: {split[0]}). '
                             'Plan with --whole-subjects.')

    prov = io.warn_if_dirty()
    ts = config.run_timestamp()
    failed = []
    for subject in subjects:
        variance_csv = None
        if variance_metrics:
            variance_csv = (config.metrics_per_window_dir(config.BIPOLAR_LEVEL_ROOT)
                            / f'sub-{subject}_bipolar_variance.csv')
            config.reset_table(variance_csv)
        session_sigs, diverged_runs, subject_ok = {}, [], True
        for token in [t for t in tokens if t.startswith(f'{subject}:')]:
            _, session, run = token.split(':')
            try:
                sig = extract_run(subject, session, run, raw_paths[token], variance_csv)
            except Exception:
                logger.error('%s failed:\n%s', token, traceback.format_exc())
                failed.append(token)
                subject_ok = False
                continue
            if sig is not None and session_sigs.setdefault(session, sig) != sig:
                # Contacts are assumed stable within a session. Each NWB carries
                # its own run's pairs, so this is recorded, not corrected.
                diverged_runs.append(run)
        if variance_metrics and subject_ok:
            # A missing run_info is what marks the subject for a rerun.
            _write_run_info(subject, diverged_runs, prov, ts)
    return failed


def _write_run_info(subject, diverged_runs, prov, ts):
    """qc/bipolar run_info, whose presence `qc.processing_status` reads as done."""
    path = config.metrics_run_info_dir(config.BIPOLAR_LEVEL_ROOT) / f'sub-{subject}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        'subject': f'sub-{subject}',
        'artifact_type': 'bipolar_variance',
        'detection_params': {'variance_window_sec': config.BIPOLAR_VARIANCE_WINDOW_SEC},
        'run_timestamp': ts,
        'git': prov,
        'pairs_diverged': bool(diverged_runs),
        'pairs_divergence_runs': diverged_runs,
    }, indent=2, default=str))


def _read_subjects(path):
    return [s.strip().replace('sub-', '') for s in Path(path).read_text().split() if s.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('plan', help='pack registry runs into array tasks')
    p.add_argument('--subjects-file', required=True)
    p.add_argument('--budget-gb', type=float, required=True,
                   help='raw NWB per task. Size it from the pilot\'s seconds per GB.')
    p.add_argument('--whole-subjects', action='store_true',
                   help='never split a subject across tasks (needed by --variance-metrics)')
    p.add_argument('--out', required=True)

    r = sub.add_parser('run', help='extract one task')
    r.add_argument('--task-file', required=True)
    r.add_argument('--task-index', type=int, required=True)
    r.add_argument('--variance-metrics', action='store_true',
                   help='also rebuild the bipolar variance QC metric for these subjects')

    m = sub.add_parser('manifest', help='write the tree manifest.json')
    m.add_argument('--subjects-file', required=True)

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    if args.cmd == 'plan':
        tasks = plan(_read_subjects(args.subjects_file), args.budget_gb, args.whole_subjects)
        Path(args.out).write_text(''.join(' '.join(t) + '\n' for t in tasks))
        print(f'{sum(map(len, tasks))} runs in {len(tasks)} tasks -> {args.out}  '
              f'(sbatch --array=0-{len(tasks) - 1})')
    elif args.cmd == 'run':
        lines = Path(args.task_file).read_text().splitlines()
        tokens = lines[args.task_index].split()
        logger.info('task %d: %d runs', args.task_index, len(tokens))
        failed = run_task(tokens, args.variance_metrics)
        if failed:
            logger.error('%d of %d runs failed: %s', len(failed), len(tokens), failed)
            sys.exit(1)
    else:
        subjects = _read_subjects(args.subjects_file)
        path = io.write_manifest(
            config.BIPOLAR_PSD_FULLRES_ROOT, params=unit_params(), script=SCRIPT,
            parents=[args.subjects_file], subjects=[f'sub-{s}' for s in subjects],
            extra={'freqs_hz': stored_freqs().tolist(),
                   'layout': f'{DATA_PATH}: (n_windows, n_pairs, n_freqs); pair names at '
                             f'{ELECTRODE_LOCATION_PATH}; frequencies are freqs_hz here',
                   'row_time': 'row i covers [i*hop, i*hop + window_sec) s from the '
                               "run's session_start_time"})
        print(f'wrote {path}')


if __name__ == '__main__':
    main()
