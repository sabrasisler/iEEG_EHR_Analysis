#!/usr/bin/env python3
"""
Fused bipolar re-reference + Welch PSD pass: single raw-NWB read per run.

While the (transient, never-persisted) bipolar-referenced trace is in memory,
computes BOTH:
  1. A continuous per-2s-window variance metric per bipolar channel (metric
     ONLY, no thresholding -- qc_scripts/build_bipolar_exclusions.py owns
     that, same metric/threshold split as the raw_voltage pipeline), written
     to qc/bipolar/metrics/per_window/sub-XXX.csv.
  2. A Welch PSD per 60s outer window (default), band-averaged into 50
     log-spaced frequency bins, written to an NWB file under
     derivatives/preprocessed/bipolar_fft/sub-XXX/ses-XXX/.

No exclusion/masking of any kind happens here -- see qc_scripts/CONTEXT.md's
metric/threshold split. No QC ever runs on the FFT/PSD output; the mask-aware
bipolar exclusion step (qc_scripts/build_bipolar_exclusions.py) reads ONLY the
variance-metric CSVs written here, never the PSD/NWB files.

Usage:
  python -m preprocessing.run_pipeline_bipolar --subjects 217,222
  python -m preprocessing.run_pipeline_bipolar --subjects 217,222 --outer-sec 60 --inner-sec 2
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from pynwb import NWBFile, NWBHDF5IO
from pynwb.file import Subject
from pynwb.misc import DecompositionSeries, FrequencyBandsTable
from pynwb.base import TimeSeries
from hdmf.backends.hdf5.h5_utils import H5DataIO

from qc_scripts import config, io_utils
from preprocessing import bipolar_reref


def _variance_rows(subject, session, run, pairs, var_result):
    window_start = var_result['window_start']
    window_end = var_result['window_end']
    metric = var_result['metric_value']   # (n_windows, n_pairs)
    n_windows, n_pairs = metric.shape
    if n_windows == 0 or n_pairs == 0:
        return None

    channel = np.repeat([p['location'] for p in pairs], n_windows)
    anode = np.repeat([p['anode_location'] for p in pairs], n_windows)
    cathode = np.repeat([p['cathode_location'] for p in pairs], n_windows)
    ws = np.tile(window_start, n_pairs)
    we = np.tile(window_end, n_pairs)
    mv = metric.T.reshape(-1)   # pair0's windows, then pair1's, ... matches `channel`'s repeat order

    return pd.DataFrame({
        'subject_id': f'sub-{subject}', 'session_id': f'ses-{session}', 'run_id': f'run-{run}',
        'channel': channel, 'anode_channel': anode, 'cathode_channel': cathode,
        'window_start_time': ws, 'window_end_time': we,
        'metric_value': mv, 'artifact_type': 'bipolar_variance',
    })


def _metrics_output_path(level_root, subject):
    return config.metrics_per_window_dir(level_root) / f'sub-{subject}_bipolar_variance.csv'


def _chunk_shape(log_power_shape, outer_sec, psd_chunk_max_hours):
    """
    See CONTEXT/plan discussion: PSD rows are already collapsed to 1/outer_sec
    (default 1/minute), so a channel's ENTIRE run is only tens-to-a-few-hundred
    KB -- far below where sub-chunking time would help (unlike dense raw
    voltage). Default: one chunk = one channel's whole run. Only cap for
    unusually long recordings so no single chunk balloons past ~1MB.
    """
    n_time, n_pairs, n_bins = log_power_shape
    if psd_chunk_max_hours is None:
        rows_per_chunk = n_time
    else:
        rows_per_chunk = min(n_time, max(1, int(round(psd_chunk_max_hours * 3600.0 / outer_sec))))
    return (max(1, rows_per_chunk), 1, n_bins)


def write_bipolar_psd_nwb(nwb_in, filtered_elec_df, pairs, psd_result, bin_edges,
                           welch_params, out_path, sidecar_extra):
    """Writes one NWB per run: bipolar electrode table + a DecompositionSeries
    of log-spaced-bin PSD + a broadband_log_power TimeSeries. No bipolar time
    series is written (transient only). No separate sidecar JSON -- ALL
    provenance (git, run_timestamp, bin edges, line-noise config, source_nwb,
    pairs_diverged, hdf5_chunk_shape) is embedded directly in the
    DecompositionSeries' `description` field instead, so nothing is lost but
    no per-run file clutter accumulates (git/timestamp/params are identical
    across every run anyway -- also recorded once per subject in
    qc/bipolar/metrics/run_info/sub-XXX.json)."""
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
        session_description=nwb_in.session_description + ' - bipolar re-referenced, Welch log-bin PSD',
        identifier=nwb_in.identifier + '_bipolar_psd',
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

    bipolar_elec_df = bipolar_reref.create_bipolar_electrode_table(filtered_elec_df, pairs)
    standard_columns = ['location', 'group', 'group_name']
    custom_columns = [c for c in bipolar_elec_df.columns if c not in standard_columns]
    for col in custom_columns:
        nwb_out.add_electrode_column(name=col, description=f'Custom column: {col}')
    for _, row in bipolar_elec_df.iterrows():
        kwargs = {'group': elec_group}
        for col in bipolar_elec_df.columns:
            if col not in ('group', 'group_name'):
                kwargs[col] = row[col]
        nwb_out.add_electrode(**kwargs)

    electrode_region = nwb_out.create_electrode_table_region(
        region=list(range(len(pairs))), description='bipolar electrode pairs')

    ecephys_module = nwb_out.create_processing_module(
        name='ecephys', description='Bipolar re-referenced Welch PSD (log-spaced frequency bins)')

    n_bins = len(bin_edges) - 1
    # DecompositionSeries.bands must be a FrequencyBandsTable (not a generic
    # DynamicTable) -- confirmed via a Sherlock smoke test. Its add_band has
    # allow_extra=True, so the custom `contains_line_noise` column (added
    # before any rows) DOES forward through add_band -- also confirmed live.
    bands_table = FrequencyBandsTable()
    bands_table.add_column(name='contains_line_noise', description='bin overlaps a line-noise harmonic '
                            '(60/120/180/240 Hz) +/- the configured guard band')
    for i in range(n_bins):
        lo, hi = float(bin_edges[i]), float(bin_edges[i + 1])
        bands_table.add_band(
            band_name=f'bin_{i:02d}', band_limits=(lo, hi),
            band_mean=float(np.sqrt(lo * hi)), band_stdev=float((hi - lo) / 2),
            contains_line_noise=bool(psd_result['contains_line_noise'][i]),
        )

    log_power = psd_result['log_power']
    chunk_shape = _chunk_shape(log_power.shape, welch_params['outer_window_sec'],
                                welch_params.get('psd_chunk_max_hours'))
    log_power_io = H5DataIO(
        data=log_power, chunks=chunk_shape, compression='gzip', compression_opts=4,
    )

    full_provenance = dict(welch_params)
    full_provenance.update(sidecar_extra)
    full_provenance['hdf5_chunk_shape'] = list(chunk_shape)
    description_params = json.dumps(full_provenance, default=str)

    decomp_series = DecompositionSeries(
        name='psd_log_bins',
        description=f'Bipolar Welch PSD, log-spaced frequency bins. Params: {description_params}',
        data=log_power_io,
        metric='power',
        unit='log10(V^2/Hz)',
        bands=bands_table,
        rate=1.0 / welch_params['outer_window_sec'],
        source_channels=electrode_region,
    )
    ecephys_module.add(decomp_series)

    broadband_series = TimeSeries(
        name='broadband_log_power',
        description='Mean log10-power across non-line-noise-flagged bins, per outer window/channel.',
        data=psd_result['broadband_log_power'],
        unit='log10(V^2/Hz)',
        rate=1.0 / welch_params['outer_window_sec'],
    )
    ecephys_module.add(broadband_series)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with NWBHDF5IO(str(out_path), 'w') as io_out:
        io_out.write(nwb_out)
    print(f"    wrote {out_path}", flush=True)


def process_session(subject, session, runs, outer_sec, inner_sec, overlap_frac,
                     bin_edges, guard_hz, psd_chunk_max_hours, psd_out_root,
                     skip_variance_metrics, metrics_out_path, prov, ts, n_workers=1):
    session_pairs_sig = None
    pairs_diverged = False
    diverged_runs = []

    for session_, run, nwb_path in runs:
        print(f"  sub-{subject} ses-{session_} run-{run}: loading...", flush=True)
        try:
            data_v, channel_names, sfreq, elec_df, elec_indices = \
                io_utils.load_all_channels_with_electrodes(nwb_path)
        except Exception as e:
            print(f"  WARNING: failed to read {nwb_path} ({e!r}); skipping this run.", flush=True)
            continue

        pairs, filtered_elec_df = bipolar_reref.derive_pairs(elec_df)
        if not pairs:
            print(f"  WARNING: no bipolar pairs derived for run-{run}; skipping.", flush=True)
            del data_v
            continue

        sig = bipolar_reref.pairs_signature(pairs)
        if session_pairs_sig is None:
            session_pairs_sig = sig
            this_run_diverged = False
        else:
            this_run_diverged = (sig != session_pairs_sig)
            if this_run_diverged:
                print(f"  WARNING: run-{run}'s bipolar pairs differ from this session's first run "
                      f"(contacts assumed stable within a session -- proceeding with THIS run's own "
                      f"pairs; accepted simplification, see plan).", flush=True)
                pairs_diverged = True
                diverged_runs.append(run)

        bipolar_v = bipolar_reref.rereference(data_v, elec_indices, pairs)
        del data_v

        if not skip_variance_metrics:
            var_result = bipolar_reref.compute_variance_windows(
                bipolar_v, sfreq, window_sec=config.BIPOLAR_VARIANCE_WINDOW_SEC)
            df = _variance_rows(subject, session_, run, pairs, var_result)
            if df is not None:
                config.append_table(df, metrics_out_path)

        psd_result = bipolar_reref.compute_welch_log_bins(
            bipolar_v, sfreq, outer_sec, inner_sec, overlap_frac, bin_edges, guard_hz,
            line_freqs=config.PSD_LINE_NOISE_FREQS_HZ, n_workers=n_workers)
        del bipolar_v

        if psd_result['log_power'].shape[0] == 0:
            # Run is shorter than one outer window (e.g. <60s) -- zero PSD
            # rows. H5DataIO's chunk shape always requests >=1 row, which
            # HDF5 rejects when the data itself has 0 rows ("Chunk shape must
            # not be greater than data shape"). Nothing meaningful to store
            # for a run this short anyway -- skip it, same pattern as the
            # existing "no bipolar pairs derived" skip above.
            print(f"  WARNING: run-{run} produced 0 PSD windows (run shorter than "
                  f"outer_sec={outer_sec}s); skipping PSD write for this run.", flush=True)
            continue

        welch_params = {
            'outer_window_sec': outer_sec, 'inner_segment_sec': inner_sec,
            'overlap_frac': overlap_frac, 'window_function': config.PSD_WINDOW_FN,
            'scaling': 'density', 'psd_chunk_max_hours': psd_chunk_max_hours,
        }
        sidecar_extra = {
            'git': prov,
            'run_timestamp': ts,
            'rereferencing_method': 'bipolar_adjacent_contact',
            'log_bin_edges_hz': np.asarray(bin_edges).tolist(),
            'n_bins': len(bin_edges) - 1,
            'f_min_hz': float(bin_edges[0]), 'f_max_hz': float(bin_edges[-1]),
            'line_noise_freqs_hz': list(config.PSD_LINE_NOISE_FREQS_HZ),
            'line_noise_guard_hz': guard_hz,
            'exclusion_applied': False,
            'source_nwb': str(nwb_path),
            'pairs_diverged_from_session_first_run': this_run_diverged,
        }

        out_dir = Path(psd_out_root) / f'sub-{subject}' / f'ses-{session_}'
        out_path = out_dir / f'sub-{subject}_ses-{session_}_run-{run}_bipolar_psd.nwb'

        # Second, metadata-only NWB open for session_description/identifier/subject/etc --
        # lazy-loaded by pynwb, does NOT re-read the electrical series' data array, so this
        # is not a second raw-data pass in the sense the metric/threshold split cares about.
        io_in = NWBHDF5IO(str(nwb_path), 'r')
        nwb_in = io_in.read()
        write_bipolar_psd_nwb(nwb_in, filtered_elec_df, pairs, psd_result, bin_edges,
                               welch_params, out_path, sidecar_extra)
        io_in.close()

    return pairs_diverged, diverged_runs


def _write_run_info(level_root, subject, pairs_diverged, diverged_runs, params, prov, ts):
    rdir = config.metrics_run_info_dir(level_root)
    rdir.mkdir(parents=True, exist_ok=True)
    info = {
        'subject': f'sub-{subject}',
        'artifact_type': 'bipolar_variance',
        'detection_params': params,
        'run_timestamp': ts,
        'git': prov,
        'pairs_diverged': pairs_diverged,
        'pairs_divergence_runs': diverged_runs,
    }
    path = rdir / f'sub-{subject}.json'
    with open(path, 'w') as f:
        json.dump(info, f, indent=2, default=str)
    print(f"  Wrote {path}", flush=True)


def run(subjects, level_root, psd_out_root, outer_sec, inner_sec, overlap_frac,
        n_bins, f_min, f_max, guard_hz, psd_chunk_max_hours, skip_variance_metrics, n_workers=1):
    bin_edges = bipolar_reref.log_bin_edges(n_bins, f_min, f_max)
    prov = config.warn_if_dirty()
    ts = config.run_timestamp()

    detection_params = {
        'variance_window_sec': config.BIPOLAR_VARIANCE_WINDOW_SEC,
        'outer_window_sec': outer_sec, 'inner_segment_sec': inner_sec,
        'overlap_frac': overlap_frac, 'n_bins': n_bins, 'f_min_hz': f_min, 'f_max_hz': f_max,
        'line_noise_freqs_hz': list(config.PSD_LINE_NOISE_FREQS_HZ), 'line_noise_guard_hz': guard_hz,
        'psd_chunk_max_hours': psd_chunk_max_hours,
    }

    failed_subjects = []
    for subject in subjects:
        print(f"=== sub-{subject} ===", flush=True)
        try:
            metrics_out_path = _metrics_output_path(level_root, subject)
            if not skip_variance_metrics:
                config.reset_table(metrics_out_path)

            session_runs = io_utils.get_session_runs(subject)
            sessions = sorted(set(s for s, _, _ in session_runs))

            subj_pairs_diverged = False
            subj_diverged_runs = []
            for session in sessions:
                runs = [(s, r, p) for s, r, p in session_runs if s == session]
                diverged, diverged_runs = process_session(
                    subject, session, runs, outer_sec, inner_sec, overlap_frac, bin_edges, guard_hz,
                    psd_chunk_max_hours, psd_out_root, skip_variance_metrics, metrics_out_path, prov, ts,
                    n_workers=n_workers)
                subj_pairs_diverged = subj_pairs_diverged or diverged
                subj_diverged_runs.extend(diverged_runs)

            _write_run_info(level_root, subject, subj_pairs_diverged, subj_diverged_runs,
                             detection_params, prov, ts)
        except Exception as e:
            # One subject's failure must not take down the rest of a batched
            # array task (BATCH_SIZE subjects run sequentially in this one
            # process) -- log it, move on. run_info.json's absence for this
            # subject is itself the marker that it needs a rerun (see
            # qc_scripts.processing_status).
            import traceback
            print(f"  ERROR: sub-{subject} failed, skipping to next subject: {e!r}", flush=True)
            traceback.print_exc()
            failed_subjects.append(subject)

    if failed_subjects:
        print(f"=== {len(failed_subjects)} subject(s) failed and were skipped: "
              f"{failed_subjects} ===", flush=True)
    return failed_subjects


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--subjects', required=True,
                     help='Comma-separated explicit subject IDs, e.g. 217,222')
    ap.add_argument('--level-root', default=str(config.BIPOLAR_LEVEL_ROOT),
                     help=f'qc/bipolar level root for the variance metric (default: {config.BIPOLAR_LEVEL_ROOT})')
    ap.add_argument('--psd-out-root', default=str(config.BIPOLAR_PSD_DERIV_ROOT),
                     help=f'Derivatives root for PSD NWB output (default: {config.BIPOLAR_PSD_DERIV_ROOT})')
    ap.add_argument('--outer-sec', type=float, default=config.PSD_OUTER_WINDOW_SEC)
    ap.add_argument('--inner-sec', type=float, default=config.PSD_INNER_SEGMENT_SEC)
    ap.add_argument('--overlap', type=float, default=config.PSD_OVERLAP_FRAC)
    ap.add_argument('--n-bins', type=int, default=config.PSD_N_LOG_BINS)
    ap.add_argument('--f-min', type=float, default=config.PSD_FREQ_MIN_HZ)
    ap.add_argument('--f-max', type=float, default=config.PSD_FREQ_MAX_HZ)
    ap.add_argument('--guard-hz', type=float, default=config.PSD_LINE_NOISE_GUARD_HZ)
    ap.add_argument('--psd-chunk-max-hours', type=float, default=config.PSD_HDF5_CHUNK_MAX_HOURS,
                     help='Cap HDF5 chunk size for unusually long runs (default: None = whole run per channel)')
    ap.add_argument('--skip-variance-metrics', action='store_true',
                     help='Escape hatch to skip writing the (cheap) bipolar variance metric CSV')
    ap.add_argument('--n-workers', type=int, default=1,
                     help='Parallelize the Welch PSD computation across this many channels at once '
                          '(ProcessPoolExecutor). Default 1 = sequential (matches the originally '
                          'validated single-threaded path). Match to --cpus-per-task in the sbatch.')
    args = ap.parse_args()

    subjects = [s.strip() for s in args.subjects.split(',')]
    print(f"Running bipolar reref+PSD on {len(subjects)} subject(s): {subjects} "
          f"(n_workers={args.n_workers})", flush=True)
    failed = run(subjects, args.level_root, args.psd_out_root, args.outer_sec, args.inner_sec,
                 args.overlap, args.n_bins, args.f_min, args.f_max, args.guard_hz,
                 args.psd_chunk_max_hours, args.skip_variance_metrics, n_workers=args.n_workers)
    if failed:
        sys.exit(1)   # non-zero exit for Slurm/sacct visibility, but every subject that COULD
                       # run already did -- this only marks the task, doesn't undo prior work.


if __name__ == '__main__':
    main()
