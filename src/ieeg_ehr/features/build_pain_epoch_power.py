#!/usr/bin/env python3
"""
P1.1 cache builder: slice the continuous bipolar PSD to pain-event epochs and
store it PER WINDOW.

Emits two artifacts per subject/session into the base unit
`features/pain/psd_epochs/epoch-<N>min-pre/`:

  cache/sub-XXX_ses-YY_epochs.parquet   per-window log-power, long format
  epoch_defs/sub-XXX_ses-YY_defs.parquet  tiny index: one row per epoch

WHAT THIS DOES NOT DO (all of it deliberate — architecture.md PART 1, and the
2026-07-27 decisions):

  - **No averaging.** The old version averaged log-power over the epoch, which
    forced the log-vs-linear and normalize-before-vs-after choices at cache
    time. Both are view axes now (view_registry AXIS 3/4). Averaging before
    normalizing is not the same as normalizing before averaging (Jensen), so
    the cache has to keep per-window granularity for those to stay free
    recomputes.
  - **No normalization.** Baseline and z-scoring are view axes 2 and 3.
  - **No QC mask, and no mask column.** The cache stores raw slices; the
    raw-voltage mask is a view-time join on (run, channel, 60s bin). This
    decouples the cache from the mask entirely: switching masks no longer
    invalidates ~47 GB of cache, and P0.1 no longer blocks building it.
  - **No line-noise filtering.** The old version skipped `contains_line_noise`
    bins. Bin i's line-noise status is a fixed property of the frequency grid
    (bin edges vs 60 Hz harmonics), recoverable at view time from the manifest,
    so dropping those rows here would bake in a choice for no storage win worth
    having.
  - **No EPOCH_MAX_EXCLUDED_FRAC.** That threshold is mask-derived, so it moved
    to the view layer with the mask.
  - **Non-finite values are stored, not dropped.** A dead channel can have
    log10(0) = -inf at some bins without tripping raw-voltage QC. The old code
    dropped those channel-epochs; "raw slices only" means we keep them. They
    are COUNTED and the count goes in the sidecar, so the hazard is visible
    rather than silent. Views must handle non-finite.

Memory: runs are opened one at a time and each epoch is written as its own
Parquet row group via a streaming ParquetWriter, so peak memory is one run's
PSD plus one epoch's frame — not the whole subject. The largest subject
(sub-256: 199 pairs x 137 events) is ~409M rows / 1.6 GB, which would not fit
comfortably any other way.

Run on a dev/interactive Slurm shell (never the login node):
    module load python/3.12
    source $GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate
    python -m ieeg_ehr.features.build_pain_epoch_power --subjects 071
"""

import argparse
import logging
import warnings

import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO

from ieeg_ehr import config, io
from ieeg_ehr.qc import psd_timing

logger = logging.getLogger(__name__)

# Long-format cache schema (architecture.md PART 1). Kept narrow on purpose:
# everything constant within an epoch (subject, session, run, pain score) lives
# in the epoch_defs index and joins on epoch_id, rather than being repeated
# across hundreds of millions of rows.
CACHE_COLUMNS = ['epoch_id', 'window_idx', 'channel', 'bin', 'log_power']


def _run_time_index(nwb_path):
    """Timing metadata for one run's PSD, WITHOUT reading the data array.

    Two-pass design: this cheap pass builds the run index so each pain event can
    be assigned to a run, and only the runs that actually carry an epoch get
    opened again for their (potentially hundreds of MB) data.
    """
    with NWBHDF5IO(str(nwb_path), 'r') as handle:
        nwb = handle.read()
        decomp = nwb.processing['ecephys']['psd_log_bins']
        n_time = decomp.data.shape[0]
        rate = float(decomp.rate)
        starting_time = float(decomp.starting_time)
        session_start = nwb.session_start_time
    run_seconds = starting_time + np.arange(n_time) / rate
    dts = pd.to_datetime(session_start) + pd.to_timedelta(run_seconds, unit='s')
    return {'path': nwb_path, 'n_time': n_time, 'rate': rate,
            'starting_time': starting_time, 'datetimes': dts.tz_localize(None)}


def _load_run_arrays(nwb_path):
    """The data + per-pair/per-bin metadata for one run. Called once per run
    that carries at least one epoch."""
    with NWBHDF5IO(str(nwb_path), 'r') as handle:
        nwb = handle.read()
        decomp = nwb.processing['ecephys']['psd_log_bins']
        log_power = decomp.data[:]              # (n_time, n_pairs, n_bins)
        bands = decomp.bands.to_dataframe()
        lo = bands['band_limits'].apply(lambda t: t[0]).to_numpy()
        hi = bands['band_limits'].apply(lambda t: t[1]).to_numpy()
        bin_edges = np.concatenate([lo, hi[-1:]])
        contains_line_noise = bands['contains_line_noise'].to_numpy(dtype=bool)
        elec = nwb.electrodes.to_dataframe()
        channels = list(elec['location'])
        dk = (list(elec['Desikan_Killiany_anode'])
              if 'Desikan_Killiany_anode' in elec.columns else [None] * len(channels))
    return {'log_power': log_power, 'bin_edges': bin_edges,
            'contains_line_noise': contains_line_noise,
            'channels': channels, 'dk_anode': dk}


def _assign_epochs(pain_df, run_index, epoch_minutes):
    """One row per usable pain event: which run covers its pre-event window, and
    which PSD rows that window spans.

    An epoch is dropped if no run covers it, or if the window would straddle a
    run boundary (the PSD is discontinuous across runs, so a straddling epoch
    would silently mix two recordings).
    """
    epochs, n_no_match, n_boundary = [], 0, 0
    for _, row in pain_df.iterrows():
        pain_time = row['date']
        window_start = pain_time - pd.Timedelta(minutes=epoch_minutes)
        hit = None
        for run_id, meta in run_index.items():
            dts = meta['datetimes']
            if len(dts) and dts[0] <= pain_time <= dts[-1]:
                if window_start < dts[0]:
                    n_boundary += 1
                    hit = 'boundary'
                    break
                sel = np.where((dts >= window_start) & (dts < pain_time))[0]
                if len(sel) == 0:
                    break
                hit = (run_id, int(sel[0]), int(sel[-1]) + 1, len(sel))
                break
        if hit is None or hit == 'boundary':
            if hit is None:
                n_no_match += 1
            continue
        run_id, r0, r1, n_win = hit
        epochs.append({'run': run_id, 'row_start': r0, 'row_stop': r1,
                       'n_windows': n_win, 'pain_event_id': int(row['pain_event_id']),
                       'pain_score': row['max_pain'], 'pain_time': pain_time,
                       'window_start': window_start})
    return epochs, n_no_match, n_boundary


def build_subject_session(subject, session, epoch_minutes, overwrite=False,
                          nonstandard_hop='refuse'):
    import pyarrow as pa
    import pyarrow.parquet as pq

    cache_path = config.pain_epoch_cache_path(subject, session, epoch_minutes)
    defs_path = config.pain_epoch_defs_path(subject, session, epoch_minutes)
    if cache_path.exists() and not overwrite:
        logger.info('sub-%s ses-%s: cache exists, skipping (use --overwrite)', subject, session)
        return None

    scores = config.pain_scores_csv(subject, session)
    if not scores.exists():
        logger.warning('sub-%s ses-%s: no pain scores at %s', subject, session, scores)
        return None
    pain_df = pd.read_csv(scores, parse_dates=['date'])
    if 'pain_event_id' not in pain_df.columns:
        pain_df = pain_df.reset_index().rename(columns={'index': 'pain_event_id'})
    pain_df = pain_df.dropna(subset=['date', 'max_pain'])

    registry = pd.read_csv(config.FILE_REGISTRY_CSV)
    runs = registry[(registry.sub_id == f'sub-{subject}')
                    & (registry.ses_id == f'ses-{session}')].run_id.unique()

    run_index = {}
    for r in runs:
        rid = str(r).replace('run-', '')
        p = config.bipolar_psd_nwb_path(subject, session, rid)
        if p.exists():
            run_index[rid] = _run_time_index(p)
    if not run_index:
        logger.warning('sub-%s ses-%s: no bipolar_fft runs on disk, skipping', subject, session)
        return None

    epochs, n_no_match, n_boundary = _assign_epochs(pain_df, run_index, epoch_minutes)
    if not epochs:
        logger.warning('sub-%s ses-%s: %d pain events, none usable (%d no run, %d boundary)',
                       subject, session, len(pain_df), n_no_match, n_boundary)
        return None

    for i, e in enumerate(epochs):
        e['epoch_id'] = i

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    defs_path.parent.mkdir(parents=True, exist_ok=True)

    schema = pa.schema([
        ('epoch_id', pa.int32()),
        ('window_idx', pa.int16()),
        ('channel', pa.dictionary(pa.int16(), pa.string())),
        ('bin', pa.int8()),
        ('log_power', pa.float32()),
    ])

    writer = pq.ParquetWriter(cache_path, schema, compression='snappy')
    n_rows = n_nonfinite = 0
    bin_edges = contains_line_noise = None
    n_channels_by_epoch = {}
    try:
        # Group by run so each run's PSD is read exactly once.
        by_run = {}
        for e in epochs:
            by_run.setdefault(e['run'], []).append(e)

        for run_id, run_epochs in by_run.items():
            arrays = _load_run_arrays(run_index[run_id]['path'])
            if bin_edges is None:
                bin_edges = arrays['bin_edges']
                contains_line_noise = arrays['contains_line_noise']
            lp = arrays['log_power']
            channels = arrays['channels']
            n_pairs, n_bins = lp.shape[1], lp.shape[2]

            for e in run_epochs:
                block = lp[e['row_start']:e['row_stop']]        # (n_win, n_pairs, n_bins)
                n_win = block.shape[0]
                n_channels_by_epoch[e['epoch_id']] = n_pairs
                n_nonfinite += int((~np.isfinite(block)).sum())

                # C-order ravel of (win, pair, bin) -> the index columns are the
                # matching repeat/tile pattern. Built with numpy rather than a
                # Python loop: this is ~3M rows per epoch.
                tbl = pa.table({
                    'epoch_id': pa.array(np.full(block.size, e['epoch_id'], dtype=np.int32)),
                    'window_idx': pa.array(np.repeat(np.arange(n_win, dtype=np.int16),
                                                     n_pairs * n_bins)),
                    'channel': pa.array(np.tile(np.repeat(channels, n_bins), n_win)
                                        ).dictionary_encode(),
                    'bin': pa.array(np.tile(np.arange(n_bins, dtype=np.int8), n_win * n_pairs)),
                    'log_power': pa.array(
                        block.reshape(-1).astype(config.CACHE_FLOAT_DTYPE, copy=False)),
                }, schema=schema)
                writer.write_table(tbl)
                n_rows += block.size
            del arrays, lp
    finally:
        writer.close()

    # epoch_start_sec / hop_sec make the epoch's windows locatable in RUN-RELATIVE
    # SECONDS, which is the coordinate the 60s QC mask grid is keyed on:
    #     run_seconds(window_idx) = epoch_start_sec + window_idx * hop_sec
    # Stored here, in the tiny per-epoch index, and NOT as cache columns: they are
    # constant within a run, and the cache reaches 409M rows for one subject, so a
    # per-row copy would add gigabytes to repeat one scalar (see CACHE_COLUMNS).
    # Storing them in this derived form rather than as raw starting_time/rate means
    # a consumer reconstructs nothing and cannot get the arithmetic wrong.
    defs = pd.DataFrame([{
        'epoch_id': e['epoch_id'], 'subject_id': f'sub-{subject}',
        'session_id': f'ses-{session}', 'run_id': f"run-{e['run']}",
        'pain_event_id': e['pain_event_id'], 'pain_score': e['pain_score'],
        'pain_time': e['pain_time'], 'window_start': e['window_start'],
        'row_start': e['row_start'], 'row_stop': e['row_stop'],
        'n_windows': e['n_windows'], 'n_channels': n_channels_by_epoch[e['epoch_id']],
        'epoch_start_sec': (run_index[e['run']]['starting_time']
                            + e['row_start'] / run_index[e['run']]['rate']),
        'hop_sec': 1.0 / run_index[e['run']]['rate'],
    } for e in epochs])

    epoch_params = {'epoch_minutes_before': epoch_minutes, 'anchor': 'pain_score_time',
                    'masked': False, 'averaged': False, 'normalized': False}
    io.write_table(defs, defs_path, kind='table',
                   script='ieeg_ehr/features/build_pain_epoch_power.py',
                   params=epoch_params, subjects=[f'sub-{subject}'])

    io.write_sidecar(cache_path, kind='table',
                     script='ieeg_ehr/features/build_pain_epoch_power.py',
                     params=dict(epoch_params, dtype=str(config.CACHE_FLOAT_DTYPE),
                                 schema=CACHE_COLUMNS),
                     parents=[str(run_index[r]['path']) for r in by_run],
                     subjects=[f'sub-{subject}'],
                     extra={'n_rows': int(n_rows), 'n_epochs': len(epochs),
                            'n_pain_events': int(len(pain_df)),
                            'n_no_matching_run': n_no_match,
                            'n_boundary_drop': n_boundary,
                            'n_nonfinite_values': int(n_nonfinite)})

    logger.info('sub-%s ses-%s: %d epochs, %d rows -> %s (%.2f GB)%s',
                subject, session, len(epochs), n_rows, cache_path.name,
                cache_path.stat().st_size / 1e9,
                f'  [{n_nonfinite} non-finite]' if n_nonfinite else '')
    return cache_path, bin_edges, contains_line_noise


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--subjects', nargs='+', required=True,
                    help='Subject IDs, e.g. 071 085 (or sub-071).')
    ap.add_argument('--session', default='01')
    ap.add_argument('--epoch-minutes', type=float, default=None,
                    help=f'Pre-event window length (default: config {config.EPOCH_MINUTES_BEFORE}).')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--write-manifest', action='store_true',
                    help='Write the base unit manifest.json. One task in an array should '
                         'set this; it describes the unit, not the subject.')
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    warnings.filterwarnings('ignore', category=UserWarning, module='pynwb')

    epoch_minutes = args.epoch_minutes if args.epoch_minutes is not None else config.EPOCH_MINUTES_BEFORE
    io.warn_if_dirty()

    bin_edges = line_noise = None
    for s in args.subjects:
        subject = s.replace('sub-', '')
        for session in ([args.session] if args.session != 'all'
                        else _sessions_for(subject)):
            result = build_subject_session(subject, session, epoch_minutes,
                                           overwrite=args.overwrite)
            if result and bin_edges is None:
                _, bin_edges, line_noise = result

    if args.write_manifest and bin_edges is not None:
        unit = config.pain_epoch_unit_dir(epoch_minutes)
        io.write_manifest(unit, script='ieeg_ehr/features/build_pain_epoch_power.py',
                          params={'epoch_minutes_before': epoch_minutes,
                                  'anchor': 'pain_score_time',
                                  'window_sec': config.PSD_WINDOW_SEC,
                                  'overlap_frac': config.PSD_OVERLAP_FRAC,
                                  'n_log_bins': config.PSD_N_LOG_BINS,
                                  'dtype': str(config.CACHE_FLOAT_DTYPE),
                                  'schema': CACHE_COLUMNS,
                                  'masked': False, 'averaged': False, 'normalized': False},
                          extra={'bin_edges_hz': [float(x) for x in bin_edges],
                                 'contains_line_noise': [bool(x) for x in line_noise],
                                 'note': 'Cache is RAW slices: no QC mask, no line-noise '
                                         'filtering, no averaging, no normalization. Masking '
                                         'is a view-time join; line-noise status is derivable '
                                         'from bin_edges_hz above.'})
        logger.info('wrote unit manifest -> %s', unit / 'manifest.json')


def _sessions_for(subject):
    reg = pd.read_csv(config.FILE_REGISTRY_CSV)
    ses = reg[reg.sub_id == f'sub-{subject}'].ses_id.unique()
    return [str(s).replace('ses-', '') for s in ses]


if __name__ == '__main__':
    main()
