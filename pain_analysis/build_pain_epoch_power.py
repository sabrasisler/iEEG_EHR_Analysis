"""
Expensive-ish cache step: for each subject, find each pain-score event's
5-minute pre-event PSD epoch, apply the raw_voltage QC mask (monopolar,
translated onto bipolar pairs), average log-power per channel/freq-bin over
the epoch, and write one CSV per subject/session
(pain_analysis/cache/sub-XXX_ses-YY_epoch_channel_power.csv).

This is the only step in pain_analysis/ that touches NWB. Region grouping,
subject weighting, and delta-from-baseline all happen later in
plot_pain_heatmaps.py so they can be iterated on without re-running this.

Each subject/session CSV gets a sidecar `*.provenance.json` recording the
mask label, epoch parameters, and git commit/dirty state used to generate
it, so any cache file on Oak can be traced back to the code + inputs that
produced it.

Run on a dev/interactive Slurm shell (never the login node):
    module load python/3.12
    source /home/groups/ckeller1/venvs/ieeg_ehr_analysis/bin/activate
    python -m pain_analysis.build_pain_epoch_power --subjects 071 085
"""

import argparse
import json
import logging

import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO

from pain_analysis import config
from qc_scripts import config as qc_config
from qc_scripts import io_utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

BIN_SEC = 60.0

CACHE_COLUMNS = [
    'subject', 'session', 'run', 'pain_event_id', 'pain_score', 'pain_bin',
    'channel', 'dk_anode_label', 'freq_bin_index', 'bin_low_hz', 'bin_high_hz',
    'mean_log_power', 'n_time_rows_used', 'frac_excluded',
]


def load_pain_scores(subject, session):
    csv_path = config.pain_scores_csv(subject, session)
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
    df['pain_bin'] = df['max_pain'].apply(config.pain_bin_for_score)
    df = df.reset_index(drop=True)
    df['pain_event_id'] = df.index
    return df


def load_mask(subject, session, mask_label):
    mask_path = config.mask_csv(subject, session, mask_label)
    if not mask_path.exists():
        logger.warning('No raw_voltage mask for sub-%s ses-%s at %s', subject, session, mask_path)
        return None
    return pd.read_csv(mask_path, usecols=['run_id', 'channel', 'bin_start', 'excluded'])


def _load_run_psd(nwb_path):
    """Open one bipolar_fft NWB and pull everything needed: PSD log-power,
    frequency bins, per-pair channel names + DK anode labels, and enough
    timing metadata to reconstruct both run-relative and absolute
    timestamps for each PSD row."""
    io = NWBHDF5IO(str(nwb_path), 'r')
    nwb = io.read()
    decomp = nwb.processing['ecephys']['psd_log_bins']

    log_power = decomp.data[:]  # (n_time, n_pairs, n_bins)
    bands = decomp.bands.to_dataframe()
    lo = bands['band_limits'].apply(lambda t: t[0]).to_numpy()
    hi = bands['band_limits'].apply(lambda t: t[1]).to_numpy()
    bin_edges = np.concatenate([lo, hi[-1:]])
    contains_line_noise = bands['contains_line_noise'].to_numpy(dtype=bool)

    elec_df = nwb.electrodes.to_dataframe()
    channel_names = list(elec_df['location'])
    if 'Desikan_Killiany_anode' in elec_df.columns:
        dk_anode_labels = list(elec_df['Desikan_Killiany_anode'])
    else:
        # Some subjects' electrode tables lack DK atlas registration entirely
        # (e.g. no volumetric parcellation run for them) -- treat every
        # channel as unmapped (None) rather than crashing; region_for_dk_label
        # already treats NaN/None as "drop, log count", so these channels
        # just won't contribute to any ROI region downstream, same as an
        # occipital/white-matter channel would.
        logger.warning('No Desikan_Killiany_anode column in electrode table for %s -- '
                        'treating all %d channels as unmapped', nwb_path, len(channel_names))
        dk_anode_labels = [None] * len(channel_names)
    rate = float(decomp.rate)
    starting_time = float(decomp.starting_time)
    session_start_time = nwb.session_start_time

    io.close()

    n_time = log_power.shape[0]
    run_seconds = starting_time + np.arange(n_time) / rate
    run_datetimes = pd.to_datetime(session_start_time) + pd.to_timedelta(run_seconds, unit='s')
    run_datetimes = run_datetimes.tz_localize(None)

    return {
        'log_power': log_power,
        'bin_edges': bin_edges,
        'contains_line_noise': contains_line_noise,
        'channel_names': channel_names,
        'dk_anode_labels': dk_anode_labels,
        'run_seconds': run_seconds,
        'run_datetimes': run_datetimes,
    }


def _find_matching_run(pain_time, window_start, runs_psd):
    """Return the run dict whose PSD covers [window_start, pain_time), or
    None. Returns ('boundary', None) instead of (None, None) if pain_time
    falls inside a run's range but window_start does not (i.e. the epoch
    would cross into a prior run) -- caller should count that separately
    from a plain no-match."""
    for run in runs_psd:
        dts = run['run_datetimes']
        if dts[0] <= pain_time <= dts[-1]:
            if window_start >= dts[0]:
                return 'match', run
            return 'boundary', None
    return 'no_match', None


def _excluded_mask(run, run_id_full, epoch_rows, mask_df):
    """(n_epoch_rows, n_pairs) bool array: True where either the anode or
    cathode monopolar channel is excluded in the raw_voltage mask at that
    row's enclosing 60s bin. All-False if mask_df is None."""
    n_sel = len(epoch_rows)
    channel_names = run['channel_names']
    n_pairs = len(channel_names)

    if mask_df is None or n_sel == 0:
        return np.zeros((n_sel, n_pairs), dtype=bool)

    anode_mono = np.array([c.split('-', 1)[0] for c in channel_names])
    cathode_mono = np.array([c.split('-', 1)[1] for c in channel_names])
    bin_starts = np.floor(run['run_seconds'][epoch_rows] / BIN_SEC) * BIN_SEC

    row_idx, pair_idx = np.meshgrid(np.arange(n_sel), np.arange(n_pairs), indexing='ij')
    long_df = pd.DataFrame({
        'bin_start': bin_starts[row_idx.ravel()],
        'anode_channel': anode_mono[pair_idx.ravel()],
        'cathode_channel': cathode_mono[pair_idx.ravel()],
    })

    run_mask = mask_df[mask_df['run_id'] == run_id_full]
    anode_excl = long_df.merge(
        run_mask.rename(columns={'channel': 'anode_channel'}),
        on=['bin_start', 'anode_channel'], how='left',
    )['excluded'].fillna(False).to_numpy()
    cathode_excl = long_df.merge(
        run_mask.rename(columns={'channel': 'cathode_channel'}),
        on=['bin_start', 'cathode_channel'], how='left',
    )['excluded'].fillna(False).to_numpy()

    return (anode_excl | cathode_excl).reshape(n_sel, n_pairs)


def process_subject_session(subject, session, mask_label, epoch_minutes, max_excluded_frac):
    pain_df = load_pain_scores(subject, session)
    if pain_df is None or pain_df.empty:
        logger.info('sub-%s ses-%s: no pain scores, skipping', subject, session)
        return None

    session_runs = io_utils.get_session_runs(subject, session)
    mask_df = load_mask(subject, session, mask_label)

    runs_psd = []
    for _, run, _raw_nwb_path in session_runs:
        nwb_path = config.bipolar_psd_nwb_path(subject, session, run)
        if not nwb_path.exists():
            logger.warning('sub-%s ses-%s run-%s: no bipolar_fft NWB at %s', subject, session, run, nwb_path)
            continue
        run_data = _load_run_psd(nwb_path)
        run_data['run'] = run
        run_data['run_id_full'] = f'run-{run}'
        runs_psd.append(run_data)

    if not runs_psd:
        logger.warning('sub-%s ses-%s: no usable bipolar_fft runs, skipping', subject, session)
        return None

    n_no_match = 0
    n_boundary_drop = 0
    n_dropped_channel_epochs = 0
    out_rows = []

    for _, pain_row in pain_df.iterrows():
        pain_bin = pain_row['pain_bin']
        if pain_bin is None:
            continue
        pain_time = pain_row['date']
        window_start = pain_time - pd.Timedelta(minutes=epoch_minutes)

        status, run = _find_matching_run(pain_time, window_start, runs_psd)
        if status == 'no_match':
            n_no_match += 1
            continue
        if status == 'boundary':
            n_boundary_drop += 1
            continue

        dts = run['run_datetimes']
        row_mask = (dts >= window_start) & (dts < pain_time)
        epoch_rows = np.where(row_mask)[0]
        if len(epoch_rows) == 0:
            n_no_match += 1
            continue

        excluded = _excluded_mask(run, run['run_id_full'], epoch_rows, mask_df)
        epoch_log_power = run['log_power'][epoch_rows]  # (n_epoch_rows, n_pairs, n_bins)
        n_epoch_rows = epoch_log_power.shape[0]

        for pair_i, channel in enumerate(run['channel_names']):
            row_excluded = excluded[:, pair_i]
            n_kept = int((~row_excluded).sum())
            frac_excluded = 1.0 - (n_kept / n_epoch_rows)
            if frac_excluded > max_excluded_frac:
                n_dropped_channel_epochs += 1
                continue

            kept_power = epoch_log_power[~row_excluded, pair_i, :]  # (n_kept, n_bins)
            if not np.all(np.isfinite(kept_power)):
                # A dead/flat channel can have literally zero stored linear
                # power at some bins (log10(0) = -inf) without ever tripping
                # the raw_voltage QC mask (e.g. a channel that's flat only at
                # this specific frequency/time, not overall). One -inf here
                # would otherwise poison this channel's mean_log_power for
                # every downstream aggregate/plot that touches it -- drop the
                # whole channel-epoch instead, same as an over-excluded one.
                n_dropped_channel_epochs += 1
                continue
            mean_log_power = kept_power.mean(axis=0)  # (n_bins,)
            dk_label = run['dk_anode_labels'][pair_i]

            for bin_i in range(len(mean_log_power)):
                if run['contains_line_noise'][bin_i]:
                    continue
                out_rows.append((
                    subject, session, run['run'], int(pain_row['pain_event_id']),
                    pain_row['max_pain'], pain_bin, channel, dk_label, bin_i,
                    run['bin_edges'][bin_i], run['bin_edges'][bin_i + 1],
                    mean_log_power[bin_i], n_kept, frac_excluded,
                ))

    logger.info(
        'sub-%s ses-%s: %d pain events, %d no matching run, %d dropped (epoch crosses run boundary), '
        '%d channel-epochs dropped (exclusion frac > %.2f)',
        subject, session, len(pain_df), n_no_match, n_boundary_drop,
        n_dropped_channel_epochs, max_excluded_frac,
    )

    if not out_rows:
        return None

    out_df = pd.DataFrame(out_rows, columns=CACHE_COLUMNS)
    out_path = config.epoch_channel_power_csv(subject, session)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    logger.info('sub-%s ses-%s: wrote %d rows to %s', subject, session, len(out_df), out_path)

    _write_provenance(subject, session, mask_label, epoch_minutes, max_excluded_frac, {
        'n_pain_events': len(pain_df),
        'n_no_matching_run': n_no_match,
        'n_boundary_drop': n_boundary_drop,
        'n_dropped_channel_epochs': n_dropped_channel_epochs,
        'n_rows_written': len(out_df),
    })
    return out_path


def _write_provenance(subject, session, mask_label, epoch_minutes, max_excluded_frac, counts):
    provenance = {
        'script': 'pain_analysis/build_pain_epoch_power.py',
        'git': qc_config.git_provenance(),
        'subject': subject,
        'session': session,
        'params': {
            'mask_label': mask_label or config.DEFAULT_MASK_LABEL,
            'mask_dir': str(config.mask_dir(mask_label)),
            'epoch_minutes': epoch_minutes,
            'max_excluded_frac': max_excluded_frac,
            'pain_bin_edges': config.PAIN_BIN_EDGES,
        },
        'inputs': {
            'pain_scores_csv': str(config.pain_scores_csv(subject, session)),
            'bipolar_psd_deriv_root': str(config.BIPOLAR_PSD_DERIV_ROOT),
        },
        'counts': counts,
    }
    prov_path = config.epoch_channel_power_provenance_json(subject, session)
    prov_path.write_text(json.dumps(provenance, indent=2))


def process_subject(subject, mask_label, epoch_minutes, max_excluded_frac):
    sessions = sorted({s for s, _run, _p in io_utils.get_session_runs(subject)})
    if not sessions:
        logger.warning('sub-%s: no runs found in file registry, skipping', subject)
        return
    for session in sessions:
        process_subject_session(subject, session, mask_label, epoch_minutes, max_excluded_frac)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subjects', nargs='+', default=None,
                         help='Subject IDs without sub- prefix (default: config.exploratory_subjects()).')
    parser.add_argument('--mask-label', default=None,
                         help=f'Raw-voltage mask label (default: {config.DEFAULT_MASK_LABEL}).')
    parser.add_argument('--epoch-minutes', type=float, default=config.EPOCH_MINUTES_BEFORE)
    parser.add_argument('--max-excluded-frac', type=float, default=config.EPOCH_MAX_EXCLUDED_FRAC)
    args = parser.parse_args()

    subjects = args.subjects or config.exploratory_subjects()
    logger.info('Processing %d subjects: %s', len(subjects), subjects)

    failed = []
    for subject in subjects:
        try:
            process_subject(subject, args.mask_label, args.epoch_minutes, args.max_excluded_frac)
        except Exception:
            # One subject's unexpected failure (e.g. a malformed/unusual NWB)
            # must not take down the rest of the batch -- log the full
            # traceback and keep going, then summarize failures at the end.
            logger.exception('sub-%s: unhandled error, skipping this subject', subject)
            failed.append(subject)

    if failed:
        logger.warning('%d/%d subjects failed and were skipped: %s', len(failed), len(subjects), failed)


if __name__ == '__main__':
    main()
