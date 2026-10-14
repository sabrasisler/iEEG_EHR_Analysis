"""Sliding-window ROI x band d_z after the first report of each pain_change pair.

    python -m ieeg_ehr.features.build_pain_change_trajectory --subjects 183 196 \\
        --reference-run <pain_change run dir> --roi-scheme roi_v3_ins \\
        --mask-level bipolar --mask-label std10_rv-gross-std3_satmargin15_sw_logz4

The extraction half of the pain_change prediction. Scoring is
`analysis/pain_change_prediction.py` and reruns per weights file without touching
this. Pairs are consecutive reports within a session, MIN_GAP_MIN-MAX_GAP_MIN
apart (`change_score.build_pairs`), so every pair's second report t2 falls inside
the horizon. With t1 the first report and a window end t = t1 + k min,
k = 0..HORIZON_MIN:

    z(t)  per channel x native frequency: the mean over the 2 s windows that START
          in [t - 5 min, t) of (log10 power - mu) / sd, the epoch rule of
          build_pain_epoch_power
    d(t)  z(t) - z(t1), averaged over the native bins of each band after the
          notch (`axes.aggregate_bands`), then over each ROI's channels

THE BASELINE IS THE STORED d_z's. mu and sd are pass 1 of
`build_pain_epoch_fullres_zscore` (`session_baseline`) on the same fullres epoch
cache: every masked window of every pain epoch in the session. Only the windows
being scored come from the continuous `preprocessed/bipolar_psd` NWBs. Power stays
log10 throughout, as in the model the weights come from.

MASKING is the epoch rule per channel: masked windows drop out, and a channel whose
excluded fraction of the 5-min window exceeds `max_excluded_frac` is NaN at that
step. A window no run covers, or whose run failed extraction, counts as excluded.
A pair whose t1 window leaves no ROI channel is dropped.

HOW. The windows sit on a 1 s grid from each run's start and reports are charted
to the minute, so a 5-min window is exactly five clock-minute blocks. Per minute
and channel x frequency the extractor keeps the sum and count of finite z, and per
channel the count of unmasked windows present. Summing five blocks gives the same
nanmean as the 300 windows, to rounding. Pairs are visited in t1 order and blocks
before the current window are freed, so memory holds about one horizon of minutes.

VALIDATION. At t = t2 the channel-level band d(t) must reproduce `d_z` from the
stored z-score tables (`pain_change.session_rows`) for every pair and channel,
NaN pattern included. A max absolute difference above VALIDATION_TOL raises.
Channel-level values exist only in memory.

OUTPUT (`config.pain_change_trajectory_dir`), per subject:
    pairs_sub-XXX.parquet  one row per pair: times, scores, per-ROI channel
                           counts, analgesics in [t1, t2), the next report
    dz_sub-XXX.parquet     one row per pair x step: t_min, overlaps_baseline,
                           masked_frac, and one float32 column per ROI x band
                           (`<roi>|<band>`), NaN where the subject lacks the cell
The validation statistics go in each dz file's sidecar.

No seizure or sleep annotations exist in this dataset (data_sop.md 4.2), so there
are no annotation-overlap columns.
"""

import argparse
import dataclasses
import json
import logging
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from ieeg_ehr import config, io
from ieeg_ehr.analysis import change_score, fullres_cells, med_state, pain_change, view_tables
from ieeg_ehr.analysis.run_bandpower_mixed import BAND_SETS
from ieeg_ehr.analysis.run_mixed_model_pilot import roi_maps
from ieeg_ehr.preprocessing.run_pipeline_bipolar import DATA_PATH, ELECTRODE_LOCATION_PATH
from ieeg_ehr.qc import mask_projection
from ieeg_ehr.views import axes, fullres_reader, view_config
from ieeg_ehr.views import build_pain_epoch_fullres_zscore as zview

logger = logging.getLogger(__name__)

SCRIPT = 'ieeg_ehr/features/build_pain_change_trajectory.py'

MIN_GAP_MIN = 30.0
MAX_GAP_MIN = 180.0
WINDOW_MIN = 5
STEP_MIN = 1
HORIZON_MIN = 180
VALIDATION_TOL = 1e-5
WINDOWS_PER_MINUTE = 60
BAND_SET = 'paper_bands_6_hg200'
LABEL = f'dz-w{WINDOW_MIN}m-s{STEP_MIN}m-h{HORIZON_MIN}m'
STARTING_TIME_PATH = DATA_PATH.replace('/data', '/starting_time')


def cell_column(roi, band):
    return f'{roi}|{band}'


def n_channels_column(roi):
    return f'n_channels|{roi}'


# ============================================================================
# Runs and minute blocks
# ============================================================================

@dataclasses.dataclass
class Run:
    run_id: str
    path: Path
    start: pd.Timestamp
    n_windows: int
    channels: list
    first_row: int = 0


def session_runs(subject, session):
    """Every bipolar_psd run of a session, in time order, with its window grid."""
    pattern = config.bipolar_psd_fullres_nwb_path(subject, session, '*')
    runs = []
    for path in sorted(pattern.parent.glob(pattern.name)):
        with h5py.File(path, 'r') as f:
            start = pd.Timestamp(f['session_start_time'].asstr()[()]).tz_localize(None)
            rate = float(f[STARTING_TIME_PATH].attrs['rate'])
            offset = float(f[STARTING_TIME_PATH][()])
            if rate != 1.0 or offset != 0.0 or start != start.floor('s'):
                raise ValueError(f'{path.name}: rate {rate}, starting_time {offset}, start '
                                 f'{start}. The minute-block grid needs 1 s windows '
                                 'from a whole-second run start.')
            runs.append(Run(path.name.split('_run-')[1].split('_')[0], path, start,
                            int(f[DATA_PATH].shape[0]),
                            list(f[ELECTRODE_LOCATION_PATH].asstr()[:])))
    if not runs:
        raise FileNotFoundError(f'no bipolar_psd runs in {pattern.parent}')
    runs = sorted(runs, key=lambda r: r.start)
    # Runs can share seconds: consecutive runs at their boundary (sub-143), or two
    # recording systems during a switch (sub-211). Any earlier run keeps a shared
    # second, as an epoch is assigned to the earlier run, and a later run's
    # duplicate rows are skipped. A run with no new second contributes nothing.
    covered = runs[0].start + pd.Timedelta(seconds=runs[0].n_windows - 1)
    for run in runs[1:]:
        run.first_row = min(run.n_windows,
                            max(0, int((covered - run.start).total_seconds()) + 1))
        if run.first_row == run.n_windows:
            logger.warning('run-%s lies inside earlier runs and is skipped', run.run_id)
        covered = max(covered, run.start + pd.Timedelta(seconds=run.n_windows - 1))
    return runs


class MinuteBlocks:
    """Per clock minute: z sums and counts per channel x bin, unmasked windows per channel."""

    def __init__(self, runs, sess, mean, sd, channels, bins):
        self.runs, self.sess = runs, sess
        index = {c: i for i, c in enumerate(sess.session_channels)}
        self.mean = mean[[index[c] for c in channels]][:, bins]
        self.sd = sd[[index[c] for c in channels]][:, bins]
        self.channels, self.bins = channels, np.asarray(bins)
        self.slot = {c: i for i, c in enumerate(channels)}
        self.blocks = {}
        self.chunk = config.PSD_FULLRES_HDF5_CHUNK_WINDOWS
        for run in runs:
            known = sess.channels_by_run.get(f'run-{run.run_id}')
            if known is not None and known != run.channels:
                raise ValueError(f'run-{run.run_id}: bipolar_psd pair names differ from '
                                 'channel_meta; the baseline rows would mislabel channels')

    def window(self, m0, n):
        """(S, N, V) stacked over the n minutes from m0: (n, C, F), (n, C, F), (n, C)."""
        for m in [m for m in self.blocks if m < m0]:
            del self.blocks[m]
        minutes = pd.date_range(m0, periods=n, freq='min')
        missing = [m for m in minutes if m not in self.blocks]
        if missing:
            if missing != list(minutes[-len(missing):]):
                raise ValueError(f'minutes from {m0} requested out of time order')
            self._compute(missing[0], len(missing))
        return tuple(np.stack([self.blocks[m][i] for m in minutes]) for i in range(3))

    def _compute(self, m0, n):
        c, f = len(self.channels), len(self.bins)
        s = np.zeros((n, c, f), dtype=config.CACHE_ACCUMULATE_DTYPE)
        cnt = np.zeros((n, c, f), dtype=np.int32)
        v = np.zeros((n, c), dtype=np.int32)
        m1 = m0 + pd.Timedelta(minutes=n)
        for run in self.runs:
            i0 = max(run.first_row, int(np.ceil((m0 - run.start).total_seconds())))
            i1 = min(run.n_windows, int(np.ceil((m1 - run.start).total_seconds())))
            if i0 >= i1:
                continue
            cols = [j for j, ch in enumerate(run.channels) if ch in self.slot]
            rows = np.array([self.slot[run.channels[j]] for j in cols], dtype=int)
            names = [run.channels[j] for j in cols]
            offset = int((run.start - m0).total_seconds())
            with h5py.File(run.path, 'r') as fh:
                data = fh[DATA_PATH]
                b0 = i0
                while b0 < i1:
                    b1 = min((b0 // self.chunk + 1) * self.chunk, i1)
                    self._fold(data[b0:b1][:, cols][:, :, self.bins], b0, b1, offset,
                               run, names, rows, s, cnt, v)
                    b0 = b1
        if (v > WINDOWS_PER_MINUTE).any():
            raise ValueError(f'more than {WINDOWS_PER_MINUTE} windows in one minute after '
                             f'{m0}: two runs overlap in time')
        for k, m in enumerate(pd.date_range(m0, periods=n, freq='min')):
            self.blocks[m] = (s[k], cnt[k], v[k])

    def _fold(self, raw, b0, b1, offset, run, names, rows, s, cnt, v):
        block = raw.astype(config.CACHE_ACCUMULATE_DTYPE)
        block[~np.isfinite(block)] = np.nan
        excluded = (mask_projection.project_pair_mask_to_windows(
                        self.sess.mask, f'run-{run.run_id}', names,
                        np.arange(b0, b1, dtype=float))
                    if self.sess.mask is not None
                    else np.zeros((b1 - b0, len(names)), dtype=bool))
        block[excluded, :] = np.nan
        z = axes.normalize(block, self.mean[rows], self.sd[rows], 'zscore_vs_baseline')
        finite = np.isfinite(z)
        minute = (offset + np.arange(b0, b1)) // WINDOWS_PER_MINUTE
        starts = np.flatnonzero(np.r_[True, np.diff(minute) != 0])
        k = minute[starts]
        s[k[:, None], rows] += np.add.reduceat(np.where(finite, z, 0.0), starts, axis=0)
        cnt[k[:, None], rows] += np.add.reduceat(finite.astype(np.int32), starts, axis=0)
        v[k[:, None], rows] += np.add.reduceat((~excluded).astype(np.int32), starts, axis=0)


# ============================================================================
# One pair
# ============================================================================

def window_means(s, cnt, v, max_excluded_frac):
    """Per step k: z mean over minute blocks [k, k + WINDOW_MIN), and excluded fraction.

    (n_steps, C, F) z with masked-out channels NaN, and (n_steps, C) fraction.
    """
    n_steps = s.shape[0] - WINDOW_MIN + 1
    zsum = np.stack([s[k:k + WINDOW_MIN].sum(axis=0) for k in range(n_steps)])
    zcnt = np.stack([cnt[k:k + WINDOW_MIN].sum(axis=0) for k in range(n_steps)])
    frac = 1.0 - (np.stack([v[k:k + WINDOW_MIN].sum(axis=0) for k in range(n_steps)])
                  / float(WINDOW_MIN * WINDOWS_PER_MINUTE))
    with np.errstate(invalid='ignore', divide='ignore'):
        zmean = np.where(zcnt > 0, zsum / zcnt, np.nan)
    zmean[frac > max_excluded_frac] = np.nan
    return zmean, frac


def pair_trajectory(blocks, t1, bin_table, bands, max_excluded_frac):
    """(channel band d, excluded fraction) per step: (n_steps, C, n_bands), (n_steps, C)."""
    s, cnt, v = blocks.window(t1 - pd.Timedelta(minutes=WINDOW_MIN),
                              WINDOW_MIN + HORIZON_MIN)
    zmean, frac = window_means(s, cnt, v, max_excluded_frac)
    d = zmean - zmean[0]
    n_steps, c, f = d.shape
    dband, names = axes.aggregate_bands(d.reshape(n_steps * c, f), bin_table,
                                        bands=bands, is_difference=True)
    if names != list(bands):
        raise ValueError(f'bands {names} are not {list(bands)}')
    return dband.reshape(n_steps, c, len(names)), frac


def roi_cells(dband, channels, region_of, regions):
    """(n_steps, R * n_bands) ROI means and (R,) contributing channels at t1."""
    n_steps, c, nb = dband.shape
    values = dband.transpose(1, 0, 2).reshape(c, n_steps * nb)
    out, _ = axes.aggregate_regions(values, channels, region_of, regions,
                                    is_difference=True)
    _, counts = axes.aggregate_regions(dband[0], channels, region_of, regions,
                                       is_difference=True)
    return out.reshape(len(regions), n_steps, nb).transpose(1, 0, 2).reshape(
        n_steps, len(regions) * nb), counts


# ============================================================================
# Pair metadata
# ============================================================================

def report_times(subject, session):
    """Every charted pain report of a session: (time, score), in time order."""
    scores = pd.read_csv(config.pain_scores_csv(subject, session), parse_dates=['date'])
    return (scores.dropna(subset=['date', 'max_pain'])
            .sort_values('date')[['date', 'max_pain']].reset_index(drop=True))


def pair_metadata(pairs, defs, admin, reports):
    """One session's pairs with times, scores, analgesics in [t1, t2) and the next report."""
    when = defs.set_index('epoch_id')['pain_time']
    rows = []
    for p in pairs.itertuples():
        t1, t2 = when[p.e1], when[p.e2]
        horizon = t1 + pd.Timedelta(minutes=HORIZON_MIN)
        given = admin[(admin['taken_dt'] >= t1) & (admin['taken_dt'] < t2)]
        later = reports[(reports['date'] > t2) & (reports['date'] <= horizon)]
        rows.append({
            'pair_id': f'{p.subject_id}_ses-{p.session}_e{p.e1}',
            'subject': p.subject_id, 'session': p.session, 'e1': p.e1, 'e2': p.e2,
            't1': t1, 't2': t2, 'gap_min': p.gap_m,
            'pain1': p.pain_1, 'pain2': p.pain_2, 'delta_pain': int(p.d_pain),
            'analgesic_in_interval': bool(len(given)),
            'analgesic_classes': '|'.join(sorted(given['level2'].unique())),
            'n_med': p.n_med, 'h_since_prior': p.h_since_prior,
            'next_report_in_window': bool(len(later)),
            'next_report_min': ((later['date'].iloc[0] - t1) / pd.Timedelta(minutes=1)
                                if len(later) else np.nan),
            'next_report_score': later['max_pain'].iloc[0] if len(later) else np.nan,
        })
    return pd.DataFrame(rows)


# ============================================================================
# One subject
# ============================================================================

def validate(dband, channels, k2, stored, stored_index):
    """|d(t2) - d_z| per channel x band, raising on any NaN-pattern disagreement."""
    have = dict(zip(stored_index['channel_uid'].str.split('|').str[-1], stored))
    diffs, nan_mismatch = [], 0
    for j, ch in enumerate(channels):
        mine = dband[k2, j]
        ref = have.get(ch, np.full(mine.shape, np.nan))
        both = np.isfinite(mine) & np.isfinite(ref)
        nan_mismatch += int((np.isfinite(mine) != np.isfinite(ref)).sum())
        diffs.append(np.abs(mine[both] - ref[both]))
    return np.concatenate(diffs), nan_mismatch


def extract_subject(subject, vc, reference, out_dir, admin, all_defs, max_pairs=None):
    """Write pairs and dz for one subject; returns the dz sidecar's validation stats.

    `max_pairs` is a smoke test: stop after that many pairs and write nothing.
    """
    t0 = time.time()
    sid = f'sub-{subject}'
    zdir = zview.zscore_dir(vc)
    zpaths = sorted(zdir.glob(f'zmean_{sid}_ses-*.parquet'))
    if not zpaths:
        raise FileNotFoundError(f'no z-score tables for {sid} in {zdir}')
    roi_by_subject, _ = roi_maps(zpaths, {sid}, reference['roi_scheme'],
                                 insula_threshold=reference['insula_threshold'])
    if sid not in roi_by_subject:
        raise ValueError(f'{sid}: no ROI-labelled channel')
    region_of = roi_by_subject[sid]
    regions = view_tables.roi_regions_for({'roi_scheme': reference['roi_scheme']})
    bands = BAND_SETS[BAND_SET]
    freq_table, _ = fullres_cells.analysis_freq_table()
    bin_table = freq_table.reset_index()
    max_excl = vc.resolved().max_excluded_frac
    min_windows = zview.zscore_params(vc, fullres_reader.n_freqs())['baseline_min_windows']
    columns = [cell_column(r, b) for r in regions for b in bands]

    defs = all_defs[all_defs['subject'] == subject]
    pairs = change_score.build_pairs(defs, admin, MIN_GAP_MIN, MAX_GAP_MIN)
    if not np.allclose(pairs['gap_m'], pairs['gap_m'].round()):
        raise ValueError(f'{sid}: a report gap is not a whole minute; t2 would fall '
                         'between steps')

    pair_rows, dz_frames, diffs = [], [], []
    stats = {'n_pairs_built': len(pairs), 'n_pairs_no_t1_channel': 0, 'nan_mismatch': 0}
    for zpath in zpaths:
        _, session = fullres_cells.subject_session_of(zpath)
        sp = pairs[pairs['session'] == session].reset_index(drop=True)
        if sp.empty:
            continue
        sess = zview.open_session(subject, session, vc)
        mean, sd, _ = zview.session_baseline(sess, min_windows)
        logger.info('%s ses-%s: baseline done, %.0fs so far', sid, session, time.time() - t0)
        channels = [c for c in sess.session_channels if c in region_of]
        blocks = MinuteBlocks(session_runs(subject, session), sess, mean, sd,
                              channels, freq_table.index.to_numpy())
        stored_index, stored = pain_change.session_rows(zpath, sp, region_of,
                                                        freq_table, bands)
        meta = pair_metadata(sp, defs[defs['session'] == session],
                             admin[(admin['subject'] == subject)
                                   & (admin['session'] == session)],
                             report_times(subject, session))
        for i in np.argsort(meta['t1'].to_numpy(), kind='stable'):
            row = meta.iloc[i]
            dband, frac = pair_trajectory(blocks, row['t1'], bin_table, bands, max_excl)
            if not np.isfinite(dband[0]).any():
                stats['n_pairs_no_t1_channel'] += 1
                continue
            k2 = int(round(row['gap_min'] / STEP_MIN))
            mine = stored_index['pair_id'].to_numpy() == sp.at[i, 'pair_id']
            d, nm = validate(dband, channels, k2, stored[mine], stored_index[mine])
            diffs.append(d)
            stats['nan_mismatch'] += nm
            cells, counts = roi_cells(dband, channels, region_of, regions)
            steps = np.arange(dband.shape[0]) * STEP_MIN
            dz = pd.DataFrame(cells, columns=columns)
            dz.insert(0, 'pair_id', row['pair_id'])
            dz.insert(1, 't_min', steps)
            dz.insert(2, 'overlaps_baseline', steps < WINDOW_MIN)
            dz.insert(3, 'masked_frac', frac.mean(axis=1))
            dz_frames.append(dz)
            pair_rows.append({**row.to_dict(), **{n_channels_column(r): int(n)
                                                  for r, n in zip(regions, counts)}})
            if max_pairs and len(pair_rows) >= max_pairs:
                break
        logger.info('%s ses-%s: %d pairs, %.0fs so far', sid, session, len(sp),
                    time.time() - t0)
        if max_pairs and len(pair_rows) >= max_pairs:
            break

    diffs = np.concatenate(diffs) if diffs else np.array([])
    stats.update({'n_pairs': len(pair_rows), 'n_validated_values': int(diffs.size),
                  'validation_max_abs_diff': float(diffs.max()) if diffs.size else None,
                  'validation_median_abs_diff': float(np.median(diffs)) if diffs.size else None,
                  'validation_tol': VALIDATION_TOL, 'elapsed_sec': round(time.time() - t0, 1)})
    logger.info('%s validation: %d values, max |diff| %.3g, median %.3g, '
                '%d NaN-pattern mismatches', sid, diffs.size,
                stats['validation_max_abs_diff'] or np.nan,
                stats['validation_median_abs_diff'] or np.nan, stats['nan_mismatch'])
    if stats['nan_mismatch'] or (diffs.size and diffs.max() > VALIDATION_TOL):
        raise AssertionError(f'{sid}: d(t2) does not reproduce the stored d_z: {stats}')
    if not pair_rows:
        raise ValueError(f'{sid}: no pair kept')
    if max_pairs:
        return stats

    params = extraction_params(vc, reference)
    parents = [str(zdir), io.manifest_ref(config.BIPOLAR_PSD_FULLRES_ROOT),
               str(reference['run_dir'])]
    pairs_out = pd.DataFrame(pair_rows)
    io.write_table(pairs_out, out_dir / f'pairs_{sid}.parquet', params=params,
                   parents=parents + [str(med_state.ADMIN_TABLE)], subjects=[sid],
                   script=SCRIPT, kind='feature')
    io.write_table(pd.concat(dz_frames, ignore_index=True), out_dir / f'dz_{sid}.parquet',
                   params=params, parents=parents, subjects=[sid], script=SCRIPT,
                   kind='feature', float_dtype=config.CACHE_FLOAT_DTYPE, extra=stats)
    return stats


# ============================================================================
# CLI
# ============================================================================

def load_reference(run_dir):
    """What the extraction must share with the fit its weights come from."""
    prov = json.loads((Path(run_dir) / 'provenance.json').read_text())
    if prov['params']['band_set'] != BAND_SET:
        raise ValueError(f'{run_dir} fit {prov["params"]["band_set"]}, not {BAND_SET}')
    split = prov.get('insula_split') or {}
    thresholds = split.get('thresholds', {})
    return {'run_dir': str(run_dir), 'roi_scheme': prov['params']['roi_scheme'],
            'view_config': prov['params']['view_config'],
            'insula_threshold': (thresholds['all'] if list(thresholds) == ['all']
                                 else thresholds or None),
            'subjects': prov['subjects']}


def extraction_params(vc, reference):
    """The config that changes the output, hashed into the directory name."""
    return {'zscore': zview.zscore_params(vc, fullres_reader.n_freqs()),
            'roi_scheme': reference['roi_scheme'],
            'insula_threshold': reference['insula_threshold'],
            'band_set': BAND_SET, 'notch_bins_removed': fullres_cells.analysis_freq_table()[1],
            'min_gap_min': MIN_GAP_MIN, 'max_gap_min': MAX_GAP_MIN,
            'window_min': WINDOW_MIN, 'step_min': STEP_MIN, 'horizon_min': HORIZON_MIN,
            'reference_run': Path(reference['run_dir']).name,
            'source': 'preprocessed/bipolar_psd', 'dtype': str(config.CACHE_FLOAT_DTYPE)}


def output_dir(vc, reference):
    return config.pain_change_trajectory_dir(LABEL, io.config_hash(extraction_params(vc, reference)))


def build_arg_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--subjects', nargs='+', required=True,
                    help="subject ids, or 'all' for the reference run's subjects "
                         'that are still discovery')
    ap.add_argument('--task-index', type=int, default=None,
                    help='extract only the subject at this index (a Slurm array task)')
    ap.add_argument('--reference-run', required=True,
                    help='the pain_change run whose fit the weights come from')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--max-pairs', type=int, default=None,
                    help='smoke test: validate this many pairs per subject, write nothing')
    view_config.add_view_arguments(ap)
    ap.set_defaults(baseline='all_pain_epochs', freq='fullres', region='none')
    return ap


def resolve_subjects(requested, reference):
    if requested == ['all']:
        subjects = sorted(set(s.replace('sub-', '') for s in reference['subjects'])
                          & set(config.discovery_subjects()))
    else:
        subjects = requested
    return config.assert_split_allowed(subjects, 'discovery')


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    io.warn_if_dirty()

    reference = load_reference(args.reference_run)
    vc = view_config.from_args(args)
    if vc.provenance() != reference['view_config']:
        raise ValueError('view arguments differ from the reference run:\n'
                         f'  here      {vc.provenance()}\n'
                         f'  reference {reference["view_config"]}')
    subjects = resolve_subjects(args.subjects, reference)
    if args.task_index is not None:
        subjects = [subjects[args.task_index]]

    out_dir = output_dir(vc, reference)
    admin = med_state.load_admin_table(subclasses=med_state.DRUG_SETS['analgesics'])
    defs = med_state.load_epoch_defs(subjects=subjects)
    if args.max_pairs:
        for subject in subjects:
            print(subject, extract_subject(subject, vc, reference, out_dir, admin, defs,
                                           max_pairs=args.max_pairs))
        return 0
    if not (out_dir / io.MANIFEST_NAME).exists():
        io.write_manifest(out_dir, params=extraction_params(vc, reference),
                          parents=[str(zview.zscore_dir(vc)), reference['run_dir']],
                          script=SCRIPT)
    for subject in subjects:
        if (out_dir / f'dz_sub-{subject}.parquet').exists() and not args.overwrite:
            logger.info('sub-%s: exists, skipping', subject)
            continue
        extract_subject(subject, vc, reference, out_dir, admin, defs)
    io.log_analysis(f'pain_change sliding-window d_z, {len(subjects)} subject(s)', out_dir)
    print(out_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())
