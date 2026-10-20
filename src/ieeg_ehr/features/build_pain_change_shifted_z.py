"""Per-assessment z-scored epoch means over a window that ends SHIFT minutes before the report.

    python -m ieeg_ehr.features.build_pain_change_shifted_z --subjects 183 196 \\
        --shift-min 5 --mask-level bipolar \\
        --mask-label std10_rv-gross-std3_satmargin15_sw_logz4

For every pain report at t, the mean over the 2 s windows that START in
[t - SHIFT - 5 min, t - SHIFT) of (log10 power - mu) / sd, per channel x native
frequency: the stored z-score tables' epoch rule (`build_pain_epoch_fullres_zscore`),
moved SHIFT minutes earlier. `pain_change --second-epoch-shift-min` takes a pair's
second epoch from these tables, so d_z compares [t1 - 5, t1) with
[t2 - SHIFT - 5, t2 - SHIFT).

THE SCALE IS THE STORED TABLES'. mu and sd are pass 1 of the stored build
(`session_baseline`: every masked window of every pain epoch in the session), and
the windows come from the continuous `preprocessed/bipolar_psd` NWBs through the
same minute blocks, mask projection and excluded-fraction rule as
`build_pain_change_trajectory`.

VALIDATION. Each report gets one stack of minute blocks [t - SHIFT - 5, t). Its last
5 minutes are the stored window [t - 5, t), which must reproduce the stored table
for every epoch x channel, NaN pattern included, or the task raises. A session with
fewer than MIN_EPOCHS epochs is skipped: no pair can use its shifted window, and its
baseline is that one epoch, whose small SD lifts float32 rounding above the
tolerance (sub-039: max |diff| 3.9e-5 where every multi-epoch session stays under
2.3e-6).

OUTPUT: `shifted_zscore_dir(vc, shift)`, beside the stored tables, one
`zmean_sub-XXX_ses-YY.parquet` per subject-session in their schema (epoch_id,
channel, pain_score, mask_excluded_frac, one float32 column per native frequency),
rows for the same epoch x channel pairs as the stored table.
"""

import argparse
import dataclasses
import logging
import sys
import time

import numpy as np
import pandas as pd

from ieeg_ehr import config, io
from ieeg_ehr.analysis import fullres_cells
from ieeg_ehr.features.build_pain_change_trajectory import (
    VALIDATION_TOL, WINDOW_MIN, MinuteBlocks, session_runs, window_means)
from ieeg_ehr.views import fullres_reader, view_config
from ieeg_ehr.views import build_pain_epoch_fullres_zscore as zview

logger = logging.getLogger(__name__)

SCRIPT = 'ieeg_ehr/features/build_pain_change_shifted_z.py'
#: A pair needs two epochs, so a session with fewer gets no shifted table.
MIN_EPOCHS = 2


def window_label(shift_min):
    """`pre10to5` for the window [t - 10, t - 5) min."""
    return f'pre{shift_min + WINDOW_MIN:g}to{shift_min:g}'


def shifted_params(vc, shift_min):
    """The config that changes the output, hashed into the directory name."""
    vc = dataclasses.replace(vc, roi_scheme='default')
    return {**zview.zscore_params(vc, fullres_reader.n_freqs()),
            'window': f'[t - {shift_min + WINDOW_MIN:g}, t - {shift_min:g}) min',
            'shift_min': shift_min, 'source': 'preprocessed/bipolar_psd',
            'stored_zscore_dir': zview.zscore_dir(vc).name}


def shifted_zscore_dir(vc, shift_min, must_exist=True):
    """The shifted tables for these view args, next to the stored ones."""
    vc = dataclasses.replace(vc, roi_scheme='default')
    params = shifted_params(vc, shift_min)
    out = config.pain_change_zscore_dir(
        f'{zview.VIEW_LABEL_PREFIX}-{vc.scheme_code}-{window_label(shift_min)}',
        io.config_hash(params))
    if must_exist:
        if not out.exists():
            raise SystemExit(f'no shifted z-score tables at {out}. Build them first:\n'
                             '    sbatch sbatch/build_pain_change_shifted_z_array.sbatch')
        io.check_view_fresh(out, view_config=params,
                            cache_manifest=config.fullres_epoch_unit_dir(
                                vc.resolved().epoch_minutes))
    return out


def build_session(subject, session, vc, shift_min, out_dir):
    """Write one subject-session's shifted table; returns its validation stats."""
    t0 = time.time()
    stored_path = zview.zscore_dir(vc) / f'zmean_sub-{subject}_ses-{session}.parquet'
    stored = io.read_table(stored_path, on_stale='warn')
    freq_cols = fullres_reader.freq_columns()
    sess = zview.open_session(subject, session, vc)
    min_windows = zview.zscore_params(vc, fullres_reader.n_freqs())['baseline_min_windows']
    mean, sd, _ = zview.session_baseline(sess, min_windows)
    channels = sess.session_channels
    blocks = MinuteBlocks(session_runs(subject, session), sess, mean, sd, channels,
                          np.arange(len(freq_cols)))
    slot = {c: i for i, c in enumerate(channels)}
    max_excl = vc.resolved().max_excluded_frac

    epochs = (sess.defs[sess.defs['epoch_id'].isin(stored['epoch_id'])]
              [['epoch_id', 'pain_time']].drop_duplicates('epoch_id')
              .assign(pain_time=lambda d: pd.to_datetime(d['pain_time']))
              .sort_values(['pain_time', 'epoch_id'], kind='stable'))
    if not (epochs['pain_time'] == epochs['pain_time'].dt.floor('min')).all():
        raise ValueError(f'sub-{subject} ses-{session}: a report is not on a whole '
                         'minute, so its window is not whole minute blocks')

    n_minutes = shift_min + WINDOW_MIN
    frames, diffs, nan_mismatch = [], [], 0
    for ep in epochs.itertuples(index=False):
        rows = stored[stored['epoch_id'] == ep.epoch_id]
        idx = [slot[c] for c in rows['channel']]
        s, cnt, v = blocks.window(ep.pain_time - pd.Timedelta(minutes=n_minutes), n_minutes)
        shifted, frac = window_means(s[:WINDOW_MIN], cnt[:WINDOW_MIN], v[:WINDOW_MIN],
                                     max_excl)
        check, _ = window_means(s[shift_min:], cnt[shift_min:], v[shift_min:], max_excl)
        want = rows[freq_cols].to_numpy(dtype=config.CACHE_ACCUMULATE_DTYPE)
        got = check[0][idx]
        nan_mismatch += int((np.isnan(want) != np.isnan(got)).sum())
        both = np.isfinite(want) & np.isfinite(got)
        diffs.append(np.abs(want[both] - got[both]))
        frame = pd.DataFrame(shifted[0][idx].astype(config.CACHE_FLOAT_DTYPE),
                             columns=freq_cols)
        frame.insert(0, 'epoch_id', ep.epoch_id)
        frame.insert(1, 'channel', rows['channel'].to_numpy())
        frame.insert(2, 'pain_score', rows['pain_score'].to_numpy())
        frame.insert(3, 'mask_excluded_frac', frac[0][idx])
        frames.append(frame)

    diffs = np.concatenate(diffs) if diffs else np.array([])
    stats = {'n_epochs': len(epochs), 'n_validated_values': int(diffs.size),
             'validation_max_abs_diff': float(diffs.max()) if diffs.size else None,
             'validation_median_abs_diff': float(np.median(diffs)) if diffs.size else None,
             'nan_mismatch': nan_mismatch, 'validation_tol': VALIDATION_TOL,
             'elapsed_sec': round(time.time() - t0, 1)}
    logger.info('sub-%s ses-%s: %d epochs, validation %d values, max |diff| %.3g, '
                '%d NaN-pattern mismatches, %.0fs', subject, session, len(epochs),
                diffs.size, stats['validation_max_abs_diff'] or np.nan, nan_mismatch,
                stats['elapsed_sec'])
    if nan_mismatch or (diffs.size and diffs.max() > VALIDATION_TOL):
        raise AssertionError(f'sub-{subject} ses-{session}: the unshifted window does not '
                             f'reproduce {stored_path.name}: {stats}')

    out_path = out_dir / stored_path.name
    io.write_table(pd.concat(frames, ignore_index=True), out_path, kind='view',
                   script=SCRIPT, params=shifted_params(vc, shift_min),
                   float_dtype=config.CACHE_FLOAT_DTYPE,
                   parents=[str(stored_path), io.manifest_ref(config.BIPOLAR_PSD_FULLRES_ROOT)],
                   subjects=[f'sub-{subject}'], extra=stats)
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--subjects', nargs='+', required=True,
                    help="subject ids, or 'all' for every discovery subject with stored tables")
    ap.add_argument('--task-index', type=int, default=None)
    ap.add_argument('--shift-min', type=int, default=5)
    ap.add_argument('--overwrite', action='store_true')
    view_config.add_view_arguments(ap)
    ap.set_defaults(baseline='all_pain_epochs', freq='fullres', region='none')
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    io.warn_if_dirty()
    if args.shift_min < 1:
        ap.error('--shift-min must be at least 1: 0 is the stored tables')

    vc = view_config.from_args(args)
    stored = sorted(zview.zscore_dir(vc).glob('zmean_sub-*_ses-*.parquet'))
    discovery = set(config.discovery_subjects())
    if args.subjects == ['all']:
        subjects = sorted({fullres_cells.subject_session_of(p)[0] for p in stored} & discovery)
    else:
        subjects = [s.replace('sub-', '') for s in args.subjects]
    subjects = config.assert_split_allowed(subjects, 'discovery')
    if args.task_index is not None:
        subjects = [subjects[args.task_index]]

    out_dir = shifted_zscore_dir(vc, args.shift_min, must_exist=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not io.sidecar_path(out_dir).exists():
        io.write_view_sidecar(out_dir, view_config=shifted_params(vc, args.shift_min),
                              script=SCRIPT,
                              cache_manifest=config.fullres_epoch_unit_dir(
                                  vc.resolved().epoch_minutes))
    for subject in subjects:
        for path in [p for p in stored
                     if fullres_cells.subject_session_of(p)[0] == subject]:
            session = fullres_cells.subject_session_of(path)[1]
            if (out_dir / path.name).exists() and not args.overwrite:
                logger.info('sub-%s ses-%s: exists, skipping', subject, session)
                continue
            n_epochs = io.read_table(path, columns=['epoch_id'], on_stale='ignore')[
                'epoch_id'].nunique()
            if n_epochs < MIN_EPOCHS:
                logger.info('sub-%s ses-%s: %d epoch(s), no pair can use it, skipping',
                            subject, session, n_epochs)
                continue
            build_session(subject, session, vc, args.shift_min, out_dir)
    io.log_analysis(f'pain_change shifted z tables {window_label(args.shift_min)}, '
                    f'{len(subjects)} subject(s)', out_dir)
    print(out_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())
