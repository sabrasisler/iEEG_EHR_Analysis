"""Per-assessment z tables for the post-report pairing, against the post-report baseline.

    python -m ieeg_ehr.features.build_pain_change_post_z --subjects 183 196 \\
        --mask-level bipolar --mask-label std10_rv-gross-std3_satmargin15_sw_logz4

For every pain report at t, two epoch means of (log10 power - mu) / sd per
channel x native frequency, over the 2 s windows that START in

    post  [t, t + 5 min)
    pre   [t - 5 min, t)

with mu and sd the AXIS 2 `post_report_epochs` baseline: every masked window of
every post epoch in the session, per channel x frequency. Sample SD, at least
`baseline_min_windows` windows, and a channel-epoch above `max_excluded_frac` left
out, the rule `session_baseline` applies to the pre-report windows.
`pain_change --first-epoch-post` pairs the post epoch of t1 with the pre epoch of
t2, so d_z compares [t1, t1 + 5) with [t2 - 5, t2), and consecutive pairs share no
neural window.

HOW. One pass over the continuous `preprocessed/bipolar_psd` NWBs through
`MinuteBlocks` (`build_pain_change_trajectory`), on the WORKING scale of the
stored tables (`session_baseline`, all pre-report pain epochs): per report, the ten
minute blocks [t - 5, t + 5). Their z sums, z^2 sums and counts give the post
baseline on the working scale, m and s, and every epoch mean converts exactly,
z = (z_work - m) / s, because z is linear in log power over the same windows. The
working scale keeps the sums near unit size, so the second moment does not cancel.

VALIDATION. The pre window on the working scale must reproduce the stored table
for every epoch x channel to Z_VALIDATION_TOL, NaN pattern included, or the task
raises: the check and its tolerance are `build_pain_change_shifted_z`'s.

OUTPUT: `post_zscore_dir(vc, 'post')` and `post_zscore_dir(vc, 'pre')`, one
`zmean_sub-XXX_ses-YY.parquet` per subject-session in each, in the stored schema
(epoch_id, channel, pain_score, mask_excluded_frac, one float32 column per native
frequency), rows for the same epoch x channel pairs as the stored table.
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
from ieeg_ehr.features.build_pain_change_shifted_z import Z_VALIDATION_TOL
from ieeg_ehr.features.build_pain_change_trajectory import (
    WINDOW_MIN, MinuteBlocks, session_runs, window_means)
from ieeg_ehr.views import fullres_reader, view_config
from ieeg_ehr.views import build_pain_epoch_fullres_zscore as zview

logger = logging.getLogger(__name__)

SCRIPT = 'ieeg_ehr/features/build_pain_change_post_z.py'
BASELINE = 'post_report_epochs'
WINDOW_LABELS = {'post': f'post0to{WINDOW_MIN}', 'pre': f'pre{WINDOW_MIN}to0'}


def working_vc(vc):
    """The stored tables' view args: same mask and epochs, all-pain-epoch baseline."""
    return dataclasses.replace(vc, baseline='all_pain_epochs', roi_scheme='default')


def post_params(vc):
    """The config that changes the output, hashed into the directory names."""
    vc = dataclasses.replace(vc, roi_scheme='default')
    return {**zview.zscore_params(vc, fullres_reader.n_freqs()),
            'windows': {'post': f'[t, t + {WINDOW_MIN}) min',
                        'pre': f'[t - {WINDOW_MIN}, t) min'},
            'source': 'preprocessed/bipolar_psd',
            'working_zscore_dir': zview.zscore_dir(working_vc(vc)).name}


def post_zscore_dir(vc, window, must_exist=True):
    """The `window` ('post' or 'pre') tables for these view args."""
    if vc.baseline != BASELINE:
        raise ValueError(f'the post-report tables have baseline {BASELINE}, not {vc.baseline}')
    vc = dataclasses.replace(vc, roi_scheme='default')
    params = post_params(vc)
    out = config.pain_change_zscore_dir(
        f'{zview.VIEW_LABEL_PREFIX}-{vc.scheme_code}-{WINDOW_LABELS[window]}',
        io.config_hash(params))
    if must_exist:
        if not out.exists():
            raise SystemExit(f'no post-report z tables at {out}. Build them first:\n'
                             '    sbatch sbatch/build_pain_change_post_z_array.sbatch')
        io.check_view_fresh(out, view_config=params,
                            cache_manifest=config.fullres_epoch_unit_dir(
                                vc.resolved().epoch_minutes))
    return out


def post_baseline(n, zsum, z2sum, min_windows):
    """(m, s) per channel x frequency from window counts, z sums and z^2 sums.

    Sample SD, NaN below `min_windows` windows or at zero spread, as
    `axes.BaselineAccumulator.finalize`.
    """
    with np.errstate(invalid='ignore', divide='ignore'):
        m = np.where(n >= 1, zsum / n, np.nan)
        var = (z2sum - n * m * m) / np.maximum(n - 1, 1)
        s = np.sqrt(np.clip(var, 0.0, None))
    s = np.where((n >= min_windows) & (s > 0), s, np.nan)
    return m, s


def build_session(subject, session, vc, out_dirs):
    """Write one subject-session's post and pre tables; returns the stats."""
    t0 = time.time()
    wvc = working_vc(vc)
    stored_path = zview.zscore_dir(wvc) / f'zmean_sub-{subject}_ses-{session}.parquet'
    stored = io.read_table(stored_path, on_stale='warn')
    freq_cols = fullres_reader.freq_columns()
    sess = zview.open_session(subject, session, wvc)
    min_windows = zview.zscore_params(wvc, fullres_reader.n_freqs())['baseline_min_windows']
    mean, sd, _ = zview.session_baseline(sess, min_windows)
    channels = sess.session_channels
    blocks = MinuteBlocks(session_runs(subject, session), sess, mean, sd, channels,
                          np.arange(len(freq_cols)))
    slot = {c: i for i, c in enumerate(channels)}
    max_excl = wvc.resolved().max_excluded_frac

    epochs = (sess.defs[sess.defs['epoch_id'].isin(stored['epoch_id'])]
              [['epoch_id', 'pain_time']].drop_duplicates('epoch_id')
              .assign(pain_time=lambda d: pd.to_datetime(d['pain_time']))
              .sort_values(['pain_time', 'epoch_id'], kind='stable'))
    if not (epochs['pain_time'] == epochs['pain_time'].dt.floor('min')).all():
        raise ValueError(f'sub-{subject} ses-{session}: a report is not on a whole '
                         'minute, so its windows are not whole minute blocks')

    shape = (len(channels), len(freq_cols))
    n = np.zeros(shape, dtype=np.int64)
    zsum, z2sum = (np.zeros(shape, dtype=config.CACHE_ACCUMULATE_DTYPE) for _ in range(2))
    working, diffs, nan_mismatch, n_post_kept = {}, [], 0, 0
    w = WINDOW_MIN
    for ep in epochs.itertuples(index=False):
        s, cnt, v, s2 = blocks.window(ep.pain_time - pd.Timedelta(minutes=w), 2 * w)
        pre, pre_frac = window_means(s[:w], cnt[:w], v[:w], max_excl)
        post, post_frac = window_means(s[w:], cnt[w:], v[w:], max_excl)
        keep = post_frac[0] <= max_excl
        n[keep] += cnt[w:].sum(axis=0)[keep]
        zsum[keep] += s[w:].sum(axis=0)[keep]
        z2sum[keep] += s2[w:].sum(axis=0)[keep]
        n_post_kept += int(keep.sum())

        rows = stored[stored['epoch_id'] == ep.epoch_id]
        idx = [slot[c] for c in rows['channel']]
        want = rows[freq_cols].to_numpy(dtype=config.CACHE_ACCUMULATE_DTYPE)
        got = pre[0][idx]
        nan_mismatch += int((np.isnan(want) != np.isnan(got)).sum())
        both = np.isfinite(want) & np.isfinite(got)
        diffs.append(np.abs(want[both] - got[both]))
        working[ep.epoch_id] = (rows, idx, {'pre': (pre[0], pre_frac[0]),
                                            'post': (post[0], post_frac[0])})

    diffs = np.concatenate(diffs) if diffs else np.array([])
    m, s = post_baseline(n, zsum, z2sum, min_windows)
    stats = {'n_epochs': len(epochs), 'n_post_channel_epochs_in_baseline': n_post_kept,
             'baseline_windows_median': float(np.median(n.min(axis=1))),
             'baseline_cells_nan': int(np.isnan(s).sum()),
             'n_validated_values': int(diffs.size),
             'validation_max_abs_diff': float(diffs.max()) if diffs.size else None,
             'validation_median_abs_diff': float(np.median(diffs)) if diffs.size else None,
             'nan_mismatch': nan_mismatch, 'validation_tol': Z_VALIDATION_TOL,
             'elapsed_sec': round(time.time() - t0, 1)}
    logger.info('sub-%s ses-%s: %d epochs, %d post channel-epochs in the baseline, '
                'validation %d values, max |diff| %.3g, %d NaN-pattern mismatches, %.0fs',
                subject, session, len(epochs), n_post_kept, diffs.size,
                stats['validation_max_abs_diff'] or np.nan, nan_mismatch,
                stats['elapsed_sec'])
    if nan_mismatch or (diffs.size and diffs.max() > Z_VALIDATION_TOL):
        raise AssertionError(f'sub-{subject} ses-{session}: the pre window does not '
                             f'reproduce {stored_path.name}: {stats}')

    for window, out_dir in out_dirs.items():
        frames = []
        for epoch_id, (rows, idx, by_window) in working.items():
            z_work, frac = by_window[window]
            z = (z_work[idx] - m[idx]) / s[idx]
            frame = pd.DataFrame(z.astype(config.CACHE_FLOAT_DTYPE), columns=freq_cols)
            frame.insert(0, 'epoch_id', epoch_id)
            frame.insert(1, 'channel', rows['channel'].to_numpy())
            frame.insert(2, 'pain_score', rows['pain_score'].to_numpy())
            frame.insert(3, 'mask_excluded_frac', frac[idx])
            frames.append(frame)
        io.write_table(pd.concat(frames, ignore_index=True), out_dir / stored_path.name,
                       kind='view', script=SCRIPT, params=post_params(vc),
                       float_dtype=config.CACHE_FLOAT_DTYPE,
                       parents=[str(stored_path),
                                io.manifest_ref(config.BIPOLAR_PSD_FULLRES_ROOT)],
                       subjects=[f'sub-{subject}'], extra={'window': window, **stats})
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--subjects', nargs='+', required=True,
                    help="subject ids, or 'all' for every discovery subject with stored tables")
    ap.add_argument('--task-index', type=int, default=None)
    ap.add_argument('--overwrite', action='store_true')
    view_config.add_view_arguments(ap)
    ap.set_defaults(baseline='all_pain_epochs', freq='fullres', region='none')
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    io.warn_if_dirty()

    vc = dataclasses.replace(view_config.from_args(args), baseline=BASELINE)
    stored = sorted(zview.zscore_dir(working_vc(vc)).glob('zmean_sub-*_ses-*.parquet'))
    discovery = set(config.discovery_subjects())
    if args.subjects == ['all']:
        subjects = sorted({fullres_cells.subject_session_of(p)[0] for p in stored} & discovery)
    else:
        subjects = [s.replace('sub-', '') for s in args.subjects]
    subjects = config.assert_split_allowed(subjects, 'discovery')
    if args.task_index is not None:
        subjects = [subjects[args.task_index]]

    out_dirs = {window: post_zscore_dir(vc, window, must_exist=False)
                for window in WINDOW_LABELS}
    for out_dir in out_dirs.values():
        out_dir.mkdir(parents=True, exist_ok=True)
        if not io.sidecar_path(out_dir).exists():
            io.write_view_sidecar(out_dir, view_config=post_params(vc), script=SCRIPT,
                                  cache_manifest=config.fullres_epoch_unit_dir(
                                      vc.resolved().epoch_minutes))
    for subject in subjects:
        for path in [p for p in stored
                     if fullres_cells.subject_session_of(p)[0] == subject]:
            session = fullres_cells.subject_session_of(path)[1]
            if (all((d / path.name).exists() for d in out_dirs.values())
                    and not args.overwrite):
                logger.info('sub-%s ses-%s: exists, skipping', subject, session)
                continue
            build_session(subject, session, vc, out_dirs)
    io.log_analysis(f'pain_change post-report z tables (post0to5 + pre5to0, '
                    f'{BASELINE} baseline), {len(subjects)} subject(s)', out_dirs['post'])
    print(*out_dirs.values())
    return 0


if __name__ == '__main__':
    sys.exit(main())
