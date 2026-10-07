"""Per-window z-scored full-resolution spectra, averaged per epoch. One row per (epoch, channel).

    python -m ieeg_ehr.views.build_pain_epoch_fullres_zscore --subjects 019 \\
        --mask-label std10_rv-gross-std3_satmargin15_sw_logz4

The input of the `pain_change` question (PLANNING.md, named questions). For each
subject-session, each channel and each native 0.5 Hz frequency:

    pass 1  mean and SD of log power over every masked 2 s window of EVERY pain
            epoch in the session (AXIS 2 `all_pain_epochs`)
    pass 2  z-score each 2 s window against that mean and SD (AXIS 3), then
            average the z-scores over the epoch's windows (AXIS 4)

Two streaming passes because the baseline must be complete before any window can
be normalized, and a subject's cache (~3.6 GB float32) does not fit in memory as
float64. The baseline is keyed on channel NAME across runs, as in
`build_pain_epoch_view`, so a channel keeps one baseline when montages differ
between runs.

Under this baseline the window-weighted mean of every channel x frequency is 0
by construction, and its window SD is 1. An epoch's value is therefore where
that epoch sits within the session's own distribution, in units of the session's
window-to-window SD. Because the baseline is a fixed per-(channel, frequency)
scalar, the epoch mean of z equals (epoch mean of log power - mu) / sd (registry
AXIS 3 note). Pass 2 still normalizes per window, so the per-window values exist
for any later within-epoch question without changing this builder.

A MATERIALIZED view, on the same grounds as `build_pain_epoch_fullres_mean`:
recomputing means two passes over the ~226 GB per-window cache, and the
`pain_change` analysis plus its later plots all read it. It lives inside the
question, `analysis/pain/pain_change/zscore_epochs/` (config.pain_change_zscore_dir),
because `pain_change` is its only consumer. The line-noise notch is
NOT applied here. It stays a view-time decision for the consumer
(`fullres_reader.notch_freqs`).
"""

import argparse
import dataclasses
import logging
import sys
import time

import numpy as np
import pandas as pd

from ieeg_ehr import config, io
from ieeg_ehr.views import axes, fullres_reader, view_config
from ieeg_ehr.views import build_pain_epoch_fullres_mean as meanview

logger = logging.getLogger(__name__)

SCRIPT = 'ieeg_ehr/views/build_pain_epoch_fullres_zscore.py'

VIEW_LABEL_PREFIX = 'fullresz'


def zscore_params(vc, n_freqs):
    """The config that changes the output, hashed into the view's directory name."""
    r = vc.resolved()
    return {
        'metric': 'epoch_mean_window_zscore',
        'source_unit': 'psd_epochs_fullres',
        'epoch_minutes': r.epoch_minutes,
        'mask_level': vc.mask_level,
        'mask_label': vc.mask_label,
        'max_excluded_frac': r.max_excluded_frac,
        'domain': 'log',
        'baseline': vc.baseline,
        'normalization': vc.normalization,
        'epoch_agg': 'mean',
        'baseline_min_windows': 2,
        'n_freqs': int(n_freqs),
        'dtype': str(config.CACHE_FLOAT_DTYPE),
        'accumulate_dtype': str(config.CACHE_ACCUMULATE_DTYPE),
    }


def zscore_dir(vc):
    """The z-score table directory these view args describe, checked for freshness.

    The builder runs with the default ROI scheme and `scheme_code` includes the
    ROI scheme, so the label is built from a copy with it reset.
    """
    vc = dataclasses.replace(vc, roi_scheme='default')
    epoch_minutes = vc.resolved().epoch_minutes
    params = zscore_params(vc, fullres_reader.n_freqs(epoch_minutes))
    out = config.pain_change_zscore_dir(f'{VIEW_LABEL_PREFIX}-{vc.scheme_code}',
                                        io.config_hash(params))
    if not out.exists():
        raise SystemExit(f'no z-score tables at {out}. Build them first:\n'
                         '    sbatch --array=0-80 sbatch/build_fullres_zscore_array.sbatch')
    io.check_view_fresh(out, view_config=params,
                        cache_manifest=config.fullres_epoch_unit_dir(epoch_minutes))
    return out


def build_subject_session(subject, session, vc, overwrite=False):
    """(path, stats) when built, (path, None) when it already exists, None on no cache."""
    t0 = time.time()
    epoch_minutes = vc.resolved().epoch_minutes

    freqs = fullres_reader.freqs_hz(epoch_minutes)
    params = zscore_params(vc, len(freqs))
    out_dir = config.pain_change_zscore_dir(
        f'{VIEW_LABEL_PREFIX}-{vc.scheme_code}', io.config_hash(params))
    out_path = out_dir / f'zmean_sub-{subject}_ses-{session}.parquet'
    if out_path.exists() and not overwrite:
        logger.info('sub-%s ses-%s: exists, skipping', subject, session)
        return out_path, None

    try:
        pf, cache_path = fullres_reader.open_cache(subject, session, epoch_minutes)
    except FileNotFoundError as e:
        logger.warning('sub-%s ses-%s: %s -- skipping', subject, session, e)
        return None

    defs = fullres_reader.load_defs(subject, session, epoch_minutes)
    defs = defs.sort_values('epoch_id').reset_index(drop=True)
    present = meanview._epochs_present(pf)
    rgs = fullres_reader.verify_layout(pf, defs, len(freqs), epochs_present=present)

    mask = fullres_reader.load_mask(subject, session, vc)
    channels_by_run = meanview._channels_by_run(subject, session, epoch_minutes)
    session_channels = sorted({c for chans in channels_by_run.values() for c in chans})
    channel_index = {c: i for i, c in enumerate(session_channels)}
    rows_by_run = {run: np.array([channel_index[c] for c in chans], dtype=int)
                   for run, chans in channels_by_run.items()}

    def epochs(epoch_filter=None):
        return fullres_reader.iter_epochs(
            pf, defs, rgs, mask=mask, channels_by_run=channels_by_run,
            view_config=vc, epoch_filter=epoch_filter, epoch_minutes=epoch_minutes)

    # ---------------- pass 1: baseline ----------------
    acc = axes.BaselineAccumulator(len(session_channels), len(freqs))
    for ep, block, _kept, _frac in epochs(axes.baseline_epoch_filter(vc.baseline)):
        acc.update(block, rows=rows_by_run[ep['run_id']])
    if acc.n_epochs == 0:
        raise ValueError(f'sub-{subject} ses-{session}: no {vc.baseline} epochs, '
                         'so there is no baseline to z-score against')
    base_mean, base_sd = acc.finalize(min_windows=params['baseline_min_windows'])
    n_base_windows = acc.count.min(axis=1)

    # ---------------- pass 2: z per window, mean per epoch ----------------
    stats = {'n_epochs': 0, 'n_baseline_epochs': acc.n_epochs,
             'n_channels_session': len(session_channels),
             'n_channels_no_baseline': int((~np.isfinite(base_sd).any(axis=1)).sum()),
             'n_all_nan_channel_epochs': 0, 'n_nonfinite_input': 0}
    frames = []
    for ep, block, _kept, frac in epochs():
        rows = rows_by_run[ep['run_id']]
        # The cache holds a few isolated non-finite values (one window at one
        # frequency). The baseline already skips them; masking them here keeps
        # one -inf from turning the whole epoch mean at that frequency into -inf.
        nonfinite = ~np.isfinite(block) & ~np.isnan(block)
        stats['n_nonfinite_input'] += int(nonfinite.sum())
        block[nonfinite] = np.nan
        z = axes.normalize(block, base_mean[rows], base_sd[rows], vc.normalization)
        zmean = axes.epoch_mean(z)                              # (n_pairs, n_freq)
        stats['n_all_nan_channel_epochs'] += int((~np.isfinite(zmean).any(axis=1)).sum())

        frame = pd.DataFrame(zmean.astype(config.CACHE_FLOAT_DTYPE),
                             columns=fullres_reader.freq_columns(epoch_minutes))
        frame.insert(0, 'epoch_id', int(ep['epoch_id']))
        frame.insert(1, 'channel', channels_by_run[ep['run_id']])
        frame.insert(2, 'pain_score', float(ep['pain_score']))
        frame.insert(3, 'n_windows_used', (~np.isnan(block[:, :, 0])).sum(axis=0))
        frame.insert(4, 'mask_excluded_frac', frac)
        frame.insert(5, 'n_baseline_windows', n_base_windows[rows])
        frames.append(frame)
        stats['n_epochs'] += 1

    table = pd.concat(frames, ignore_index=True)
    io.write_table(table, out_path, kind='view', script=SCRIPT, params=params,
                   float_dtype=config.CACHE_FLOAT_DTYPE,
                   parents=[io.manifest_ref(config.fullres_epoch_unit_dir(epoch_minutes)),
                            str(cache_path)],
                   subjects=[f'sub-{subject}'],
                   extra={**stats, 'elapsed_sec': round(time.time() - t0, 1)})
    # On the view DIRECTORY and only if absent, so it does not overwrite the
    # table's own sidecar (see build_pain_epoch_fullres_mean.py).
    if not io.sidecar_path(out_dir).exists():
        io.write_view_sidecar(
            out_dir, view_config=params, script=SCRIPT,
            cache_manifest=config.fullres_epoch_unit_dir(epoch_minutes))

    logger.info('sub-%s ses-%s: %d epochs (%d in baseline) x %d channels -> %s '
                '(%.0fs), %d channels without a baseline, %d all-NaN channel-epochs, '
                '%d non-finite input values masked',
                subject, session, stats['n_epochs'], acc.n_epochs,
                len(session_channels), out_path.name, time.time() - t0,
                stats['n_channels_no_baseline'], stats['n_all_nan_channel_epochs'],
                stats['n_nonfinite_input'])
    return out_path, stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--subjects', nargs='+', required=True)
    ap.add_argument('--session', default='all')
    ap.add_argument('--overwrite', action='store_true')
    view_config.add_view_arguments(ap)
    ap.set_defaults(baseline='all_pain_epochs', freq='fullres', region='none')
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    io.warn_if_dirty()

    vc = view_config.from_args(args)
    if vc.normalization != 'zscore_vs_baseline':
        ap.error(f'--normalization {vc.normalization!r} refused: this view IS the '
                 'z-scored epoch mean. Use build_pain_epoch_fullres_mean for raw.')

    n_built, n_existing, out_dir = 0, 0, None
    for s in args.subjects:
        subject = s.replace('sub-', '')
        for session in ([args.session] if args.session != 'all'
                        else meanview._sessions_for(subject)):
            r = build_subject_session(subject, session, vc, overwrite=args.overwrite)
            if r is None:
                continue
            out_dir = r[0].parent
            if r[1] is None:
                n_existing += 1
            else:
                n_built += 1
    if n_built:
        io.log_analysis(f'full-res per-window z-scored epoch means ({vc.baseline}), '
                        f'{n_built} subject-session(s)', out_dir)
    # An existing table is a success: a rerun of a finished array must not fail.
    return 0 if n_built or n_existing else 1


if __name__ == '__main__':
    sys.exit(main())
