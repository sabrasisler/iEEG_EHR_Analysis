"""QC-masked, epoch-averaged full-resolution spectra. One row per (epoch, channel).

    python -m ieeg_ehr.views.build_pain_epoch_fullres_mean --subjects 019

A MATERIALIZED view of `features/pain/psd_epochs_fullres/` -- which the default is
explicitly against (`architecture.md` PART 2: recompute, don't save). This is the
exception the rule anticipates: "materialize ONLY when recompute is measured slow
AND something depends on it." Both halves hold here.

  - MEASURED SLOW: the per-window cache is ~226 GB and a full-cohort pass takes
    ~7 min of pure decode. The epoch mean is ~1.2 GB -- a ~190x reduction, because
    averaging 300 windows is what the reduction IS.
  - SOMETHING DEPENDS ON IT: specparam (BG.3) cannot use per-window data at all. A
    single 2 s Hann periodogram is chi-squared with 2 dof, so its standard
    deviation EQUALS its mean -- ~100% variance per bin, and a peak fit on that is
    fitting noise. Averaging ~300 half-overlapping windows (~150 independent) cuts
    that ~12x and is what makes a spectrum smooth enough to parameterize. And
    FOOOF's parameters are a sweep axis, so every arm would otherwise re-read
    226 GB.

So this sits AFTER view AXIS 4 and before anything that fits a spectrum.

WHAT IT DOES AND DOES NOT DO. Masking: yes (that is a choice-independent data
fact, and it must happen BEFORE averaging or an artifact window contaminates the
mean). Normalization: NO -- deliberately. A baseline-subtracted or z-scored
spectrum's aperiodic component is not the aperiodic component, and its peaks are
not its peaks; `build_pain_epoch_slope.py` makes the same refusal for the same
reason. Normalization stays a downstream view of this.

AVERAGING IS LINEAR-THEN-LOG, IN FLOAT64. The cache stores log10 power, so the
mean is `log10(mean(10**x))`, not `mean(x)`. Those are different quantities -- the
second is a geometric mean -- and the arithmetic one is what a power spectrum
means. Both the exponentiation and the accumulation run in float64
(`CACHE_LINEAR_DOMAIN_DTYPE`, `CACHE_ACCUMULATE_DTYPE`): the worst stored
log-power is ~-36.8, barely a decade above float32's smallest normal, and a
float32 accumulator over ~300 windows holds only ~6 significant figures (P0.6).
"""

import argparse
import logging
import sys
import warnings
import time

import numpy as np
import pandas as pd

from ieeg_ehr import config, io
from ieeg_ehr.views import axes, cache_reader, fullres_reader, view_config

logger = logging.getLogger(__name__)

SCRIPT = 'ieeg_ehr/views/build_pain_epoch_fullres_mean.py'

#: The human half of the view directory name. Prefixed so a mean view can never
#: be confused with a power or slope view in the same `views/` tree.
VIEW_LABEL_PREFIX = 'fullresmean'


def mean_params(vc, n_freqs):
    """The config that changes the output, hashed into the view's directory name.

    `epoch_agg` is pinned to linear_then_log rather than taken from the ViewConfig:
    this artifact IS the arithmetic epoch mean, and a geometric one would be a
    different artifact that happens to have the same shape.
    """
    return {
        'metric': 'epoch_mean_log_power',
        'source_unit': 'psd_epochs_fullres',
        'epoch_minutes': vc.resolved().epoch_minutes,
        'mask_level': vc.mask_level,
        'mask_label': vc.mask_label,
        'max_excluded_frac': vc.resolved().max_excluded_frac,
        'epoch_agg': 'linear_then_log',
        'normalization': 'none',
        'domain': 'log',
        'n_freqs': int(n_freqs),
        'dtype': str(config.CACHE_FLOAT_DTYPE),
        'accumulate_dtype': str(config.CACHE_ACCUMULATE_DTYPE),
    }


def epoch_mean_log(block):
    """(n_win, n_pairs, n_freq) log10 -> (n_pairs, n_freq) log10 of the MEAN.

    NaN-aware on the window axis, because masking sets whole (window, channel)
    cells to NaN rather than deleting them -- every pair keeps its slot on axis 1
    so `channels` stays synchronised with the array.

    A channel whose every window was masked yields all-NaN here rather than a
    number, which is the honest answer; the caller records the count.
    """
    linear = np.power(10.0, np.asarray(block, dtype=config.CACHE_LINEAR_DOMAIN_DTYPE))
    with np.errstate(divide='ignore', invalid='ignore'), \
            warnings.catch_warnings():
        # "Mean of empty slice" is the EXPECTED outcome for a channel whose every
        # window the QC mask excluded, and NaN is the answer we want there. Left
        # as a warning it fires once per fully-masked channel-epoch and buries the
        # log; the count is reported in the sidecar instead.
        warnings.filterwarnings('ignore', message='Mean of empty slice',
                                category=RuntimeWarning)
        mean = np.nanmean(linear, axis=0, dtype=config.CACHE_ACCUMULATE_DTYPE)
        return np.log10(mean)


def build_subject_session(subject, session, vc, epoch_minutes=None, overwrite=False):
    """Returns (path, stats) or None."""
    t0 = time.time()
    epoch_minutes = vc.resolved().epoch_minutes if epoch_minutes is None else epoch_minutes

    freqs = fullres_reader.freqs_hz(epoch_minutes)
    params = mean_params(vc, len(freqs))
    out_dir = config.fullres_epoch_views_dir(
        f'{VIEW_LABEL_PREFIX}-{vc.scheme_code}', io.config_hash(params), epoch_minutes)
    out_path = out_dir / f'mean_sub-{subject}_ses-{session}.parquet'
    if out_path.exists() and not overwrite:
        logger.info('sub-%s ses-%s: exists, skipping', subject, session)
        return None

    try:
        pf, cache_path = fullres_reader.open_cache(subject, session, epoch_minutes)
    except FileNotFoundError as e:
        logger.warning('sub-%s ses-%s: %s -- skipping', subject, session, e)
        return None

    defs = fullres_reader.load_defs(subject, session, epoch_minutes)
    defs = defs.sort_values('epoch_id').reset_index(drop=True)
    defs['pain_bin'] = axes.assign_pain_bins(defs, vc.pain_bins)

    # An epoch the extractor skipped (short raw read) is in defs but not in the
    # cache, so the layout walk is told which epochs to expect rather than
    # inferring a mismatch as corruption.
    present = _epochs_present(pf)
    rgs = fullres_reader.verify_layout(pf, defs, len(freqs), epochs_present=present)

    mask = fullres_reader.load_mask(subject, session, vc)
    channels_by_run = _channels_by_run(subject, session, epoch_minutes)

    rows, stats = [], {'n_epochs': 0, 'n_all_nan_channels': 0, 'n_channels': 0}
    for ep, block, kept, frac in fullres_reader.iter_epochs(
            pf, defs, rgs, mask=mask, channels_by_run=channels_by_run,
            view_config=vc, epoch_minutes=epoch_minutes):
        channels = channels_by_run[ep['run_id']]
        mean = epoch_mean_log(block)                      # (n_pairs, n_freq)
        allnan = ~np.isfinite(mean).any(axis=1)
        stats['n_all_nan_channels'] += int(allnan.sum())
        stats['n_channels'] = mean.shape[0]

        frame = pd.DataFrame(mean.astype(config.CACHE_FLOAT_DTYPE),
                             columns=fullres_reader.freq_columns(epoch_minutes))
        frame.insert(0, 'epoch_id', int(ep['epoch_id']))
        frame.insert(1, 'channel', channels)
        frame.insert(2, 'pain_score', float(ep['pain_score']))
        frame.insert(3, 'pain_bin', ep['pain_bin'])
        frame.insert(4, 'n_windows_used', (~np.isnan(block[:, :, 0])).sum(axis=0))
        frame.insert(5, 'mask_excluded_frac', frac)
        rows.append(frame)
        stats['n_epochs'] += 1

    if not rows:
        logger.warning('sub-%s ses-%s: no epochs produced a mean', subject, session)
        return None

    table = pd.concat(rows, ignore_index=True)
    io.write_table(table, out_path, kind='view', script=SCRIPT, params=params,
                   float_dtype=config.CACHE_FLOAT_DTYPE,
                   parents=[io.manifest_ref(config.fullres_epoch_unit_dir(epoch_minutes)),
                            str(cache_path)],
                   subjects=[f'sub-{subject}'],
                   extra={**stats, 'elapsed_sec': round(time.time() - t0, 1),
                          'freqs_hz_first_last': [float(freqs[0]), float(freqs[-1])]})
    # On the view DIRECTORY, not the table file, and only if absent -- pointing it
    # at `out_path` overwrites the provenance sidecar `write_table` just wrote and
    # discards every `extra` counter (see build_pain_epoch_fooof.py for the case
    # where that silently zeroed a reported number). Same pattern as
    # `build_pain_epoch_slope.py:360`.
    if not io.sidecar_path(out_dir).exists():
        io.write_view_sidecar(
            out_dir, view_config=params, script=SCRIPT,
            cache_manifest=config.fullres_epoch_unit_dir(epoch_minutes))

    logger.info('sub-%s ses-%s: %d epochs x %d channels -> %s (%.1f MB, %.0fs)%s',
                subject, session, stats['n_epochs'], stats['n_channels'],
                out_path.name, out_path.stat().st_size / 1e6, time.time() - t0,
                f"  [{stats['n_all_nan_channels']} all-NaN channel-epochs]"
                if stats['n_all_nan_channels'] else '')
    return out_path, stats


def _epochs_present(parquet_file):
    """epoch_ids actually in the cache, read from the index column only."""
    ids = set()
    for i in range(parquet_file.num_row_groups):
        t = parquet_file.read_row_groups([i], columns=['epoch_id'])
        ids.update(np.unique(t.column('epoch_id').to_numpy()).tolist())
    return sorted(int(x) for x in ids)


def _channels_by_run(subject, session, epoch_minutes):
    """{run_id: [pair names in cache column order]} from this unit's channel_meta."""
    path = config.fullres_epoch_channel_meta_path(subject, session, epoch_minutes)
    if not path.exists():
        path = config.pain_epoch_channel_meta_path(subject, session, epoch_minutes)
    if not path.exists():
        raise FileNotFoundError(
            f'no channel_meta for sub-{subject} ses-{session}. Masking needs the '
            'pair names, and an empty list would read as "nothing excluded" -- the '
            'failure mode that makes incomplete QC look like clean data. Build it '
            f'with `python -m ieeg_ehr.views.channel_meta --subjects {subject}`.')
    meta = io.read_table(path, on_stale='ignore')
    return {run: list(g.sort_values('pair_index')['channel'])
            for run, g in meta.groupby('run_id')}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--subjects', nargs='+', required=True)
    ap.add_argument('--session', default='01')
    ap.add_argument('--overwrite', action='store_true')
    view_config.add_view_arguments(ap)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    io.warn_if_dirty()

    vc = view_config.from_args(args)
    if vc.normalization != 'none':
        ap.error(
            f'--normalization {vc.normalization!r} refused. This artifact is the '
            'RAW epoch-mean spectrum, because it is what specparam and the slope '
            'fit read: a baseline-subtracted or z-scored spectrum\'s aperiodic '
            'component is not the aperiodic component, and dividing each bin by '
            'its own baseline SD rescales the y-axis per frequency. Normalization '
            'is a downstream view of this one.')

    n_ok = 0
    out_dir = None
    for s in args.subjects:
        subject = s.replace('sub-', '')
        for session in ([args.session] if args.session != 'all'
                        else _sessions_for(subject)):
            r = build_subject_session(subject, session, vc,
                                      overwrite=args.overwrite)
            if r is not None:
                n_ok += 1
                out_dir = r[0].parent
    if n_ok and out_dir is not None:
        io.log_analysis(f'full-res epoch-mean spectra, {n_ok} subject-session(s)',
                        out_dir)
    return 0 if n_ok else 1


def _sessions_for(subject):
    reg = pd.read_csv(config.FILE_REGISTRY_CSV)
    return [str(s).replace('ses-', '')
            for s in reg[reg.sub_id == f'sub-{subject}'].ses_id.unique()]


if __name__ == '__main__':
    sys.exit(main())
