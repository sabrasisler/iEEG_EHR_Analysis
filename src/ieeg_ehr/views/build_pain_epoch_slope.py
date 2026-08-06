#!/usr/bin/env python3
"""
BG.2: per-epoch 1/f slope, per channel, averaged into regions.

The sibling of `build_pain_epoch_view.py`. Same cache, same mask, same cohort
gate, same epoch definitions, same output layout -- one number per (epoch,
region) instead of a spectrum:

    epoch's 2s windows -> mean log-power per (channel, freq bin)   [AXIS 4]
      -> OLS of log-power on log10(f), PER CHANNEL                 [the slope]
      -> arithmetic mean of slopes within an ROI                   [AXIS 6]

A SEPARATE SCRIPT, not a flag on the view builder, for two reasons. The
architecture doc calls this a sibling table; and adding a parameter to the view
builder would change its config_hash and therefore orphan the five power views
already on Oak, which nothing about a new derived quantity should do.

ONE PASS, NOT TWO. The view builder's first pass exists only to accumulate the
0-pain baseline. A slope needs raw log-power (see below), so there is no baseline
to build and the pass is skipped -- this runs at roughly half the cost of a power
view of the same subject.

WHY RAW LOG-POWER, AND WHY THAT IS ENFORCED
-------------------------------------------
`--normalization none` is the only value accepted, and the script refuses the
others rather than warning. A z-scored spectrum's slope is not the aperiodic
slope: dividing each bin by its own baseline SD rescales the y-axis
bin-by-bin, which tilts the line by whatever the SD's own frequency dependence
happens to be. `baseline_subtract` is better behaved -- it gives the CHANGE in
tilt versus the 0-pain baseline, a real and interesting quantity -- but it is a
different quantity with a different name, and it cannot be plotted against a
'none' violin because 0-pain would be 0 by construction (the circularity
docs/cluster_permutation.md 6 and plot_band_violin_view.py both document).

WHY PER CHANNEL AND NOT PER REGION-AVERAGED SPECTRUM
-----------------------------------------------------
Region-averaging raw log power is linear-then-log -- log10(mean_c 10**x_c) --
which is dominated by the loudest channel in the region. Fitting THAT gives
approximately the loudest channel's slope, dressed up as a regional one. Fitting
each channel and averaging the slopes weights channels equally, which is what the
figure claims to show. Slopes average arithmetically; see views/aperiodic.py.

Run on Slurm, never the login node:
    python -m ieeg_ehr.views.build_pain_epoch_slope --subjects 019 \\
        --mask-label std10_rv-gross-std3_satmargin15_sw_logz4 --roi-scheme roi_v2
"""

import argparse
import dataclasses
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ieeg_ehr import config, io
from ieeg_ehr.config import cohorts, roi_schemes
from ieeg_ehr.qc import psd_timing
from ieeg_ehr.views import (aperiodic, axes, cache_reader, channel_meta,
                            view_config as vc)

logger = logging.getLogger(__name__)

# The human half of the views/ directory name. Prefixed so a slope view can never
# be mistaken for -- or globbed alongside -- a power view of the same axes.
VIEW_LABEL_PREFIX = 'slope'


def slope_params(args, drop_bins, fit_bin_idx):
    """The slope-specific half of the hashed config.

    Hashed, not merely logged: two fit ranges are two different numbers for the
    same epoch, and they must not be able to land in the same directory.
    """
    return {
        'metric': 'aperiodic_slope',
        'fit_lo_hz': args.fit_lo_hz,
        'fit_hi_hz': args.fit_hi_hz,
        'fit_drop_line_noise_bins': not args.keep_line_noise_bins,
        'fit_line_noise_bins_dropped': [int(b) for b in drop_bins],
        'fit_bins_available': [int(b) for b in fit_bin_idx],
        'min_fit_bins': args.min_fit_bins,
        'min_span_decades': args.min_span_decades,
        'fit_level': 'per_channel_then_region_mean',
        'slope_note':
            'OLS of epoch-mean log10 power on log10(geometric bin centre), fit per '
            'bipolar channel and then averaged ARITHMETICALLY within an ROI. A '
            'broadband tilt over the fitted range, NOT a knee-free specparam '
            'exponent: the range spans the low-frequency knee and the alpha/beta '
            'peaks. More negative = steeper spectrum.',
    }


def build_subject_slopes(subject, session, view_config, args, epoch_minutes=None,
                         nonstandard_hop='refuse'):
    """Returns (epoch_table, subject_table, stats) for one subject/session."""
    t0 = time.time()
    epoch_minutes = epoch_minutes or view_config.epoch_minutes
    n_bins = config.PSD_N_LOG_BINS

    # Same stale-PSD gate as the view builder: a subject written by the superseded
    # 60 s outer-window design stores log(Welch mean of ~59 segments), which is a
    # different y and therefore a different slope, and no view can undo it
    # (DECISIONS.md 2026-07-28).
    psd_timing.assert_subject_ok(subject, policy=nonstandard_hop)

    defs = cache_reader.load_defs(subject, session, epoch_minutes)
    parquet_file, cache_path = cache_reader.open_cache(subject, session, epoch_minutes)
    row_group_map = cache_reader.verify_layout(parquet_file, defs, n_bins)

    meta = channel_meta.build(subject, session, defs['run_id'].unique(), epoch_minutes)
    channels_by_run = {run: channel_meta.channels_for_run(meta, run)[0]
                       for run in defs['run_id'].unique()}
    region_of = (channel_meta.region_map(meta, view_config.roi_scheme)
                 if view_config.region != 'none' else {})
    mask = cache_reader.load_mask(subject, session, view_config)

    bin_table = cache_reader.bin_edges(epoch_minutes)
    drop_bins = ([] if args.keep_line_noise_bins
                 else list(cache_reader.line_noise_bins(epoch_minutes)))
    fit_bin_idx, log_f = aperiodic.fit_bins(bin_table, args.fit_lo_hz, args.fit_hi_hz,
                                            drop_bins=drop_bins)

    defs = defs.sort_values('epoch_id').reset_index(drop=True)
    defs['pain_bin'] = axes.assign_pain_bins(defs, view_config.pain_bins)

    regions = (roi_schemes.roi_regions(view_config.roi_scheme)
               if view_config.region != 'none' else [])

    stats = {
        'subject_id': f'sub-{subject}', 'session_id': f'ses-{session}',
        'n_epochs_total': int(len(defs)),
        'cache_bytes': int(cache_path.stat().st_size),
        'rows_read': 0, 'n_channel_epochs_dropped_coverage': 0,
        'n_channel_epochs_fit': 0, 'n_channel_epochs_unfittable': 0,
        'mask_excluded_frac': float('nan'),
        'subject_relative_threshold': axes.subject_relative_threshold(defs),
        'n_fit_bins_available': int(fit_bin_idx.size),
    }

    rows, r2_all, excl_fracs = [], [], []
    for epoch_row, block, kept, frac in cache_reader.iter_epochs(
            parquet_file, defs, n_bins, mask, channels_by_run, view_config,
            row_group_map):
        channels = channels_by_run[epoch_row['run_id']]
        stats['rows_read'] += int(block.size)
        stats['n_channel_epochs_dropped_coverage'] += int((~kept).sum())
        excl_fracs.append(frac)

        # AXIS 4 first: the epoch mean of log-power, per (channel, bin). Fitting
        # this is exactly averaging the per-window slopes -- an OLS slope is a
        # fixed-weight linear functional of y (views/aperiodic.py).
        values = axes.epoch_mean(block)                       # (n_pairs, n_bins)
        fit = aperiodic.fit_slopes(values[:, fit_bin_idx], log_f,
                                   min_bins=args.min_fit_bins,
                                   min_span_decades=args.min_span_decades)

        ok = np.isfinite(fit['slope'])
        stats['n_channel_epochs_fit'] += int(ok.sum())
        stats['n_channel_epochs_unfittable'] += int((~ok).sum())
        r2_all.append(fit['r2'][ok])

        if view_config.region == 'none':
            labels = channels
            slope_by_label = fit['slope']
            r2_by_label = fit['r2']
            counts = np.isfinite(fit['slope']).astype(int)
        else:
            labels = regions
            slope_by_label, counts = aperiodic.average_by_region(
                fit['slope'], channels, region_of, regions)
            # r2 averaged over the SAME channels, so a region's fit quality is
            # reported on the fits that actually built its slope.
            r2_by_label, _ = aperiodic.average_by_region(
                np.where(ok, fit['r2'], np.nan), channels, region_of, regions)

        for i, label in enumerate(labels):
            if not np.isfinite(slope_by_label[i]):
                continue
            rows.append((epoch_row['epoch_id'], int(epoch_row['pain_event_id']),
                         float(epoch_row['pain_score']), epoch_row['pain_bin'],
                         label, float(slope_by_label[i]), float(r2_by_label[i]),
                         int(counts[i])))

    if excl_fracs:
        stats['mask_excluded_frac'] = float(np.nanmean(np.concatenate(excl_fracs)))
    r2_all = np.concatenate(r2_all) if r2_all else np.array([])
    stats['r2_median'] = float(np.nanmedian(r2_all)) if r2_all.size else float('nan')
    stats['r2_p05'] = float(np.nanpercentile(r2_all, 5)) if r2_all.size else float('nan')

    epoch_table = pd.DataFrame(rows, columns=[
        'epoch_id', 'pain_event_id', 'pain_score', 'pain_bin', 'region',
        'slope', 'r2', 'n_channels'])
    epoch_table.insert(0, 'session_id', f'ses-{session}')
    epoch_table.insert(0, 'subject_id', f'sub-{subject}')

    if epoch_table.empty:
        subject_table = epoch_table.copy()
    else:
        subject_table = (epoch_table
                         .groupby(['subject_id', 'session_id', 'pain_bin', 'region'],
                                  dropna=False)
                         .agg(slope=('slope', 'mean'), r2=('r2', 'mean'),
                              n_epochs=('epoch_id', 'nunique'),
                              n_channels=('n_channels', 'max'))
                         .reset_index())

    stats['elapsed_sec'] = round(time.time() - t0, 2)
    stats['n_rows_epoch_table'] = int(len(epoch_table))
    stats['sec_per_gb'] = round(stats['elapsed_sec'] / (stats['cache_bytes'] / 1e9), 1)
    if view_config.region != 'none':
        mapped = sum(1 for v in region_of.values() if v is not None)
        stats['n_channels_mapped'] = mapped
        stats['n_channels_unmapped'] = len(region_of) - mapped
        if mapped == 0:
            # Same distinction the view builder draws: no localization at all is an
            # upstream data gap, localization outside the scheme is a scheme choice.
            # Both give an empty table, so saying which happened is the whole value.
            if meta['dk_anode'].isna().all():
                logger.error(
                    'sub-%s ses-%s: NO Desikan_Killiany_anode labels -- no anatomical '
                    'localization, so the slope table will be EMPTY. Use --region none '
                    'to get per-channel slopes for this subject.', subject, session)
            else:
                logger.error(
                    'sub-%s ses-%s: DK labels present but none fall inside ROI scheme '
                    '%r -- the slope table will be EMPTY.', subject, session,
                    view_config.roi_scheme)

    logger.info('sub-%s ses-%s: %d epochs, %.2f GB, %.1fs (%.1f s/GB), %d channel-epochs '
                'fit / %d unfittable, median r2 %.3f, mask-excluded %.3f',
                subject, session, len(defs), stats['cache_bytes'] / 1e9,
                stats['elapsed_sec'], stats['sec_per_gb'], stats['n_channel_epochs_fit'],
                stats['n_channel_epochs_unfittable'], stats['r2_median'],
                stats['mask_excluded_frac'])
    return epoch_table, subject_table, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--subjects', nargs='+', default=None,
                    help='Explicit subject IDs, still CHECKED against --split.')
    ap.add_argument('--split', default='discovery',
                    choices=list(cohorts.SPLITS) + ['heldout'])
    ap.add_argument('--session', default='01')
    ap.add_argument('--out-dir', default=None,
                    help='Override the destination. Default is the base unit\'s '
                         'views/ directory. Must resolve to Oak, never the repo.')
    ap.add_argument('--view-label', default=None)
    ap.add_argument('--no-save', action='store_true')
    ap.add_argument('--print-out-dir', action='store_true',
                    help='Print the resolved directory and exit. How the plot job '
                         'finds the view without hard-coding its config_hash.')
    ap.add_argument('--nonstandard-hop', choices=['refuse', 'drop', 'allow'],
                    default='refuse')

    g = ap.add_argument_group('the fit (config/psd_params.py)')
    g.add_argument('--fit-lo-hz', type=float, default=config.SLOPE_FIT_LO_HZ)
    g.add_argument('--fit-hi-hz', type=float, default=config.SLOPE_FIT_HI_HZ)
    g.add_argument('--keep-line-noise-bins', action='store_true',
                   help='Include the bins flagged contains_line_noise in the fit. '
                        'Default DROPS them: a notch or a harmonic peak sitting on '
                        'the regression line pulls the slope by an amount that has '
                        'nothing to do with physiology.')
    g.add_argument('--min-fit-bins', type=int, default=config.SLOPE_MIN_FIT_BINS)
    g.add_argument('--min-span-decades', type=float, default=config.SLOPE_MIN_SPAN_DECADES)

    vc.add_view_arguments(ap)
    args = ap.parse_args()

    logging.basicConfig(level=(logging.WARNING if args.print_out_dir else logging.INFO),
                        format='%(asctime)s %(levelname)s %(message)s')

    if args.normalization != 'none':
        raise SystemExit(
            f'--normalization {args.normalization!r} is refused. A slope is fit on RAW '
            'log-power: z-scoring rescales every frequency bin by its own baseline SD, '
            'which tilts the line by the SD\'s frequency dependence, and '
            'baseline_subtract gives the CHANGE in tilt -- a different quantity that '
            'also cannot be drawn against a 0-pain violin. Pass --normalization none.')
    if args.domain != 'log':
        raise SystemExit('--domain must be log: the fit IS log-power vs log-frequency.')
    if args.freq != 'log_bins_50':
        raise SystemExit('--freq must be log_bins_50: the fit needs the individual '
                         'bins, and canonical_bands has already collapsed them.')

    view = vc.from_args(args)
    # Keep the recorded ViewConfig honest about the line-noise bins: they do not
    # contribute to any number this script emits.
    view = dataclasses.replace(view, drop_line_noise_bins=not args.keep_line_noise_bins)

    unit = config.pain_epoch_unit_dir(view.epoch_minutes)
    bin_table = cache_reader.bin_edges(view.epoch_minutes)
    drop_bins = ([] if args.keep_line_noise_bins
                 else list(cache_reader.line_noise_bins(view.epoch_minutes)))
    fit_bin_idx, _ = aperiodic.fit_bins(bin_table, args.fit_lo_hz, args.fit_hi_hz,
                                        drop_bins=drop_bins)

    view_params = dict(view.provenance(), split=args.split,
                       **slope_params(args, drop_bins, fit_bin_idx))
    label = args.view_label or f'{VIEW_LABEL_PREFIX}-{view.scheme_code}'
    out_dir = (Path(args.out_dir) if args.out_dir else
               config.pain_epoch_views_dir(label, io.config_hash(view_params),
                                           view.epoch_minutes))
    if args.print_out_dir:
        print(out_dir)
        return

    io.warn_if_dirty()
    logger.info('view config: %s', view.to_dict())
    logger.info('fit: %.3g-%.3g Hz -> %d of %d bins (%d line-noise dropped), '
                'min %d bins / %.2g decades',
                args.fit_lo_hz, args.fit_hi_hz, fit_bin_idx.size,
                config.PSD_N_LOG_BINS, len(drop_bins), args.min_fit_bins,
                args.min_span_decades)

    if args.subjects:
        subjects = cohorts.assert_split_allowed(args.subjects, args.split)
    else:
        subjects = cohorts.subjects_for_split(
            args.split, available=cohorts.subjects_with_epoch_cache(view.epoch_minutes))
        if not subjects:
            raise SystemExit(f'no subjects in split={args.split!r} have an epoch cache')
    logger.info('split=%s -> %d subject(s)', args.split, len(subjects))
    logger.info('slope tables -> %s', out_dir)

    all_stats = []
    for s in subjects:
        subject = s.replace('sub-', '')
        epoch_table, subject_table, stats = build_subject_slopes(
            subject, args.session, view, args, nonstandard_hop=args.nonstandard_hop)
        all_stats.append(stats)
        if args.no_save:
            continue
        # SAME FILE NAMES as the power view (`view_epochs_*` / `view_subject_*`), so
        # analysis/view_tables.load_view_tables reads these unchanged. They are told
        # apart by the DIRECTORY and by params['metric'] in the sidecar, never by a
        # filename -- which is also why the plot script checks that key.
        for name, table in (('epochs', epoch_table), ('subject', subject_table)):
            io.write_table(
                table, out_dir / f'view_{name}_sub-{subject}_ses-{args.session}.parquet',
                kind='view', script='ieeg_ehr/views/build_pain_epoch_slope.py',
                params=view_params, parents=[io.manifest_ref(unit)],
                subjects=[f'sub-{subject}'], extra={'stats': stats})

    stats_df = pd.DataFrame(all_stats)
    logger.info('\n%s', stats_df.to_string(index=False))
    if not args.no_save:
        # One stats file per subject/session: this runs as a Slurm array into a
        # shared directory, and a single shared name would be rewritten by every
        # task (the same reason build_pain_epoch_view.py does this).
        suffix = (f'sub-{subjects[0].replace("sub-", "")}_ses-{args.session}'
                  if len(subjects) == 1 else f'{len(subjects)}subj_ses-{args.session}')
        io.write_table(stats_df, out_dir / f'view_stats_{suffix}.parquet', kind='table',
                       script='ieeg_ehr/views/build_pain_epoch_slope.py',
                       params=view_params, subjects=[f'sub-{s}' for s in subjects],
                       extra={'cohort': cohorts.cohort_provenance()})
        if not io.sidecar_path(out_dir).exists():
            io.write_view_sidecar(out_dir, view_config=view_params, cache_manifest=unit,
                                  script='ieeg_ehr/views/build_pain_epoch_slope.py')
        logger.info('slope tables + stats -> %s', out_dir)


if __name__ == '__main__':
    main()
