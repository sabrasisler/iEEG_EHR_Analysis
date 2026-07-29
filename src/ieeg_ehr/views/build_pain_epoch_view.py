#!/usr/bin/env python3
"""
P1.3 view builder: per-window cache -> analysis-ready region x frequency table.

Runs the registry chain in order, per subject/session:

    baseline (0-pain windows, masked)
      -> normalize PER 2s WINDOW
      -> mean over the epoch's ~300 windows        (per channel, per freq bin)
      -> mean over channels within an ROI          (per epoch)
      -> mean over epochs within a pain level      (per subject)

Two streaming passes over the cache. Pass 1 reads only the 0-pain epochs and
accumulates the per-(channel, bin) baseline; pass 2 reads every epoch and produces
values. Two passes rather than one because the baseline must be complete before any
window can be normalized against it, and holding every epoch in memory to avoid
the second read is not an option at 409M rows for the largest subject.

0-pain epochs are carried through pass 2 as well, not just consumed as baseline:
they are the 'none' level, and they normalize to ~0 by construction -- which makes
them a free correctness check (a 'none' row far from 0 means the baseline leaked).

Output lands in the base unit's views/ directory by default --
`features/pain/psd_epochs/epoch-5min-pre/views/<scheme_code>_<config_hash>/` --
which is a DISPOSABLE performance cache, deletable at any time (architecture.md
PART 2). Materializing it is justified here and not by default elsewhere: the
build is measured at ~34 s/GB, a full-cohort run is a 59-task array, and two plot
scripts now read the tables instead of recomputing. A figure a human reads is NOT
this; that is an analysis output under config.analysis_run_dir().

Run on Slurm, never the login node:
    python -m ieeg_ehr.views.build_pain_epoch_view --subjects 090 \\
        --mask-label std10_rv-gross-std3_satmargin15_sw_logz4
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ieeg_ehr import config, io
from ieeg_ehr.config import cohorts, roi_schemes
from ieeg_ehr.qc import psd_timing
from ieeg_ehr.views import axes, cache_reader, channel_meta, view_config as vc

logger = logging.getLogger(__name__)


def build_subject_view(subject, session, view_config, epoch_minutes=None,
                       collect_epochs=True, nonstandard_hop='refuse'):
    """Returns (epoch_table, subject_table, stats) for one subject/session."""
    t0 = time.time()
    epoch_minutes = epoch_minutes or view_config.epoch_minutes
    n_bins = config.PSD_N_LOG_BINS

    # Was this subject's PSD written by the CURRENT windowing design? ONE lookup in
    # the cached qc/psd_timing/ table -- no NWB reads, nothing per epoch. Runs from
    # the superseded 60 s outer-window design store log(Welch-mean of ~59 segments)
    # instead of log(single 2 s segment), which freezes the log-vs-linear averaging
    # choice into the file where no view can undo it (DECISIONS.md 2026-07-28 +
    # correction). Default refuses; an UNAUDITED subject also refuses, because
    # silence must not read as approval.
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
    drop_bins = (cache_reader.line_noise_bins(epoch_minutes)
                 if view_config.drop_line_noise_bins else np.array([], dtype=int))

    defs = defs.sort_values('epoch_id').reset_index(drop=True)
    defs['pain_bin'] = axes.assign_pain_bins(defs, view_config.pain_bins)

    stats = {
        'subject_id': f'sub-{subject}', 'session_id': f'ses-{session}',
        'n_epochs_total': int(len(defs)),
        'n_baseline_epochs': int(defs.apply(axes.is_baseline_epoch, axis=1).sum()),
        'cache_bytes': int(cache_path.stat().st_size),
        'rows_read': 0, 'n_channel_epochs_dropped_coverage': 0,
        'n_nonfinite_input': 0, 'mask_excluded_frac': float('nan'),
        'subject_relative_threshold': axes.subject_relative_threshold(defs),
    }

    # ---------------- pass 1: baseline over 0-pain windows ----------------
    # SESSION-WIDE, keyed on channel NAME. Channel ORDER is per run (montages can
    # differ between runs), so each run's pair axis is mapped onto a shared
    # session-wide channel index. Keying on (run, pair_index) instead would give
    # every run its own baseline and strand every epoch in a run containing no
    # 0-pain event -- 4 of sub-019's 49 epochs -- which is not what registry
    # AXIS 2 ("the subject's 0-pain epoch windows") describes.
    session_channels = sorted({c for chans in channels_by_run.values() for c in chans})
    channel_index = {c: i for i, c in enumerate(session_channels)}
    rows_by_run = {run: np.array([channel_index[c] for c in chans], dtype=int)
                   for run, chans in channels_by_run.items()}

    acc = axes.BaselineAccumulator(len(session_channels), n_bins)
    excl_fracs = []
    for epoch_row, block, kept, frac in cache_reader.iter_epochs(
            parquet_file, defs, n_bins, mask, channels_by_run, view_config,
            row_group_map, epoch_filter=axes.is_baseline_epoch):
        acc.update(axes.to_domain(block, view_config.domain),
                   rows=rows_by_run[epoch_row['run_id']])
        stats['rows_read'] += int(block.size)
        excl_fracs.append(frac)

    if view_config.normalization != 'none' and acc.n_epochs == 0:
        raise ValueError(
            f'sub-{subject} ses-{session}: no 0-pain epochs, so '
            f'normalization={view_config.normalization!r} has no baseline. Either this '
            'subject cannot enter a baseline-normalized view, or use '
            '--normalization none.'
        )

    baseline_mean, baseline_sd = acc.finalize()
    stats['n_baseline_channels'] = int((acc.count >= 2).any(axis=1).sum())
    stats['n_channels_session'] = len(session_channels)

    # ---------------- pass 2: values for every epoch ----------------
    regions = roi_schemes.roi_regions(view_config.roi_scheme) if view_config.region != 'none' else []
    rows = []
    n_dropped = 0
    for epoch_row, block, kept, frac in cache_reader.iter_epochs(
            parquet_file, defs, n_bins, mask, channels_by_run, view_config,
            row_group_map):
        run = epoch_row['run_id']
        channels = channels_by_run[run]
        stats['n_nonfinite_input'] += int((~np.isfinite(block)).sum())
        n_dropped += int((~kept).sum())
        excl_fracs.append(frac)

        block = axes.to_domain(block, view_config.domain)
        if view_config.normalization != 'none':
            # Gather this run's channels out of the session-wide baseline. A
            # channel with too few baseline windows carries NaN here, so its
            # z-scores become NaN rather than a fabricated number.
            base_rows = rows_by_run[run]
            block = axes.normalize(block, baseline_mean[base_rows],
                                   baseline_sd[base_rows], view_config.normalization)

        values = axes.epoch_mean(block)                       # (n_pairs, n_bins)
        if drop_bins.size:
            values[:, drop_bins] = np.nan

        if view_config.freq == 'canonical_bands':
            values, col_names = axes.aggregate_bands(
                values, bin_table, is_difference=view_config.is_difference,
                domain=view_config.domain)
            col_index = col_names
        else:
            col_index = list(range(values.shape[1]))

        if view_config.region == 'none':
            labels, counts = channels, np.ones(len(channels), dtype=int)
            grid = values
        else:
            grid, counts = axes.aggregate_regions(
                values, channels, region_of, regions,
                is_difference=view_config.is_difference, domain=view_config.domain)
            labels = regions

        if not collect_epochs:
            continue
        # Long format, dropping all-NaN cells so the table stays small.
        for i, label in enumerate(labels):
            finite = np.isfinite(grid[i, :])
            if not finite.any():
                continue
            for j in np.flatnonzero(finite):
                rows.append((epoch_row['epoch_id'], int(epoch_row['pain_event_id']),
                             float(epoch_row['pain_score']), epoch_row['pain_bin'],
                             label, col_index[j], float(grid[i, j]), int(counts[i])))

    stats['n_channel_epochs_dropped_coverage'] = n_dropped
    if excl_fracs:
        stats['mask_excluded_frac'] = float(np.nanmean(np.concatenate(excl_fracs)))

    epoch_table = pd.DataFrame(rows, columns=[
        'epoch_id', 'pain_event_id', 'pain_score', 'pain_bin', 'region',
        'freq_bin_index', 'value', 'n_channels'])
    epoch_table.insert(0, 'session_id', f'ses-{session}')
    epoch_table.insert(0, 'subject_id', f'sub-{subject}')

    # Final collapse: mean over epochs within (pain_bin, region, freq bin).
    if epoch_table.empty:
        subject_table = epoch_table.copy()
    else:
        subject_table = (epoch_table
                         .groupby(['subject_id', 'session_id', 'pain_bin', 'region',
                                   'freq_bin_index'], dropna=False)
                         .agg(value=('value', 'mean'),
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
        stats['n_channels_no_dk_label'] = int(
            meta.loc[meta['dk_anode'].isna(), 'channel'].nunique())
        logger.info('sub-%s ses-%s: %d/%d channels map to an ROI (%d dropped as '
                    'white matter / occipital / unlabeled)', subject, session, mapped,
                    len(region_of), len(region_of) - mapped)
        if mapped == 0:
            # Distinguish "no DK localization at all" from "localized, but every
            # parcel falls outside the ROI set" -- the first is an upstream data gap
            # that no ROI scheme can fix, the second is a scheme choice. Both
            # produce an empty table, so saying which one happened is the whole
            # value of this message.
            if meta['dk_anode'].isna().all():
                logger.error(
                    'sub-%s ses-%s: NO Desikan_Killiany_anode labels in the NWB '
                    'electrodes table (all %d rows null) -- this subject has no '
                    'anatomical localization and CANNOT enter a region-level view. '
                    'The view table will be EMPTY. Use --region none to analyse it '
                    'per channel.', subject, session, len(meta))
            else:
                logger.error(
                    'sub-%s ses-%s: DK labels present but NONE fall inside ROI scheme '
                    '%r -- the view table will be EMPTY. Check the scheme, not the data.',
                    subject, session, view_config.roi_scheme)
    logger.info('sub-%s ses-%s: %d epochs, %.2f GB, %.1fs (%.1f s/GB), '
                'mask-excluded %.3f, %d channel-epochs dropped for coverage',
                subject, session, len(defs), stats['cache_bytes'] / 1e9,
                stats['elapsed_sec'], stats['sec_per_gb'],
                stats['mask_excluded_frac'], n_dropped)
    return epoch_table, subject_table, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--subjects', nargs='+', default=None,
                    help='Explicit subject IDs. Still CHECKED against --split, so '
                         'naming a non-discovery subject is refused rather than '
                         'quietly allowed. Omit to use the whole split.')
    ap.add_argument('--split', default='discovery',
                    choices=list(cohorts.SPLITS) + ['heldout'],
                    help='Cohort gate (P0.2). Default discovery -- the LOCKED, '
                         'permanent exploratory set. "heldout" raises: the matched '
                         'hold-out is built offline on the PHI side (P4).')
    ap.add_argument('--session', default='01')
    ap.add_argument('--out-dir', default=None,
                    help='Override the destination. Default is the base unit\'s '
                         'views/ directory, config.pain_epoch_views_dir(scheme_code, '
                         'config_hash) -- a disposable performance cache, deletable. '
                         'Pass this only for a one-off; must resolve to Oak/scratch, '
                         'never the repo.')
    ap.add_argument('--view-label', default=None,
                    help='Override the human half of the views directory name '
                         "(default: the view's scheme_code, e.g. 'blsub-rel'). The "
                         'config_hash half is never overridable.')
    ap.add_argument('--no-save', action='store_true',
                    help='Compute and report only. Default SAVES, so the numbers can be '
                         'inspected before a figure is trusted.')
    ap.add_argument('--print-out-dir', action='store_true',
                    help='Print the resolved views directory and exit, without '
                         'reading or writing anything. How a downstream plot job '
                         'finds a view without hard-coding its config_hash.')
    ap.add_argument('--nonstandard-hop', choices=['refuse', 'drop', 'allow'], default='refuse',
                    help='What to do when a subject was not written by the current PSD '
                         'windowing design. Default refuses (DECISIONS.md 2026-07-28).')
    vc.add_view_arguments(ap)
    args = ap.parse_args()

    logging.basicConfig(level=(logging.WARNING if args.print_out_dir else logging.INFO),
                        format='%(asctime)s %(levelname)s %(message)s')
    view = vc.from_args(args)

    # The split is IN the hashed params, not just in the provenance text: a
    # discovery view and an all-subjects view of the same axes are different
    # artifacts holding different subject sets, and they must not share a
    # directory. Hashing the same dict the tables are stamped with also keeps the
    # directory's config_hash equal to its contents' -- if the two could differ,
    # the folder name would be a claim nothing verifies.
    unit = config.pain_epoch_unit_dir(view.epoch_minutes)
    view_params = dict(view.provenance(), split=args.split)
    out_dir = (Path(args.out_dir) if args.out_dir else
               config.pain_epoch_views_dir(args.view_label or view.scheme_code,
                                           io.config_hash(view_params),
                                           view.epoch_minutes))

    # Resolved BEFORE the cohort gate and before any data is touched, so a plotting
    # job can ask where a view lives without building it. This is the reason the
    # sbatch scripts do not spell the config_hash themselves: ONE code path
    # computes it, so the builder and the plotter cannot disagree about the
    # directory. Prints the path and nothing else -- it is consumed by $(...).
    if args.print_out_dir:
        print(out_dir)
        return

    io.warn_if_dirty()
    logger.info('view config: %s', view.to_dict())

    # Resolve the cohort gate BEFORE touching any data. An explicit --subjects list
    # is still validated against the split: without that check, --split would be
    # advisory and hand-naming a hold-out-eligible subject would work, which cannot
    # be undone once its data has been seen.
    if args.subjects:
        subjects = cohorts.assert_split_allowed(args.subjects, args.split)
    else:
        subjects = cohorts.subjects_for_split(
            args.split, available=cohorts.subjects_with_epoch_cache(view.epoch_minutes))
        if not subjects:
            raise SystemExit(f'no subjects in split={args.split!r} have an epoch cache')
    logger.info('split=%s -> %d subject(s): %s', args.split, len(subjects), subjects)
    logger.info('view tables -> %s', out_dir)

    all_stats = []
    for s in subjects:
        subject = s.replace('sub-', '')
        epoch_table, subject_table, stats = build_subject_view(
            subject, args.session, view, nonstandard_hop=args.nonstandard_hop)
        all_stats.append(stats)
        if args.no_save:
            continue
        for name, table in (('epochs', epoch_table), ('subject', subject_table)):
            io.write_table(
                table, out_dir / f'view_{name}_sub-{subject}_ses-{args.session}.parquet',
                kind='view', script='ieeg_ehr/views/build_pain_epoch_view.py',
                params=view_params, parents=[io.manifest_ref(unit)],
                subjects=[f'sub-{subject}'], extra={'stats': stats})

    stats_df = pd.DataFrame(all_stats)
    logger.info('\n%s', stats_df.to_string(index=False))
    if not args.no_save:
        # ONE STATS FILE PER SUBJECT/SESSION, not one per run. This script is
        # invoked as a Slurm array of one subject per task, all writing into the
        # same views/ directory; a shared `view_stats.parquet` would be rewritten
        # by every task and end up describing only whichever finished last. Same
        # reason metrics_run_info_dir() is per-subject (config/paths.py).
        suffix = (f'sub-{subjects[0].replace("sub-", "")}_ses-{args.session}'
                  if len(subjects) == 1 else f'{len(subjects)}subj_ses-{args.session}')
        io.write_table(stats_df, out_dir / f'view_stats_{suffix}.parquet', kind='table',
                       script='ieeg_ehr/views/build_pain_epoch_view.py',
                       params=view_params,
                       subjects=[f'sub-{s}' for s in subjects],
                       extra={'cohort': cohorts.cohort_provenance()})

        # The directory-level staleness sidecar (architecture.md PART 2): cache
        # manifest digest + view config + view commit, so a later load can refuse
        # a view built from a cache or a code version that has since moved.
        #
        # WRITE-ONCE, because every array task reaches this line. The content is
        # identical from every task (it describes the view, not the subject), so a
        # lost update costs nothing -- but write_sidecar is a plain write_text, and
        # 16 concurrent writers could let a reader see a half-written file.
        if not io.sidecar_path(out_dir).exists():
            io.write_view_sidecar(out_dir, view_config=view_params, cache_manifest=unit,
                                  script='ieeg_ehr/views/build_pain_epoch_view.py')
        logger.info('view tables + stats -> %s', out_dir)


if __name__ == '__main__':
    main()
