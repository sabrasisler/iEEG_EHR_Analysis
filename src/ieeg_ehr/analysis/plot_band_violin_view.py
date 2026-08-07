#!/usr/bin/env python3
"""
One band, every pain level, one dot per subject: violins of within-subject
normalized band power, one panel per region.

TWO NORMALIZATIONS, AND THE CHOICE IS THE WHOLE DESIGN (`--normalize`)
----------------------------------------------------------------------
**view_baseline** — plot the view's own values, already referenced to each
subject's 0-pain baseline. The 0-pain violin then sits at ~0 BY CONSTRUCTION. That
sounds like a defect and is in fact the reason to want it: it is not EXACTLY 0,
because the baseline pools 2 s WINDOWS while a plotted value averages EPOCHS and QC
masking leaves epochs with unequal window counts (docs/labnotebook 2026-08-05). The
residual spread is therefore a VISIBLE NOISE FLOOR on the same axis as low and high
-- you can see what "no effect" looks like in this pipeline and read the other two
against it. Values are directly comparable to the heatmaps.

**within_subject** — for reading the three levels as peers, where 0-pain pinned near
zero is unhelpful. The reference must then be something that is NOT one of the three,
so each subject is standardized against THEIR OWN OVERALL LEVEL:

    for each (subject, region):
        take the band power of EVERY epoch, all pain levels pooled
        z = (epoch - mean over those epochs) / SD over those epochs
    then average z within each pain level -> one value per subject per level

Symmetric across the three levels, so none of them is privileged, and it removes
the between-subject scale -- which is the thing that makes raw power unplottable
here. Between-subject spread in raw log power is ~2 log units from electrode
impedance and amplifier gain alone, against a pain effect of order 0.1.

TWO HONEST COSTS, neither of which invalidates the comparison:
  - Pooling all levels into the SD means that SD contains some of the
    between-level variance the figure is looking for, so effects are slightly
    SHRUNK. It is the same shrinkage for every subject, so comparisons between
    levels remain valid; the absolute z is not comparable to the heatmaps' z.
  - It is a per-epoch standardization, not per-window. The principled per-window
    version is `baseline: whole_session`, an axis value that docs/view_registry.md
    already defines and ViewConfig still raises NotImplementedError for. Doing it
    there would be reusable and would not need this script -- see TASKS.md.

ONE DOT IS ONE SUBJECT. Not one epoch: epochs are nested within subjects, and
treating them as independent is the pseudo-replication that would inflate any
statistic computed off this figure (the same reason the cluster test takes a
56-subject matrix and not a ~700-epoch one).

BAND EDGES come from config.CANONICAL_BANDS_HZ, which is what the rest of the code
uses. NOTE architecture.md PART 0 records an unresolved discrepancy: that doc claims
beta is 15-25 Hz while the code says 13-30. The resolved edges are written into
provenance so a later decision is traceable rather than silently reinterpreting
this figure.

Run on Slurm, never the login node:
    python -m ieeg_ehr.analysis.plot_band_violin_view --view-dir <raw view> --band beta
"""

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ieeg_ehr import config, io
from ieeg_ehr.analysis import view_tables
from ieeg_ehr.features import common
from ieeg_ehr.views import cache_reader

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_TYPE = 'band_violin'


def epochs_to_band(epoch_tables, bin_labels, band, bands=None, drop_bins=(),
                   is_difference=False):
    """Per-epoch band value, aggregated over the bins whose CENTRE falls in the band.

    THE AGGREGATION DEPENDS ON WHAT THE VALUES ARE, exactly as
    views/axes.aggregate_bands branches:

    - RAW log power (`is_difference=False`): LINEAR-THEN-LOG. A mean of logs is a
      geometric mean, which is not the band's average power, so bins are
      exponentiated, averaged, and re-logged (registry AXIS 5,
      preprocessing/bipolar_bands.py). Getting this wrong is silent and biases every
      band downward.
    - A DIFFERENCE of logs or a z-score (`is_difference=True`): ARITHMETIC mean. It
      is already dimensionless, and exponentiating a z-score would be meaningless.

    Uses the GEOMETRIC-mean bin centre for band membership because the bins are
    log-spaced; an arithmetic centre would drift toward the high edge of each bin.
    """
    bands = bands or config.CANONICAL_BANDS_HZ
    fmin, fmax = bands[band]
    centres = np.sqrt(bin_labels['bin_low_hz'] * bin_labels['bin_high_hz'])
    in_band = centres[(centres >= fmin) & (centres < fmax)].index
    in_band = [b for b in in_band if b not in set(drop_bins)]
    if not in_band:
        raise SystemExit(f'no frequency bin centre falls inside {band} '
                         f'({fmin}-{fmax} Hz) after dropping line-noise bins')
    logger.info('%s = %g-%g Hz -> %d bin(s): %s Hz', band, fmin, fmax, len(in_band),
                [f'{bin_labels.loc[b, "bin_low_hz"]:.0f}' for b in in_band])

    rows = epoch_tables[epoch_tables['freq_bin_index'].isin(in_band)].copy()
    keys = ['subject_id', 'epoch_id', 'pain_bin', 'region']

    if is_difference:
        out = (rows.groupby(keys, dropna=False)['value'].mean().reset_index()
               .rename(columns={'value': 'band_value'}))
    else:
        # float64 before exponentiating: the cache is float32 and the worst stored
        # log-power is ~-36.8, barely a decade above float32's smallest normal (P0.6).
        rows['linear'] = np.power(10.0, rows['value'].to_numpy(dtype=np.float64))
        out = rows.groupby(keys, dropna=False)['linear'].mean().reset_index()
        with np.errstate(divide='ignore'):
            out['band_value'] = np.log10(out['linear'])
        out = out.drop(columns='linear')
    return out, in_band




def plot_violin_grid(subject_values, regions, panels, title, out_path, value_label,
                     ncols=4, footnote=None):
    """One panel per region; one violin per pain level; one dot per subject."""
    subjects = sorted(subject_values['subject'].unique())
    colour = common.subject_color_map(subjects)

    nrows = int(np.ceil(len(regions) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 3.0 * nrows),
                             sharey=True, squeeze=False)
    flat = axes.ravel()

    for ax, region in zip(flat, regions):
        rows = subject_values[subject_values['region'] == region]
        ax.axhline(0, color='0.35', linewidth=0.8, zorder=1)
        common.draw_seaborn_violin_with_subject_dots(
            ax, rows, colour, value_col='value', pain_bins=panels)
        n = rows['subject'].nunique()
        ax.set_title(f'{region}  (n={n} subj)', fontsize=9)
        ax.tick_params(labelsize=8)
        # seaborn labels the y axis with the value COLUMN name ('value'), which
        # duplicates the figure-level label 21 times and says nothing.
        ax.set_ylabel('')
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)

    for ax in flat[len(regions):]:
        ax.set_visible(False)

    fig.supylabel(value_label, fontsize=9)
    fig.suptitle(title, fontsize=12)
    if footnote:
        fig.text(0.01, -0.01, footnote, ha='left', va='top', fontsize=6, color='0.25')
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info('Wrote %s', out_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--view-dir', required=True,
                    help='A view directory. An un-normalized (raw) view pairs with '
                         '--normalize within_subject; a baseline_subtract or '
                         'zscore_vs_baseline view pairs with --normalize '
                         'view_baseline. --normalize auto picks for you.')
    ap.add_argument('--band', default='beta',
                    choices=sorted(config.CANONICAL_BANDS_HZ),
                    help='Band edges from config.CANONICAL_BANDS_HZ. NOTE '
                         'architecture.md PART 0 records an unresolved discrepancy '
                         'for beta (code 13-30 Hz, that doc 15-25).')
    ap.add_argument('--pain-bin-scheme', choices=list(view_tables.PANELS),
                    default='subject_relative')
    ap.add_argument('--min-subjects', type=int, default=8)
    ap.add_argument('--normalize', choices=['auto', 'within_subject', 'view_baseline'],
                    default='auto',
                    help="How the values are normalized. 'view_baseline' uses the "
                         "view's own 0-pain referencing, which puts the 0-pain violin "
                         'at ~0 by construction -- deliberately, so it reads as a '
                         'visible noise floor, and so values are comparable to the '
                         "heatmaps. 'within_subject' standardizes against each "
                         "subject's mean/SD over ALL epochs pooled, so no level is "
                         'its own reference; that one needs the un-normalized view. '
                         "'auto' picks by what the view is.")
    ap.add_argument('--min-epochs', type=int, default=4,
                    help='A (subject, region) needs this many epochs before its SD '
                         'is treated as a scale. Below it, the subject is dropped '
                         'rather than given a z built on nothing.')
    ap.add_argument('--keep-line-noise-bins', action='store_true')
    ap.add_argument('--ncols', type=int, default=4)
    ap.add_argument('--run-name', default=None)
    view_tables.add_output_arguments(ap)
    args = ap.parse_args()

    view_dir = Path(args.view_dir)
    io.warn_if_dirty()

    epoch_tables, epoch_paths = view_tables.load_view_tables(view_dir, 'epochs')
    subject_tables, subject_paths = view_tables.load_view_tables(view_dir, 'subject')
    view_params, view = view_tables.view_params_from(subject_paths)
    logger.info('view: %s', {k: view_params.get(k) for k in
                             ('normalization', 'domain', 'mask_label', 'pain_bins',
                              'roi_scheme')})

    # TWO NORMALIZATIONS, and which one is right depends on what you want to read.
    #
    # view_baseline: take the view's own values, already referenced to each
    #   subject's 0-pain baseline. The 'none' violin then sits at ~0 BY
    #   CONSTRUCTION -- but not exactly, and that is the point: the epoch-weighting
    #   asymmetry (docs/labnotebook 2026-08-05) gives it a small real spread, so it
    #   becomes a VISIBLE NOISE FLOOR on the same axis. low and high are read
    #   against it, and the values are directly comparable to the heatmaps' z.
    #
    # within_subject: this script standardizes against each subject's own mean/SD
    #   over ALL epochs, pooled across levels, so no level is its own reference.
    #   Needs the un-normalized view. Absolute values are NOT comparable to the
    #   heatmaps.
    view_is_normalized = view is not None and view.is_difference
    if args.normalize == 'auto':
        mode = 'view_baseline' if view_is_normalized else 'within_subject'
    else:
        mode = args.normalize

    if mode == 'view_baseline' and not view_is_normalized:
        raise SystemExit(
            '--normalize view_baseline needs a view that HAS a baseline, but '
            f'{view_dir} was built with normalization='
            f'{(view.normalization if view else None)!r}. Point at a '
            'baseline_subtract or zscore_vs_baseline view, or use --normalize '
            'within_subject.')
    if mode == 'within_subject' and view_is_normalized:
        raise SystemExit(
            f'--normalize within_subject on a {view.normalization!r} view would '
            'standardize an already-baselined quantity twice. Use the un-normalized '
            'view for that mode, or --normalize view_baseline for this one.')
    logger.info('normalization mode: %s (view normalization=%r)', mode,
                view.normalization if view else None)

    # 'none' is drawn HERE, unlike in the spectra/heatmaps, because nothing in this
    # figure's normalization makes it the reference.
    panels = [b for b in config.pain_bin_order(args.pain_bin_scheme)
              if b in set(epoch_tables['pain_bin'])]
    logger.info('violins per panel: %s', panels)

    epoch_minutes = view_params.get('epoch_minutes')
    bin_labels = cache_reader.bin_edges(epoch_minutes).set_index('freq_bin_index')
    drop_bins = ([] if args.keep_line_noise_bins
                 else list(cache_reader.line_noise_bins(epoch_minutes)))

    band_epochs, band_bins = epochs_to_band(epoch_tables, bin_labels, args.band,
                                            drop_bins=drop_bins,
                                            is_difference=view_is_normalized)
    # Both helpers live in view_tables so this figure and plot_slope_violin.py
    # cannot standardize differently -- that shared-definition rule is the whole
    # reason that module exists.
    if mode == 'within_subject':
        band_epochs = view_tables.within_subject_z(
            band_epochs, 'band_value', min_epochs=args.min_epochs)
        value_col = 'z'
    else:
        # The view already normalized it; averaging over epochs is all that is left.
        value_col = 'band_value'
    subject_values = view_tables.subject_level(band_epochs, panels,
                                              value_col=value_col)

    roi_regions = view_tables.roi_regions_for(view_params)
    regions, per_region, below = view_tables.regions_by_min_subjects(
        subject_values, panels, roi_regions, args.min_subjects)
    if not regions:
        raise SystemExit(f'no region has >= {args.min_subjects} subjects in every '
                         f'level {panels}')
    logger.info('%d region(s) plotted: %s', len(regions),
                {r: int(per_region[r]) for r in regions})

    if not args.view_scheme:
        args.view_scheme = ((view.scheme_code if view is not None else 'unknown')
                            + f'-{args.band}')
    run_dir = view_tables.resolve_run_dir(
        args, OUTPUT_TYPE, view, run_name=args.run_name or f'discovery_{args.band}')
    logger.info('run dir: %s', run_dir)

    subjects = sorted(subject_values['subject'].unique())
    band_params = {
        'band': args.band,
        'band_hz': list(config.CANONICAL_BANDS_HZ[args.band]),
        'freq_bins_in_band': [int(b) for b in band_bins],
        'normalize_mode': mode,
        'normalization_note': (
            'view_baseline: the values are the VIEW\'s own, referenced to each '
            'subject\'s 0-pain baseline. The 0-pain violin therefore sits at ~0 by '
            'construction -- but NOT exactly, because the baseline pools windows '
            'while a reported value averages epochs (see docs/labnotebook '
            '2026-08-05); that residual spread is a visible NOISE FLOOR and the '
            'other levels are read against it. Values ARE comparable to the '
            'heatmaps.'
            if mode == 'view_baseline' else
            'within_subject: each epoch minus that (subject, region) mean over ALL '
            'epochs, divided by its SD, POOLED across pain levels so no level is its '
            'own reference. Absolute values are NOT comparable to the heatmaps.'),
        'unit_of_observation': 'one subject (mean over that subject\'s epochs)',
        'min_epochs_for_scale': args.min_epochs,
        'line_noise_bins_dropped': [int(b) for b in drop_bins],
        'band_edge_caveat':
            'Edges from config.CANONICAL_BANDS_HZ. architecture.md PART 0 records an '
            'unresolved discrepancy (that doc gives beta as 15-25 Hz, the code 13-30).',
    }

    io.write_table(subject_values, run_dir / 'subject_band_values.parquet', kind='table',
                   script='ieeg_ehr/analysis/plot_band_violin_view.py',
                   params={**view_params, **band_params},
                   parents=[io.parent_ref(p, digest=False) for p in epoch_paths],
                   subjects=subjects)

    io.write_run_provenance(
        run_dir, script='ieeg_ehr/analysis/plot_band_violin_view.py',
        params={**vars(args), 'view_params': view_params, **band_params},
        parents=[io.parent_ref(p, digest=False) for p in epoch_paths + subject_paths],
        subjects=subjects,
        extra={'panels': panels, 'regions_plotted': regions,
               'n_subjects_per_region': {r: int(per_region[r]) for r in regions},
               'regions_below_floor': below, 'roi_regions': roi_regions,
               'status': 'EXPLORATORY nomination, not a finding '
                         '(CLAUDE.md; pending P2.6 FREEZE)'},
    )

    lo, hi = config.CANONICAL_BANDS_HZ[args.band]
    if mode == 'view_baseline':
        label = f'{args.band} — {view.value_label if view else "value"}'
        subtitle = f'vs each subject\'s 0-pain baseline ({view_params.get("normalization")})'
        footnote = (
            'One dot = one subject (mean over that subject\'s epochs), NOT one epoch.\n'
            'Values are the view\'s own, referenced to each subject\'s 0-pain baseline, '
            'so the 0-pain violin is ~0 BY CONSTRUCTION -- read it as a visible NOISE '
            'FLOOR, not as a result. It is not exactly 0 because the baseline pools '
            '2s WINDOWS while a plotted value averages EPOCHS, and QC masking leaves '
            'epochs with unequal window counts (docs/labnotebook 2026-08-05).\n'
            'EXPLORATORY, discovery cohort -- NOMINATIONS, NOT FINDINGS.')
    else:
        label = f'{args.band} power (within-subject z)'
        subtitle = 'within-subject z, all levels pooled in the reference'
        footnote = (
            'One dot = one subject (mean over that subject\'s epochs), NOT one epoch. '
            'z is within-subject: each epoch minus that subject/region mean over ALL '
            'epochs, divided by its SD -- pooled across pain levels so no level is '
            'its own reference, which is why 0-pain is drawable here. Absolute z is '
            'therefore NOT comparable to the heatmaps\' z.\n'
            'EXPLORATORY, discovery cohort -- NOMINATIONS, NOT FINDINGS.')

    plot_violin_grid(
        subject_values, regions, panels,
        f'{args.band} ({lo:g}-{hi:g} Hz) by pain level — {len(subjects)} subjects, '
        f'{subtitle}',
        run_dir / f'{args.band}_violin_by_region.png',
        label, ncols=args.ncols, footnote=footnote)

    io.log_analysis(f'{args.band}-band violins by region, within-subject z, '
                    f'{len(regions)} regions, n={len(subjects)}', run_dir)
    logger.info('figure + provenance -> %s', run_dir)


if __name__ == '__main__':
    main()
