#!/usr/bin/env python3
"""
One band, every pain level, one dot per subject: violins of within-subject
normalized band power, one panel per region.

WHY THIS NEEDS A DIFFERENT NORMALIZATION FROM EVERY OTHER FIGURE HERE
---------------------------------------------------------------------
The heatmaps and spectra reference each subject to their own 0-pain baseline, which
is the right thing when you are plotting low and high. It CANNOT work here, because
this figure draws 'none' as one of its violins: if the 0-pain epochs define the
baseline, their normalized value is 0 by construction, and the first violin would
be a spike at zero sitting beside two real distributions. That is the same
circularity that makes the cluster test's `none` bin a noise floor rather than a
control (docs/cluster_permutation.md 6).

So the reference has to be something that is NOT one of the three levels being
drawn. This script standardizes each subject against THEIR OWN OVERALL LEVEL:

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


def epochs_to_band(epoch_tables, bin_labels, band, bands=None, drop_bins=()):
    """Per-epoch band power, aggregated over the bins whose CENTRE falls in the band.

    LINEAR-THEN-LOG. A mean of log values is a geometric mean, which is not the
    band's average power -- so bins are exponentiated, averaged, and re-logged, the
    same convention preprocessing/bipolar_bands.py and registry AXIS 5 use. Getting
    this wrong is silent and biases every band downward.

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
    # float64 before exponentiating: the cache is float32 and the worst stored
    # log-power is ~-36.8, barely a decade above float32's smallest normal (P0.6).
    rows['linear'] = np.power(10.0, rows['value'].to_numpy(dtype=np.float64))
    out = (rows.groupby(['subject_id', 'epoch_id', 'pain_bin', 'region'],
                        dropna=False)['linear'].mean().reset_index())
    with np.errstate(divide='ignore'):
        out['band_log_power'] = np.log10(out['linear'])
    return out.drop(columns='linear'), in_band


def within_subject_z(band_epochs, min_epochs=4):
    """z-score each epoch against its (subject, region) mean/SD over ALL epochs.

    Pooled over pain levels ON PURPOSE -- see the module docstring. A
    (subject, region) with fewer than `min_epochs` epochs, or zero variance, yields
    NaN rather than a fabricated z: a standard deviation from two epochs is not a
    scale, and dividing by it would manufacture enormous values from nothing.
    """
    grouped = band_epochs.groupby(['subject_id', 'region'], dropna=False)['band_log_power']
    stats = grouped.agg(subject_mean='mean', subject_sd=lambda s: s.std(ddof=1),
                        n_epochs='size').reset_index()
    merged = band_epochs.merge(stats, on=['subject_id', 'region'], how='left')

    usable = (merged['n_epochs'] >= min_epochs) & (merged['subject_sd'] > 0)
    merged['z'] = np.where(usable,
                           (merged['band_log_power'] - merged['subject_mean'])
                           / merged['subject_sd'].replace(0, np.nan), np.nan)
    dropped = int((~usable).sum())
    if dropped:
        logger.info('%d/%d epoch rows have no usable within-subject scale '
                    '(< %d epochs for that subject/region, or zero SD)',
                    dropped, len(merged), min_epochs)
    return merged


def subject_level(band_z, panels):
    """One value per (subject, region, pain_bin): mean z over that subject's epochs."""
    out = (band_z[band_z['pain_bin'].isin(panels)]
           .groupby(['subject_id', 'region', 'pain_bin'], dropna=False)
           .agg(value=('z', 'mean'), n_epochs=('epoch_id', 'nunique'))
           .reset_index()
           .dropna(subset=['value']))
    out['subject'] = out['subject_id']          # the violin helper's column name
    return out


def plot_violin_grid(subject_values, regions, panels, title, out_path, value_label,
                     ncols=4):
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
    fig.text(0.01, -0.01,
             'One dot = one subject (mean over that subject\'s epochs), NOT one '
             'epoch. z is within-subject: each epoch minus that subject/region mean '
             'over ALL epochs, divided by its SD -- pooled across pain levels so no '
             'level is its own reference, which is why 0-pain is drawable here. '
             'Absolute z is therefore NOT comparable to the heatmaps\' z.\n'
             'EXPLORATORY, discovery cohort -- NOMINATIONS, NOT FINDINGS.',
             ha='left', va='top', fontsize=6, color='0.25')
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info('Wrote %s', out_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--view-dir', required=True,
                    help='A view directory. Use the UN-NORMALIZED (raw) view: this '
                         'script does its own within-subject standardization, and '
                         'starting from an already-baseline-referenced view would '
                         'make the 0-pain violin 0 by construction.')
    ap.add_argument('--band', default='beta',
                    choices=sorted(config.CANONICAL_BANDS_HZ),
                    help='Band edges from config.CANONICAL_BANDS_HZ. NOTE '
                         'architecture.md PART 0 records an unresolved discrepancy '
                         'for beta (code 13-30 Hz, that doc 15-25).')
    ap.add_argument('--pain-bin-scheme', choices=list(view_tables.PANELS),
                    default='subject_relative')
    ap.add_argument('--min-subjects', type=int, default=8)
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

    # Refuse a view that is already baseline-referenced: 'none' would be 0 by
    # construction and the whole point of this figure would be lost. A loud stop,
    # because the resulting figure would look plausible.
    if view is not None and view.is_difference:
        raise SystemExit(
            f'--view-dir is a {view.normalization!r} view, which is already '
            'referenced to each subject\'s 0-pain baseline -- so the 0-pain violin '
            'would be exactly 0 and the other two would be measured against it. '
            'Use the un-normalized view (--normalization none) and let this script '
            'do the within-subject standardization.')

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
                                            drop_bins=drop_bins)
    band_z = within_subject_z(band_epochs, min_epochs=args.min_epochs)
    subject_values = subject_level(band_z, panels)

    roi_regions = view_tables.roi_regions_for(view_params)
    counts = (subject_values.groupby(['region', 'pain_bin'])['subject'].nunique()
              .unstack('pain_bin').reindex(columns=panels))
    per_region = counts.min(axis=1, skipna=False).fillna(0).astype(int)
    regions = [r for r in roi_regions if per_region.get(r, 0) >= args.min_subjects]
    below = {r: int(per_region.get(r, 0)) for r in roi_regions
             if 0 < per_region.get(r, 0) < args.min_subjects}
    if below:
        logger.info('%d region(s) below the %d-subject floor, not plotted: %s',
                    len(below), args.min_subjects, below)
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
        'normalization_note':
            'Within-subject z: each epoch minus that (subject, region) mean over ALL '
            'epochs, divided by its SD, POOLED across pain levels so that no level '
            'is its own reference. Deliberately NOT the view-level baseline, which '
            'would make the 0-pain violin 0 by construction. Absolute z is not '
            'comparable to the heatmaps.',
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
    plot_violin_grid(
        subject_values, regions, panels,
        f'{args.band} ({lo:g}-{hi:g} Hz) by pain level — {len(subjects)} subjects, '
        f'within-subject z',
        run_dir / f'{args.band}_violin_by_region.png',
        f'{args.band} power (within-subject z)', ncols=args.ncols)

    io.log_analysis(f'{args.band}-band violins by region, within-subject z, '
                    f'{len(regions)} regions, n={len(subjects)}', run_dir)
    logger.info('figure + provenance -> %s', run_dir)


if __name__ == '__main__':
    main()
