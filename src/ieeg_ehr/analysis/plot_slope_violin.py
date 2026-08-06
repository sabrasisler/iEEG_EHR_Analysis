#!/usr/bin/env python3
"""
1/f slope by pain level: violins of zero / low / high, one dot per subject,
one panel per region.

Reads the slope tables from `views/build_pain_epoch_slope.py` -- one value per
(epoch, region), fit per channel and averaged into the region -- and collapses
them the way every figure in this project collapses things: to ONE VALUE PER
SUBJECT. A dot is a subject, never an epoch. Epochs are nested within subjects,
and treating them as independent is the pseudo-replication that would inflate any
statistic computed off the figure.

WHAT THE Y AXIS IS
------------------
Two figures are written from the same table, and they answer different questions:

  *_z.png      within-subject z (PRIMARY). Each epoch's slope minus that
               (subject, region) mean over ALL epochs, divided by its SD, pooled
               across pain levels. Pooled so that no level is its own reference,
               which is the only reason 'none' can be drawn at all -- see
               analysis/view_tables.within_subject_z. Removes the between-subject
               offset, which is what makes the comparison legible.

  *_native.png the slope itself, in decades of power per decade of frequency.
               Interpretable on sight (a value near -2 is a 1/f^2 spectrum) and it
               shows the between-subject spread the z figure deliberately hides.
               A pain effect of order 0.05 against a between-subject spread of
               order 1 is why the z version is the primary.

MORE NEGATIVE = STEEPER. The y axis is NOT inverted; read a downward shift as the
spectrum tilting steeper (relatively more low-frequency power), which is the
direction usually reported for increased inhibition / reduced excitation.

WHAT THIS FIGURE IS FOR, next to the band violins: a band effect and an aperiodic
effect are not distinguishable on a heatmap. A broadband tilt shows up as a
same-signed change in every band, and the 2026-08-05 21-region run found exactly
that kind of block in the sensorimotor strip. If the effect is a tilt, it appears
here; if it is a genuine narrowband oscillation, it does not.

EXPLORATORY, discovery cohort. Nominations, not findings.

Run on Slurm, never the login node:
    python -m ieeg_ehr.analysis.plot_slope_violin --view-dir <slope view>
"""

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from ieeg_ehr import config, io
from ieeg_ehr.analysis import view_tables
from ieeg_ehr.features import common

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_TYPE = 'slope_violin'


def plot_violin_grid(subject_values, regions, panels, title, out_path, value_label,
                     footnote, ncols=4, reference=None):
    """One panel per region; one violin per pain level; one dot per subject."""
    subjects = sorted(subject_values['subject'].unique())
    colour = common.subject_color_map(subjects)

    nrows = int(np.ceil(len(regions) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 3.0 * nrows),
                             sharey=True, squeeze=False)
    flat = axes.ravel()

    for ax, region in zip(flat, regions):
        rows = subject_values[subject_values['region'] == region]
        if reference is not None:
            ax.axhline(reference, color='0.35', linewidth=0.8, zorder=1)
        common.draw_seaborn_violin_with_subject_dots(
            ax, rows, colour, value_col='value', pain_bins=panels)
        ax.set_title(f'{region}  (n={rows["subject"].nunique()} subj)', fontsize=9)
        ax.tick_params(labelsize=8)
        # seaborn labels the y axis with the value COLUMN name, which duplicates
        # the figure-level label once per panel and says nothing.
        ax.set_ylabel('')
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)

    for ax in flat[len(regions):]:
        ax.set_visible(False)

    fig.supylabel(value_label, fontsize=9)
    fig.suptitle(title, fontsize=12)
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
                    help='A SLOPE view directory from build_pain_epoch_slope.py.')
    ap.add_argument('--pain-bin-scheme', choices=list(view_tables.PANELS),
                    default='subject_relative')
    ap.add_argument('--min-subjects', type=int, default=8)
    ap.add_argument('--min-epochs', type=int, default=4,
                    help='A (subject, region) needs this many epochs before its SD is '
                         'treated as a scale. Below it the subject is dropped rather '
                         'than given a z built on nothing.')
    ap.add_argument('--min-r2', type=float, default=None,
                    help='Drop (epoch, region) rows whose mean fit r2 is below this. '
                         'Default: keep everything and report the distribution -- the '
                         'metric is stored per fit precisely so the threshold is a '
                         'cheap downstream choice, not a baked-in one.')
    ap.add_argument('--ncols', type=int, default=4)
    ap.add_argument('--run-name', default=None)
    view_tables.add_output_arguments(ap)
    args = ap.parse_args()

    view_dir = Path(args.view_dir)
    io.warn_if_dirty()

    epoch_tables, epoch_paths = view_tables.load_view_tables(view_dir, 'epochs')
    subject_paths = sorted(view_dir.glob('view_subject_sub-*.parquet'))
    view_params, view = view_tables.view_params_from(subject_paths or epoch_paths)

    # Refuse a POWER view. The two view kinds share file names on purpose (so the
    # loader is shared), which means the guard has to be on the recorded metric.
    # Without it this would happily plot a `value` column of log-power and label
    # the axis "slope".
    if view_params.get('metric') != 'aperiodic_slope':
        raise SystemExit(
            f'--view-dir does not hold slope tables (params[metric]='
            f'{view_params.get("metric")!r}, expected \'aperiodic_slope\'). Build one '
            'with `python -m ieeg_ehr.views.build_pain_epoch_slope`.')
    if 'slope' not in epoch_tables.columns:
        raise SystemExit(f'no `slope` column in {view_dir}')

    logger.info('view: %s', {k: view_params.get(k) for k in
                             ('normalization', 'mask_label', 'pain_bins', 'roi_scheme',
                              'fit_lo_hz', 'fit_hi_hz', 'min_fit_bins')})

    r2 = epoch_tables['r2'].to_numpy(dtype=float)
    logger.info('fit r2 over %d (epoch, region) rows: median %.3f, 5th pct %.3f, '
                'min %.3f', len(r2), float(np.nanmedian(r2)),
                float(np.nanpercentile(r2, 5)), float(np.nanmin(r2)))
    if args.min_r2 is not None:
        before = len(epoch_tables)
        epoch_tables = epoch_tables[epoch_tables['r2'] >= args.min_r2]
        logger.info('r2 >= %.2f keeps %d/%d rows', args.min_r2, len(epoch_tables), before)
        if epoch_tables.empty:
            raise SystemExit(f'no rows survive --min-r2 {args.min_r2}')

    # 'none' IS drawn here, unlike in the spectra and heatmaps: nothing in this
    # figure's standardization makes the 0-pain epochs their own reference.
    panels = [b for b in config.pain_bin_order(args.pain_bin_scheme)
              if b in set(epoch_tables['pain_bin'])]
    logger.info('violins per panel: %s', panels)

    epoch_z = view_tables.within_subject_z(epoch_tables, 'slope',
                                           min_epochs=args.min_epochs)
    z_values = view_tables.subject_level(epoch_z, panels, value_col='z')
    native_values = view_tables.subject_level(epoch_z, panels, value_col='slope')

    roi_regions = view_tables.roi_regions_for(view_params)
    regions, per_region, below = view_tables.regions_by_min_subjects(
        z_values, panels, roi_regions, args.min_subjects)
    if not regions:
        raise SystemExit(f'no region has >= {args.min_subjects} subjects in every '
                         f'level {panels}')
    logger.info('%d region(s) plotted: %s', len(regions),
                {r: int(per_region[r]) for r in regions})

    if not args.view_scheme:
        args.view_scheme = (view.scheme_code if view is not None else 'unknown')
    run_dir = view_tables.resolve_run_dir(args, OUTPUT_TYPE, view,
                                          run_name=args.run_name or 'discovery')
    logger.info('run dir: %s', run_dir)

    subjects = sorted(z_values['subject'].unique())
    lo, hi = view_params.get('fit_lo_hz'), view_params.get('fit_hi_hz')
    fig_params = {
        'unit_of_observation': "one subject (mean over that subject's epochs)",
        'min_epochs_for_scale': args.min_epochs,
        'min_r2_applied': args.min_r2,
        'r2_median': float(np.nanmedian(r2)),
        'normalization_note':
            'Within-subject z: each epoch minus that (subject, region) mean over ALL '
            'epochs, divided by its SD, POOLED across pain levels so no level is its '
            'own reference. Not the view-level baseline, which would make the 0-pain '
            'violin 0 by construction. Absolute z is not comparable to the heatmaps.',
        'sign_convention': 'more negative = steeper spectrum',
    }

    # Both value columns in ONE table, so the two figures are provably the same
    # subjects and the same epochs.
    merged = z_values.merge(
        native_values.rename(columns={'value': 'slope'})[
            ['subject_id', 'region', 'pain_bin', 'slope']],
        on=['subject_id', 'region', 'pain_bin'], how='left').rename(columns={'value': 'z'})
    io.write_table(merged, run_dir / 'subject_slope_values.parquet', kind='table',
                   script='ieeg_ehr/analysis/plot_slope_violin.py',
                   params={**view_params, **fig_params},
                   parents=[io.parent_ref(p, digest=False) for p in epoch_paths],
                   subjects=subjects)

    io.write_run_provenance(
        run_dir, script='ieeg_ehr/analysis/plot_slope_violin.py',
        params={**vars(args), 'view_params': view_params, **fig_params},
        parents=[io.parent_ref(p, digest=False) for p in epoch_paths + subject_paths],
        subjects=subjects,
        extra={'panels': panels, 'regions_plotted': regions,
               'n_subjects_per_region': {r: int(per_region[r]) for r in regions},
               'regions_below_floor': below, 'roi_regions': roi_regions,
               'status': 'EXPLORATORY nomination, not a finding '
                         '(CLAUDE.md; pending P2.6 FREEZE)'},
    )

    shared = (f'One dot = one subject (mean over that subject\'s epochs), NOT one '
              f'epoch. Slope = OLS of log10 power on log10 frequency over '
              f'{lo:g}-{hi:g} Hz, fit PER CHANNEL and averaged within the region; '
              f'line-noise bins excluded. MORE NEGATIVE = STEEPER.\n'
              f'EXPLORATORY, discovery cohort -- NOMINATIONS, NOT FINDINGS.')

    plot_violin_grid(
        z_values, regions, panels,
        f'1/f slope by pain level — {len(subjects)} subjects, within-subject z',
        run_dir / 'slope_violin_by_region_z.png',
        '1/f slope (within-subject z)',
        'z is within-subject: each epoch minus that subject/region mean over ALL '
        'epochs, divided by its SD -- pooled across pain levels so no level is its '
        'own reference, which is why 0-pain is drawable here. ' + shared,
        ncols=args.ncols, reference=0)

    plot_violin_grid(
        native_values, regions, panels,
        f'1/f slope by pain level — {len(subjects)} subjects, native units',
        run_dir / 'slope_violin_by_region_native.png',
        '1/f slope (decades power per decade Hz)',
        'Native units, so the between-subject spread the z figure removes is '
        'visible here -- it is large relative to any pain effect, which is why the '
        'z version is the primary. ' + shared,
        ncols=args.ncols)

    io.log_analysis(f'1/f slope violins by region ({lo:g}-{hi:g} Hz fit), '
                    f'{len(regions)} regions, n={len(subjects)}', run_dir)
    logger.info('figures + provenance -> %s', run_dir)


if __name__ == '__main__':
    main()
