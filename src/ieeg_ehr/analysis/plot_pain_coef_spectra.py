#!/usr/bin/env python3
"""
Continuous-pain spectra: one subplot per region, ONE line -- the group-mean
regression of log power on pain score against frequency, with a +-SEM ribbon.

The line-plot companion to the `contpain` heatmap, and the same relationship to it
that the z-score spectra have to the z-score heatmap: a colour scale says WHERE, a
quantitative axis says WHAT SHAPE. A broadband tilt, a narrowband beta effect and a
1/f rotation all look like "some red, some blue" on a heatmap and look like three
different curves here.

ONE LINE, NOT TWO OR THREE. The binned figures draw a line per pain level because
the pain axis is categorical there. Here pain is CONTINUOUS and already inside the
statistic: each subject contributes one coefficient per frequency bin, fitted across
all of their epochs. There is no level left to separate by, so the panel shows the
single curve and its ribbon.

READING THE SIGN. Above the y=0 line, power RISES with pain; below it, falls. Sign
is read from the axis rather than from colour, because the curve crosses zero and a
single colour cannot encode a direction that changes along the line. That is the one
thing the heatmap does better -- hence both figures.

Reuses `plot_pain_view_spectra.plot_grid` rather than reimplementing the grid, so
the two spectra figures cannot drift apart in axis ticks, line-noise masking or
isolated-point handling.

Run on Slurm, never the login node:
    python -m ieeg_ehr.analysis.plot_pain_coef_spectra --view-dir <RAW view>
"""

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd

from ieeg_ehr import io
from ieeg_ehr.analysis import pain_coef, view_tables
from ieeg_ehr.analysis.plot_pain_view_spectra import plot_grid, spectra_table
from ieeg_ehr.views import cache_reader

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_TYPE = 'region_spectrum'

# The single series' name and colour. Deliberately NOT one of the pain-level hues:
# this is not a pain level, and reusing 'high' crimson would imply it was. Dark
# slate reads as "the estimate" and lets the y=0 line carry the direction.
SERIES = 'pain_coef'
SERIES_COLOR = {'pain_coef': '#33404d'}


def coef_stats(coef, subjects, regions, freq_bins):
    """(region, freq_bin_index) -> mean, sd, sem, n_subjects ACROSS subjects.

    Shaped like `view_tables.subject_stats` output, with a single constant
    `pain_bin` column, so `plot_grid` and `spectra_table` can consume it unchanged.

    EQUAL-WEIGHTED across subjects, the project's standing rule: the subject is the
    unit of replication, so one with 200 contacts must not outvote one with 30. sd
    is the sample SD (ddof=1), so a cell backed by a single subject yields NaN and
    therefore no ribbon rather than a fabricated zero-width one.
    """
    rows = []
    for ri, region in enumerate(regions):
        for bi, b in enumerate(freq_bins):
            col = coef[:, ri, bi]
            col = col[np.isfinite(col)]
            n = col.size
            sd = float(col.std(ddof=1)) if n > 1 else np.nan
            rows.append({'pain_bin': SERIES, 'region': region,
                         'freq_bin_index': b,
                         'mean': float(col.mean()) if n else np.nan,
                         'sd': sd,
                         'sem': sd / np.sqrt(n) if n > 1 else np.nan,
                         'n_subjects': n})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--view-dir', required=True,
                    help='An UN-NORMALIZED view (--normalization none).')
    ap.add_argument('--run-name', default=None)
    ap.add_argument('--min-subjects', type=int, default=8)
    ap.add_argument('--min-epochs', type=int, default=pain_coef.MIN_EPOCHS)
    ap.add_argument('--min-range', type=float, default=pain_coef.MIN_RANGE)
    ap.add_argument('--min-non-modal', type=int, default=pain_coef.MIN_NON_MODAL)
    ap.add_argument('--spread', choices=['sem', 'sd', 'none'], default='sem')
    ap.add_argument('--ncols', type=int, default=4)
    ap.add_argument('--free-y', action='store_true')
    ap.add_argument('--keep-line-noise-bins', action='store_true')
    view_tables.add_output_arguments(ap)
    args = ap.parse_args()

    view_dir = Path(args.view_dir)
    io.warn_if_dirty()

    epoch_tables, epoch_paths = view_tables.load_view_tables(view_dir, 'epochs')
    _, subject_paths = view_tables.load_view_tables(view_dir, 'subject')
    view_params, view = view_tables.view_params_from(subject_paths)
    logger.info('view: %s', {k: view_params.get(k) for k in
                             ('normalization', 'domain', 'mask_label', 'roi_scheme')})

    if view is not None and view.is_difference:
        raise SystemExit(
            f'--view-dir is a {view.normalization!r} view. The regression must run on '
            'raw log power -- a baseline has already removed part of what it is meant '
            'to measure. Use --normalization none.')

    roi_regions = view_tables.roi_regions_for(view_params)
    epoch_minutes = view_params.get('epoch_minutes')
    bin_labels = cache_reader.bin_edges(epoch_minutes).set_index('freq_bin_index')
    freq_bins = bin_labels.index.tolist()
    line_noise = ([] if args.keep_line_noise_bins
                  else [b for b in cache_reader.line_noise_bins(epoch_minutes)
                        if b in freq_bins])
    logger.info('masking %d line-noise bin(s) from display: %s Hz', len(line_noise),
                [f'{bin_labels.loc[b, "bin_low_hz"]:.0f}' for b in line_noise])

    coef, subjects, _, diagnostics = pain_coef.subject_coef_matrix(
        epoch_tables, roi_regions, freq_bins,
        min_epochs=args.min_epochs, min_range=args.min_range,
        min_non_modal=args.min_non_modal)

    stats = coef_stats(coef, subjects, roi_regions, freq_bins)
    regions, per_region = view_tables.regions_with_min_subjects(
        stats, [SERIES], args.min_subjects, regions=roi_regions)
    if not regions:
        raise SystemExit(f'no region has >= {args.min_subjects} subjects')
    logger.info('%d region(s) plotted: %s', len(regions),
                {r: int(per_region[r]) for r in regions})

    if not args.view_scheme:
        roi_code = (view.scheme_code.rsplit('-', 1)[-1]
                    if view is not None and '-' in view.scheme_code else 'roidefault')
        args.view_scheme = f'contpain-{roi_code}'
    run_dir = view_tables.resolve_run_dir(
        args, OUTPUT_TYPE, view, run_name=args.run_name or 'discovery_contpain')
    logger.info('run dir: %s', run_dir)

    params = {**view_params, 'min_epochs': args.min_epochs,
              'min_range': args.min_range, 'min_non_modal': args.min_non_modal,
              'quantity': 'pain_coef'}
    io.write_table(spectra_table(stats, bin_labels, [SERIES], line_noise),
                   run_dir / 'pain_coef_spectra_table.parquet', kind='table',
                   script='ieeg_ehr/analysis/plot_pain_coef_spectra.py',
                   params=params,
                   parents=[io.parent_ref(p, digest=False) for p in epoch_paths],
                   subjects=subjects)
    io.write_table(diagnostics, run_dir / 'subject_diagnostics.parquet', kind='table',
                   script='ieeg_ehr/analysis/plot_pain_coef_spectra.py', params=params)

    io.write_run_provenance(
        run_dir, script='ieeg_ehr/analysis/plot_pain_coef_spectra.py',
        params={**vars(args), 'view_params': view_params},
        parents=[io.parent_ref(p, digest=False) for p in epoch_paths + subject_paths],
        subjects=subjects,
        extra={'quantity': 'pain_coef = OLS slope of log10(V^2/Hz) on pain score, '
                           'per subject per region per frequency bin',
               'n_subjects': len(subjects), 'regions_plotted': regions,
               'n_subjects_per_region': {r: int(per_region[r]) for r in regions},
               'line_noise_bins_masked': [int(b) for b in line_noise],
               'exclusions': diagnostics.loc[~diagnostics['included'],
                                             ['subject_id', 'excluded_because']]
                             .to_dict('records'),
               'status': 'EXPLORATORY nomination, not a finding '
                         '(CLAUDE.md; pending P2.6 FREEZE)'})

    plot_grid(stats, regions, per_region, [SERIES], bin_labels, line_noise,
              f'Continuous pain: d(log power)/d(pain score) — n={len(subjects)} subjects',
              run_dir / 'group_pain_coef_spectra.png',
              'd log10(V^2/Hz) per pain point',
              spread=None if args.spread == 'none' else args.spread,
              ncols=args.ncols, share_y=not args.free_y, zero_line=True,
              colors=SERIES_COLOR, legend_title='Regression coefficient')

    io.log_analysis(f'continuous-pain regression spectra (pain_coef), '
                    f'{len(regions)} regions, n={len(subjects)}', run_dir)
    logger.info('figure + provenance -> %s', run_dir)


if __name__ == '__main__':
    main()
