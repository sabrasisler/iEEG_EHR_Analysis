"""Unpooled per-subject pain slopes for every cell of a grid run.

Sign consistency -- what fraction of subjects slope the same way as the group --
needs each subject's OWN slope. The obvious shortcut is the model's BLUPs, which
already exist for every cell, and it does not work: partial pooling drags every
subject toward the group, so BLUP-based consistency comes out at 0.8-1.0 almost
everywhere and separates nothing. The pilot's raw per-subject fits, by contrast,
ran 50-94% and cleanly split real cells from null ones.

So this recomputes the unpooled fits. It is NOT a refit of the mixed model -- each
subject gets an ordinary least-squares line through their own epochs, which is
microseconds. The cost here is entirely the Lustre read of the per-channel view,
which is why it is an array over regions like the grid itself.

    python -m ieeg_ehr.analysis.compute_subject_slopes_grid \
        --run-dir <grid run> --region-index N
    python -m ieeg_ehr.analysis.compute_subject_slopes_grid \
        --run-dir <grid run> --stage collect
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ieeg_ehr import io
from ieeg_ehr.analysis import reference_run
from ieeg_ehr.analysis.plot_mixed_model_subject_lines import epoch_level, subject_slopes
from ieeg_ehr.analysis.run_mixed_model_grid import resolve_cohort
from ieeg_ehr.analysis.run_mixed_model_pilot import load_cell_frames, resolve_view_dir

logger = logging.getLogger(__name__)


def grid_axes(run_dir):
    """(ordered regions, bins) taken from the run's own cells table.

    Read from the artifact rather than from the ROI registry, so this can only
    ever compute slopes for cells the run actually contains -- if the run was
    filtered, the filter is inherited for free.
    """
    cells = io.read_table(Path(run_dir) / 'grid_cells.parquet', on_stale='warn')
    regions = sorted(cells['region'].unique())
    return cells, regions


def stage_fit(args):
    run_dir = Path(args.run_dir)
    cells, regions = grid_axes(run_dir)
    if not 0 <= args.region_index < len(regions):
        raise SystemExit(f'--region-index {args.region_index} outside 0..{len(regions)-1}')
    region = regions[args.region_index]

    todo = cells[cells['region'] == region]
    bins = sorted(todo['freq_bin_index'].unique())
    logger.info('region %d/%d: %s, %d bins', args.region_index, len(regions),
                region, len(bins))

    ref = reference_run.load(args.reference_run)
    view_dir = resolve_view_dir(None, mask_label=ref.view_params.get('mask_label'),
                                roi_scheme=ref.view_params.get('roi_scheme', 'roi_v2'))
    paths, _, _, subjects, roi_by_subject, _ = resolve_cohort(ref, view_dir)

    frames = load_cell_frames(paths, subjects, {(region, int(b)) for b in bins},
                              roi_by_subject)

    parts = []
    for b in bins:
        df = frames.get((region, int(b)))
        if df is None or df.empty:
            continue
        s = subject_slopes(epoch_level(df))
        s['region'] = region
        s['freq_bin_index'] = int(b)
        parts.append(s)

    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=['subject', 'slope', 'se', 'p', 'region', 'freq_bin_index'])
    io.write_table(out, run_dir / 'subject_slopes' / f'region_{args.region_index:03d}.parquet',
                   params={'region': region, 'n_bins': len(bins),
                           'source': 'unpooled per-subject OLS, not BLUPs'},
                   parents=[str(run_dir / 'grid_cells.parquet')],
                   script='ieeg_ehr/analysis/compute_subject_slopes_grid.py')
    logger.info('%s: %d subject-slope rows over %d bins', region, len(out), len(bins))


def stage_collect(args):
    run_dir = Path(args.run_dir)
    cells, regions = grid_axes(run_dir)

    parts, missing = [], []
    for i, region in enumerate(regions):
        p = run_dir / 'subject_slopes' / f'region_{i:03d}.parquet'
        if not p.exists():
            missing.append(region)
            continue
        parts.append(io.read_table(p, on_stale='ignore'))
    if missing:
        logger.error('MISSING per-region slope files: %s', missing)
    if not parts:
        raise SystemExit('nothing to collect')

    slopes = pd.concat(parts, ignore_index=True)

    # Sign consistency per cell, plus the count actually contributing a slope --
    # a subject with a single distinct pain score has no line and must not be
    # counted in the denominator as if they had disagreed.
    merged = slopes.merge(cells[['region', 'freq_bin_index', 'beta_nrs_within']],
                          on=['region', 'freq_bin_index'], how='inner')
    ok = merged[merged['slope'].notna() & merged['beta_nrs_within'].notna()].copy()
    ok['agrees'] = np.sign(ok['slope']) == np.sign(ok['beta_nrs_within'])
    cons = (ok.groupby(['region', 'freq_bin_index'])
            .agg(frac_sign_consistent=('agrees', 'mean'),
                 n_with_slope=('agrees', 'size'),
                 n_subject_p05=('p', lambda s: int((s < 0.05).sum())))
            .reset_index())

    io.write_table(slopes, run_dir / 'subject_slopes.parquet',
                   params={'source': 'unpooled per-subject OLS'},
                   parents=[str(run_dir / 'grid_cells.parquet')],
                   subjects=sorted(slopes['subject'].dropna().unique()),
                   script='ieeg_ehr/analysis/compute_subject_slopes_grid.py')
    io.write_table(cons, run_dir / 'sign_consistency.parquet',
                   params={'definition': 'fraction of subjects whose UNPOOLED slope '
                                         'shares the sign of the group fixed effect'},
                   parents=[str(run_dir / 'subject_slopes.parquet')],
                   script='ieeg_ehr/analysis/compute_subject_slopes_grid.py')

    logger.info('%d cells with sign consistency | median %.3f | range %.3f-%.3f',
                len(cons), cons['frac_sign_consistent'].median(),
                cons['frac_sign_consistent'].min(), cons['frac_sign_consistent'].max())
    io.log_analysis('unpooled per-subject slopes + sign consistency for the '
                    'mixed-model grid (EXPLORATORY)', run_dir)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--stage', choices=['fit', 'collect'], default='fit')
    ap.add_argument('--region-index', type=int, default=None)
    ap.add_argument('--reference-run', default=str(reference_run.CONTPAIN_HEATMAP))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    io.warn_if_dirty()

    if args.stage == 'collect':
        stage_collect(args)
    else:
        if args.region_index is None:
            raise SystemExit('--stage fit needs --region-index')
        stage_fit(args)


if __name__ == '__main__':
    main()
