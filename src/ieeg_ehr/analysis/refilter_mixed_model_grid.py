"""Refilter an existing mixed-model grid and redo FDR, WITHOUT refitting.

Every cell in the grid is fitted independently, so which cells you keep changes
nothing about any beta, SE, LRT statistic or BLUP -- it changes only the
multiple-comparison FAMILY. Dropping cells and recomputing BH is therefore exactly
equivalent to having fitted only the kept cells, at none of the cost. Refitting to
change a filter would be a waste.

WHAT IS DROPPED

- UNRESOLVABLE BINS. The frequency axis is 50 log-spaced bins from 1 Hz, but the
  PSD comes from 2-second windows, so the true resolution is a flat 0.5 Hz. Below
  ~4.7 Hz a log bin is narrower than that and contains no FFT frequency of its own;
  the cache builder fills it from the nearest neighbour, making it an exact
  duplicate of a neighbour. At 5-minute epochs that is bins {1, 2, 4, 5, 7, 10}.
  Keeping them padded the BH family with 126 copies across 21 regions and made one
  repeated number read as broadband low-frequency structure.
- REGIONS BELOW A COVERAGE FLOOR, as a whole.

Both criteria are recorded in the new run's provenance, and the parent run is
referenced rather than modified -- the original stays exactly as it was produced.

    python -m ieeg_ehr.analysis.refilter_mixed_model_grid \
        --parent-run <grid run dir> --min-subjects 10
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from ieeg_ehr import config, io
from ieeg_ehr.analysis.run_mixed_model_grid import (
    OUTPUT_TYPE, QUESTION, VIEW_SCHEME, add_fdr, write_methods)
from ieeg_ehr.views import cache_reader

logger = logging.getLogger(__name__)

RUN_NAME = 'fullgrid_filtered'


def refilter(cells, drop_bins, min_subjects):
    """(kept cells, a per-region report of what happened and why)."""
    n0 = len(cells)
    coverage = cells.groupby('region')['n_subjects'].max()

    dropped_bin = cells['freq_bin_index'].isin(drop_bins)
    thin_region = cells['region'].map(coverage) < min_subjects

    report = pd.DataFrame({
        'n_subjects': coverage,
        'kept': ~(coverage < min_subjects),
    }).reset_index().sort_values('n_subjects')

    kept = cells[~dropped_bin & ~thin_region].copy()
    logger.info('%d cells in -> %d out (%d unresolvable-bin, %d thin-region)',
                n0, len(kept), int(dropped_bin.sum()),
                int((~dropped_bin & thin_region).sum()))
    logger.info('regions excluded at min_subjects=%d: %s', min_subjects,
                sorted(report.loc[~report['kept'], 'region']) or 'none')
    return kept, report


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--parent-run', required=True,
                    help='An existing grid run directory containing grid_cells.parquet.')
    ap.add_argument('--min-subjects', type=int, default=10)
    ap.add_argument('--epoch-minutes', type=float, default=5.0)
    ap.add_argument('--keep-unresolvable-bins', action='store_true')
    ap.add_argument('--question', default=QUESTION)
    ap.add_argument('--view-scheme', default=VIEW_SCHEME)
    ap.add_argument('--run-name', default=RUN_NAME)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    io.warn_if_dirty()

    parent = Path(args.parent_run)
    cells = io.read_table(parent / 'grid_cells.parquet', on_stale='warn')

    drop_bins = ([] if args.keep_unresolvable_bins
                 else [int(b) for b in cache_reader.unresolvable_bins(args.epoch_minutes)])
    logger.info('unresolvable bins: %s', drop_bins)

    kept, report = refilter(cells, drop_bins, args.min_subjects)
    if kept.empty:
        raise SystemExit('nothing survived the filter')

    # The whole point: BH is recomputed on the SURVIVING family, not inherited.
    kept = kept.drop(columns=[c for c in kept.columns if c.startswith(
        ('p_bh', 'p_lrt_bh'))], errors='ignore')
    kept = add_fdr(kept, 'p', 'p')
    kept = add_fdr(kept, 'p_lrt_mixture', 'p_lrt')

    run_dir = config.analysis_run_dir(question=args.question,
                                      output_type=OUTPUT_TYPE,
                                      view_scheme=args.view_scheme,
                                      run_name=args.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    params = {'parent_run': str(parent), 'min_subjects': args.min_subjects,
              'unresolvable_bins_removed': drop_bins,
              'refit': False,
              'note': 'Cells are copied unchanged from the parent; only the BH '
                      'family was recomputed on the surviving cells.'}
    io.write_table(kept, run_dir / 'grid_cells.parquet', params=params,
                   parents=[str(parent / 'provenance.json'),
                            str(parent / 'grid_cells.parquet')],
                   script='ieeg_ehr/analysis/refilter_mixed_model_grid.py')
    io.write_table(report, run_dir / 'region_coverage.parquet', params=params,
                   script='ieeg_ehr/analysis/refilter_mixed_model_grid.py')

    blup_path = parent / 'grid_blups.parquet'
    if blup_path.exists():
        blups = io.read_table(blup_path, on_stale='ignore')
        keys = set(zip(kept['region'], kept['freq_bin_index']))
        blups = blups[[(r, b) in keys
                       for r, b in zip(blups['region'], blups['freq_bin'])]]
        io.write_table(blups, run_dir / 'grid_blups.parquet', params=params,
                       parents=[str(blup_path)],
                       script='ieeg_ehr/analysis/refilter_mixed_model_grid.py')

    io.write_run_provenance(
        run_dir, script='ieeg_ehr/analysis/refilter_mixed_model_grid.py',
        params=params, parents=[str(parent / 'provenance.json')],
        extra={'status': 'EXPLORATORY, derived from a parent grid run by filtering '
                         'and re-running FDR. NO REFIT.',
               'p_provisional': 'Parametric Wald p, inherited from the parent. The '
                                'pilot permutation put the null z SD at ~1.03, so '
                                'these are approximately calibrated.'})

    manifest = kept[['region', 'freq_bin_index']].copy()
    manifest['above_coverage_floor'] = True
    write_methods(run_dir, kept, manifest, missing=[])

    sig = int((kept['p_bh_reject'] == True).sum())  # noqa: E712
    sig_lrt = int((kept['p_lrt_bh_reject'] == True).sum())  # noqa: E712
    logger.info('=' * 70)
    logger.info('REFILTERED GRID: %d cells, %d regions x %d bins',
                len(kept), kept['region'].nunique(), kept['freq_bin_index'].nunique())
    logger.info('  fixed effect BH-significant  : %d (was %d over %d cells)',
                sig, int((cells.get('p_bh_reject') == True).sum()), len(cells))  # noqa: E712
    logger.info('  heterogeneity BH-significant : %d', sig_lrt)
    logger.info('=' * 70)

    io.log_analysis('mixed-model grid refiltered (duplicate bins + coverage floor), '
                    'FDR recomputed, no refit (EXPLORATORY)', run_dir)
    print(run_dir)


if __name__ == '__main__':
    main()
