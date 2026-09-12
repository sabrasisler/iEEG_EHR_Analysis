"""Cluster permutation test on the change-score medication map.

WHY THIS AND NOT THE MASS-UNIVARIATE BH. The group grid treats 798 region x
frequency cells as 798 tests. They are not: log-spaced bins are drawn from the
same FFT, and regions share subjects and channels. The diagnostics made the
symptom visible -- 22-38% of cells fell under p=.05 where a global null expects
5%. BH across a correlated family does not repair that; it renames it. A cluster
test uses the correlation instead of fighting it, by making a contiguous run of
frequency bins the unit of inference.

WHAT IS TESTED. One value per subject per region per bin: that subject's own
`med_between` coefficient from `--stage subjects`. The null is that this is
symmetric about zero -- i.e. a dose moves power neither up nor down. Adjacency is
along FREQUENCY only, within a region; region rows are not neighbours, because
the heatmap's row order is a display choice rather than an anatomical graph.

WHAT IT LICENSES. "This cluster differs from zero SOMEWHERE inside it" -- never
"this bin does". Cluster extent is not a localisation claim, and per CLAUDE.md a
p-value never travels without an effect size, so every cluster row here carries
the mean and peak per-subject beta and the subject count behind it.

Read `docs/cluster_permutation.md` before changing any parameter.

    python -m ieeg_ehr.analysis.run_change_score_cluster --run-dir <arm>
"""

import argparse
import glob
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ieeg_ehr import config, io
from ieeg_ehr.analysis import cluster_permutation as cp
from ieeg_ehr.analysis import view_tables

logger = logging.getLogger(__name__)

#: Level-3 output type. DELIBERATELY NOT `univariate_analysis`: this is a
#: different inference, not another view of the univariate one. The whole reason
#: it exists is that the mass-univariate family is wrong for a correlated grid,
#: so filing its output inside that tree would bury the distinction exactly where
#: a reader is most likely to conflate the two. The source run is recorded as a
#: provenance parent instead.
OUTPUT_TYPE = 'cluster_permutation'
QUESTION = 'psd_physiology'
VIEW_SCHEME = 'med_change_score'

DISCLAIMER = ('EXPLORATORY -- discovery cohort, NOMINATIONS NOT FINDINGS. '
              'Cluster inference localises to a CLUSTER, never to a bin.')


def load_subject_slopes(run_dir):
    files = [f for f in sorted(glob.glob(f'{run_dir}/subject_slopes/*.parquet'))
             if not f.endswith('provenance.json')]
    if not files:
        raise SystemExit(
            f'no subject_slopes/*.parquet in {run_dir}.\n'
            'Run the per-subject stage first:\n'
            '    STAGE=subjects sbatch ... sbatch/change_score_array.sbatch')
    df = pd.concat([io.read_table(f, on_stale='ignore') for f in files],
                   ignore_index=True)
    return df[df['ok'].fillna(False)] if 'ok' in df.columns else df


def to_cube(slopes, regions, bins):
    """(n_subject, n_region, n_bin) of per-subject med_between coefficients."""
    subjects = sorted(slopes['subject'].unique())
    s_ix = {s: i for i, s in enumerate(subjects)}
    r_ix = {r: i for i, r in enumerate(regions)}
    b_ix = {b: i for i, b in enumerate(bins)}

    x = np.full((len(subjects), len(regions), len(bins)), np.nan)
    for row in slopes.itertuples():
        r = r_ix.get(row.region)
        b = b_ix.get(int(row.freq_bin_index))
        if r is None or b is None:
            continue
        x[s_ix[row.subject], r, b] = row.med_beta
    return x, subjects


def cluster_table(res, regions, bins, bin_low, x):
    """One row per cluster, with the EFFECT SIZE alongside every p-value."""
    rows = []
    for c in res['clusters']:
        r, lo, hi = c['region_idx'], c['bin_lo'], c['bin_hi']
        block = x[:, r, lo:hi + 1]
        per_subject = np.nanmean(block, axis=1)
        finite = np.isfinite(per_subject)
        rows.append({
            'region': regions[r],
            'bin_lo': bins[lo], 'bin_hi': bins[hi], 'n_bins': c['n_bins'],
            'hz_lo': float(bin_low.get(bins[lo], np.nan)),
            'hz_hi': float(bin_low.get(bins[hi], np.nan)),
            'sign': c['sign'], 'mass': c['mass'], 'peak_t': c['peak_t'],
            'p_within_region': c['p_within_region'],
            'region_p_bh': c.get('region_p_bh', np.nan),
            'p_global': c['p_global'],
            'sig_two_stage': c['sig_two_stage'], 'sig_global': c['sig_global'],
            # Effect size, in the units of the thing modelled: d log10 power
            # attributable to a dose falling between the two assessments.
            'mean_beta': float(np.nanmean(block)),
            'peak_abs_beta': float(np.nanmax(np.abs(block))),
            'frac_subjects_same_sign': float(
                np.mean(np.sign(per_subject[finite]) == c['sign'])
                if finite.any() else np.nan),
            'n_subjects': int(finite.sum()),
        })
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(['sig_two_stage', 'p_within_region'],
                              ascending=[False, True]).reset_index(drop=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--roi-scheme', default='roi_v2')
    ap.add_argument('--n-perm', type=int, default=10000)
    ap.add_argument('--min-cluster-bins', type=int, default=3)
    ap.add_argument('--min-subjects', type=int, default=8)
    ap.add_argument('--alpha', type=float, default=0.05)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--statistic', choices=['t', 'yuen'], default='t')
    ap.add_argument('--question', default=QUESTION)
    ap.add_argument('--view-scheme', default=VIEW_SCHEME)
    ap.add_argument('--run-name', default=None,
                    help='defaults to cluster_<source run dir name>')
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    io.warn_if_dirty()

    run_dir = Path(args.run_dir)
    slopes = load_subject_slopes(run_dir)
    cells = io.read_table(run_dir / 'grid_cells.parquet', on_stale='warn')

    regions = [r for r in view_tables.roi_regions_for({'roi_scheme': args.roi_scheme})
               if r in set(slopes['region'])]
    bins = sorted(int(b) for b in slopes['freq_bin_index'].unique())
    bin_low = (cells.drop_duplicates('freq_bin_index')
               .set_index('freq_bin_index')['freq_bin_low'].to_dict())

    x, subjects = to_cube(slopes, regions, bins)
    n_map = np.isfinite(x).sum(axis=0)
    logger.info('cube: %d subjects x %d regions x %d bins | median %d subjects/cell',
                len(subjects), len(regions), len(bins), int(np.median(n_map)))

    res = cp.cluster_test(x, alpha=args.alpha, min_extent=args.min_cluster_bins,
                          n_perm=args.n_perm, seed=args.seed,
                          statistic=args.statistic, min_subjects=args.min_subjects)

    table = cluster_table(res, regions, bins, bin_low, x)
    n_two = int(table['sig_two_stage'].sum()) if len(table) else 0
    n_glob = int(table['sig_global'].sum()) if len(table) else 0
    logger.info('=' * 70)
    logger.info('%d candidate clusters | %d significant two-stage | %d global',
                len(table), n_two, n_glob)
    if len(table):
        logger.info('\n%s', table.head(15).to_string(index=False))
    logger.info('=' * 70)

    params = {'n_perm': args.n_perm, 'min_cluster_bins': args.min_cluster_bins,
              'min_subjects': args.min_subjects, 'alpha': args.alpha,
              'seed': args.seed, 'statistic': args.statistic,
              'n_subjects': len(subjects), 'n_regions': len(regions),
              'n_bins': len(bins), 'source_run': run_dir.name}

    # Its OWN level-3 output type, never inside `univariate_analysis`. The source
    # run is a provenance parent, not a parent directory.
    out_dir = config.analysis_run_dir(
        question=args.question, output_type=OUTPUT_TYPE,
        view_scheme=args.view_scheme,
        run_name=args.run_name or f'cluster_{run_dir.name}')
    out_dir.mkdir(parents=True, exist_ok=True)

    io.write_table(table, out_dir / 'clusters.csv', params=params,
                   parents=[str(run_dir / 'subject_slopes'),
                            str(run_dir / 'provenance.json')],
                   subjects=sorted(subjects),
                   script='ieeg_ehr/analysis/run_change_score_cluster.py')

    # The per-subject mean map is the effect-size substrate the outlines sit on.
    mean_map = pd.DataFrame(np.nanmean(x, axis=0), index=regions, columns=bins)
    mean_map.index.name = 'region'
    io.write_table(mean_map.reset_index(), out_dir / 'subject_mean_med_beta.csv',
                   params=params,
                   script='ieeg_ehr/analysis/run_change_score_cluster.py')

    io.write_run_provenance(
        out_dir, script='ieeg_ehr/analysis/run_change_score_cluster.py',
        params=params, subjects=sorted(subjects),
        parents=[str(run_dir / 'provenance.json')],
        extra={'status': 'EXPLORATORY cluster permutation, NOT a finding',
               'licenses': 'A significant cluster means the effect differs from '
                           'zero SOMEWHERE inside it. Cluster extent is NOT a '
                           'localisation claim and no single bin inside it is '
                           'thereby significant.',
               'why_not_univariate': 'The 798-cell BH family is wrong for this '
                                     'grid: log-spaced bins come from one FFT '
                                     'and regions share subjects and channels. '
                                     'Measured symptom: 22-38% of cells under '
                                     'p=.05 where a global null expects 5%.'})
    io.log_analysis('change-score cluster permutation over frequency on the '
                    'per-subject med_between map (EXPLORATORY)', out_dir)
    print(out_dir)


if __name__ == '__main__':
    main()
