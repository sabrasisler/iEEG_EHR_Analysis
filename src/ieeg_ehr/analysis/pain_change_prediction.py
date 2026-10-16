"""pain_change prediction: invert the d_pain fixed effects into a predicted change in pain.

    python -m ieeg_ehr.analysis.pain_change_prediction weights \\
        --source-run <pain_change run dir> --name insample_painchange_20261006-211218
    python -m ieeg_ehr.analysis.pain_change_prediction score \\
        --trajectory-dir <features/pain/pain_change_trajectory/...> \\
        --weights insample_painchange_20261006-211218

The scoring half of the sliding-window prediction. The extraction
(`features/build_pain_change_trajectory.py`) runs once. Scoring reruns per
weights file, so a leave-one-subject-out or split-half set needs only a new file.

`weights` exports one row per CONVERGED ROI x band cell of a `pain_change` fit:
`roi, band, beta` (the `d_pain` fixed effect), `var_resid`, `n_subjects`, `n_pairs`.
From a `pain_change_loo` run it exports the same schema plus `heldout_subject`, one
set of rows per fold, and `predict` scores each subject with the fold that
excluded it.

A fit that records its intercept (`pain_change_loo`) also exports `alpha`, `gamma`
and `delta` (intercept, `pain_1_within` and `gap_h` effects), and its centering,
the per-subject pain_1 mean of each cell, beside the weights as
`<name>_centering.csv`.

`score` computes, per pair and window end t, over the cells the subject has
(a NaN cell drops out of both sums):

    raw   dpain_hat(t) = sum_c beta_c dz_c(t) / sum_c beta_c^2
    adj   the same on r_c(t) = dz_c(t) - alpha_c - gamma_c pain_1_within_c
                               - delta_c (t - t1) in hours

the least-squares inversion of the model with d_pain unknown. `pain_1_within_c` is
pain1 minus the subject's centering mean for that cell, the one the fit used.
Each comes `uniform` and `ivw` (every term divided by var_resid_c). The adjusted
prediction needs alpha and the centering, and below EXTRAPOLATE_BELOW_MIN it is
`extrapolated`, as the fit saw no shorter gap. Output lands in
`analysis/pain/pain_change/prediction/<data token>/<run token>_<timestamp>/`, the
tokens of the fit the weights came from (docs/run_tokens.md), with the raw uniform
figures of `plot_pain_change_prediction`. A weights file is named
`<data token>_<source run>` unless `--name` says otherwise.

EXPLORATORY and descriptive: no significance testing. Discovery cohort only.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ieeg_ehr import config, io
from ieeg_ehr.analysis.change_score import DEFAULT_MIN_GAP_MIN
from ieeg_ehr.features.build_pain_change_trajectory import cell_column

logger = logging.getLogger(__name__)

SCRIPT = 'ieeg_ehr/analysis/pain_change_prediction.py'
WEIGHT_COLUMNS = ['roi', 'band', 'beta', 'var_resid', 'n_subjects', 'n_pairs']
#: Fit columns the adjusted prediction needs, and their weights-file names.
ADJUST_TERMS = {'intercept_beta': 'alpha', 'baseline_beta': 'gamma', 'gap_beta': 'delta'}
#: The pain_change fit's shortest gap: an adjusted step before it is extrapolated.
EXTRAPOLATE_BELOW_MIN = DEFAULT_MIN_GAP_MIN

#: Default delta-pain bins, inclusive at both ends, in display order.
DPAIN_BINS = {'large decrease': (-10, -3), 'small decrease': (-2, -1),
              'no change': (0, 0), 'small increase': (1, 2), 'large increase': (3, 10)}


def assign_bins(delta_pain, bins=None):
    """The bin label of each integer delta_pain, an ordered categorical."""
    bins = bins or DPAIN_BINS
    labels = pd.Series(pd.NA, index=delta_pain.index, dtype=object)
    for label, (lo, hi) in bins.items():
        labels[(delta_pain >= lo) & (delta_pain <= hi)] = label
    if labels.isna().any():
        raise ValueError(f'delta_pain values outside every bin: '
                         f'{sorted(delta_pain[labels.isna()].unique())}')
    return pd.Categorical(labels, categories=list(bins), ordered=True)


def count_table(pairs, bins=None):
    """N pairs and N subjects per integer delta_pain and per bin."""
    df = pairs.assign(bin=assign_bins(pairs['delta_pain'], bins))
    by_delta = (df.groupby('delta_pain').agg(n_pairs=('pair_id', 'size'),
                                             n_subjects=('subject', 'nunique'))
                .reset_index().rename(columns={'delta_pain': 'group'})
                .assign(level='delta_pain'))
    by_bin = (df.groupby('bin', observed=False).agg(n_pairs=('pair_id', 'size'),
                                                    n_subjects=('subject', 'nunique'))
              .reset_index().rename(columns={'bin': 'group'}).assign(level='bin'))
    return pd.concat([by_delta, by_bin], ignore_index=True)[
        ['level', 'group', 'n_pairs', 'n_subjects']]


# ============================================================================
# Weights
# ============================================================================

def weights_source(source_run):
    """`cells_loo.csv` of a `pain_change_loo` run, else `cells.csv` of a `pain_change` run."""
    loo = Path(source_run) / 'cells_loo.csv'
    return loo if loo.exists() else Path(source_run) / 'cells.csv'


def export_weights(source_run):
    """The converged cells of a pain_change run, in the weights schema.

    From a leave-one-subject-out run, one set per fold with `heldout_subject`,
    the everyone-in gate fold left out.
    """
    from ieeg_ehr.analysis.pain_change_loo import NO_HOLDOUT
    cells = io.read_table(weights_source(source_run), on_stale='warn')
    columns = list(WEIGHT_COLUMNS)
    if 'heldout_subject' in cells.columns:
        cells = cells[cells['heldout_subject'] != NO_HOLDOUT]
        columns.append('heldout_subject')
    if set(ADJUST_TERMS) <= set(cells.columns):
        columns += list(ADJUST_TERMS.values())
    keep = (cells['converged'].astype(bool) & np.isfinite(cells['dpain_beta'])
            & np.isfinite(cells['var_resid']))
    w = cells[keep].rename(columns={'region': 'roi', 'freq': 'band', 'dpain_beta': 'beta',
                                    **ADJUST_TERMS})
    logger.info('%d of %d cells converged and exported', len(w), len(cells))
    return w[columns].reset_index(drop=True)


def export_centering(source_run):
    """`roi, band, subject, pain_1_mean` from a `pain_change_loo` run, or None."""
    path = Path(source_run) / 'centering.csv'
    if not path.exists():
        return None
    return (io.read_table(path, on_stale='warn')
            .rename(columns={'region': 'roi', 'freq': 'band'}))


def centering_path(weights_name):
    return config.pain_change_prediction_weights_path(f'{weights_name}_centering')


# ============================================================================
# Scoring
# ============================================================================

def _invert(x, beta, inv_var):
    """(uniform, ivw) least-squares d_pain per row, over the finite cells of x."""
    finite = np.isfinite(x)
    b = np.where(finite, beta, 0.0)
    x = np.where(finite, x, 0.0)
    with np.errstate(invalid='ignore', divide='ignore'):
        return ((b * x).sum(axis=1) / (b ** 2).sum(axis=1),
                (b * x * inv_var).sum(axis=1) / (b ** 2 * inv_var).sum(axis=1))


def predict(dz_cells, pairs, weights, centering=None):
    """(pair_id, t_min) x the raw and, given alpha and centering, adjusted predictions.

    Also `n_cells_used` and `sum_beta2` of the raw uniform prediction per step.
    Weights with a `heldout_subject` column score each subject with its own fold.
    """
    adjusted = centering is not None and 'alpha' in weights.columns
    meta = pairs.set_index('pair_id')
    parts = []
    for subject, rows in dz_cells.groupby(dz_cells['pair_id'].map(meta['subject']),
                                          sort=False):
        w = (weights[weights['heldout_subject'] == subject]
             if 'heldout_subject' in weights.columns else weights)
        if w.empty:
            raise ValueError(f'no weights fold holds out {subject}')
        cols = [cell_column(r, b) for r, b in zip(w['roi'], w['band'])]
        missing = sorted(set(cols) - set(rows.columns))
        if missing:
            raise ValueError(f'weights name cells absent from dz_cells: {missing}')
        x = rows[cols].to_numpy(dtype=config.CACHE_ACCUMULATE_DTYPE)
        beta, inv_var = w['beta'].to_numpy(), 1.0 / w['var_resid'].to_numpy()
        out = rows[['pair_id', 't_min']].copy()
        out['n_cells_used'] = np.isfinite(x).sum(axis=1)
        out['sum_beta2'] = (np.where(np.isfinite(x), beta, 0.0) ** 2).sum(axis=1)
        out['pred_raw_uniform'], out['pred_raw_ivw'] = _invert(x, beta, inv_var)
        if adjusted:
            mean = (centering[centering['subject'] == subject]
                    .set_index(['roi', 'band'])['pain_1_mean']
                    .reindex(list(zip(w['roi'], w['band']))).to_numpy())
            pain_1_within = rows['pair_id'].map(meta['pain1']).to_numpy()[:, None] - mean
            gap_h = rows['t_min'].to_numpy()[:, None] / 60.0
            r = (x - w['alpha'].to_numpy() - w['gamma'].to_numpy() * pain_1_within
                 - w['delta'].to_numpy() * gap_h)
            out['pred_adj_uniform'], out['pred_adj_ivw'] = _invert(r, beta, inv_var)
            out['extrapolated'] = out['t_min'] < EXTRAPOLATE_BELOW_MIN
        parts.append(out)
    return pd.concat(parts, ignore_index=True)


def load_trajectory(trajectory_dir, with_dz=True):
    """(pairs, dz_cells) for every discovery subject in an extraction directory.

    `with_dz=False` skips the dz files and returns None in their place.
    """
    trajectory_dir = Path(trajectory_dir)
    discovery = {f'sub-{s}' for s in config.discovery_subjects()}
    pairs, dz = [], []
    for path in sorted(trajectory_dir.glob('pairs_sub-*.parquet')):
        sid = path.stem.split('_')[1]
        if sid not in discovery:
            logger.warning('SPLIT GATE: %s is not discovery, skipped', sid)
            continue
        pairs.append(io.read_table(path, on_stale='warn'))
        if with_dz:
            dz.append(io.read_table(trajectory_dir / f'dz_{sid}.parquet', on_stale='warn'))
    if not pairs:
        raise SystemExit(f'no discovery pairs_sub-*.parquet in {trajectory_dir}')
    return (pd.concat(pairs, ignore_index=True),
            pd.concat(dz, ignore_index=True) if with_dz else None)


def storage_report(trajectory_dir, n_pairs, reference_run):
    """Bytes of the dz files, and that size scaled to every eligible discovery pair."""
    from ieeg_ehr.features.build_pain_change_trajectory import MAX_GAP_MIN, MIN_GAP_MIN
    size = sum(p.stat().st_size for p in Path(trajectory_dir).glob('dz_sub-*.parquet'))
    index = pd.read_csv(Path(reference_run) / 'pair_index.csv', dtype={'subject': str})
    eligible = index[index['gap_m'].between(MIN_GAP_MIN, MAX_GAP_MIN)
                     & index['subject'].isin(config.discovery_subjects())]
    return {'dz_bytes': int(size), 'n_pairs': int(n_pairs),
            'bytes_per_pair': size / n_pairs,
            'n_eligible_discovery_pairs': len(eligible),
            'n_eligible_discovery_subjects': int(eligible['subject'].nunique()),
            'projected_dz_bytes': int(size / n_pairs * len(eligible))}


def write_methods(run_dir, params, n_pairs, subjects, storage):
    text = f"""# pain_change sliding-window prediction

EXPLORATORY, discovery cohort, descriptive. No significance testing.

## What is plotted

For each consecutive-report pair, `d_z(t)` is the ROI x band change in z from the
5-min window before report 1 to the 5-min window ending at t, for
t = t1 ... t1 + 3 h in 1-min steps (`{params['trajectory_dir']}`). The predicted
change in pain inverts the `d_pain` fixed effects of `{params['weights']}`:

```
raw  dpain_hat(t) = sum_c beta_c d_z_c(t) / sum_c beta_c^2
adj  the same on d_z_c(t) - alpha_c - gamma_c pain_1_within_c - delta_c (t - t1) in h
```

over the converged cells the subject has, each `uniform` and `ivw` (every term
divided by var_resid_c). Adjusted prediction: {'yes' if params['adjusted'] else
'no, the weights carry no intercept or centering'}. Below {EXTRAPOLATE_BELOW_MIN:g}
min the adjusted prediction is extrapolated: the fit saw no shorter gap.

## Data

{len(subjects)} subjects, {n_pairs} pairs. dz files: {storage['dz_bytes'] / 1e6:.1f} MB,
{storage['bytes_per_pair'] / 1e3:.0f} KB per pair, so about
{storage['projected_dz_bytes'] / 1e6:.0f} MB for all {storage['n_eligible_discovery_pairs']}
eligible pairs of {storage['n_eligible_discovery_subjects']} discovery subjects.

## Figures

Subject means first, then mean +/- SEM across subjects. Groups under
`min_subjects` are dropped. The shaded 0-5 min overlaps the baseline window.
"""
    (run_dir / 'METHODS.md').write_text(text)


def source_tokens(weights_name):
    """(data token, run token) of the fit a weights file came from."""
    path = config.pain_change_prediction_weights_path(weights_name)
    source = Path(io.read_sidecar(path)['params']['source_run'])
    return source.parent.name, source.name.rsplit('_', 1)[0]


def score(trajectory_dir, weights_name):
    """Write one scoring run beside its fit's tokens; returns its directory."""
    weights_path = config.pain_change_prediction_weights_path(weights_name)
    weights = io.read_table(weights_path, on_stale='warn')
    centering = (io.read_table(centering_path(weights_name), on_stale='warn')
                 if centering_path(weights_name).exists() else None)
    pairs, dz = load_trajectory(trajectory_dir)
    preds = predict(dz, pairs, weights, centering)
    traj_manifest = json.loads((Path(trajectory_dir) / io.MANIFEST_NAME).read_text())
    reference_run = next(p['path'] for p in traj_manifest['parents']
                         if Path(p['path']).name == traj_manifest['params']['reference_run'])

    data_token, run_token = source_tokens(weights_name)
    run_dir = config.analysis_run_dir(question=config.PAIN_CHANGE_QUESTION,
                                      output_type=config.PAIN_CHANGE_PREDICTION_OUTPUT,
                                      view_scheme=data_token, run_name=run_token)
    subjects = sorted(pairs['subject'].unique())
    params = {'trajectory_dir': str(trajectory_dir), 'weights': weights_name,
              'n_weight_cells': len(weights), 'bins': DPAIN_BINS,
              'adjusted': 'pred_adj_uniform' in preds.columns}
    parents = [str(trajectory_dir), str(weights_path)] + (
        [str(centering_path(weights_name))] if centering is not None else [])
    counts = count_table(pairs)
    print(counts.to_string(index=False))
    io.write_table(preds, run_dir / f'predictions_{weights_name}.csv', params=params,
                   parents=parents, subjects=subjects, script=SCRIPT)
    io.write_table(counts, run_dir / 'counts.csv', params=params, parents=parents,
                   subjects=subjects, script=SCRIPT)
    storage = storage_report(trajectory_dir, len(pairs), reference_run)
    write_methods(run_dir, params, len(pairs), subjects, storage)
    io.write_run_provenance(
        run_dir, script=SCRIPT, params=params, parents=parents, subjects=subjects,
        extra={'status': 'EXPLORATORY, descriptive, NOT a finding', 'storage': storage})
    io.log_analysis(f'pain_change sliding-window prediction, weights {weights_name}, '
                    f'{len(subjects)} subjects, {len(pairs)} pairs', run_dir)
    return run_dir


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
    w = sub.add_parser('weights', help='export a pain_change fit as a weights file')
    w.add_argument('--source-run', required=True)
    w.add_argument('--name', default=None,
                   help='weights file name; default <data token>_<source run>')
    s = sub.add_parser('score', help='score an extraction with a weights file, then plot')
    s.add_argument('--trajectory-dir', required=True)
    s.add_argument('--weights', required=True, help='weights file name')
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    io.warn_if_dirty()

    if args.command == 'weights':
        source = Path(args.source_run)
        args.name = args.name or f'{source.parent.name}_{source.name}'
        weights = export_weights(args.source_run)
        path = io.write_table(weights, config.pain_change_prediction_weights_path(args.name),
                              params={'source_run': str(args.source_run),
                                      'filter': 'converged, finite beta and var_resid'},
                              parents=[str(weights_source(args.source_run))],
                              script=SCRIPT)
        print(path)
        centering = export_centering(args.source_run)
        if centering is not None:
            print(io.write_table(centering, centering_path(args.name),
                                 params={'source_run': str(args.source_run)},
                                 parents=[str(Path(args.source_run) / 'centering.csv')],
                                 script=SCRIPT))
        return 0

    from ieeg_ehr.analysis import plot_pain_change_prediction as plots
    run_dir = score(args.trajectory_dir, args.weights)
    plots.render_all(run_dir, args.weights)
    print(run_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())
