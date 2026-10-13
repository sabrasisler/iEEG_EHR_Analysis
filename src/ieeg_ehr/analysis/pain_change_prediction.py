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
A held-out set uses the same schema plus `heldout_subject`, one set of rows per
fold, and `predict` scores each subject with the fold that excluded it.

`score` computes, per pair and window end t, over the cells the subject has
(a NaN cell drops out of both sums):

    uniform   dpain_hat(t) = sum_c beta_c dz_c(t) / sum_c beta_c^2
    ivw       the same with each term divided by var_resid_c

the least-squares inversion of dz_c = beta_c * dpain. The adjusted prediction
(intercept, pain_1_within and gap_h removed first) is out for now: the fit did not
export its intercept. Output lands in
`analysis/pain/pain_change/prediction/<run_name>_<timestamp>/` with the figures of
`plot_pain_change_prediction`.

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
from ieeg_ehr.features.build_pain_change_trajectory import cell_column

logger = logging.getLogger(__name__)

SCRIPT = 'ieeg_ehr/analysis/pain_change_prediction.py'
RUN_NAME = 'painpred'
WEIGHT_COLUMNS = ['roi', 'band', 'beta', 'var_resid', 'n_subjects', 'n_pairs']
PREDICTIONS = ('uniform', 'ivw')

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

def export_weights(source_run):
    """The converged cells of a pain_change run, in the weights schema."""
    cells = io.read_table(Path(source_run) / 'cells.csv', on_stale='warn')
    keep = (cells['converged'].astype(bool) & np.isfinite(cells['dpain_beta'])
            & np.isfinite(cells['var_resid']))
    w = cells[keep].rename(columns={'region': 'roi', 'freq': 'band', 'dpain_beta': 'beta'})
    logger.info('%d of %d cells converged and exported', len(w), len(cells))
    return w[WEIGHT_COLUMNS].reset_index(drop=True)


# ============================================================================
# Scoring
# ============================================================================

def predict(dz_cells, pairs, weights):
    """(pair_id, t_min) x the uniform and inverse-variance predictions.

    Also `n_cells_used` and `sum_beta2` (uniform) per step. Weights with a
    `heldout_subject` column score each subject with its own fold.
    """
    subject_of = pairs.set_index('pair_id')['subject']
    parts = []
    for subject, rows in dz_cells.groupby(dz_cells['pair_id'].map(subject_of), sort=False):
        w = (weights[weights['heldout_subject'] == subject]
             if 'heldout_subject' in weights.columns else weights)
        if w.empty:
            raise ValueError(f'no weights fold holds out {subject}')
        cols = [cell_column(r, b) for r, b in zip(w['roi'], w['band'])]
        missing = sorted(set(cols) - set(rows.columns))
        if missing:
            raise ValueError(f'weights name cells absent from dz_cells: {missing}')
        x = rows[cols].to_numpy(dtype=config.CACHE_ACCUMULATE_DTYPE)
        finite = np.isfinite(x)
        beta = np.where(finite, w['beta'].to_numpy(), 0.0)
        x = np.where(finite, x, 0.0)
        inv_var = 1.0 / w['var_resid'].to_numpy()
        out = rows[['pair_id', 't_min']].copy()
        out['n_cells_used'] = finite.sum(axis=1)
        out['sum_beta2'] = (beta ** 2).sum(axis=1)
        with np.errstate(invalid='ignore', divide='ignore'):
            out['pred_uniform'] = (beta * x).sum(axis=1) / out['sum_beta2']
            out['pred_ivw'] = ((beta * x * inv_var).sum(axis=1)
                               / (beta ** 2 * inv_var).sum(axis=1))
        parts.append(out)
    return pd.concat(parts, ignore_index=True)


def load_trajectory(trajectory_dir):
    """(pairs, dz_cells) for every discovery subject in an extraction directory."""
    trajectory_dir = Path(trajectory_dir)
    discovery = {f'sub-{s}' for s in config.discovery_subjects()}
    pairs, dz = [], []
    for path in sorted(trajectory_dir.glob('pairs_sub-*.parquet')):
        sid = path.stem.split('_')[1]
        if sid not in discovery:
            logger.warning('SPLIT GATE: %s is not discovery, skipped', sid)
            continue
        pairs.append(io.read_table(path, on_stale='warn'))
        dz.append(io.read_table(trajectory_dir / f'dz_{sid}.parquet', on_stale='warn'))
    if not pairs:
        raise SystemExit(f'no discovery pairs_sub-*.parquet in {trajectory_dir}')
    return pd.concat(pairs, ignore_index=True), pd.concat(dz, ignore_index=True)


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
uniform  dpain_hat(t) = sum_c beta_c d_z_c(t) / sum_c beta_c^2
ivw      the same with each term divided by var_resid_c
```

over the converged cells the subject has. The adjusted prediction is not computed:
the fit did not export its intercept.

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


def score(trajectory_dir, weights_name, run_name=RUN_NAME):
    """Write one scoring run; returns its directory."""
    weights_path = config.pain_change_prediction_weights_path(weights_name)
    weights = io.read_table(weights_path, on_stale='warn')
    pairs, dz = load_trajectory(trajectory_dir)
    preds = predict(dz, pairs, weights)
    traj_manifest = json.loads((Path(trajectory_dir) / io.MANIFEST_NAME).read_text())
    reference_run = next(p['path'] for p in traj_manifest['parents']
                         if Path(p['path']).name == traj_manifest['params']['reference_run'])

    run_dir = config.analysis_run_dir(question=config.PAIN_CHANGE_QUESTION,
                                      output_type=config.PAIN_CHANGE_PREDICTION_OUTPUT,
                                      run_name=run_name)
    subjects = sorted(pairs['subject'].unique())
    params = {'trajectory_dir': str(trajectory_dir), 'weights': weights_name,
              'n_weight_cells': len(weights), 'bins': DPAIN_BINS}
    parents = [str(trajectory_dir), str(weights_path)]
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
    w.add_argument('--name', required=True)
    s = sub.add_parser('score', help='score an extraction with a weights file, then plot')
    s.add_argument('--trajectory-dir', required=True)
    s.add_argument('--weights', required=True, help='weights file name')
    s.add_argument('--run-name', default=RUN_NAME)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    io.warn_if_dirty()

    if args.command == 'weights':
        weights = export_weights(args.source_run)
        path = io.write_table(weights, config.pain_change_prediction_weights_path(args.name),
                              params={'source_run': str(args.source_run),
                                      'filter': 'converged, finite beta and var_resid'},
                              parents=[str(Path(args.source_run) / 'cells.csv')],
                              script=SCRIPT)
        print(path)
        return 0

    from ieeg_ehr.analysis import plot_pain_change_prediction as plots
    run_dir = score(args.trajectory_dir, args.weights, args.run_name)
    for weighting in PREDICTIONS:
        plots.render_all(run_dir, weights_name=args.weights, weighting=weighting)
    plots.render_dz(run_dir)
    print(run_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())
