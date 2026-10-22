"""Per-subject tracking: does each patient's predicted change in pain follow their reports?

    python -m ieeg_ehr.analysis.pain_change_tracking --run-dir <scoring run> \\
        --prediction raw --weighting uniform

Reads one `pain_change_prediction score` run, which must use leave-one-subject-out
weights, and writes nothing it reads. Per pair, the predicted change at t2 is the
mean prediction over the steps within DOSE_WINDOW_MIN of t2, the dose-response
definition (`plot_pain_change_prediction.near_t2`).

A subject enters with at least `--min-pairs` pairs and `--min-levels` distinct
delta_pain values. Per subject, across its pairs: Spearman rho of predicted on
actual delta pain (primary), Pearson r, the OLS slope, n_pairs, the delta_pain
range, and the mean n_cells_used and sum_beta2 of the same steps. MDD comes from
`dx_state.subject_labels` (90-day ICD window, a billing artefact, see there).

NULL. Per subject, delta_pain is shuffled across its pairs `--n-perm` times and
rho recomputed. Each subject's band (the middle of its own null) and permutation
percentile are stored, the forest plot draws every subject against its own band,
and the histogram draws the pooled null. All of it is descriptive: consecutive
pairs share an assessment and so an epoch, so a per-subject percentile is not an
independent test, and no group p-value is computed (CLAUDE.md, Sweeps). The null is rebuilt
from `--seed`, not stored.

Output: `<run>/tracking/<prediction>_<weighting>_<timestamp>/`
`per_subject_tracking.csv`, `figures/*.png`, SUMMARY.md, provenance.json.

EXPLORATORY. Discovery cohort only. Nominations, not findings.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from scipy import stats  # noqa: E402

from ieeg_ehr import config, io  # noqa: E402
from ieeg_ehr.analysis import dx_state  # noqa: E402
from ieeg_ehr.analysis import plot_pain_change_prediction as pplot  # noqa: E402
from ieeg_ehr.analysis.pain_change import FIGURES_SUBDIR  # noqa: E402
from ieeg_ehr.config.paths import _run_stamp  # noqa: E402

logger = logging.getLogger(__name__)

SCRIPT = 'ieeg_ehr/analysis/pain_change_tracking.py'
SUBDIR = 'tracking'
RHO_BINS = np.linspace(-1, 1, 21)
#: Percentiles of each subject's own null drawn as its band in the forest plot.
NULL_BAND = (2.5, 97.5)
#: Categorical slots: above the band, below it. Inside stays neutral.
ABOVE, BELOW = '#2a78d6', '#eb6834'
#: Sequential blue ramp (steps 250 -> 650) for sum_beta2.
BETA2_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    'sum_beta2', ['#86b6ef', '#2a78d6', '#104281'])
#: Categorical slots 1 and 2 for the MDD split.
MDD_COLOURS = {False: '#2a78d6', True: '#eb6834'}


# ============================================================================
# Metrics
# ============================================================================

def _ranks(a):
    return stats.rankdata(a, axis=-1)


def _row_corr(x, y):
    """Pearson r of x against each row of y."""
    xc = x - x.mean()
    yc = y - y.mean(axis=-1, keepdims=True)
    return (yc @ xc) / np.sqrt((yc ** 2).sum(axis=-1) * (xc ** 2).sum())


def subject_metrics(pp, n_perm, rng):
    """One row of tracking metrics and the permutation null of rho, for one subject."""
    pred, actual = pp['value'].to_numpy(), pp['delta_pain'].to_numpy(dtype=float)
    rho = stats.spearmanr(pred, actual).statistic
    # Spearman is Pearson on ranks. Shuffling the ranks of delta_pain keeps its ties.
    rank_pred, rank_actual = _ranks(pred), _ranks(actual)
    null = _row_corr(rank_pred, rng.permuted(np.tile(rank_actual, (n_perm, 1)), axis=1))
    row = {'n_pairs': len(pp), 'n_levels': int(pp['delta_pain'].nunique()),
           'delta_min': int(actual.min()), 'delta_max': int(actual.max()),
           'spearman_rho': rho, 'pearson_r': stats.pearsonr(pred, actual).statistic,
           'slope': np.polyfit(actual, pred, 1)[0],
           'perm_percentile': 100 * float(np.mean(null < rho) + 0.5 * np.mean(null == rho)),
           'null_median': float(np.median(null)),
           'null_lo': float(np.percentile(null, NULL_BAND[0])),
           'null_hi': float(np.percentile(null, NULL_BAND[1])),
           'mean_sum_beta2': pp['sum_beta2'].mean(),
           'mean_n_cells_used': pp['n_cells_used'].mean()}
    return row, null


def mdd_labels(subjects):
    """{subject_id: bool}, the dx_state default label. Unknown subjects are absent."""
    bare = [s.replace('sub-', '') for s in subjects]
    labels, _ = dx_state.subject_labels(dx_state.load_diagnoses(subjects=bare),
                                        condition='mdd')
    return labels.set_index('subject_id')['dx_state'].astype(bool).to_dict()


def tracking_table(df, min_pairs, min_levels, n_perm, seed):
    """(per-subject table, pooled null, excluded {subject: reason})."""
    per_pair = pplot.near_t2(df, columns=('value', 'sum_beta2', 'n_cells_used'))
    per_pair = per_pair.dropna(subset=['value'])
    rng = np.random.default_rng(seed)
    rows, nulls, excluded = [], [], {}
    for subject, pp in per_pair.groupby('subject', sort=True):
        n_levels = pp['delta_pain'].nunique()
        if len(pp) < min_pairs or n_levels < min_levels:
            excluded[subject] = f'{len(pp)} pairs and {n_levels} distinct values'
            continue
        row, null = subject_metrics(pp, n_perm, rng)
        rows.append({'subject': subject, **row})
        nulls.append(null)
    for subject, reason in excluded.items():
        logger.info('%s excluded: %s (min %d pairs, %d values)', subject, reason,
                    min_pairs, min_levels)
    table = pd.DataFrame(rows)
    mdd = mdd_labels(table['subject'])
    table['mdd'] = table['subject'].map(mdd)
    if table['mdd'].isna().any():
        logger.warning('MDD unknown: %s', table.loc[table['mdd'].isna(), 'subject'].tolist())
    return table, np.concatenate(nulls), excluded


# ============================================================================
# Figures
# ============================================================================

def _rho_axes(ax, table, null):
    """Histogram of rho with the pooled null scaled to the same count, lines at 0 and median."""
    pplot._style(ax)
    ax.hist(table['spearman_rho'], bins=RHO_BINS, color=pplot.NEUTRAL, alpha=0.85,
            edgecolor='white', linewidth=1, label='subjects')
    weights = np.full(len(null), len(table) / len(null))
    ax.hist(null, bins=RHO_BINS, weights=weights, histtype='step', color=pplot.MUTED,
            linewidth=1.2, label='pooled permutation null')
    pplot._zero(ax, 'x')
    ax.axvline(table['spearman_rho'].median(), color=pplot.INK, lw=1.2, ls='--',
               label='median rho')
    ax.set_xlim(-1, 1)
    ax.set_ylabel('N subjects', color=pplot.INK)


def _annotation(table, null):
    return (f'{len(table)} subjects. rho > 0: {100 * (table["spearman_rho"] > 0).mean():.0f}%. '
            f'Median rho {table["spearman_rho"].median():.2f}, null median {np.median(null):.2f}.')


def band_counts(table):
    """(n above their own null band, n below)."""
    return (int((table['spearman_rho'] > table['null_hi']).sum()),
            int((table['spearman_rho'] < table['null_lo']).sum()))


def fig_rho_forest(table, out_dir, title):
    """One row per subject: its own null band and its observed rho, sorted by rho."""
    t = table.sort_values('spearman_rho').reset_index(drop=True)
    above = t['spearman_rho'] > t['null_hi']
    below = t['spearman_rho'] < t['null_lo']
    colour = np.where(above, ABOVE, np.where(below, BELOW, pplot.NEUTRAL))
    y = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(6.4, 0.6 + 0.19 * len(t)))
    pplot._style(ax)
    ax.hlines(y, t['null_lo'], t['null_hi'], color=pplot.GRID, lw=4.5,
              label=f'own null, middle {NULL_BAND[1] - NULL_BAND[0]:g}%')
    pplot._zero(ax, 'x')
    ax.scatter(t['spearman_rho'], y, s=26, c=colour, edgecolor='white', linewidth=0.5,
               zorder=3)
    for c, lab in ((ABOVE, 'above own null'), (pplot.NEUTRAL, 'inside'),
                   (BELOW, 'below own null')):
        ax.scatter([], [], s=26, c=c, label=lab)
    ax.set_yticks(y)
    ax.set_yticklabels([f'{s} ({n})' for s, n in zip(t['subject'], t['n_pairs'])],
                       fontsize=6.5)
    ax.set_ylim(-1, len(t))
    ax.set_xlim(-1, 1)
    ax.set_xlabel('Spearman rho, predicted vs actual Δpain at t2', color=pplot.INK)
    pplot._title(ax, title)
    pplot._legend(ax, loc='upper left', bbox_to_anchor=(1.01, 1))
    n_above, n_below = band_counts(t)
    tail = (100 - NULL_BAND[1]) / 100
    pplot._footer(fig, f'{len(t)} subjects (n pairs in brackets). Above own null: {n_above}, '
                       f'below: {n_below}. Chance expects about {tail * len(t):.1f} each. '
                       'Descriptive: pairs share assessments.')
    pplot._save(fig, out_dir, 'tracking_rho_forest')


def fig_rho(table, null, out_dir, title):
    fig, ax = plt.subplots(figsize=(6, 4))
    _rho_axes(ax, table, null)
    ax.set_xlabel('Spearman rho, predicted vs actual Δpain at t2', color=pplot.INK)
    pplot._title(ax, title)
    pplot._legend(ax, loc='upper left')
    pplot._footer(fig, _annotation(table, null))
    pplot._save(fig, out_dir, 'tracking_rho_hist')


def _strip(ax, table, colour, label):
    """One dot per subject at its rho, under the histogram."""
    rng = np.random.default_rng(0)
    pplot._style(ax)
    ax.scatter(table['spearman_rho'], rng.uniform(-0.3, 0.3, len(table)), s=22, c=colour,
               edgecolor='white', linewidth=0.5)
    ax.set_yticks([])
    ax.set_ylim(-0.6, 0.6)
    ax.set_xlim(-1, 1)
    ax.set_xlabel('Spearman rho, predicted vs actual Δpain at t2', color=pplot.INK)
    ax.set_ylabel(label, color=pplot.INK, rotation=0, ha='right', va='center')


def fig_rho_by_beta2(table, null, out_dir, title):
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(6.4, 5), sharex=True,
                                      gridspec_kw={'height_ratios': [3, 1]})
    _rho_axes(top, table, null)
    pplot._title(top, title)
    pplot._legend(top, loc='upper left')
    norm = Normalize(table['mean_sum_beta2'].min(), table['mean_sum_beta2'].max())
    _strip(bottom, table, BETA2_CMAP(norm(table['mean_sum_beta2'])), 'subjects')
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=BETA2_CMAP),
                      ax=[top, bottom], pad=0.02)
    cb.set_label('mean sum beta^2', color=pplot.INK)
    cb.outline.set_visible(False)
    r = stats.spearmanr(table['mean_sum_beta2'], table['spearman_rho']).statistic
    pplot._footer(fig, _annotation(table, null)
                  + f' Spearman(sum beta^2, rho) across subjects: {r:.2f}.')
    pplot._save(fig, out_dir, 'tracking_rho_by_sum_beta2')


def fig_rho_by_mdd(table, null, out_dir, title):
    known = table.dropna(subset=['mdd'])
    fig, ax = plt.subplots(figsize=(6, 4))
    pplot._style(ax)
    groups = [known.loc[known['mdd'] == m, 'spearman_rho'] for m in (False, True)]
    ax.hist(groups, bins=RHO_BINS, stacked=True, color=[MDD_COLOURS[False], MDD_COLOURS[True]],
            edgecolor='white', linewidth=1,
            label=[f'no MDD code (n={len(groups[0])}, median {groups[0].median():.2f})',
                   f'MDD code (n={len(groups[1])}, median {groups[1].median():.2f})'])
    weights = np.full(len(null), len(known) / len(null))
    ax.hist(null, bins=RHO_BINS, weights=weights, histtype='step', color=pplot.MUTED,
            linewidth=1.2, label='pooled permutation null')
    pplot._zero(ax, 'x')
    ax.set_xlim(-1, 1)
    ax.set_xlabel('Spearman rho, predicted vs actual Δpain at t2', color=pplot.INK)
    ax.set_ylabel('N subjects', color=pplot.INK)
    pplot._title(ax, title)
    pplot._legend(ax, loc='upper left')
    pplot._footer(fig, f'MDD: an ICD code within {dx_state.DEFAULT_WINDOW_DAYS} days before '
                       'admission, a billing artefact (dx_state). '
                       f'{len(table) - len(known)} subjects unknown.')
    pplot._save(fig, out_dir, 'tracking_rho_by_mdd')


def slope_panel(ax, table):
    """Histogram of the per-subject OLS slopes of predicted on actual Δpain, median inlaid."""
    pplot._style(ax)
    lo, hi = np.nanpercentile(table['slope'], [0, 100])
    ax.hist(table['slope'], bins=np.linspace(min(lo, -1), max(hi, 2), 25),
            color=pplot.NEUTRAL, alpha=0.85, edgecolor='white', linewidth=1)
    pplot._zero(ax, 'x')
    ax.text(0.97, 0.97, f'Median slope {table["slope"].median():.2f}', transform=ax.transAxes,
            ha='right', va='top', fontsize=9, color=pplot.INK)
    ax.set_xlabel('Predicted Δpain per point of true Δpain', color=pplot.INK)
    ax.set_ylabel('Patients', color=pplot.INK)


def fig_slope(table, out_dir, title):
    fig, ax = plt.subplots(figsize=(6, 4))
    slope_panel(ax, table)
    pplot._title(ax, title)
    pplot._footer(fig, f'{len(table)} subjects.')
    pplot._save(fig, out_dir, 'tracking_slope_hist')


# ============================================================================
# CLI
# ============================================================================

def write_summary(run_dir, table, null, excluded, params):
    r = stats.spearmanr(table['mean_sum_beta2'], table['spearman_rho']).statistic
    known = table.dropna(subset=['mdd'])
    text = f"""# Per-subject tracking

EXPLORATORY, descriptive. Scoring run `{params['run_dir']}`, weights
`{params['weights']}`, prediction `{params['prediction']}`, weighting
`{params['weighting']}`.

- Included: {len(table)} subjects (at least {params['min_pairs']} pairs and
  {params['min_levels']} distinct delta_pain values). Excluded: {len(excluded)}
  ({', '.join(f'{s} with {why}' for s, why in excluded.items()) or 'none'}).
- Against each subject's own null (middle {NULL_BAND[1] - NULL_BAND[0]:g}%): {band_counts(table)[0]} above,
  {band_counts(table)[1]} below, about {(100 - NULL_BAND[1]) / 100 * len(table):.1f} each expected by chance.
- rho > 0 in {100 * (table['spearman_rho'] > 0).mean():.0f}% of subjects. Median rho
  {table['spearman_rho'].median():.2f} against a pooled null median of {np.median(null):.2f}
  ({params['n_perm']} within-subject shuffles each, seed {params['seed']}).
- Median Pearson r {table['pearson_r'].median():.2f}, median slope {table['slope'].median():.2f}.
- Mean sum beta^2 vs rho across subjects: Spearman {r:.2f}.
- MDD code: median rho {known.loc[known['mdd'].astype(bool), 'spearman_rho'].median():.2f}
  (n={int(known['mdd'].astype(bool).sum())}) vs {known.loc[~known['mdd'].astype(bool), 'spearman_rho'].median():.2f}
  (n={int((~known['mdd'].astype(bool)).sum())}).

Permutation percentiles are descriptive. Consecutive pairs share an assessment, so
they are not independent tests, and no group p-value is computed.
"""
    (run_dir / 'SUMMARY.md').write_text(text)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-dir', required=True, help='a pain_change_prediction score run')
    ap.add_argument('--prediction', choices=['raw', 'adj'], default='raw')
    ap.add_argument('--weighting', choices=['uniform', 'ivw'], default='uniform')
    ap.add_argument('--min-pairs', type=int, default=10)
    ap.add_argument('--min-levels', type=int, default=3)
    ap.add_argument('--n-perm', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    io.warn_if_dirty()

    source = Path(args.run_dir)
    weights_name = json.loads((source / 'provenance.json').read_text())['params']['weights']
    weights_path = config.pain_change_prediction_weights_path(weights_name)
    if 'heldout_subject' not in pd.read_csv(weights_path, nrows=0).columns:
        raise SystemExit(f'{weights_name} is not leave-one-subject-out: this analysis '
                         'runs out of sample only')
    df = pplot.prediction_frame(source, weights_name, args.prediction, args.weighting)
    table, null, excluded = tracking_table(df, args.min_pairs, args.min_levels,
                                           args.n_perm, args.seed)

    run_dir = (source / SUBDIR
               / f'{args.prediction}_{args.weighting}_{_run_stamp()}')
    out_dir = run_dir / FIGURES_SUBDIR
    params = {'run_dir': str(source), 'weights': weights_name,
              'prediction': args.prediction, 'weighting': args.weighting,
              'min_pairs': args.min_pairs, 'min_levels': args.min_levels,
              'n_perm': args.n_perm, 'seed': args.seed,
              't2_window_min': pplot.DOSE_WINDOW_MIN,
              'mdd_window_days': dx_state.DEFAULT_WINDOW_DAYS}
    subjects = sorted(table['subject'])
    io.write_table(table, run_dir / 'per_subject_tracking.csv', params=params,
                   parents=[str(source / f'predictions_{weights_name}.csv')],
                   subjects=subjects, script=SCRIPT)
    title = f'Per-subject tracking, {args.prediction}, {args.weighting}, {weights_name}'
    fig_rho_forest(table, out_dir, title)
    fig_rho(table, null, out_dir, title)
    fig_rho_by_beta2(table, null, out_dir, title)
    fig_rho_by_mdd(table, null, out_dir, title)
    fig_slope(table, out_dir, title)
    write_summary(run_dir, table, null, excluded, params)
    io.write_run_provenance(run_dir, script=SCRIPT, params=params, parents=[str(source)],
                            subjects=subjects,
                            extra={'status': 'EXPLORATORY, descriptive, NOT a finding',
                                   'excluded': excluded})
    io.log_analysis(f'pain_change per-subject tracking, {args.prediction} {args.weighting}, '
                    f'{len(table)} subjects', run_dir)
    print((run_dir / 'SUMMARY.md').read_text())
    print(run_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())
