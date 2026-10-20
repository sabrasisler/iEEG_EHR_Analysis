"""Does an autonomic measure explain the pain-change slopes? Two scatters over the cells.

    python -m ieeg_ehr.analysis.plot_pain_change_autonomic \\
        --run-dir <pain_change -dhr or -dhrv run> --base-run <same model without it> \\
        --main-run <the run whose heatmap marks the significant cells>

The autonomic term (`dHR` or `dHRV`) is the one in `--run-dir`'s formula. Writes
into `<run-dir>/figures/` (docs/run_tokens.md):

- `scatter_dpain-raw-vs-adj.png`: one dot per cell, x = `d_pain` beta of
  `--base-run`, y = `d_pain` beta of `--run-dir` (adjusted for the term), with
  the identity line. Dots on the diagonal mean the term explains none of the slope.
- `scatter_dpain-vs-<term>.png`: the same cells, x = `d_pain` beta and y = the
  term's beta, both of `--run-dir`. Cells spread along x with a near-zero
  correlation mean the two effects dissociate. It shows dissociation only, not
  that the term fails to explain the pain slope.
- `summary_dpain-<term>.png`: both scatters as two panels, every band in one
  panel, coloured delta (light green) to high gamma (dark green).
- `pairs_dpain-<term>.png`: no neural data. A: every pair of `--run-dir`'s
  `pair_index.csv`, the term's change against the change in pain, with the mean
  per pain change and the within-subject correlation. B: each subject's own slope
  of the term's change on the pain change, with its 95% CI.

The scatters take one panel per band, shared axes. Cells outlined are BH-significant for `d_pain`
in `--main-run` (`dpain_bh_reject`), the cells of the main heatmap. Both fits
must come from one `autonomics/` folder, so they share rows. The per-cell table
and the attenuation numbers go to `<run-dir>/autonomic_compare.csv`.
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ieeg_ehr import io
from ieeg_ehr.analysis.pain_change import (AUTONOMIC_TERMS, FIGURES_SUBDIR, TERM_PREFIX,
                                           model_terms)

logger = logging.getLogger(__name__)

SCRIPT = 'ieeg_ehr/analysis/plot_pain_change_autonomic.py'
DOT = '#2a78d6'
INK = '#1a1a19'
MUTED = '#6b6a64'
GRID = '#e4e3dd'
SURFACE = '#fcfcfb'
#: Autonomic term -> (name in titles, axis label of its slope, short name, unit).
TERM_LABELS = {'dHR': ('heart-rate change', 'Heart-rate slope (Δz per bpm)', 'HR', 'bpm'),
               'dHRV': ('HRV change', 'HRV slope (Δz per Δ ln RMSSD)', 'HRV', 'ln RMSSD')}
#: A subject's own slope in `pairs_dpain-<term>.png` needs this many pairs.
MIN_SUBJECT_PAIRS = 5
#: Pain changes with fewer pairs get no mean in panel A.
MIN_BIN_PAIRS = 10
BAND_LABELS = {'delta': 'Delta', 'theta': 'Theta', 'alpha': 'Alpha', 'beta': 'Beta',
               'gamma': 'Gamma', 'high_gamma': 'High gamma'}
#: The summary figure's band ramp: `BAND_CMAP` sampled over BAND_CMAP_RANGE, delta at
#: the light end. The light end stays above the near-white steps so delta shows on white.
BAND_CMAP = 'Greens'
BAND_CMAP_RANGE = (0.35, 1.0)
BAND_SYMBOLS = {'delta': 'δ', 'theta': 'θ', 'alpha': 'α', 'beta': 'β', 'gamma': 'γ',
                'high_gamma': 'hγ'}


def compare_cells(base, adjusted, main, prefix):
    """One row per cell: raw and adjusted d_pain beta, the autonomic beta (`prefix`, renamed
    `auto_*`), main-heatmap significance."""
    keep = ['region', 'freq', 'dpain_beta', 'dpain_se']
    auto = adjusted[keep + [f'{prefix}_beta', f'{prefix}_se', f'{prefix}_bh_reject']].rename(
        columns={f'{prefix}_{c}': f'auto_{c}' for c in ('beta', 'se', 'bh_reject')})
    df = base[keep].merge(auto, on=['region', 'freq'], suffixes=('_raw', '_adj'))
    df = df.merge(main[['region', 'freq', 'dpain_bh_reject']].rename(
        columns={'dpain_bh_reject': 'main_significant'}), on=['region', 'freq'], how='left')
    df['main_significant'] = df['main_significant'].fillna(False).astype(bool)
    df['shift_in_se'] = (df['dpain_beta_adj'] - df['dpain_beta_raw']) / df['dpain_se_raw']
    df['pct_change'] = 100 * (df['dpain_beta_adj'] - df['dpain_beta_raw']) / df['dpain_beta_raw']
    return df


def summary(df):
    """The attenuation and dissociation numbers the figures print."""
    raw, adj = df['dpain_beta_raw'].to_numpy(), df['dpain_beta_adj'].to_numpy()
    sig = df['main_significant']
    return {
        'n_cells': len(df), 'n_main_significant': int(sig.sum()),
        # A percent change is unstable where the raw slope is near zero, so the
        # median is over the cells the main heatmap marks.
        'median_pct_change_main_significant': float(df.loc[sig, 'pct_change'].median()),
        # Least squares through the origin over every cell: 1 = no attenuation.
        'slope_adj_on_raw': float((raw * adj).sum() / (raw ** 2).sum()),
        'median_abs_shift_in_se': float(df['shift_in_se'].abs().median()),
        'r_dpain_auto': float(np.corrcoef(df['dpain_beta_adj'], df['auto_beta'])[0, 1]),
    }


def facet_grid(df, bands, x, y, xlabel, ylabel, title, subtitle, out, identity):
    lo = min(df[x].min(), df[y].min()) if identity else None
    hi = max(df[x].max(), df[y].max()) if identity else None
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.4), sharex=True, sharey=True)
    for ax, band in zip(axes.flat, bands):
        cells = df[df['freq'] == band]
        sig = cells['main_significant']
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.axhline(0, color=GRID, linewidth=1.2, zorder=1)
        ax.axvline(0, color=GRID, linewidth=1.2, zorder=1)
        if identity:
            pad = 0.05 * (hi - lo)
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=MUTED,
                    linewidth=1.2, linestyle='--', zorder=2)
        ax.scatter(cells.loc[~sig, x], cells.loc[~sig, y], s=40, color=DOT, alpha=0.75,
                   edgecolors=SURFACE, linewidths=1.5, zorder=3)
        ax.scatter(cells.loc[sig, x], cells.loc[sig, y], s=56, color=DOT,
                   edgecolors=INK, linewidths=1.6, zorder=4)
        name = BAND_LABELS.get(band, band)
        r = np.corrcoef(cells[x], cells[y])[0, 1] if len(cells) > 2 else np.nan
        ax.set_title(f'{name}   r = {r:.2f}' if not identity else name,
                     fontsize=11, color=INK, loc='left')
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=9)
    for ax in axes[1]:
        ax.set_xlabel(xlabel, color=INK, fontsize=10)
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel, color=INK, fontsize=10)
    handles = [plt.Line2D([], [], marker='o', linestyle='', markersize=7, color=DOT,
                          markeredgecolor=SURFACE, label='cell'),
               plt.Line2D([], [], marker='o', linestyle='', markersize=8, color=DOT,
                          markeredgecolor=INK, markeredgewidth=1.6,
                          label='BH-significant in the main d_pain heatmap')]
    if identity:
        handles.append(plt.Line2D([], [], color=MUTED, linestyle='--', label='identity'))
    fig.legend(handles=handles, loc='lower center', ncol=len(handles), frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(title, x=0.01, ha='left', fontsize=13, color=INK)
    fig.text(0.01, 0.935, subtitle, ha='left', fontsize=9.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor='white')
    plt.close(fig)


def summary_figure(df, bands, stats, term, out):
    """Panel A: raw vs adjusted pain slope, identity line. Panel B: pain vs autonomic slope.
    Every band in one panel, delta to high gamma light to dark green; main-heatmap cells
    outlined."""
    _, auto_label, short, _ = TERM_LABELS[term]
    colours = dict(zip(bands, plt.get_cmap(BAND_CMAP)(np.linspace(*BAND_CMAP_RANGE,
                                                                   len(bands)))))
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(9.4, 5.2))

    def dots(ax, x, y):
        for sig in (False, True):
            for band in bands:
                c = df[(df['freq'] == band) & (df['main_significant'] == sig)]
                if sig:
                    ax.scatter(c[x], c[y], s=60, color=colours[band], edgecolors='black',
                               linewidths=1.1, zorder=4)
                else:
                    ax.scatter(c[x], c[y], s=24, color=colours[band], alpha=0.6,
                               edgecolors='none', zorder=3)
        ax.axhline(0, color=GRID, linewidth=1, zorder=1)
        ax.axvline(0, color=GRID, linewidth=1, zorder=1)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)

    dots(ax_a, 'dpain_beta_raw', 'dpain_beta_adj')
    lim = 1.08 * np.abs(df[['dpain_beta_raw', 'dpain_beta_adj']].to_numpy()).max()
    ax_a.plot([-lim, lim], [-lim, lim], color=MUTED, linestyle='--', linewidth=1.3, zorder=2)
    ax_a.set(xlim=(-lim, lim), ylim=(-lim, lim), aspect='equal')
    ax_a.set_xlabel('Pain slope, original model')
    ax_a.set_ylabel(f'Pain slope, {short}-adjusted')
    ax_a.set_title(f'A  Pain slope after {short} adjustment', loc='left', fontweight='bold')
    ax_a.text(0.04, 0.96, f"slope = {stats['slope_adj_on_raw']:.2f}\n"
              f"sig. cells: {stats['median_pct_change_main_significant']:+.0f}% median",
              transform=ax_a.transAxes, va='top', fontsize=11)

    dots(ax_b, 'dpain_beta_adj', 'auto_beta')
    ax_b.set_xlabel('Pain slope (Δz / pain point)')
    ax_b.set_ylabel(auto_label.replace(' per ', ' / '))
    ax_b.set_title(f'B  Pain vs {short} slopes', loc='left', fontweight='bold')
    ax_b.text(0.04, 0.96, f"r = {stats['r_dpain_auto']:.2f}", transform=ax_b.transAxes,
              va='top', fontsize=11)

    handles = [plt.Line2D([], [], marker='o', linestyle='', markersize=8, color=colours[b],
                          label=BAND_SYMBOLS.get(b, b)) for b in bands]
    handles.append(plt.Line2D([], [], marker='o', linestyle='', markersize=8, color='white',
                              markeredgecolor='black', label='significant'))
    fig.legend(handles=handles, loc='lower center', ncol=len(handles), frameon=False,
               fontsize=11, handletextpad=0.3, columnspacing=1.4)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor='white')
    plt.close(fig)


def subject_slopes(pairs, term):
    """Per subject with MIN_SUBJECT_PAIRS pairs and varying pain: OLS slope of `term` on
    d_pain, its SE and the pair count."""
    rows = []
    for subject, g in pairs.groupby('subject_id'):
        x, y = g['d_pain'].to_numpy(float), g[term].to_numpy(float)
        if len(g) < MIN_SUBJECT_PAIRS or x.std() == 0:
            continue
        xc = x - x.mean()
        slope = (xc * (y - y.mean())).sum() / (xc ** 2).sum()
        resid = y - y.mean() - slope * xc
        se = np.sqrt((resid ** 2).sum() / (len(g) - 2) / (xc ** 2).sum())
        rows.append({'subject_id': subject, 'slope': slope, 'se': se, 'n_pairs': len(g)})
    return pd.DataFrame(rows).sort_values('slope').reset_index(drop=True)


def pair_figure(pairs, term, out):
    """Panel A: the term's change against the pain change, one dot per pair. Panel B: each
    subject's own slope. Returns the numbers it prints."""
    _, _, short, unit = TERM_LABELS[term]
    green = plt.get_cmap(BAND_CMAP)
    light, dark = green(0.55), green(0.95)
    within = pairs.assign(**{c: pairs[c] - pairs.groupby('subject_id')[c].transform('mean')
                             for c in ('d_pain', term)})
    slopes = subject_slopes(pairs, term)
    stats = {'n_pairs': len(pairs), 'n_subjects': int(pairs['subject_id'].nunique()),
             'r_within_subject': float(within['d_pain'].corr(within[term])),
             'n_subjects_with_slope': len(slopes),
             'median_subject_slope': float(slopes['slope'].median()),
             'n_subject_slopes_positive': int((slopes['slope'] > 0).sum()),
             'n_subject_cis_excluding_zero': int(
                 (np.abs(slopes['slope']) > 1.96 * slopes['se']).sum())}

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10.4, 4.9),
                                     gridspec_kw={'width_ratios': [1.15, 1]})
    rng = np.random.default_rng(0)
    ax_a.scatter(pairs['d_pain'] + rng.uniform(-0.25, 0.25, len(pairs)), pairs[term], s=12,
                 color=light, alpha=0.45, edgecolors='none', zorder=2)
    bins = pairs.groupby('d_pain')[term].agg(['mean', 'sem', 'size'])
    bins = bins[bins['size'] >= MIN_BIN_PAIRS]
    ax_a.errorbar(bins.index, bins['mean'], yerr=1.96 * bins['sem'], fmt='o', color=dark,
                  markersize=6, capsize=0, linewidth=1.6, zorder=4)
    ax_a.axhline(0, color=GRID, linewidth=1, zorder=1)
    ax_a.axvline(0, color=GRID, linewidth=1, zorder=1)
    ax_a.set_xticks(np.arange(-10, 11, 2))
    ax_a.set_xlabel('Δ pain (NRS points)')
    ax_a.set_ylabel(f'Δ {short} ({unit})')
    ax_a.set_title(f'A  {short} change against pain change', loc='left', fontweight='bold')
    ax_a.text(0.03, 0.97, f"within-subject r = {stats['r_within_subject']:.2f}\n"
              f"{stats['n_pairs']} pairs, {stats['n_subjects']} subjects\n"
              f"dark: mean ± 95% CI (≥ {MIN_BIN_PAIRS} pairs)",
              transform=ax_a.transAxes, va='top', fontsize=10,
              bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.85, 'pad': 2}, zorder=5)

    y = np.arange(len(slopes))
    ax_b.errorbar(slopes['slope'], y, xerr=1.96 * slopes['se'], fmt='o', color=dark,
                  ecolor=light, markersize=4, linewidth=1.4, capsize=0, zorder=3)
    ax_b.axvline(0, color=MUTED, linewidth=1, zorder=1)
    ax_b.axvline(stats['median_subject_slope'], color=dark, linestyle='--', linewidth=1.2,
                 zorder=2)
    # The axis fits the slopes: a subject with little pain variance has a CI wide enough
    # to flatten every other one, so long CIs run off the edge.
    reach = 1.35 * np.abs(slopes['slope']).max()
    ax_b.set_xlim(-reach, reach)
    ax_b.set_yticks([])
    ax_b.set_ylabel('Subjects, sorted by slope')
    ax_b.set_xlabel(f'Δ {short} per pain point ({unit})')
    ax_b.set_title(f'B  Each subject\'s {short}-pain slope', loc='left', fontweight='bold')
    ax_b.text(0.97, 0.03, f"median {stats['median_subject_slope']:+.2f} (dashed)\n"
              f"{stats['n_subject_slopes_positive']} of {stats['n_subjects_with_slope']} positive\n"
              f"{stats['n_subject_cis_excluding_zero']} CIs exclude 0",
              transform=ax_b.transAxes, va='bottom', ha='right', fontsize=10,
              bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.85, 'pad': 2}, zorder=5)
    for ax in (ax_a, ax_b):
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor='white')
    plt.close(fig)
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-dir', required=True, help='the pain_change run with dHR or dHRV')
    ap.add_argument('--base-run', required=True, help='the same model without it')
    ap.add_argument('--main-run', required=True,
                    help='the run whose d_pain heatmap marks the significant cells')
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    run_dir, base_dir, main_dir = Path(args.run_dir), Path(args.base_run), Path(args.main_run)
    params = json.loads((run_dir / 'provenance.json').read_text())['params']
    base_params = json.loads((base_dir / 'provenance.json').read_text())['params']
    if run_dir.parent != base_dir.parent or not params.get('autonomic'):
        raise SystemExit('--run-dir and --base-run must sit in one autonomics/ folder, '
                         'so both fit the same rows')
    (term,) = [t for t in model_terms(params['model']) if t in AUTONOMIC_TERMS]
    if set(model_terms(base_params['model'])) & set(AUTONOMIC_TERMS):
        raise SystemExit(f'--base-run has an autonomic term: {base_params["formula"]}')
    prefix = TERM_PREFIX[term]
    name, auto_label, _, _ = TERM_LABELS[term]

    pairs = {d: pd.read_csv(d / 'pair_index.csv', usecols=['subject_id', 't1', 't2'])
             for d in (run_dir, base_dir)}
    keys = {d: set(map(tuple, p.to_numpy())) for d, p in pairs.items()}
    n_diff = len(keys[run_dir] ^ keys[base_dir])
    logger.info('pairs: %d with dHR, %d without, %d in only one', len(keys[run_dir]),
                len(keys[base_dir]), n_diff)

    read = lambda d: io.read_table(d / 'cells.csv', on_stale='warn')
    df = compare_cells(read(base_dir), read(run_dir), read(main_dir), prefix)
    stats = {'term': term, **summary(df), 'n_pairs_adjusted': len(keys[run_dir]),
             'n_pairs_base': len(keys[base_dir]), 'n_pairs_in_one_only': n_diff}
    for k, v in stats.items():
        logger.info('%s: %s', k, f'{v:.4g}' if isinstance(v, float) else v)

    bands = list(params['bands'])
    figures = run_dir / FIGURES_SUBDIR
    parents = [str(base_dir / 'cells.csv'), str(run_dir / 'cells.csv'),
               str(main_dir / 'cells.csv')]
    io.write_table(df, run_dir / 'autonomic_compare.csv', params=vars(args),
                   parents=parents, script=SCRIPT, extra={'summary': stats})

    raw_adj = figures / 'scatter_dpain-raw-vs-adj.png'
    facet_grid(df, bands, 'dpain_beta_raw', 'dpain_beta_adj',
               'Pain slope, original model (Δz per pain point)',
               f'Pain slope, adjusted for {term}',
               f'Pain slopes before and after adjusting for {name}',
               f"{stats['n_cells']} region x band cells. Slope of adjusted on original "
               f"{stats['slope_adj_on_raw']:.2f} (1 = no attenuation). Median change in the "
               f"{stats['n_main_significant']} main-heatmap cells "
               f"{stats['median_pct_change_main_significant']:+.1f}%. Median shift "
               f"{stats['median_abs_shift_in_se']:.2f} SE.",
               raw_adj, identity=True)
    pain_auto = figures / f'scatter_dpain-vs-{prefix}.png'
    facet_grid(df, bands, 'dpain_beta_adj', 'auto_beta',
               'Pain slope (Δz per pain point)', auto_label,
               f'Pain and {name} effects across cells',
               f"Both from the {term} model. Correlation across all {stats['n_cells']} "
               f"cells r = {stats['r_dpain_auto']:.2f}. Dissociation only: this does not "
               f'test whether {name} explains the pain slope.',
               pain_auto, identity=False)
    combined = figures / f'summary_dpain-{prefix}.png'
    summary_figure(df, bands, stats, term, combined)
    for out in (raw_adj, pain_auto, combined):
        io.write_sidecar(out, kind='figure', script=SCRIPT, params=vars(args),
                         parents=parents, extra={'summary': stats})
    pairs_out = figures / f'pairs_dpain-{prefix}.png'
    pair_stats = pair_figure(
        pd.read_csv(run_dir / 'pair_index.csv', usecols=['subject_id', 'd_pain', term]),
        term, pairs_out)
    logger.info('pairs: %s', pair_stats)
    io.write_sidecar(pairs_out, kind='figure', script=SCRIPT, params=vars(args),
                     parents=[str(run_dir / 'pair_index.csv')], extra={'summary': pair_stats})
    io.log_analysis(f'pain_change autonomic scatters (raw vs {term}-adjusted d_pain, '
                    f'd_pain vs {term})', figures)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
