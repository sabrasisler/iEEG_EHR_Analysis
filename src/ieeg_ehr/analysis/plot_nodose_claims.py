"""One figure per claim for the no-dose pain-encoding analysis.

Three claims, three figures, deliberately separate -- each is a different kind of
argument and pooling them into one panel grid would blur which evidence supports
which.

`fig_replication`  -- the headline. DOSED-only vs NO-DOSE-only per-cell betas.
    Those two subsets are DISJOINT, so agreement between them is independent
    replication. The full-run-vs-no-dose comparison is shown only as a faded
    supplement, because no-dose pairs are a SUBSET of the full run and much of
    that r = 0.956 is the same rows agreeing with themselves.

    Both halves are noisy, so r is ATTENUATED. The panel therefore carries the
    attenuation CEILING -- sqrt(reliability_A * reliability_B), estimated from
    the betas and their SEs -- because an observed r of 0.5 against a ceiling of
    0.55 is near-perfect replication, and without the ceiling the figure could be
    read as refuting the claim it exists to support.

`fig_power`  -- why BH collapsed from 62 cells to 2-3 without the effect
    changing. The dosed pairs carry the large pain swings, so removing them
    removes the predictor variance the slope needs. The zero spike is drawn as
    its OWN bar: 43% of no-dose pairs have d_pain exactly 0, which is the
    mechanism and is invisible at cell level.

`fig_scope`  -- what "no dose" does and does not mean. ECDF of time since the
    prior dose, so 58% within 1 h and 78% within 4 h are readable off an axis
    rather than asserted in prose.

    python -m ieeg_ehr.analysis.plot_nodose_claims
"""

import argparse
import glob
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ieeg_ehr import config, io
from ieeg_ehr.analysis.plot_change_score_diagnostics import (COLOR_BAD, COLOR_OK,
                                                             INK, INK_MUTED)

logger = logging.getLogger(__name__)

BASE = None          # resolved in main()
KEY = ['region', 'freq_bin_index']
COLOR_MUTED = '#9AA4B2'

DISCLAIMER = ('EXPLORATORY -- discovery cohort, NOMINATIONS NOT FINDINGS. '
              'Cells are correlated; counts of cells are not sample sizes.')


def find_run(base, pattern):
    hits = sorted(glob.glob(f'{base}/{pattern}'))
    return Path(hits[-1]) if hits else None


def converged(run_dir):
    d = io.read_table(Path(run_dir) / 'grid_cells.parquet', on_stale='warn')
    return d[d['converged'].fillna(False)].copy()


def reliability(beta, se):
    """Fraction of between-cell beta variance that is SIGNAL, not estimation noise.

    var(observed) = var(true) + mean(se^2) for independent errors, so the
    reliability is (var(observed) - mean(se^2)) / var(observed). This is what
    caps the correlation achievable between two independent noisy halves; the
    ceiling on r is sqrt(rel_A * rel_B).
    """
    beta = np.asarray(beta, dtype=float)
    se = np.asarray(se, dtype=float)
    ok = np.isfinite(beta) & np.isfinite(se)
    if ok.sum() < 3:
        return np.nan
    v = float(np.var(beta[ok], ddof=1))
    noise = float(np.mean(se[ok] ** 2))
    return float(max((v - noise) / v, 0.0)) if v > 0 else np.nan


def bh_count(p, q):
    """Number of BH rejections at level q."""
    p = np.sort(np.asarray(p, dtype=float))
    p = p[np.isfinite(p)]
    m = len(p)
    if not m:
        return 0
    k = np.arange(1, m + 1)
    below = p <= k / m * q
    return int(k[below].max()) if below.any() else 0


# ============================================================================

def fig_replication(dose, nodose, full, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    has_dose = dose is not None
    ncol = 2 if has_dose else 1
    fig, axes = plt.subplots(1, ncol, figsize=(7.2 * ncol, 6.4), squeeze=False)

    def panel(ax, a, b, la, lb, color, headline):
        j = (a[KEY + ['dpain_beta', 'dpain_se']]
             .rename(columns={'dpain_beta': 'ba', 'dpain_se': 'sa'})
             .merge(b[KEY + ['dpain_beta', 'dpain_se', 'dpain_bh_reject']]
                    .rename(columns={'dpain_beta': 'bb', 'dpain_se': 'sb'}),
                    on=KEY))
        if j.empty:
            ax.set_visible(False)
            return None
        r = float(j['ba'].corr(j['bb']))
        sign = float((np.sign(j['ba']) == np.sign(j['bb'])).mean())
        ratio = float(j['bb'].abs().median() / j['ba'].abs().median())
        rel_a, rel_b = reliability(j['ba'], j['sa']), reliability(j['bb'], j['sb'])
        ceiling = float(np.sqrt(rel_a * rel_b)) if np.isfinite(rel_a * rel_b) else np.nan

        # 1/SE weighting: a cell estimated from few pairs should not look like a
        # cell estimated from many. Combined SE of the two halves.
        w = 1.0 / np.sqrt(j['sa'] ** 2 + j['sb'] ** 2)
        size = 6 + 34 * (w - w.min()) / max(w.max() - w.min(), 1e-12)

        ax.axhline(0, color='#DDDDDD', lw=1)
        ax.axvline(0, color='#DDDDDD', lw=1)
        lim = float(np.nanmax(np.abs(np.concatenate([j['ba'], j['bb']])))) * 1.05
        ax.plot([-lim, lim], [-lim, lim], color=INK_MUTED, lw=1.2, ls='--',
                label='identity', zorder=1)
        ax.scatter(j['ba'], j['bb'], s=size, c=color, alpha=0.45,
                   linewidths=0, zorder=2, label=f'cell (n={len(j)})')
        marked = j[j['dpain_bh_reject'] == True]                  # noqa: E712
        if len(marked):
            ax.scatter(marked['ba'], marked['bb'], s=44, facecolors='none',
                       edgecolors=INK, linewidths=1.1, zorder=3,
                       label=f'BH-significant in {lb} ({len(marked)})')
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel(f'{la}   d_pain beta', fontsize=9, color=INK)
        ax.set_ylabel(f'{lb}   d_pain beta', fontsize=9, color=INK)
        ax.set_title(headline, fontsize=10, color=INK)
        txt = (f'r = {r:.3f}\n'
               f'attenuation ceiling = {ceiling:.3f}\n'
               f'  (reliability {rel_a:.2f} / {rel_b:.2f})\n'
               f'sign agreement = {sign:.1%}\n'
               f'median |beta| ratio = {ratio:.2f}')
        ax.text(0.03, 0.97, txt, transform=ax.transAxes, va='top', fontsize=8.5,
                color=INK, bbox=dict(facecolor='white', edgecolor='#DDDDDD',
                                     boxstyle='round,pad=0.4'))
        ax.tick_params(labelsize=8, colors=INK_MUTED)
        for s in ax.spines.values():
            s.set_color('#CCCCCC')
        ax.legend(fontsize=7.5, frameon=False, loc='lower right')
        return {'comparison': f'{la} vs {lb}', 'n_cells': len(j), 'r': r,
                'reliability_a': rel_a, 'reliability_b': rel_b,
                'attenuation_ceiling': ceiling, 'sign_agreement': sign,
                'median_abs_beta_ratio': ratio}

    rows = []
    if has_dose:
        rows.append(panel(axes[0][0], dose, nodose, 'DOSED pairs only',
                          'NO-DOSE pairs only', COLOR_OK,
                          'INDEPENDENT REPLICATION\ndisjoint halves -- no shared rows'))
        rows.append(panel(axes[0][1], full, nodose, 'ALL pairs',
                          'NO-DOSE pairs only', COLOR_MUTED,
                          'SUPPLEMENT (shared data)\nno-dose pairs are a SUBSET of "all"'))
    else:
        rows.append(panel(axes[0][0], full, nodose, 'ALL pairs',
                          'NO-DOSE pairs only', COLOR_MUTED,
                          'SUPPLEMENT (shared data) -- dosed-only run not ready'))

    fig.suptitle('Does the pain effect replicate where no dose was given?\n'
                 'r is ATTENUATED by the noise in each half -- read it against '
                 'the ceiling, not against 1.0.', fontsize=12, color=INK)
    fig.text(0.5, 0.005, DISCLAIMER, ha='center', fontsize=7, color=INK_MUTED)
    fig.tight_layout(rect=[0, 0.03, 1, 0.90])
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return pd.DataFrame([r for r in rows if r])


def fig_power(full_pairs, full_cells, nodose_cells, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(19, 4.9))

    # -- panel 1: where the pain swings live -------------------------------
    ax = axes[0]
    dosed = full_pairs[full_pairs['med_between'] == 1]['d_pain']
    undosed = full_pairs[full_pairs['med_between'] == 0]['d_pain']
    lim = float(np.nanmax(np.abs(full_pairs['d_pain'])))
    edges = np.arange(-lim - 0.5, lim + 1.5, 1.0)
    for vals, color, lab in ((undosed, COLOR_OK, 'no dose between'),
                             (dosed, COLOR_BAD, 'dose between')):
        nz = vals[vals != 0]
        ax.hist(nz, bins=edges, density=True, color=color, alpha=0.55,
                label=f'{lab}  (sd={vals.std():.2f}, n={len(vals)})')
    # The zero spike gets its OWN bar -- it is 43% of the no-dose pairs and a
    # smoothed density hides exactly the thing that explains the power loss.
    for vals, color, off in ((undosed, COLOR_OK, -0.22), (dosed, COLOR_BAD, 0.02)):
        frac = float((vals == 0).mean())
        ax.bar(off, frac, width=0.2, color=color, edgecolor=INK, linewidth=0.8,
               zorder=5)
        ax.text(off + 0.1, frac, f' {frac:.0%}\n at 0', fontsize=8, va='bottom',
                ha='center', color=INK)
    ax.set_xlabel('change in pain (NRS points)', fontsize=9, color=INK)
    ax.set_ylabel('density / fraction', fontsize=9, color=INK)
    ax.set_title('Dosed pairs carry the LARGE pain swings\n'
                 'removing them removes the predictor variance',
                 fontsize=10, color=INK)
    ax.legend(fontsize=8, frameon=False)

    # -- panel 2: SE inflation ---------------------------------------------
    ax = axes[1]
    j = (full_cells[KEY + ['dpain_se']].rename(columns={'dpain_se': 'se_full'})
         .merge(nodose_cells[KEY + ['dpain_se']]
                .rename(columns={'dpain_se': 'se_nodose'}), on=KEY))
    ratio = (j['se_nodose'] / j['se_full']).replace([np.inf, -np.inf], np.nan).dropna()
    ax.hist(ratio, bins=40, color=COLOR_OK, edgecolor='white', linewidth=0.6)
    ax.axvline(1.0, color=INK_MUTED, lw=1.2, ls=':', label='no change')
    ax.axvline(float(ratio.median()), color=INK, lw=1.6,
               label=f'median = {ratio.median():.2f}')
    ax.set_xlabel('SE ratio  (no-dose / all pairs)', fontsize=9, color=INK)
    ax.set_ylabel('cells', fontsize=9, color=INK)
    ax.set_title(f'Standard errors inflate\n{len(ratio)} cells fitted in both',
                 fontsize=10, color=INK)
    ax.legend(fontsize=8, frameon=False)

    # -- panel 3: BH count vs q --------------------------------------------
    ax = axes[2]
    qs = np.linspace(0.01, 0.30, 60)
    for cells, color, lab in ((full_cells, COLOR_BAD, 'all pairs'),
                              (nodose_cells, COLOR_OK, 'no-dose only')):
        counts = [bh_count(cells['dpain_p'], q) for q in qs]
        ax.plot(qs, counts, color=color, lw=2, label=lab)
    ax.axvline(0.05, color=INK_MUTED, lw=1.2, ls=':', label='q = 0.05')
    ax.set_xlabel('BH q', fontsize=9, color=INK)
    ax.set_ylabel('cells rejected', fontsize=9, color=INK)
    ax.set_title('Rejections vs threshold\n'
                 'a steep no-dose climb just above .05 = a threshold story',
                 fontsize=10, color=INK)
    ax.legend(fontsize=8, frameon=False)

    for ax in axes:
        ax.tick_params(labelsize=8, colors=INK_MUTED)
        ax.grid(axis='y', color='#EEEEEE', lw=0.6)
        ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_color('#CCCCCC')

    fig.suptitle('Why BH collapsed from 62 cells to 2-3 while the effect did not '
                 'change', fontsize=12, color=INK)
    fig.text(0.5, 0.005, DISCLAIMER, ha='center', fontsize=7, color=INK_MUTED)
    fig.tight_layout(rect=[0, 0.03, 1, 0.90])
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return ratio


def fig_scope(full_pairs, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    rows = []
    for m, color, lab in ((0, COLOR_OK, 'no dose between'),
                          (1, COLOR_BAD, 'dose between')):
        v = full_pairs.loc[full_pairs['med_between'] == m, 'h_since_prior'].dropna()
        v = np.sort(v.to_numpy())
        if not len(v):
            continue
        y = np.arange(1, len(v) + 1) / len(v)
        ax.step(v, y, where='post', color=color, lw=2.2,
                label=f'{lab}  (n={len(v)}, median {np.median(v):.1f} h)')
        rows.append({'stratum': lab, 'n': len(v), 'median_h': float(np.median(v)),
                     'frac_within_1h': float((v <= 1).mean()),
                     'frac_within_4h': float((v <= 4).mean())})
    for x, lab in ((1, '1 h'), (4, '4 h')):
        ax.axvline(x, color=INK_MUTED, lw=1.1, ls=':')
        ax.text(x, 0.02, f' {lab}', fontsize=8, color=INK_MUTED)
    ax.set_xscale('log')
    ax.set_xlabel('hours since the PRIOR dose (log scale)', fontsize=9, color=INK)
    ax.set_ylabel('cumulative fraction of pairs', fontsize=9, color=INK)
    ax.set_title('"No dose between" does NOT mean unmedicated\n'
                 'the no-dose group is the MORE recently dosed one',
                 fontsize=11, color=INK)
    ax.tick_params(labelsize=8, colors=INK_MUTED)
    ax.grid(color='#EEEEEE', lw=0.6)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color('#CCCCCC')
    ax.legend(fontsize=8.5, frameon=False, loc='lower right')
    fig.text(0.5, 0.005, DISCLAIMER, ha='center', fontsize=7, color=INK_MUTED)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--base', default=None)
    ap.add_argument('--run-name', default='nodose_claims')
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    io.warn_if_dirty()

    base = args.base or str(config.analysis_run_dir(
        question='psd_physiology', output_type='univariate_analysis',
        view_scheme='med_change_score', run_name='').parent)

    full_dir = find_run(base, 'changescore_analgesics_min30_2*')
    nodose_dir = find_run(base, 'changescore_analgesics_min30_nodose_*')
    dose_dir = find_run(base, 'changescore_analgesics_min30_doseonly_*')
    logger.info('full=%s', full_dir)
    logger.info('nodose=%s', nodose_dir)
    logger.info('dose=%s', dose_dir)
    if full_dir is None or nodose_dir is None:
        raise SystemExit('need both the full and no-dose runs')

    full_cells, nodose_cells = converged(full_dir), converged(nodose_dir)
    dose_cells = (converged(dose_dir)
                  if dose_dir and (dose_dir / 'grid_cells.parquet').exists()
                  else None)
    if dose_cells is None:
        logger.warning('DOSE-ONLY run not collected yet -- the headline '
                       'independent-replication panel will be omitted and only '
                       'the shared-data supplement drawn.')
    full_pairs = io.read_table(full_dir / 'pair_index.parquet', on_stale='warn')

    out_dir = config.analysis_run_dir(
        question='psd_physiology', output_type='univariate_analysis',
        view_scheme='med_change_score', run_name=args.run_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    rep = fig_replication(dose_cells, nodose_cells, full_cells,
                          out_dir / 'fig_replication.png')
    logger.info('\n%s', rep.to_string(index=False))
    io.write_table(rep, out_dir / 'replication_summary.csv',
                   params={'note': 'r is attenuated; compare to the ceiling'},
                   script='ieeg_ehr/analysis/plot_nodose_claims.py')

    ratio = fig_power(full_pairs, full_cells, nodose_cells,
                      out_dir / 'fig_power.png')
    logger.info('SE ratio median %.3f over %d cells', ratio.median(), len(ratio))

    scope = fig_scope(full_pairs, out_dir / 'fig_scope.png')
    logger.info('\n%s', scope.to_string(index=False))
    io.write_table(scope, out_dir / 'scope_summary.csv',
                   params={'marks_h': [1, 4]},
                   script='ieeg_ehr/analysis/plot_nodose_claims.py')

    io.write_run_provenance(
        out_dir, script='ieeg_ehr/analysis/plot_nodose_claims.py',
        params={'full': str(full_dir), 'nodose': str(nodose_dir),
                'dose': str(dose_dir) if dose_dir else None},
        parents=[str(full_dir / 'provenance.json'),
                 str(nodose_dir / 'provenance.json')],
        extra={'status': 'EXPLORATORY, NOT a finding',
               'reading_r': 'The correlation between two disjoint halves is '
                            'attenuated by the estimation noise in each. Compare '
                            'it to the attenuation ceiling on the panel, not to '
                            '1.0.'})
    io.log_analysis('no-dose claims: replication, power loss and scope '
                    '(EXPLORATORY)', out_dir)
    print(out_dir)


if __name__ == '__main__':
    main()
