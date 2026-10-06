"""Region x band heatmap of the pain_change `d_pain` fixed effect.

    python -m ieeg_ehr.analysis.plot_pain_change_heatmap --run-dir <pain_change run>

Reads `cells.csv` and `provenance.json` from a `pain_change` run and writes
`<run>/figures/pain_change_heatmap.png`. Three panels: the `d_pain` beta (z per
NRS point) on a symmetric diverging scale with cells surviving global BH
outlined, then subjects and electrodes per region as bars, because both counts
are constant across bands. Significance is an outline only, never a colour, the
same convention as `plot_mixed_model_grid`.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from ieeg_ehr import io
from ieeg_ehr.analysis import mixed_model as mm, view_tables
from ieeg_ehr.analysis.pain_change import FIGURES_SUBDIR, FORMULA, random_effects_text
from ieeg_ehr.features import common

logger = logging.getLogger(__name__)


def grids(cells, regions, cols):
    """(beta, BH mask, n_subjects per region, n_channels per region) in fixed order."""
    def pivot(value):
        return cells.pivot_table(index='region', columns='freq', values=value,
                                 aggfunc='first').reindex(index=regions, columns=cols)
    beta = pivot('dpain_beta')
    sig = (cells.assign(sig=cells['dpain_bh_reject'].eq(True).astype(float))
           .pivot_table(index='region', columns='freq', values='sig', aggfunc='first')
           .reindex(index=regions, columns=cols).fillna(0).astype(bool))
    return beta, sig, pivot('n_subjects').max(axis=1), pivot('n_channels').max(axis=1)


def plot(cells, regions, cols, params, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    beta, sig, n_subj, n_chan = grids(cells, regions, cols)
    vmax = float(np.nanmax(np.abs(beta.to_numpy())))
    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad('0.85')

    fig, axes = plt.subplots(1, 3, figsize=(11, 0.32 * len(regions) + 2.4),
                             gridspec_kw={'width_ratios': [len(cols), 1.6, 1.6]})
    ax = axes[0]
    im = ax.imshow(beta.to_numpy(dtype=float), aspect='auto', cmap=cmap,
                   vmin=-vmax, vmax=vmax, interpolation='nearest')
    common.draw_mask_outline(ax, sig.to_numpy())
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f'{c}\n{lo:g}-{hi:g} Hz' for c, (lo, hi) in
                        ((c, params['bands'][c]) for c in cols)],
                       fontsize=7, rotation=45, ha='right')
    ax.set_yticks(range(len(regions)))
    ax.set_yticklabels(regions, fontsize=8)
    ax.set_title('d_pain fixed effect (z per NRS point)', fontsize=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label('beta', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    for ax, counts, label, colour in ((axes[1], n_subj, 'subjects', '0.55'),
                                      (axes[2], n_chan, 'electrodes', '#4a7ba7')):
        vals = counts.to_numpy(dtype=float)
        ax.barh(range(len(regions)), np.nan_to_num(vals), color=colour)
        ax.set_ylim(len(regions) - 0.5, -0.5)
        ax.set_yticks(range(len(regions)))
        ax.set_yticklabels([])
        ax.set_title(f'n {label}\nper region', fontsize=10)
        ax.tick_params(labelsize=7)
        span = np.nanmax(vals) if np.isfinite(vals).any() else 1.0
        ax.set_xlim(0, 1.25 * span)
        for i, v in enumerate(vals):
            if np.isfinite(v):
                ax.text(v + 0.03 * span, i, f'{int(v)}', va='center', fontsize=6.5,
                        color='0.3')
        if label == 'subjects':
            ax.axvline(params['min_subjects'], color='0.2', lw=0.8, ls=':')

    n_fit = int(np.isfinite(beta.to_numpy()).sum())
    fig.suptitle(f'pain_change: {FORMULA} + {random_effects_text(mm.VC_CHANGE)}',
                 fontsize=10, y=0.995)
    fig.tight_layout(rect=(0, 0.05, 1, 0.98))
    fig.text(0.01, 0.005,
             f'ROI scheme {params["roi_scheme"]}. {n_fit} cells fitted; outlines mark '
             f'BH q<0.05 across all fitted cells. Grey = not fitted (below '
             f'{params["min_subjects"]} subjects, dotted line). EXPLORATORY, '
             'discovery cohort, NOMINATIONS NOT FINDINGS. See METHODS.md.',
             fontsize=7, color='0.35', ha='left', va='bottom', wrap=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-dir', required=True)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    run_dir = Path(args.run_dir)
    params = json.loads((run_dir / 'provenance.json').read_text())['params']
    if not params.get('bands'):
        raise SystemExit('this heatmap is for a --freq canonical_bands run')
    cells = io.read_table(run_dir / 'cells.csv', on_stale='warn')
    regions = [r for r in view_tables.roi_regions_for({'roi_scheme': params['roi_scheme']})
               if r in set(cells['region'])]
    out = run_dir / FIGURES_SUBDIR / 'pain_change_heatmap.png'
    plot(cells, regions, list(params['bands']), params, out)
    io.log_analysis('pain_change d_pain heatmap', out.parent)
    logger.info('wrote %s', out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
