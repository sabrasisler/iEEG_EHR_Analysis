"""Overview figures for a (refiltered) mixed-model grid.

Two figures, deliberately at different altitudes:

`fig_grid_map` -- the whole grid at once. Region x frequency, four panels: the
pain fixed effect, where it survives FDR, how much subjects DIFFER, and how many
subjects each region has. The first panel is the direct successor to the
two-stage heatmap; the other three exist because a beta alone is not readable.
A cell that is strong, significant, homogeneous and well covered is a different
claim from one that is strong and none of those.

`fig_grid_spectra` -- one small panel per region, beta against frequency with its
95% band. A heatmap makes you compare colours across a log axis, which is exactly
where a sign flip or a narrow peak is easiest to miss; a spectrum makes both
obvious. This is the figure to read second and probably the one to think with.

NEITHER SHOWS A p-VALUE AS A COLOUR. Significance is an outline on panel 1 and
nothing else, because the heterogeneity LRT is significant in ~98% of cells and a
p-value map would be uniformly black while carrying no information. Panel 3 shows
the heterogeneity EFFECT SIZE instead -- the by-subject slope SD, in the same
units as beta, which is the comparison that means something.

    python -m ieeg_ehr.analysis.plot_mixed_model_grid --run-dir <filtered run>
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ieeg_ehr import io
from ieeg_ehr.analysis import view_tables
from ieeg_ehr.features import common

logger = logging.getLogger(__name__)

DISCLAIMER = ('EXPLORATORY -- discovery cohort, NOMINATIONS NOT FINDINGS. Parametric '
              'Wald p (pilot permutation put the null z SD at ~1.03). Not confirmed '
              'out of sample.')


def pivot(cells, value, regions, bins):
    """region x freq_bin table in a FIXED row/column order.

    Built by reindex rather than by pivot alone: a region or bin missing from the
    data must appear as an empty ROW, not vanish and silently shift every label
    below it against its colours.
    """
    p = cells.pivot_table(index='region', columns='freq_bin_index', values=value,
                          aggfunc='first')
    return p.reindex(index=regions, columns=bins)


def order_regions(cells, regions):
    """The registry order, restricted to regions actually present."""
    present = set(cells['region'])
    return [r for r in regions if r in present]


def sign_consistency(run_dir, cells, blups, regions, bins):
    """(pivot of sign consistency, a label saying which estimate it is).

    Prefers `sign_consistency.parquet` -- the UNPOOLED per-subject fits from
    `compute_subject_slopes_grid`. Falls back to the model's BLUPs only if that
    has not been computed, and says so on the panel, because the two are not
    interchangeable: partial pooling drags every subject toward the group, so
    BLUP-based consistency comes out at 0.8-1.0 almost everywhere and separates
    nothing. A panel that cannot distinguish a real cell from a null one should
    not be able to masquerade as one that can.
    """
    path = Path(run_dir) / 'sign_consistency.parquet'
    if path.exists():
        d = io.read_table(path, on_stale='warn')
        p = d.pivot_table(index='region', columns='freq_bin_index',
                          values='frac_sign_consistent')
        return p.reindex(index=regions, columns=bins), 'unpooled per-subject fits'

    logger.warning('no sign_consistency.parquet; falling back to BLUPs, which are '
                   'shrunk toward the group and will look far more consistent than '
                   'the data is. Run compute_subject_slopes_grid for the real thing.')
    b = blups.merge(cells[['region', 'freq_bin_index', 'beta_nrs_within']],
                    left_on=['region', 'freq_bin'],
                    right_on=['region', 'freq_bin_index'], how='inner')
    b = b[b['beta_nrs_within'].notna() & b['subject_slope'].notna()]
    b['agrees'] = np.sign(b['subject_slope']) == np.sign(b['beta_nrs_within'])
    frac = (b.groupby(['region', 'freq_bin_index'])['agrees'].mean()
            .rename('frac').reset_index())
    p = frac.pivot_table(index='region', columns='freq_bin_index', values='frac')
    return (p.reindex(index=regions, columns=bins),
            'SHRUNK BLUPs -- optimistic, see footnote')


def fig_grid_map(run_dir, cells, blups, regions, bins, bin_labels, out_path,
                 het_vmax=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    beta = pivot(cells, 'beta_nrs_within', regions, bins)
    sig = pivot(cells, 'p_bh_reject', regions, bins).fillna(False).astype(bool)
    # Heterogeneity as an SD, in beta's units -- a variance is unreadable next to
    # a slope, and this is the number that says how much subjects actually differ.
    het = np.sqrt(pivot(cells, 'var_subj_slope', regions, bins).astype(float))
    cons, cons_source = sign_consistency(run_dir, cells, blups, regions, bins)

    fig, axes = plt.subplots(1, 6, figsize=(27, 0.42 * len(regions) + 3.6),
                             gridspec_kw={'width_ratios': [1, 1, 1, 1, 0.34, 0.34]})

    vmax = float(np.nanmax(np.abs(beta.to_numpy())))
    div = plt.get_cmap('RdBu_r').copy()
    div.set_bad('0.85')

    # A few thin high-frequency cells run an order of magnitude above the rest
    # (max 0.181 against a median of 0.013) and are almost certainly badly
    # estimated rather than genuinely that heterogeneous. Scaling to the max
    # spends the whole colour ramp on them and flattens every real cell to the
    # bottom, so the default clips at the 95th percentile. Cells above the cap
    # saturate rather than disappear -- the title states the cap so a saturated
    # cell is never mistaken for one that merely sits at the top of the range.
    het_top = float(het_vmax if het_vmax is not None
                    else np.nanpercentile(het.to_numpy(), 95))

    # Symmetric about 0.5 so the diverging colours still mean "better/worse than
    # chance", but only as wide as the data actually goes. Floored at 0.05 so a
    # uniformly consistent grid does not get its noise amplified into structure.
    cons_span = float(np.nanmax(np.abs(cons.to_numpy(dtype=float) - 0.5)))
    cons_span = max(cons_span, 0.05) if np.isfinite(cons_span) else 0.5

    panels = [
        (axes[0], beta, div, -vmax, vmax,
         'pain fixed effect\nd log10 power per pain point', True),
        (axes[1], beta.where(sig), div, -vmax, vmax,
         f'the same, BH-significant only\n({int(sig.to_numpy().sum())} cells, q=0.05)',
         False),
        (axes[2], het, _seq_cmap(plt), 0.0, het_top,
         f'HETEROGENEITY\nbetween-subject SD of the slope (clipped at {het_top:.3f})',
         False),
        # Centred at 0.5 -- half the subjects agreeing is chance, so that is the
        # neutral point -- but NOT spanning 0 to 1. A fraction of 0 would mean
        # every subject opposes the group, which cannot happen: the group beta IS
        # a weighted average of those same subject slopes, so its sign is set by
        # them. The reachable range is ~0.5 to 1, with modest excursions below 0.5
        # when a minority with large slopes carries the mean. Spanning 0-1 spent
        # half the ramp on impossible states and rendered every real cell purple.
        (axes[3], cons, _cons_cmap(plt), 0.5 - cons_span, 0.5 + cons_span,
         f'SIGN CONSISTENCY\n{cons_source}', False),
    ]
    for ax, mat, cm, vlo, vhi, title, outline in panels:
        im = ax.imshow(mat.to_numpy(dtype=float), aspect='auto', cmap=cm,
                       vmin=vlo, vmax=vhi, interpolation='nearest')
        if outline:
            common.draw_mask_outline(ax, sig.to_numpy())
        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(len(bins)))
        ax.set_xticklabels([f'{bin_labels.loc[b, "bin_low_hz"]:.0f}' for b in bins],
                           fontsize=6, rotation=90)
        ax.set_yticks(range(len(regions)))
        ax.set_yticklabels(regions if ax is axes[0] else [], fontsize=8)
        ax.set_xlabel('frequency bin, low edge (Hz)', fontsize=8)
        common.add_band_boundary_lines(ax, bin_labels.loc[bins])
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02).ax.tick_params(labelsize=7)

    # Coverage. NOT heatmaps: both are one number per region -- verified constant
    # across all 38 bins -- and a 38-wide strip would imply they vary with
    # frequency, which is a claim the data does not make.
    for ax, col, label, colour in ((axes[4], 'n_subjects', 'subjects', '0.55'),
                                   (axes[5], 'n_channels', 'electrodes', '#4a7ba7')):
        per_region = pivot(cells, col, regions, bins).max(axis=1)
        ax.barh(range(len(regions)), per_region.to_numpy(), color=colour)
        ax.set_yticks(range(len(regions)))
        ax.set_yticklabels([])
        ax.set_ylim(len(regions) - 0.5, -0.5)
        ax.set_title(f'n {label}\nper region', fontsize=10)
        ax.set_xlabel(f'n {label}', fontsize=8)
        ax.tick_params(labelsize=7)
        vals = per_region.to_numpy()
        span = np.nanmax(vals) if np.isfinite(np.nanmax(vals)) else 1.0
        for i, v in enumerate(vals):
            if np.isfinite(v):
                ax.text(v + 0.02 * span, i, f'{int(v)}', va='center', fontsize=6.2,
                        color='0.3')

    fig.suptitle('Mixed-model pain encoding across the region x frequency grid',
                 fontsize=13)
    fig.tight_layout(rect=(0, 0.075, 1, 0.945))
    fig.text(0.01, 0.005,
             'Outlines on panel 1 mark cells surviving BH across the whole grid at '
             'q=0.05. Panel 3 is a standard DEVIATION rather than a p-value: the '
             'heterogeneity LRT is significant in ~98% of cells, so a p-map would be '
             'uniformly dark and carry no information, whereas the spread has '
             'structure. Panel 4 is centred at 0.5 because half the subjects '
             'agreeing IS chance, and is scaled to the observed spread rather than '
             '0-1: a fraction of 0 would mean every subject opposes the group, which '
             'cannot happen, since the group effect is itself a weighted average of '
             'those subject slopes. Orange means FEWER than half of subjects share '
             'the group\'s direction -- the mean is being carried by a minority with '
             'large slopes and does not describe a typical patient. Its denominator '
             'is subjects whose slope is defined at all (>=2 distinct pain scores); '
             'in the pilot that excluded nobody. Subject and electrode counts are constant '
             'across frequency, hence bars. Grey = no data. Bins narrower than the '
             '0.5 Hz FFT resolution were removed as exact duplicates.\n' + DISCLAIMER,
             fontsize=7, va='bottom', ha='left', color='0.35', wrap=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _cons_cmap(plt):
    cm = plt.get_cmap('PuOr').copy()
    cm.set_bad('0.85')
    return cm


def _seq_cmap(plt):
    cm = plt.get_cmap('viridis').copy()
    cm.set_bad('0.85')
    return cm


def fig_grid_spectra(cells, regions, bin_labels, out_path, ncol=5):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    nrow = int(np.ceil(len(regions) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 2.7 * nrow),
                             squeeze=False, sharex=True)

    ymax = float(np.nanmax(np.abs(
        np.concatenate([(cells['beta_nrs_within'] + 1.96 * cells['se']).to_numpy(),
                        (cells['beta_nrs_within'] - 1.96 * cells['se']).to_numpy()]))))

    for i, region in enumerate(regions):
        ax = axes[i // ncol][i % ncol]
        d = cells[cells['region'] == region].sort_values('freq_bin_index')
        hz = np.sqrt(d['freq_bin_low'].to_numpy() * d['freq_bin_high'].to_numpy())
        b = d['beta_nrs_within'].to_numpy(dtype=float)
        se = d['se'].to_numpy(dtype=float)

        ax.fill_between(hz, b - 1.96 * se, b + 1.96 * se, color='0.75', alpha=0.45,
                        lw=0)
        ax.plot(hz, b, color='black', lw=1.3)
        star = d['p_bh_reject'].fillna(False).to_numpy(dtype=bool)
        ax.scatter(hz[star], b[star], s=14, color='#c1442f', zorder=4)
        ax.axhline(0, color='0.6', lw=0.8, ls='--')
        ax.set_xscale('log')
        ax.set_ylim(-ymax, ymax)
        ax.set_title(f"{region}  (n={int(d['n_subjects'].max())})", fontsize=9)
        ax.tick_params(labelsize=7)
        if i % ncol == 0:
            ax.set_ylabel('beta', fontsize=8)
        if i // ncol == nrow - 1:
            ax.set_xlabel('frequency (Hz)', fontsize=8)

    for j in range(len(regions), nrow * ncol):
        axes[j // ncol][j % ncol].set_visible(False)

    fig.suptitle('Pain fixed effect vs frequency, per region\n'
                 'black = beta, band = 95% CI, red dots = BH-significant across the grid',
                 fontsize=12)
    fig.tight_layout(rect=(0, 0.045, 1, 0.94))
    fig.text(0.01, 0.005,
             'All panels share a y-scale so region amplitudes are comparable. The x '
             'axis is log-spaced, matching how the bins were built. A sign change '
             'within a region is visible here and very hard to see on a heatmap, '
             'which is the reason this figure exists alongside one.\n' + DISCLAIMER,
             fontsize=7, va='bottom', ha='left', color='0.35', wrap=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--roi-scheme', default='roi_v2')
    ap.add_argument('--het-vmax', type=float, default=None,
                    help='Colour-bar top for the heterogeneity panel, in beta units. '
                         'Default: the 95th percentile across cells.')
    ap.add_argument('--suffix', default='',
                    help='Appended to figure filenames, e.g. --suffix _v2, so an '
                         'alternative scaling does not overwrite the first render.')
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    io.warn_if_dirty()

    run_dir = Path(args.run_dir)
    cells = io.read_table(run_dir / 'grid_cells.parquet', on_stale='warn')

    regions = order_regions(cells,
                            view_tables.roi_regions_for({'roi_scheme': args.roi_scheme}))
    bins = sorted(cells['freq_bin_index'].unique())
    bin_labels = (cells.drop_duplicates('freq_bin_index')
                  .set_index('freq_bin_index')[['freq_bin_low', 'freq_bin_high']]
                  .rename(columns={'freq_bin_low': 'bin_low_hz',
                                   'freq_bin_high': 'bin_high_hz'})
                  .sort_index())
    logger.info('%d cells | %d regions | %d bins', len(cells), len(regions), len(bins))

    blups = io.read_table(run_dir / 'grid_blups.parquet', on_stale='warn')
    map_path = run_dir / f'fig_grid_map{args.suffix}.png'
    spec_path = run_dir / f'fig_grid_spectra{args.suffix}.png'

    fig_grid_map(run_dir, cells, blups, regions, bins, bin_labels, map_path,
                 het_vmax=args.het_vmax)
    logger.info('wrote %s', map_path)
    fig_grid_spectra(cells, regions, bin_labels, spec_path)
    logger.info('wrote %s', spec_path)

    io.log_analysis('mixed-model grid overview figures: region x frequency map and '
                    'per-region spectra (EXPLORATORY)', run_dir)


if __name__ == '__main__':
    main()
