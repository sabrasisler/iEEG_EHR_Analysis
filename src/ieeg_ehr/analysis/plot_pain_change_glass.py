"""Glass brain of one pain_change term: BH-significant ROIs coloured by region.

    python -m ieeg_ehr.analysis.plot_pain_change_glass --run-dir <pain_change run> \\
        [--term dpain]

Writes `<run>/figures/glass_<term token>.png` (docs/run_tokens.md): a sagittal
view over an axial one. Every electrode the run's ROI map assigns to a region is
drawn at the MNI midpoint of its bipolar pair. Electrodes of an ROI with at
least one BH-significant cell for the term (`<term>_bh_reject`, BH across all
fitted cells, failed fits excluded) take that ROI's categorical colour; every
other electrode is a small grey dot, so an empty area means no coverage, not no
effect. The legend names each coloured ROI's significant bands and how many
electrodes it draws.

Colour carries identity, not the slope: it fits a term whose significant cells
share a sign (`d_pain`, all positive in the 2026-10-15 run), and the heatmap
beside it carries the size. The script refuses a term whose significant cells
mix signs, or more ROIs than CATEGORICAL holds.

`--low-high` writes `glass_<term token>_sig.png` instead: only electrodes of an ROI
with a BH-significant cell are drawn, each dot coloured by the ROI's slope. The
LOW value is the mean beta over the ROI's significant LOW_BANDS (delta, theta,
alpha), the HIGH value over its significant high bands (beta to high gamma). An
ROI significant in both is a split dot, left half low, right half high
(`plot_domain_glass.wedge_markers`). Colours are the heatmap's: RdBu_r on +/- the
heatmap's cap (`plot_pain_change_heatmap.colour_cap`), so a half and its heatmap
cells match. The legend is a key to the halves and a colorbar, not a list of
ROIs; `glass_<term token>_sig.csv` beside it lists each drawn ROI's values.

Region assignment is `run_mixed_model_pilot.roi_maps` over the run's z-score
tables with the run's pinned insula cut, the same call `pain_change.load_rows`
makes, so the brain cannot disagree with the fit about membership. Electrodes
outside the MNI152 brain mask are dropped and counted
(`plot_domain_glass.inside_brain`).
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from ieeg_ehr import io
from ieeg_ehr.analysis.pain_change import FIGURES_SUBDIR, discovery_paths
from ieeg_ehr.analysis.plot_band_map_domains import BAND_SYMBOL
from ieeg_ehr.analysis.plot_domain_glass import electrode_coords, inside_brain, wedge_markers
from ieeg_ehr.analysis.plot_pain_change_heatmap import TERMS, colour_cap
from ieeg_ehr.analysis.run_mixed_model_pilot import roi_maps
from ieeg_ehr.config import roi_schemes

logger = logging.getLogger(__name__)

SCRIPT = 'ieeg_ehr/analysis/plot_pain_change_glass.py'

#: Categorical hues in fixed order (dataviz default palette, light mode),
#: assigned to the significant ROIs in scheme display order.
CATEGORICAL = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300',
               '#6250d6', '#e34948')
GREY = '0.80'
MNI = ['mni_x', 'mni_y', 'mni_z']
#: --low-high: the band group of the left half; every other band is high.
LOW_BANDS = ('delta', 'theta', 'alpha')
#: --low-high: the heatmap's colormap, so a dot half and its cells match.
CMAP = 'RdBu_r'


def electrodes(params, subjects):
    """One row per (subject, channel) with its MNI midpoint and the fit's ROI."""
    region_of, _ = roi_maps(discovery_paths(Path(params['zscore_dir'])), set(subjects),
                            params['roi_scheme'],
                            insula_threshold=params['insula_threshold'])
    el = electrode_coords(subjects, params['view_config']['epoch_minutes'])
    el['region'] = [region_of.get(s, {}).get(c)
                    for s, c in zip(el['subject_id'], el['channel'])]
    el = el[el['region'].notna()]
    keep = inside_brain(el[MNI].to_numpy(dtype=float))
    logger.info('%d electrodes with an ROI, %d outside the MNI152 brain mask dropped',
                len(el), int((~keep).sum()))
    return el[keep].reset_index(drop=True)


def significant_regions(cells, term, scheme):
    """[(region, [band, ...]), ...] in scheme display order, BH-significant cells only."""
    sig = cells[cells[f'{term}_bh_reject'].eq(True) & cells['converged'].eq(True)]
    signs = set(np.sign(sig[f'{term}_beta']))
    if len(signs) > 1:
        raise SystemExit(f'{term}: significant cells mix signs, so a region colour '
                         'would hide the direction; use the heatmap')
    out = [(r, list(sig.loc[sig['region'] == r, 'freq']))
           for r in roi_schemes.roi_regions(scheme) if r in set(sig['region'])]
    if len(out) > len(CATEGORICAL):
        raise SystemExit(f'{term}: {len(out)} significant ROIs, more than the '
                         f'{len(CATEGORICAL)} categorical colours')
    return out


def draw(el, regions, out_path, node_size, grey_size):
    """Sagittal over axial, grey electrodes first, then each ROI in its colour.

    ROIs are drawn largest first, so a small ROI (Auditory, S2/PO) sits on top
    of the large one around it (Lateral Temporal) instead of under it.
    """
    from nilearn import plotting

    fig = plt.figure(figsize=(5.2, 8.6))
    colour_of = {r: c for (r, _), c in zip(regions, CATEGORICAL)}
    grey = el.loc[~el['region'].isin(colour_of), MNI].to_numpy(dtype=float)
    n_of = el['region'].value_counts()
    handles, displays = [], []
    for mode, rect in (('x', (0.0, 0.58, 1.0, 0.40)), ('z', (0.08, 0.17, 0.84, 0.42))):
        d = plotting.plot_glass_brain(None, display_mode=mode, figure=fig, axes=rect,
                                      annotate=False)
        d.add_markers(grey, marker_color=GREY, marker_size=grey_size, linewidths=0)
        for region in sorted(colour_of, key=lambda r: -n_of[r]):
            colour = colour_of[region]
            pts = el.loc[el['region'] == region, MNI].to_numpy(dtype=float)
            d.add_markers(pts, marker_color=colour, marker_size=node_size,
                          edgecolors='white', linewidths=0.3)
        displays.append(d)
    for region, bands in regions:
        handles.append(mlines.Line2D(
            [], [], ls='none', marker='o', markersize=8, markerfacecolor=colour_of[region],
            markeredgecolor='white',
            label=f'{region}: {", ".join(BAND_SYMBOL[b] for b in bands)} '
                  f'(n = {n_of[region]})'))
    handles.append(mlines.Line2D(
        [], [], ls='none', marker='o', markersize=5, markerfacecolor=GREY,
        markeredgecolor='none', label=f'not significant (n = {len(grey)})'))
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.165),
               ncol=2, frameon=False, fontsize=8.5, handletextpad=0.4,
               columnspacing=1.2, labelspacing=0.6,
               title='BH-significant regions (bands)',
               title_fontproperties={'weight': 'bold', 'size': 10})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    for d in displays:
        d.close()


def low_high_values(cells, term):
    """One row per ROI with a BH-significant cell: mean significant beta per group.

    `low` and `high` are NaN where the ROI has no significant cell in that group.
    """
    sig = cells[cells[f'{term}_bh_reject'].eq(True) & cells['converged'].eq(True)]
    group = np.where(sig['freq'].isin(LOW_BANDS), 'low', 'high')
    out = (sig.assign(group=group).groupby(['region', 'group'])[f'{term}_beta'].mean()
           .unstack('group').reindex(columns=['low', 'high']))
    bands = sig.groupby('region')['freq'].agg(lambda b: ','.join(b))
    return out.assign(bands=bands).reset_index()


def draw_low_high(el, values, cap, cb_label, out_path, node_size):
    """Sagittal over axial: one dot per electrode of a drawn ROI, split low | high."""
    from nilearn import plotting

    cmap = plt.get_cmap(CMAP)
    norm = matplotlib.colors.Normalize(-cap, cap)
    halves = wedge_markers(2)
    n_of = el['region'].value_counts()
    fig = plt.figure(figsize=(5.2, 9.2))
    displays = []
    for mode, rect in (('x', (0.0, 0.58, 1.0, 0.40)), ('z', (0.08, 0.17, 0.84, 0.42))):
        d = plotting.plot_glass_brain(None, display_mode=mode, figure=fig, axes=rect,
                                      annotate=False)
        for r in sorted(values.itertuples(), key=lambda r: -n_of[r.region]):
            pts = el.loc[el['region'] == r.region, MNI].to_numpy(dtype=float)
            parts = [v for v in (r.low, r.high) if np.isfinite(v)]
            markers = halves if len(parts) == 2 else ['o']
            for m, v in zip(markers, parts):
                d.add_markers(pts, marker_color=matplotlib.colors.to_hex(cmap(norm(v))), marker_size=node_size,
                              marker=m, linewidths=0)
        displays.append(d)

    key_colour = '0.45'
    handles = [
        mlines.Line2D([], [], ls='none', marker='o', markersize=10, fillstyle='left',
                      markerfacecolor=key_colour, markerfacecoloralt='white',
                      markeredgecolor=key_colour,
                      label='left half: δ–α'),
        mlines.Line2D([], [], ls='none', marker='o', markersize=10, fillstyle='right',
                      markerfacecolor=key_colour, markerfacecoloralt='white',
                      markeredgecolor=key_colour, label='right half: β–hγ'),
        mlines.Line2D([], [], ls='none', marker='o', markersize=10,
                      markerfacecolor=key_colour, markeredgecolor=key_colour,
                      label='whole dot: one group only')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.165),
               ncol=1, frameon=False, fontsize=8.5, handletextpad=0.4,
               labelspacing=0.5)
    cax = fig.add_axes((0.25, 0.035, 0.5, 0.012))
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                      orientation='horizontal')
    cb.set_label(cb_label, fontsize=8.5)
    cb.ax.tick_params(labelsize=7.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    for d in displays:
        d.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--term', choices=list(TERMS), default='dpain')
    ap.add_argument('--node-size', type=float, default=16)
    ap.add_argument('--grey-size', type=float, default=5)
    ap.add_argument('--low-high', action='store_true',
                    help='significant ROIs only, each dot split into its low and high '
                         'band mean beta (glass_<term>_sig.png)')
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    run_dir = Path(args.run_dir)
    prov = json.loads((run_dir / 'provenance.json').read_text())
    params = prov['params']
    cells = io.read_table(run_dir / 'cells.csv', on_stale='warn')
    cells = cells.assign(region=cells['region'].replace(roi_schemes.ROI_RENAMES))
    el = electrodes(params, prov['subjects'])
    token, _, cb_label = TERMS[args.term]
    if args.low_high:
        values = low_high_values(cells, args.term)
        empty = sorted(set(values['region']) - set(el['region']))
        if empty:
            raise SystemExit(f'significant ROIs with no electrode drawn: {empty}')
        cap = colour_cap(cells, args.term)
        out = run_dir / FIGURES_SUBDIR / f'glass_{token}_sig.png'
        draw_low_high(el[el['region'].isin(values['region'])], values, cap, cb_label,
                      out, args.node_size)
        n = el['region'].value_counts()
        io.write_table(values.assign(n_electrodes=values['region'].map(n)),
                       out.with_suffix('.csv'), params=vars(args),
                       parents=[str(run_dir / 'cells.csv')], subjects=prov['subjects'],
                       script=SCRIPT)
        io.log_analysis(f'pain_change {args.term} glass brain, {len(values)} BH-significant '
                        'ROIs split low | high band mean beta', out.parent)
        logger.info('wrote %s:\n%s', out, values.to_string(index=False))
        return 0
    regions = significant_regions(cells, args.term, params['roi_scheme'])
    empty = [r for r, _ in regions if not (el['region'] == r).any()]
    if empty:
        raise SystemExit(f'significant ROIs with no electrode drawn: {empty}')

    out = run_dir / FIGURES_SUBDIR / f'glass_{token}.png'
    draw(el, regions, out, args.node_size, args.grey_size)
    io.log_analysis(f'pain_change {args.term} glass brain, {len(regions)} BH-significant '
                    f'ROIs coloured by region, {len(el)} electrodes', out.parent)
    logger.info('wrote %s', out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
