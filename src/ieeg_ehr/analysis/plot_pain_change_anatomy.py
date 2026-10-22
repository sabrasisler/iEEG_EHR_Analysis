"""Anatomy glass brain of a pain_change run: hue = lobe, shade = ROI within it.

    python -m ieeg_ehr.analysis.plot_pain_change_anatomy --run-dir <pain_change run>

Writes `<run>/glass_brain/anatomy_<timestamp>/fig_anatomy_glass.png`: one
sagittal and one axial view on one scale (`plot_domain_anatomy.draw_brains`),
with the legend on the right, at exactly `--figsize` inches. Every electrode the
run's ROI map assigns to a region is drawn at the MNI midpoint of its bipolar
pair, so the brain shows the fit's own subjects and membership
(`plot_pain_change_glass.electrodes`). ANATOMY ONLY: no slope, no p-value.

LOBES maps each ROI of the hcpexv5 scheme to one lobe, and LOBE_HUES gives each
lobe its base colour (set by the user 2026-10-22). The script refuses an ROI of
the run that LOBES does not place.
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from ieeg_ehr import io
from ieeg_ehr.analysis.plot_domain_anatomy import check_inside, draw_brains
from ieeg_ehr.analysis.plot_domain_glass import SUBDIR
from ieeg_ehr.analysis.plot_pain_change_glass import electrodes
from ieeg_ehr.config import roi_schemes

logger = logging.getLogger(__name__)

SCRIPT = 'ieeg_ehr/analysis/plot_pain_change_anatomy.py'

#: Lobe -> its ROIs in the hcpexv5 scheme, in legend order.
LOBES = {
    'Frontal': ('dPFC', 'vlPFC', 'OFC', 'vmPFC', 'Premotor', 'M1', 'Frontal operculum'),
    'Parietal': ('S1', 'SII', 'SPL + IPS', 'Supramarginal', 'Angular', 'Precuneus'),
    'Temporal': ('STG', 'MTG', 'ITG/fusiform'),
    'Occipital': ('Occipital',),
    'Cingulate': ('ACC', 'MCC', 'PCC'),
    'Insula': ('aIns', 'pIns'),
    'Medial temporal': ('Medial temporal lobe', 'Amygdala', 'Hippocampus'),
    'Subcortical': ('Ventral thalamus', 'Dorsal/medial thalamus', 'Basal ganglia'),
}
#: Base colour of each lobe. A lobe with one ROI draws it in this colour.
LOBE_HUES = {
    'Frontal': '#018700', 'Parietal': '#8c3bff', 'Temporal': '#00acc6',
    'Occipital': '#ffa52f', 'Cingulate': '#ff7ed1', 'Insula': '#d60000',
    'Medial temporal': '#a17569', 'Subcortical': '#005659',
}
#: A shade is the base colour mixed with black (t < 0) or white (t > 0) by |t|.
#: Adjacent shades of one lobe sit MIX_STEP apart in t, or MIX_MIN_SPAN / (n - 1)
#: if wider, so a two-ROI lobe still gets two distinct shades. MIX_RANGE bounds t,
#: and no shade is darker than HLS lightness DARK_FLOOR.
MIX_STEP = 0.17
MIX_MIN_SPAN = 0.6
MIX_RANGE = (-0.55, 0.60)
DARK_FLOOR = 0.15


def shades(base, n):
    """`n` shades of `base`, darkest first, centred on `base` where the bounds allow.

    Mixing toward white rather than raising HLS lightness keeps the light shades
    pastel instead of neon.
    """
    rgb = np.array(matplotlib.colors.to_rgb(base))
    if n == 1:
        return [matplotlib.colors.to_hex(rgb)]
    step = max(MIX_STEP, MIX_MIN_SPAN / (n - 1))
    span = step * (n - 1)
    lightness = (rgb.max() + rgb.min()) / 2
    floor = max(MIX_RANGE[0], DARK_FLOOR / lightness - 1)
    lo = float(np.clip(-span / 2, floor, MIX_RANGE[1] - span))
    out = []
    for t in lo + step * np.arange(n):
        mixed = rgb * (1 + t) if t < 0 else rgb + (1 - rgb) * t
        out.append(matplotlib.colors.to_hex(mixed))
    return out


def palette():
    """{ROI: hex}, each lobe's ROIs spread across shades of its hue."""
    return {roi: c for lobe, rois in LOBES.items()
            for roi, c in zip(rois, shades(LOBE_HUES[lobe], len(rois)))}


def legend_entries(colours):
    """[(handle, label)] in LOBES order, a bold header over each lobe's ROIs.

    A one-ROI lobe is a single bold entry in its colour. Returns one list per lobe.
    """
    blank = mpatches.Patch(alpha=0, linewidth=0)
    out = []
    for lobe, rois in LOBES.items():
        head = f'$\\bf{{{lobe.replace(" ", chr(92) + " ")}}}$'
        if len(rois) == 1:
            out.append([(mpatches.Patch(facecolor=colours[rois[0]], edgecolor='0.35',
                                        linewidth=0.4), head)])
            continue
        out.append([(blank, head)] + [
            (mpatches.Patch(facecolor=colours[r], edgecolor='0.35', linewidth=0.4), r)
            for r in rois])
    return out


def two_columns(groups):
    """Entries for a 2-column legend that splits only between lobes.

    matplotlib fills columns top to bottom, so the shorter column is padded with
    blank rows to the length of the longer one.
    """
    sizes = [len(g) for g in groups]
    k = min(range(1, len(groups)),
            key=lambda i: max(sum(sizes[:i]), sum(sizes[i:])))
    left = [e for g in groups[:k] for e in g]
    right = [e for g in groups[k:] for e in g]
    rows = max(len(left), len(right))
    blank = (mpatches.Patch(alpha=0, linewidth=0), '')
    return left + [blank] * (rows - len(left)) + right + [blank] * (rows - len(right))


def draw(el, colours, out_path, args):
    """Legend on the right at its measured width, the brains in what it leaves."""
    W, H = args.figsize
    pad = 0.05
    fig = plt.figure(figsize=(W, H))
    lax = fig.add_axes((0, 0, 1, 1))
    lax.axis('off')
    handles, labels = zip(*two_columns(legend_entries(colours)))
    leg = lax.legend(handles=handles, labels=labels, loc='center right',
                     bbox_to_anchor=(1 - pad / W, 0.5), ncol=2, frameon=False,
                     fontsize=args.font_legend, handlelength=1.1, handleheight=1.1,
                     columnspacing=0.8, labelspacing=0.30, borderpad=0.0,
                     handletextpad=0.5)
    fig.canvas.draw()
    leg_w = leg.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
    brain_w = W - 2 * pad - leg_w - args.legend_gap
    node_args = argparse.Namespace(node_size=args.node_size, alpha=args.alpha,
                                   annotate=False, edge_color=args.edge_color,
                                   edge_width=args.edge_width)
    displays = draw_brains(fig, el, colours, ['x', 'z'],
                           (pad / W, pad / H, brain_w / W, 1 - 2 * pad / H),
                           args.brain_gap, node_args)
    fig.canvas.draw()
    check_inside(fig)
    fig.savefig(out_path, dpi=args.dpi, facecolor='white')
    plt.close(fig)
    for d in displays:
        d.close()
    logger.info('wrote %s (%.2f x %.2f in at %d dpi)', out_path, W, H, args.dpi)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--figsize', nargs=2, type=float, default=[8.5, 4.5],
                    metavar=('W', 'H'), help='Exact output size, inches.')
    ap.add_argument('--font-legend', type=float, default=8)
    ap.add_argument('--legend-gap', type=float, default=0.10,
                    help='Inches between the brains and the legend.')
    ap.add_argument('--brain-gap', type=float, default=0.01,
                    help='Gap between the two views, fraction of figure width.')
    ap.add_argument('--node-size', type=float, default=8)
    ap.add_argument('--alpha', type=float, default=0.9)
    ap.add_argument('--edge-color', default='0.25')
    ap.add_argument('--edge-width', type=float, default=0.2)
    ap.add_argument('--dpi', type=int, default=400)
    ap.add_argument('--label', default='anatomy')
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    io.warn_if_dirty()
    run_dir = Path(args.run_dir)
    prov = json.loads((run_dir / 'provenance.json').read_text())
    params = prov['params']
    el = electrodes(params, prov['subjects'])
    unplaced = sorted(set(el['region']) - {r for rois in LOBES.values() for r in rois})
    if unplaced:
        raise SystemExit(f'ROIs of the run with no lobe in LOBES: {unplaced}')
    colours = palette()

    out_dir = run_dir / SUBDIR / f'{args.label}_{datetime.now():%Y%m%d-%H%M%S}'
    out_dir.mkdir(parents=True)
    lobe_of = {r: lobe for lobe, rois in LOBES.items() for r in rois}
    table = (el.groupby('region').agg(n_electrodes=('channel', 'size'),
                                      n_subjects=('subject_id', 'nunique'))
             .reindex(list(lobe_of)).fillna(0).astype(int).reset_index())
    table.insert(0, 'lobe', table['region'].map(lobe_of))
    table['colour'] = table['region'].map(colours)
    logger.info('\n%s', table.to_string(index=False))
    subjects = sorted(el['subject_id'].unique())
    parents = [str(run_dir / 'provenance.json')]
    io.write_table(table, out_dir / 'colour_source.csv', script=SCRIPT, parents=parents,
                   subjects=subjects,
                   params={'lobes': LOBES, 'lobe_hues': LOBE_HUES,
                           'mix_step': MIX_STEP, 'mix_min_span': MIX_MIN_SPAN,
                           'mix_range': list(MIX_RANGE), 'dark_floor': DARK_FLOOR,
                           'roi_scheme': params['roi_scheme']})
    io.write_run_provenance(out_dir, script=SCRIPT,
                            params={**vars(args), 'source_run': str(run_dir),
                                    'roi_scheme': params['roi_scheme'],
                                    'n_electrodes': int(len(el))},
                            parents=parents, subjects=subjects)
    draw(el, colours, out_dir / 'fig_anatomy_glass.png', args)
    io.log_analysis(f'pain_change anatomy glass brain: {len(el)} electrodes, '
                    f'{len(subjects)} subjects, hue = lobe, shade = ROI', out_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
