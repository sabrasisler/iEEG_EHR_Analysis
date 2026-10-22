"""Anatomy glass brain of a pain_change run: hue = lobe group, shade = ROI within it.

    python -m ieeg_ehr.analysis.plot_pain_change_anatomy --run-dir <pain_change run>

Writes `<run>/glass_brain/anatomy_<timestamp>/fig_anatomy_glass.{png,svg}`: one
sagittal and one axial view on one scale (`plot_domain_anatomy.draw_brains`),
with the legend on the right, at exactly `--figsize` inches. Every electrode the
run's ROI map assigns to a region is drawn at the MNI midpoint of its bipolar
pair, so the brain shows the fit's own subjects and membership
(`plot_pain_change_glass.electrodes`). ANATOMY ONLY: no slope, no p-value.

GROUPS maps each ROI of the hcpexv5 scheme to one group, and GROUP_COLOURS gives
each group its hue (set by the user 2026-10-22): the four DOMAIN_HUES of
`plot_domain_anatomy` plus Dark2 pink. Shades within a group come from
`plot_domain_anatomy.shades`, the same rule as the domain anatomy figure. Somatomotor ROIs sit in the
lobe that holds them: M1 and Premotor in Frontal, S1 and SII in Parietal. The
script refuses an ROI of the run that GROUPS does not place.
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

from ieeg_ehr import io
from ieeg_ehr.analysis.plot_domain_anatomy import (DOMAIN_HUES, OTHER_GREY, check_inside,
                                                   draw_brains, shades)
from ieeg_ehr.analysis.plot_domain_glass import SUBDIR
from ieeg_ehr.analysis.plot_pain_change_glass import electrodes

logger = logging.getLogger(__name__)

SCRIPT = 'ieeg_ehr/analysis/plot_pain_change_anatomy.py'

#: Group -> its ROIs in the hcpexv5 scheme, in legend order.
GROUPS = {
    'Frontal': ('dPFC', 'vlPFC', 'OFC', 'vmPFC', 'Frontal operculum', 'Premotor', 'M1'),
    'Parietal': ('S1', 'SII', 'SPL + IPS', 'Supramarginal', 'Angular', 'Precuneus'),
    'Insula & cingulate': ('aIns', 'pIns', 'ACC', 'MCC', 'PCC'),
    'Temporal': ('STG', 'MTG', 'ITG/fusiform', 'Medial temporal lobe', 'Amygdala',
                 'Hippocampus'),
    'Subcortical': ('Ventral thalamus', 'Dorsal/medial thalamus', 'Basal ganglia'),
    'Other': ('Occipital',),
}
SENSORY, AFFECTIVE, COGNITIVE, MODULATORY = DOMAIN_HUES
GROUP_COLOURS = {
    'Frontal': COGNITIVE, 'Parietal': SENSORY, 'Insula & cingulate': AFFECTIVE,
    'Temporal': '#e7298a', 'Subcortical': MODULATORY, 'Other': OTHER_GREY,
}


def palette():
    """{ROI: hex}, each group's ROIs spread across shades of its hue, Other flat grey."""
    return {roi: c for g, rois in GROUPS.items()
            for roi, c in zip(rois, [OTHER_GREY] * len(rois) if g == 'Other'
                              else shades(GROUP_COLOURS[g], len(rois)))}


def legend_entries(colours):
    """One list of (handle, label) per group: a header over the group's ROIs.

    A one-ROI group is a single header entry in its colour. `draw` bolds the headers.
    """
    blank = mpatches.Patch(alpha=0, linewidth=0)
    out = []
    for g, rois in GROUPS.items():
        swatch = [(mpatches.Patch(facecolor=colours[r], edgecolor='0.35', linewidth=0.4), r)
                  for r in rois]
        out.append([(swatch[0][0], g)] if len(rois) == 1 else [(blank, g)] + swatch)
    return out


def two_columns(groups):
    """Entries for a 2-column legend that splits only between groups.

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
    for t in leg.get_texts():
        if t.get_text() in GROUPS:
            t.set_fontweight('bold')
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
    fig.savefig(out_path.with_suffix('.svg'), facecolor='white')
    plt.close(fig)
    for d in displays:
        d.close()
    logger.info('wrote %s and .svg (%.2f x %.2f in at %d dpi)', out_path, W, H, args.dpi)


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
    group_of = {r: g for g, rois in GROUPS.items() for r in rois}
    unplaced = sorted(set(el['region']) - set(group_of))
    if unplaced:
        raise SystemExit(f'ROIs of the run with no group in GROUPS: {unplaced}')
    colours = palette()

    out_dir = run_dir / SUBDIR / f'{args.label}_{datetime.now():%Y%m%d-%H%M%S}'
    out_dir.mkdir(parents=True)
    table = (el.groupby('region').agg(n_electrodes=('channel', 'size'),
                                      n_subjects=('subject_id', 'nunique'))
             .reindex(list(group_of)).fillna(0).astype(int).reset_index())
    table.insert(0, 'group', table['region'].map(group_of))
    table['colour'] = table['region'].map(colours)
    logger.info('\n%s', table.to_string(index=False))
    subjects = sorted(el['subject_id'].unique())
    parents = [str(run_dir / 'provenance.json')]
    io.write_table(table, out_dir / 'colour_source.csv', script=SCRIPT, parents=parents,
                   subjects=subjects,
                   params={'groups': GROUPS, 'group_colours': GROUP_COLOURS,
                           'roi_scheme': params['roi_scheme']})
    io.write_run_provenance(out_dir, script=SCRIPT,
                            params={**vars(args), 'source_run': str(run_dir),
                                    'roi_scheme': params['roi_scheme'],
                                    'n_electrodes': int(len(el))},
                            parents=parents, subjects=subjects)
    draw(el, colours, out_dir / 'fig_anatomy_glass.png', args)
    io.log_analysis(f'pain_change anatomy glass brain: {len(el)} electrodes, '
                    f'{len(subjects)} subjects, hue = lobe group, shade = ROI', out_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
