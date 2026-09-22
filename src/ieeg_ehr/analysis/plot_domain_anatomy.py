#!/usr/bin/env python3
"""Where every electrode in the study is, coloured by domain and shaded by region.

    <domain run>/glass_brain/<label>_<timestamp>/fig_anatomy_glass.png

AN ANATOMY FIGURE, NOT AN EFFECT FIGURE. `plot_domain_glass` paints a model
result onto the electrodes; this paints only the ASSIGNMENT. Nothing here is a
statistic -- no slope, no p-value, no scale -- so it can be read before any
result and cannot be over-read into one.

Colour is the DOMAIN, shade is the REGION inside it. Four qualitative hues that
stay distinct for a colour-vision-deficient reader, one per processing domain,
with each domain's member ROIs spread across lightness of that hue. So Sensory
is four oranges (Thalamus, S1, S2/PO, pIns), and a reader can see both "this
is Sensory" at a glance and "this specific cluster is thalamus" on inspection.

WHY SHADES RATHER THAN 21 SEPARATE COLOURS. `plot_electrode_locations` uses
glasbey for exactly that and says in its own docstring that no 21-colour
categorical palette can be colourblind-safe. Nesting shade inside hue gives up
the ability to name a region by colour alone -- two shades of one orange are
genuinely hard to tell apart in a dense cluster -- and buys the thing that
matters more here: the DOMAIN, which is the model's unit, is legible
immediately and at any size.

THREE GREYS, FOR THREE DIFFERENT KINDS OF "not one of the four"
----------------------------------------------------------------
  Control domain      Auditory and Occipital. IN the model, as the reference
                      domain, and grey because it is the negative control --
                      the reference should recede, the same reason
                      `DOMAIN_COLOURS` greys it.
  Not in any domain   PCC, Hippocampus, Basal Ganglia, Parietal (other), MTL
                      (other), Lateral Temporal. Real regions the framework
                      does not claim, DROPPED from the model but present in
                      the region-level consistency map, and a large share of
                      the implant -- Lateral Temporal alone is the biggest
                      single ROI in the cohort. Drawing them is what stops
                      this reading as "the study only sampled these four
                      systems". `--domains-only` removes them.
  (absent)            White matter, ventricles, unlabelled. Not in the study
                      at all, so not drawn and not counted here.

Electrodes outside the MNI152 brain mask are dropped, so nothing floats
outside the outline. Output is versioned under `glass_brain/` exactly as
`plot_domain_glass` is, and never overwrites.

    python -m ieeg_ehr.analysis.plot_domain_anatomy --run-dir <domain run>

EXPLORATORY. Discovery cohort.
"""

import argparse
import colorsys
import logging
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ieeg_ehr import io
from ieeg_ehr.analysis import insula_ap
from ieeg_ehr.analysis.plot_domain_glass import (SUBDIR, electrode_coords,
                                                 inside_brain, load_run)
from ieeg_ehr.analysis.run_domain_model import domain_caveat
from ieeg_ehr.config import roi_schemes

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

SCRIPT = 'ieeg_ehr/analysis/plot_domain_anatomy.py'

#: nilearn's glass-brain projectors. Enumerated here so a bad `--display-mode`
#: fails at argparse with the valid list, rather than after the minutes of
#: electrode loading it takes to reach the first plot call. Note the absence of
#: 'lz' -- a left-only sagittal beside an axial is not one of them.
GLASS_MODES = ('l', 'r', 'x', 'y', 'z', 'lr', 'xz', 'yx', 'yz', 'lyr', 'lzr',
               'lyrz', 'lzry', 'ortho')

DISCLAIMER = 'EXPLORATORY -- discovery cohort. Anatomy only; no result here.'

#: One hue per processing domain, in the scheme's DISPLAY order. The first
#: three are ColorBrewer Dark2 (a qualitative set built to stay separable under
#: the common colour vision deficiencies); the fourth is a mid blue chosen to
#: sit apart from all three. Sabra's palette, 2026-09-21.
#:
#: Order matters and is the scheme's, not alphabetical: Sensory, Affective,
#: Cognitive, Modulatory -- the pathway order every other domain figure uses,
#: so a domain keeps its colour across the whole project's figures.
DOMAIN_HUES = ('#d95f02', '#1b9e77', '#7570b3', '#447eae')

#: EVERYTHING THAT IS NOT ONE OF THE FOUR DOMAINS, in one flat grey under one
#: legend entry. Two groups were tried first -- the Control domain in mid-greys
#: and the unassigned regions in light greys -- and collapsed at Sabra's
#: request (2026-09-22): as a figure it is one statement, "sampled, but not one
#: of the four systems", and splitting it into two grey families plus eight
#: legend lines spent most of the legend on the part of the figure that is
#: context rather than content.
#:
#: THE DISTINCTION IS NOT LOST, it moves to `colour_source.csv`, which keeps
#: one row per region with its own domain label -- so Control (the model's
#: reference domain, Auditory + Occipital) is still separable from the six
#: regions no domain claims, for anyone who needs it.
#:
#: The tone is deliberately neutral and mid: these ~1,900 contacts outnumber
#: any single domain, so a darker grey would bury the four coloured systems the
#: figure is about, and a lighter one was hard to tell from the palest domain
#: shade.
OTHER_GREY = '#b4b4b4'
OTHER_LABEL = 'Other'

#: Width of one character as a fraction of the font size, for DejaVu Sans.
#: Used ONLY to decide which domain labels fit their span, where being off by
#: a little just moves a label to the staggered row. The bottom margin is
#: measured from the rendered labels instead -- estimating it was wrong twice.
CHAR_EM = 0.34

#: (horizontal, vertical) extent in mm of each glass-brain view, from the
#: MNI152 bounding box the projectors use: x spans ~156 mm, y ~188 mm, z
#: ~156 mm.
#:
#: THIS IS WHY THE BRAINS WERE DIFFERENT SIZES. nilearn scales each view to
#: fill whatever axes rect it is handed, independently -- so equal-width rects
#: render a sagittal (188 mm across) and an axial (156 mm across) at different
#: mm-per-inch, and the sagittal comes out visibly smaller. Giving each view a
#: rect proportional to its OWN extent puts every view on one scale, which is
#: the only way two brains side by side are comparable at a glance.
BRAIN_EXTENT_MM = {'x': (188, 156), 'l': (188, 156), 'r': (188, 156),
                   'y': (156, 156), 'z': (156, 188)}


#: Lightness range for the shades within a domain, as a multiple of the base
#: colour's own lightness, and the hard ceiling on the result.
#:
#: NARROWED FROM (0.62, 1.42) AND A 0.88 CEILING on 2026-09-22: the palest
#: shade came out close enough to the grey of the non-domain electrodes that a
#: reader could not tell a pale S2/PO orange from a grey Lateral Temporal at
#: marker size. A shade only has to be distinguishable from the OTHER SHADES OF
#: ITS OWN HUE -- four of them, adjacent in the legend -- whereas it has to be
#: unmistakably chromatic against grey across the whole brain. So the range is
#: tight and the ceiling low, and saturation is pushed UP as lightness rises to
#: stop the light end washing out.
SHADE_RANGE = (0.66, 1.24)
SHADE_MAX_LIGHTNESS = 0.68


def shades(base, n, lo=SHADE_RANGE[0], hi=SHADE_RANGE[1]):
    """`n` shades of `base`, varying LIGHTNESS, darkest first.

    Hue is held fixed so every shade still reads as the same domain colour --
    that is the whole point. Saturation is held or RAISED, never lowered:
    lightening a colour in HLS desaturates it toward white, which is exactly
    how the pale end stopped being distinguishable from grey.
    """
    r, g, b = matplotlib.colors.to_rgb(base)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    if n <= 1:
        return [matplotlib.colors.to_hex((r, g, b))]
    out = []
    for f in np.linspace(lo, hi, n):
        ll = float(np.clip(l * f, 0.18, SHADE_MAX_LIGHTNESS))
        ss = float(np.clip(s * (1.0 + 0.45 * max(0.0, f - 1.0)), 0.0, 1.0))
        out.append(matplotlib.colors.to_hex(colorsys.hls_to_rgb(h, ll, ss)))
    return out


def build_palette(members, display, base_order, unassigned):
    """{region: hex} and {domain: [regions]}, shades nested inside hues.

    `display` is the DOMAIN order (hue assignment) and `base_order` the region
    order inside each domain, both taken from the scheme so a region's colour
    is a property of the scheme rather than of whatever order the data arrived
    in -- two runs of this script must not recolour the brain.
    """
    colours, groups = {}, {}
    hues = list(DOMAIN_HUES)
    hue_for = {}
    non_control = [d for d in display if d != 'Control']
    if len(non_control) > len(hues):
        raise SystemExit(f'{len(non_control)} non-control domains but only '
                         f'{len(hues)} hues in DOMAIN_HUES')
    for i, dom in enumerate(non_control):
        hue_for[dom] = hues[i]

    other = []
    for dom in display:
        regions = [r for r in base_order if r in members.get(dom, ())]
        if dom == 'Control':
            other.extend(regions)          # folded into Other, see OTHER_GREY
            continue
        groups[dom] = regions
        for r, c in zip(regions, shades(hue_for[dom], len(regions))):
            colours[r] = c

    other.extend(unassigned)
    if other:
        # ONE entry, one tone. `groups` keys the legend, so an empty region
        # list here is what makes it a single line rather than eight.
        groups[OTHER_LABEL] = []
        for r in other:
            colours[r] = OTHER_GREY
    return colours, groups, hue_for, other


def assign_regions(subjects, scheme, insula_threshold, epoch_minutes):
    """Electrodes with their base-scheme ROI, insula split applied."""
    el = electrode_coords(subjects, epoch_minutes)
    el['region'] = [roi_schemes.region_for_dk_label(
        lbl, scheme, include_coordinate_parents=True) for lbl in el['dk_anode']]
    coord = roi_schemes.coordinate_regions(scheme)
    if coord:
        contacts, _ = insula_ap.load_insula_contacts(
            subjects, epoch_minutes=epoch_minutes)
        contacts, _ = insula_ap.median_split(contacts,
                                             threshold=insula_threshold)
        look = insula_ap.channel_lookup(contacts)
        el['region'] = [look.get(s, {}).get(c) if r in coord else r
                        for s, c, r in zip(el['subject_id'], el['channel'],
                                           el['region'])]
    n0 = len(el)
    el = el[el['region'].notna()].reset_index(drop=True)
    logger.info('%d of %d electrode(s) carry a region in %r', len(el), n0,
                scheme)
    return el, n0


def glass(el, colours, groups, other, mode, out_path, title, args, caveat,
          counts):
    """The brain. One view per axes rect, laid out with a controllable gap.

    ONE `plot_markers` CALL PER (VIEW, REGION), not one per region with a
    multi-view display_mode. Handing nilearn 'xz' lets IT lay the two views
    out, and its spacing is generous -- on a 7-inch poster panel the two brains
    sat far enough apart to waste most of the width. Splitting the mode into
    single views and placing each in its own rect makes the gap a parameter
    (`--brain-gap`, negative to overlap the bounding boxes) instead of
    nilearn's default.

    Regions are drawn LARGEST GROUP FIRST within each view, so the biggest go
    down first and the small ones land on top: Lateral Temporal is 757 contacts
    and would otherwise bury Auditory's 31 entirely.
    """
    from nilearn import plotting

    views = ['x', 'y', 'z'] if mode == 'ortho' else list(mode)
    order = sorted(colours, key=lambda r: -int((el['region'] == r).sum()))

    fig = plt.figure(figsize=tuple(args.figsize))
    fw, fh = args.figsize
    brain_w, gap = args.brain_width, args.brain_gap

    # ONE SCALE FOR EVERY VIEW. Widths are proportional to each view's own
    # anatomical extent, and the common mm-per-inch is whichever of the width
    # and height budgets binds first, so the brains come out the same size
    # instead of each being stretched to fill an equal box.
    ext = [BRAIN_EXTENT_MM.get(v, (170, 170)) for v in views]
    n = len(views)
    avail_w_in = (brain_w - gap * (n - 1)) * fw
    avail_h_in = args.brain_height * fh
    mm_per_in = min(avail_w_in / sum(h for h, _ in ext),
                    avail_h_in / max(v for _, v in ext))
    widths = [h * mm_per_in / fw for h, _ in ext]
    heights = [v * mm_per_in / fh for _, v in ext]
    # Centred on the band, and centred vertically view by view, so a shorter
    # view sits on the same midline rather than on the same baseline.
    x = (brain_w - sum(widths) - gap * (n - 1)) / 2
    mid = args.brain_bottom + args.brain_height / 2

    displays = []
    for i, view in enumerate(views):
        x0, each, height = x, widths[i], heights[i]
        bottom = mid - height / 2
        x += widths[i] + gap
        d = None
        for region in order:
            rows = el[el['region'] == region]
            if rows.empty:
                continue
            coords = rows[['mni_x', 'mni_y', 'mni_z']].to_numpy(dtype=float)
            d = plotting.plot_markers(
                node_values=np.ones(len(coords)), node_coords=coords,
                node_size=args.node_size, display_mode=view, colorbar=False,
                figure=fig, axes=(x0, bottom, each, height),
                alpha=args.alpha, annotate=args.annotate,
                node_cmap=matplotlib.colors.ListedColormap([colours[region]]),
                node_vmin=0, node_vmax=2,
                node_kwargs={'edgecolors': args.edge_color,
                             'linewidths': args.edge_width})
        displays.append(d)

    # -- legend on the RIGHT ------------------------------------------------
    # COUNTS ARE NOT IN THESE LABELS. They were, and at 14 pt they doubled the
    # width of every entry for information the bar figure beside it already
    # carries per region, far more legibly than a parenthesis.
    lax = fig.add_axes((brain_w + 0.01, 0.0, 1.0 - brain_w - 0.015, 1.0))
    handles, labels = [], []
    blank = mpatches.Patch(alpha=0, linewidth=0)
    for dom in groups:
        if not groups[dom]:
            handles.append(mpatches.Patch(facecolor=OTHER_GREY,
                                          edgecolor='0.35', linewidth=0.4))
            labels.append(f'$\\bf{{{dom}}}$')
            continue
        handles.append(blank)
        labels.append(f'$\\bf{{{dom.replace(" ", chr(92) + " ")}}}$')
        for r in groups[dom]:
            handles.append(mpatches.Patch(facecolor=colours[r],
                                          edgecolor='0.35', linewidth=0.4))
            labels.append(r)
    lax.legend(handles=handles, labels=labels, loc='center left',
               ncol=args.legend_cols, frameon=False,
               fontsize=args.font_legend, handlelength=1.1, handleheight=1.1,
               columnspacing=1.0, labelspacing=0.30, borderpad=0.0,
               handletextpad=0.5)
    lax.axis('off')

    if title:
        fig.suptitle(title, fontsize=args.font_title, y=0.995)
    # bbox_inches='tight' is BACK, now that an exact output size is no longer
    # the requirement. It trims the dead margin around the brains and, more
    # to the point, guarantees the legend cannot be cut off -- the failure it
    # was removed to avoid (a figure silently saving larger than declared)
    # only mattered while the font sizes were pinned to a physical size.
    fig.savefig(out_path, dpi=args.dpi, bbox_inches='tight', pad_inches=0.04)
    plt.close(fig)
    for d in displays:
        if d is not None:
            d.close()
    logger.info('wrote %s', out_path.name)


def bars(el, colours, groups, other, n_cohort, out_path, title, args, caveat):
    """Coverage as two stacked bars sharing one x axis: reach, then depth.

    THE TWO PANELS ANSWER DIFFERENT QUESTIONS AND THEY DISAGREE, which is the
    reason to stack them rather than pick one.

      top     How many PATIENTS have at least one electrode there. This is
              what decides whether a region-level estimate is a group claim at
              all -- a region sampled in 10 of 51 patients cannot support one
              however many contacts those 10 contributed.
      bottom  How many ELECTRODES. This is what drives the within-subject
              precision of that estimate.

    A region can be wide and shallow (Hippocampus: 43 patients, 296 contacts)
    or narrow and deep (Occipital: 10 patients, 54 contacts), and the glass
    brain shows only the second -- a dense cluster there could be one patient's
    entire depth electrode. Reading the two panels together is the only way to
    tell those apart, and it is why they share an x axis.

    Colour is the SAME hex the glass brain used, taken from the same palette
    dict rather than recomputed, so a bar and its dots cannot drift apart.

    X LABELS ARE ANGLED. Vertical was tried when the figure had to be 7
    inches wide and 22 names could not fit any other way; on a wider canvas
    45 degrees is both easier to read and shorter, because a rotated label's
    footprint is its length times sin(angle).
    """
    order, boundaries, labels_dom = [], [], []
    for dom in groups:
        regions = groups[dom] or [r for r in other if r in set(el['region'])]
        regions = [r for r in regions if (el['region'] == r).any()]
        if not regions:
            continue
        if order:
            boundaries.append(len(order) - 0.5)
        labels_dom.append((dom, len(order), len(order) + len(regions) - 1))
        order.extend(regions)

    n_el = el.groupby('region').size().reindex(order).fillna(0).to_numpy()
    n_sub = (el.groupby('region')['subject_id'].nunique()
             .reindex(order).fillna(0).to_numpy())
    pct = 100.0 * n_sub / max(n_cohort, 1)
    cols = [colours[r] for r in order]
    x = np.arange(len(order))

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=tuple(args.bar_figsize))
    # EXPLICIT, NOT tight_layout: the x labels are region names set vertically
    # and tight_layout does not reserve for them reliably against a fixed
    # figure height. It also warns here, because the domain names are drawn in
    # axes-transform coordinates.
    fig.subplots_adjust(left=args.bar_left, right=0.995,
                        top=args.bar_top, bottom=0.30, hspace=0.10)

    # Y LABELS MUST FIT THE AXES HEIGHT, because a rotated label's LENGTH runs
    # along it. At 7x4 with 22 vertical region names each panel is only ~1.2
    # inches tall, and "Patients with >=1 electrode (%)" at 14 pt is 3 inches
    # of text -- the first version had the two labels printed through each
    # other. They are arguments so a taller figure can carry the full wording.
    for ax, vals, ylab, is_pct in (
            (axes[0], pct, args.ylabel_top, True),
            (axes[1], n_el, args.ylabel_bottom, False)):
        ax.bar(x, vals, color=cols, edgecolor='0.3', linewidth=0.5, width=0.82)
        # The unit lives in the TICK labels, not the axis label: a rotated
        # axis label's length runs along the axes height, and at 7x4 each
        # panel is under an inch tall, so "(%)" is three characters that do
        # not fit anywhere useful.
        ax.set_ylabel(ylab, fontsize=args.font_label)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(axis='y', labelsize=args.font_tick)
        for b in boundaries:
            ax.axvline(b, color='0.82', lw=0.9, zorder=0)
        for xi, v in zip(x, vals):
            ax.text(xi, v, f'{v:.0f}', ha='center', va='bottom',
                    fontsize=args.font_value, color='0.25',
                    rotation=args.value_rotation)
        ax.margins(y=0.20)
    axes[0].set_ylim(0, 108)
    axes[0].set_yticks([0, 50, 100])
    if args.ylabel_top_ticks_percent:
        axes[0].set_yticklabels(['0', '50', '100%'])

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(order, rotation=args.bar_rotation, ha='right',
                            rotation_mode='anchor', fontsize=args.font_tick)
    for t, r in zip(axes[1].get_xticklabels(), order):
        t.set_color('0.35' if r in other else colours[r])
    axes[1].set_xlim(-0.7, len(order) - 0.3)

    # MEASURE THE LABELS, DO NOT ESTIMATE THEM. Two guesses at "how tall is
    # 'Lateral Temporal' set vertically at 14 pt" were wrong in both
    # directions -- the first left an inch of white space below the names, the
    # second clipped them. Rendering once and asking the renderer is exact,
    # costs one draw, and keeps working at any figure size or font size.
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    h_px = max((t.get_window_extent(rend).height
                for t in axes[1].get_xticklabels()), default=0.0)
    need = h_px / fig.dpi / fig.get_size_inches()[1]
    bottom = (args.bar_bottom if args.bar_bottom is not None
              else min(0.62, need + 0.05))
    fig.subplots_adjust(bottom=bottom)

    # Domain names above the top panel, spanning their own regions. A domain
    # whose span cannot hold its name at this font size is STAGGERED onto a
    # second row rather than dropped: with one region, Modulatory has a
    # quarter-inch of span and a one-inch name, and it is the domain a reader
    # is least likely to guess from colour alone.
    #
    width, alt = fig.get_size_inches()[0], False
    span_in = (width * (0.995 - args.bar_left)) / max(len(order), 1)
    char_in = CHAR_EM * args.font_label / 72.0
    for dom, i0, i1 in labels_dom:
        fits = (i1 - i0 + 1) * span_in >= char_in * len(dom)
        y = 1.04 if fits else 1.19
        if not fits:
            alt = True
        axes[0].text((i0 + i1) / 2, y, dom,
                     transform=axes[0].get_xaxis_transform(), ha='center',
                     va='bottom', fontsize=args.font_label,
                     color=('0.35' if dom == OTHER_LABEL
                            else colours[groups[dom][0]]))
    if alt:
        # The staggered row needs headroom the default `top` does not leave.
        fig.subplots_adjust(top=min(args.bar_top, 0.84))

    if title:
        fig.suptitle(title, fontsize=args.font_title, y=0.995)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches='tight', pad_inches=0.06)
    plt.close(fig)
    w, h = fig.get_size_inches()
    logger.info('wrote %s  (%.2f x %.2f in at %d dpi)', out_path.name, w, h,
                args.dpi)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-dir', required=True,
                    help='A domain-model run; its scheme, insula threshold and '
                         'subject list are taken from its provenance.')
    ap.add_argument('--domains-only', action='store_true',
                    help='Drop the regions no domain claims. Off by default: '
                         'they are a large share of the implant and omitting '
                         'them misrepresents the coverage.')
    ap.add_argument('--display-modes', nargs='+', default=['xz', 'lyrz'],
                    choices=GLASS_MODES,
                    help="Which nilearn views. 'xz' (default) is ONE sagittal "
                         'plus one axial: the sagittal places a region on the '
                         'anterior-posterior axis and the axial shows the '
                         'hemispheric spread, which at this density is all '
                         "the information the four-panel 'lyrz' carried. Note "
                         "'x' projects BOTH hemispheres into one sagittal, "
                         "unlike 'l'/'r' which split them -- so no electrode "
                         'is dropped from the view. There is no \'lz\': '
                         'nilearn does not offer a left-only sagittal beside '
                         'an axial.')
    ap.add_argument('--annotate', action='store_true',
                    help="Draw nilearn's L/R and coordinate labels. Off by "
                         'default: they crowd a dense figure and the axial '
                         'view already orients the reader.')
    # -- POSTER GEOMETRY. Defaults are the size these are printed at, so the
    # -- figure that gets saved is the figure that gets mounted: font sizes
    # -- only mean anything relative to the physical size, and a figure
    # -- designed at 19 inches and scaled to 7 has 5 pt labels on the wall.
    ap.add_argument('--figsize', nargs=2, type=float, default=[13.0, 6.0],
                    metavar=('W', 'H'),
                    help='Glass brain figure, inches. BIG on purpose: type '
                         'size is only legible RELATIVE to the figure, and at '
                         '7 inches a 14 pt legend is 2%% of the width, which '
                         'squeezes the brains to nothing. A larger canvas '
                         'with smaller type reads better at any print size.')
    ap.add_argument('--bar-figsize', nargs=2, type=float, default=[13.0, 6.5],
                    metavar=('W', 'H'), help='Bar figure, inches.')
    ap.add_argument('--brain-width', type=float, default=0.74,
                    help='Fraction of the glass figure the brains occupy; the '
                         'rest is the legend, on the right.')
    ap.add_argument('--brain-gap', type=float, default=-0.03,
                    help='Gap between adjacent brain views, as a fraction of '
                         'figure width. NEGATIVE overlaps their bounding '
                         'boxes, which is usually what is wanted: a glass '
                         'brain does not fill its box, so touching boxes '
                         'still leave a visible gutter.')
    ap.add_argument('--brain-bottom', type=float, default=0.02)
    ap.add_argument('--brain-height', type=float, default=0.96)
    ap.add_argument('--legend-cols', type=int, default=2)
    ap.add_argument('--bar-rotation', type=float, default=45,
                    help='Angle of the region names under the bars.')
    ap.add_argument('--value-rotation', type=float, default=0,
                    help='Angle of the per-bar value labels. 0 reads best '
                         'once the bars are wide enough to hold a number.')
    ap.add_argument('--bar-left', type=float, default=0.085)
    ap.add_argument('--bar-top', type=float, default=0.90)
    ap.add_argument('--ylabel-top', default='Patients with\n\u22651 electrode (%)',
                    help='A rotated label\'s LENGTH runs along the axes '
                         'height, which at 7x4 is about 1.2 inches -- the '
                         'full "Patients with >=1 electrode (%%)" is 3 inches '
                         'of text at 14 pt and collides with the panel below. '
                         'Give the full wording on a taller figure.')
    ap.add_argument('--ylabel-bottom', default='Number of\nelectrodes')
    ap.add_argument('--ylabel-top-ticks-percent', action='store_true',
                    default=False)
    ap.add_argument('--no-percent-ticks', dest='ylabel_top_ticks_percent',
                    action='store_false')
    ap.add_argument('--bar-figsize-alt', nargs=2, type=float,
                    default=[0.0, 0.0], metavar=('W', 'H'),
                    help='A SECOND bar figure at this size, written as '
                         'fig_anatomy_bars_tall.png. 22 grouped bars with '
                         '14 pt vertical region names do not really fit 7x4 '
                         '-- the names alone take ~40%% of the height -- so '
                         'the taller version is emitted beside it rather '
                         'than instead of it. Pass 0 0 to skip.')
    ap.add_argument('--bar-bottom', type=float, default=None,
                    help='Fraction of the height reserved for the vertical '
                         'region names. Computed from the longest name and '
                         'the tick font size when not given, which is almost '
                         'always what you want -- a fixed value either clips '
                         'the names or leaves an inch of white below them.')
    # -- FONTS. Sabra's poster floor: nothing below 14 pt for a label, nothing
    # -- below 10 pt for a per-bar value.
    ap.add_argument('--font-label', type=float, default=15)
    ap.add_argument('--font-tick', type=float, default=13)
    ap.add_argument('--font-legend', type=float, default=14)
    ap.add_argument('--font-value', type=float, default=10)
    ap.add_argument('--font-title', type=float, default=16)
    ap.add_argument('--titles', action='store_true',
                    help='Draw figure titles. Off by default: a poster panel '
                         'carries its own heading, and a duplicate inside the '
                         'image wastes the height the labels need.')
    ap.add_argument('--dpi', type=int, default=400)
    ap.add_argument('--node-size', type=float, default=17)
    ap.add_argument('--alpha', type=float, default=0.9)
    ap.add_argument('--edge-color', default='0.25')
    ap.add_argument('--edge-width', type=float, default=0.3)
    ap.add_argument('--brain-dilation-mm', type=float, default=0.0)
    ap.add_argument('--label', default='anatomy')
    args = ap.parse_args()

    io.warn_if_dirty()
    run_dir = Path(args.run_dir)
    params, subjects, _, _ = load_run(run_dir)
    scheme = params['roi_scheme']
    spec = roi_schemes.domain_scheme(scheme)
    base_order = roi_schemes.roi_regions(spec['base'])

    el, n_all = assign_regions(subjects, spec['base'],
                               params.get('insula_threshold'),
                               params.get('epoch_minutes'))
    counts = {'n_electrodes_in_channel_meta': int(n_all),
              'n_no_region': int(n_all - len(el))}

    unassigned = [r for r in base_order if r not in spec['roi_to_domain']]
    if args.domains_only:
        el = el[~el['region'].isin(unassigned)].reset_index(drop=True)
        unassigned = []
    colours, groups, hue_for, other = build_palette(
        spec['members'], spec['display'], base_order, unassigned)

    coords = el[['mni_x', 'mni_y', 'mni_z']].to_numpy(dtype=float)
    keep = inside_brain(coords, args.brain_dilation_mm)
    counts['n_outside_brain_dropped'] = int((~keep).sum())
    if (~keep).any():
        logger.warning('dropped %d electrode(s) outside the MNI152 brain mask',
                       int((~keep).sum()))
    el = el[keep].reset_index(drop=True)
    counts['n_plotted'] = int(len(el))

    logger.info('domain hues: %s', hue_for)
    per_region = (el.groupby('region')
                  .agg(n_electrodes=('channel', 'size'),
                       n_subjects=('subject_id', 'nunique')))
    logger.info('\n%s', per_region.reindex(
        [r for r in base_order if r in per_region.index]).to_string())

    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    out_dir = run_dir / SUBDIR / f'{args.label}_{stamp}'
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info('versioned output dir (never overwrites): %s', out_dir)

    dom_of = dict(spec['roi_to_domain'])
    table = (per_region.reset_index()
             .assign(domain=lambda d: d['region'].map(dom_of)
                     .fillna('not in any domain'),
                     colour=lambda d: d['region'].map(colours)))
    table['_o'] = table['region'].map({r: i for i, r in enumerate(base_order)})
    table = table.sort_values('_o').drop(columns='_o')
    io.write_table(table[['domain', 'region', 'colour', 'n_electrodes',
                          'n_subjects']],
                   out_dir / 'colour_source.csv', script=SCRIPT,
                   parents=[str(run_dir / 'provenance.json')],
                   params={'domain_hues': list(DOMAIN_HUES),
                           'other_grey': OTHER_GREY,
                           'shade_range': list(SHADE_RANGE),
                           'scheme': scheme, 'base_scheme': spec['base']},
                   extra={'reading': 'the exact hex each region was drawn in; '
                                     'hue is the domain, lightness the region'})
    io.write_table(el, out_dir / 'anatomy_electrodes.csv', script=SCRIPT,
                   parents=[str(run_dir / 'provenance.json')],
                   subjects=sorted(el['subject_id'].unique()),
                   params={'base_scheme': spec['base'],
                           'brain_dilation_mm': args.brain_dilation_mm},
                   extra={'electrode_counts': counts})

    io.write_run_provenance(
        out_dir, script=SCRIPT,
        params={**vars(args), 'scheme': scheme, 'base_scheme': spec['base'],
                'source_run': str(run_dir), 'electrode_counts': counts,
                'figures': [f'fig_anatomy_glass_{m}.png'
                            for m in args.display_modes]
                           + ['fig_anatomy_bars.png',
                              'fig_anatomy_bars_tall.png'],
                'insula_threshold': params.get('insula_threshold')},
        parents=[str(run_dir / 'provenance.json')],
        subjects=sorted(el['subject_id'].unique()),
        extra={'status': DISCLAIMER, 'domain_caveat': domain_caveat(scheme),
               'palette': {'domain_hue': hue_for,
                           'region_colour': {r: colours[r] for r in colours},
                           'other_regions': sorted(other),
                           'rule': 'hue = domain, lightness = region within '
                                   'it. Everything that is not one of the four '
                                   'domains -- the Control domain AND the '
                                   'regions no domain claims -- is one flat '
                                   'grey under a single "Other" legend entry; '
                                   'colour_source.csv keeps them separable by '
                                   'their own domain label.'},
               'what_this_is': 'ANATOMY ONLY. No slope, no p-value, no scale. '
                               'It shows which electrodes the study has and '
                               'which domain each belongs to.'})

    io.log_analysis(
        f'glass brain: all {len(el)} study electrodes coloured by processing '
        'domain, shaded by region (ANATOMY, no result)', out_dir)
    # ONE FILE PER VIEW SET, named by the mode, so a two-panel version for a
    # slide and a four-panel version to crop from can coexist in one folder
    # without either being "the" file that the other overwrote.
    # NO TITLE AND NO CAPTION on either figure by default. They are poster
    # panels: the poster supplies the heading, and a caption at a size that
    # would fit here would be below the readable floor the rest of the figure
    # is built to. Everything a caption would have said -- the insula split,
    # the dropped-electrode counts, what grey means, the domain caveat -- is
    # in this folder's provenance.json and colour_source.csv.
    gtitle = (f'Electrode coverage by processing domain — {len(el)} bipolar '
              f'pairs, {el["subject_id"].nunique()} subjects'
              if args.titles else None)
    btitle = ('Coverage by region — how many patients, and how many electrodes'
              if args.titles else None)
    for mode in args.display_modes:
        glass(el, colours, groups, other, mode,
              out_dir / f'fig_anatomy_glass_{mode}.png', gtitle,
              args, domain_caveat(scheme), counts)
    bars(el, colours, groups, other, len(subjects),
         out_dir / 'fig_anatomy_bars.png', btitle,
         args, domain_caveat(scheme))
    if all(v > 0 for v in args.bar_figsize_alt):
        import copy
        alt = copy.copy(args)
        alt.bar_figsize = args.bar_figsize_alt
        # A taller panel CAN hold the full wording, so it gets it.
        alt.ylabel_top = 'Patients with\n\u22651 electrode (%)'
        alt.ylabel_bottom = 'Number of\nelectrodes'
        alt.bar_bottom = None      # recomputed for the taller figure
        bars(el, colours, groups, other, len(subjects),
             out_dir / 'fig_anatomy_bars_tall.png', btitle,
             alt, domain_caveat(scheme))
    logger.info('done -> %s', out_dir)


if __name__ == '__main__':
    main()
