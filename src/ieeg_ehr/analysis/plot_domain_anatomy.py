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
    """The brain. One `plot_markers` call per region, drawn largest group first.

    `plot_markers` takes a colormap rather than per-point colours, so one call
    per colour is the only way to get 21. DRAWN LARGEST-FIRST so the biggest
    regions go down first and the small ones land on top: Lateral Temporal is
    821 contacts and would otherwise bury Auditory's 32 entirely.
    """
    from nilearn import plotting

    order = sorted(colours, key=lambda r: -int((el['region'] == r).sum()))
    # HEIGHT IS FIXED AND THE BRAINS GET A FIXED SHARE OF IT, rather than the
    # figure being sized to the brains alone. The legend is five columns deep
    # whatever the view count, and at two panels the earlier layout let it
    # collide with the caption -- a narrower figure does not make the legend
    # shorter.
    n_panels = 3 if mode == 'ortho' else len(mode)
    fig = plt.figure(figsize=(5.3 * n_panels + 2.2, 8.6))
    display = None
    for region in order:
        rows = el[el['region'] == region]
        if rows.empty:
            continue
        coords = rows[['mni_x', 'mni_y', 'mni_z']].to_numpy(dtype=float)
        display = plotting.plot_markers(
            node_values=np.ones(len(coords)), node_coords=coords,
            node_size=args.node_size, display_mode=mode,
            colorbar=False, figure=fig, axes=(0.01, 0.42, 0.98, 0.54),
            alpha=args.alpha, annotate=args.annotate,
            node_cmap=matplotlib.colors.ListedColormap([colours[region]]),
            node_vmin=0, node_vmax=2,
            node_kwargs={'edgecolors': args.edge_color,
                         'linewidths': args.edge_width})

    # -- legend: ONE COLUMN PER DOMAIN, so the grouping is visible as layout
    # rather than only as hue. matplotlib fills a legend column-major, so the
    # lists are padded to equal length and the blanks are invisible handles.
    lax = fig.add_axes((0.02, 0.14, 0.96, 0.26))
    cols = list(groups)
    depth = max(len(groups[d]) for d in cols) + 1
    handles, labels = [], []
    blank = mpatches.Patch(alpha=0, linewidth=0)
    for dom in cols:
        if not groups[dom]:
            # A group with no member list is a SINGLE swatch -- `Other`, whose
            # eight regions share one tone and would otherwise spend eight
            # legend lines saying the same thing.
            rows = el[el['region'].isin(other)]
            handles.append(mpatches.Patch(facecolor=OTHER_GREY,
                                          edgecolor='0.35', linewidth=0.4))
            labels.append(f'$\\bf{{{dom}}}$  ({len(rows)}, '
                          f'{rows["subject_id"].nunique()} subj)')
            for _ in range(depth - 1):
                handles.append(blank)
                labels.append('')
            continue
        handles.append(blank)
        labels.append(f'$\\bf{{{dom.replace(" ", chr(92) + " ")}}}$')
        for r in groups[dom]:
            n = int((el['region'] == r).sum())
            ns = int(el.loc[el['region'] == r, 'subject_id'].nunique())
            handles.append(mpatches.Patch(facecolor=colours[r],
                                          edgecolor='0.35', linewidth=0.4))
            labels.append(f'{r}  ({n}, {ns} subj)')
        for _ in range(depth - len(groups[dom]) - 1):
            handles.append(blank)
            labels.append('')
    lax.legend(handles=handles, labels=labels, loc='upper center',
               ncol=len(cols), frameon=False, fontsize=8.2,
               handlelength=1.2, handleheight=1.0, columnspacing=1.6,
               labelspacing=0.32, borderpad=0.1)
    lax.axis('off')

    fig.suptitle(title, fontsize=14, y=0.985)
    # DELIBERATELY SHORT. Hue = domain, shade = region, grey = everything else,
    # and nothing here is a statistic -- that is the whole of what a reader
    # needs at the figure. The insula-split caveat, the dropped-electrode
    # counts and the full domain caveat are all in this folder's
    # provenance.json, which is where a caption that long belongs.
    fig.text(0.02, 0.015,
             f'{len(el)} bipolar pairs, {el["subject_id"].nunique()} subjects. Hue = processing '
             'domain, shade = region within it. Grey "Other" = the Control domain plus the '
             'regions no domain claims. ANATOMY ONLY -- no slope, no p-value. '
             f'{counts["n_outside_brain_dropped"]} electrode(s) outside the MNI152 brain mask '
             'were dropped. See provenance.json for the insula split and the full caveats.\n'
             + DISCLAIMER,
             fontsize=7.4, va='bottom', ha='left', color='0.35', wrap=True)

    fig.savefig(out_path, dpi=230, bbox_inches='tight')
    plt.close(fig)
    if display is not None:
        display.close()
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

    fig, axes = plt.subplots(2, 1, sharex=True,
                             figsize=(max(9.0, 0.46 * len(order) + 3.0), 9.0))
    # EXPLICIT, NOT tight_layout. The x labels are rotated region names up to
    # "Lateral Temporal" long, and tight_layout does not account for them
    # against a figure-level caption -- the first version put the labels
    # straight through the footnote. It also warns here, because the domain
    # names are drawn in axes-transform coordinates.
    fig.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.28,
                        hspace=0.08)

    for ax, vals, ylab in (
            (axes[0], pct, f'Patients with \u22651 electrode (%)\n(of {n_cohort})'),
            (axes[1], n_el, 'Number of electrodes')):
        ax.bar(x, vals, color=cols, edgecolor='0.3', linewidth=0.5, width=0.78)
        ax.set_ylabel(ylab, fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=8)
        # Separators at the DOMAIN boundaries, so the grouping survives even
        # for a reader who cannot separate the hues.
        for b in boundaries:
            ax.axvline(b, color='0.82', lw=0.9, zorder=0)
        for xi, v in zip(x, vals):
            ax.text(xi, v, f'{v:.0f}' + ('%' if ylab.startswith('Patients') else ''),
                    ha='center', va='bottom', fontsize=6.6, color='0.3')
        ax.margins(y=0.14)
    axes[0].set_ylim(0, 100)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(order, rotation=45, ha='right', fontsize=8.5)
    for t, r in zip(axes[1].get_xticklabels(), order):
        t.set_color('0.45' if r in other else colours[r])
    axes[1].set_xlim(-0.7, len(order) - 0.3)

    # Domain names above the top panel, spanning their own regions.
    for dom, i0, i1 in labels_dom:
        axes[0].text((i0 + i1) / 2, 1.035, dom, transform=(
            axes[0].get_xaxis_transform()), ha='center', va='bottom',
            fontsize=10, color=('0.45' if dom == OTHER_LABEL
                                else colours[groups[dom][0]]))

    fig.suptitle(title, fontsize=13, y=0.975)
    fig.text(0.01, 0.008,
             'Same colours as the glass brain. TOP is REACH (share of the '
             f'{n_cohort}-patient cohort with any electrode there), BOTTOM is DEPTH (total '
             'electrodes) — they disagree, which is why both are shown: a region sampled in '
             'few patients cannot support a group claim however many contacts those patients '
             'gave, and a dense cluster on the glass brain can be one patient\'s depth '
             'electrode. Grey "Other" = the Control domain plus the regions no domain claims.\n'
             + DISCLAIMER,
             fontsize=7.4, va='bottom', ha='left', color='0.35', wrap=True)
    fig.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)
    logger.info('wrote %s', out_path.name)


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
                            for m in args.display_modes] + ['fig_anatomy_bars.png'],
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
    for mode in args.display_modes:
        glass(el, colours, groups, other, mode,
              out_dir / f'fig_anatomy_glass_{mode}.png',
              f'Electrode coverage by processing domain — {len(el)} bipolar '
              f'pairs, {el["subject_id"].nunique()} subjects',
              args, domain_caveat(scheme), counts)
    bars(el, colours, groups, other, len(subjects),
         out_dir / 'fig_anatomy_bars.png',
         'Coverage by region — how many patients, and how many electrodes',
         args, domain_caveat(scheme))
    logger.info('done -> %s', out_dir)


if __name__ == '__main__':
    main()
