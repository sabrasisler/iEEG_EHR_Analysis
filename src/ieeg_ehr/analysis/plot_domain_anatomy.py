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

#: The Control domain, greyed for the same reason `run_domain_model`'s palette
#: greys it: it is the negative control and the reference level, and the
#: reference should recede rather than compete.
CONTROL_GREYS = ('0.38', '0.58')

#: Regions no domain claims, as LIGHT GREYS spanning this range. One flat tone
#: was tried first and read as a rendering bug: the legend showed six identical
#: swatches against six different names. Varying them costs nothing -- on the
#: brain they are still obviously one grey family, which is the only thing the
#: hue has to say -- and the legend stops looking broken.
#:
#: The range stays LIGHT on purpose. These 1,763 contacts outnumber any single
#: domain, so at full contrast they would bury the four coloured systems the
#: figure is about; they are here for coverage context, not for reading.
UNASSIGNED_GREY_RANGE = (0.66, 0.86)
UNASSIGNED_LABEL = 'not in any domain'


def shades(base, n, lo=0.62, hi=1.42):
    """`n` shades of `base`, varying LIGHTNESS only, darkest first.

    Hue and saturation are held fixed so every shade still reads as the same
    domain colour -- which is the whole point -- and only lightness carries the
    region. Clamped away from both ends: a shade at L<0.18 is black to the eye
    and one at L>0.88 disappears into the white glass brain, so either would
    silently stop being a distinguishable category.
    """
    r, g, b = matplotlib.colors.to_rgb(base)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    if n <= 1:
        return [matplotlib.colors.to_hex((r, g, b))]
    out = []
    for f in np.linspace(lo, hi, n):
        ll = float(np.clip(l * f, 0.18, 0.88))
        out.append(matplotlib.colors.to_hex(colorsys.hls_to_rgb(h, ll, s)))
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

    for dom in display:
        regions = [r for r in base_order if r in members.get(dom, ())]
        groups[dom] = regions
        if dom == 'Control':
            greys = list(CONTROL_GREYS)
            if len(regions) > len(greys):
                greys = shades('#808080', len(regions))
            for r, c in zip(regions, greys):
                colours[r] = matplotlib.colors.to_hex(
                    matplotlib.colors.to_rgb(c))
        else:
            for r, c in zip(regions, shades(hue_for[dom], len(regions))):
                colours[r] = c

    if unassigned:
        groups[UNASSIGNED_LABEL] = list(unassigned)
        lo, hi = UNASSIGNED_GREY_RANGE
        levels = ([sum(UNASSIGNED_GREY_RANGE) / 2] if len(unassigned) == 1
                  else np.linspace(lo, hi, len(unassigned)))
        for r, v in zip(unassigned, levels):
            colours[r] = matplotlib.colors.to_hex((v, v, v))
    return colours, groups, hue_for


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


def glass(el, colours, groups, out_path, title, args, caveat, counts):
    """The brain. One `plot_markers` call per region, drawn largest group first.

    `plot_markers` takes a colormap rather than per-point colours, so one call
    per colour is the only way to get 21. DRAWN LARGEST-FIRST so the biggest
    regions go down first and the small ones land on top: Lateral Temporal is
    821 contacts and would otherwise bury Auditory's 32 entirely.
    """
    from nilearn import plotting

    order = sorted(colours, key=lambda r: -int((el['region'] == r).sum()))
    fig = plt.figure(figsize=(19, 6.6))
    display = None
    for region in order:
        rows = el[el['region'] == region]
        if rows.empty:
            continue
        coords = rows[['mni_x', 'mni_y', 'mni_z']].to_numpy(dtype=float)
        display = plotting.plot_markers(
            node_values=np.ones(len(coords)), node_coords=coords,
            node_size=args.node_size, display_mode='lyrz', colorbar=False,
            figure=fig, axes=(0.01, 0.30, 0.98, 0.62), alpha=args.alpha,
            node_cmap=matplotlib.colors.ListedColormap([colours[region]]),
            node_vmin=0, node_vmax=2,
            node_kwargs={'edgecolors': args.edge_color,
                         'linewidths': args.edge_width})

    # -- legend: ONE COLUMN PER DOMAIN, so the grouping is visible as layout
    # rather than only as hue. matplotlib fills a legend column-major, so the
    # lists are padded to equal length and the blanks are invisible handles.
    lax = fig.add_axes((0.02, 0.10, 0.96, 0.18))
    cols = [d for d in groups if groups[d]]
    depth = max(len(groups[d]) for d in cols) + 1
    handles, labels = [], []
    blank = mpatches.Patch(alpha=0, linewidth=0)
    for dom in cols:
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

    fig.suptitle(title, fontsize=14, y=0.975)
    fig.text(0.02, 0.008,
             f'{len(el)} bipolar pairs from {el["subject_id"].nunique()} subjects, at the MNI '
             'MIDPOINT of each pair; the region comes from the ANODE\'s DK label, matching the '
             'model. COLOUR IS THE DOMAIN, SHADE IS THE REGION INSIDE IT. Nothing here is a '
             'result -- no slope, no p-value. Grey: Control is the model\'s reference domain '
             f'(the negative control); "{UNASSIGNED_LABEL}" are real regions the framework does '
             'not claim, dropped from the domain model but present in the region-level '
             'consistency map, and they are drawn because leaving them out would suggest the '
             'study sampled only the four coloured systems. White matter, ventricles and '
             f'unlabelled contacts are not in the study and are not drawn ({counts["n_no_region"]} '
             f'of {counts["n_electrodes_in_channel_meta"]}). '
             f'{counts["n_outside_brain_dropped"]} further electrode(s) fell outside the MNI152 '
             'brain mask and were dropped rather than clipped. ' + caveat + '\n' + DISCLAIMER,
             fontsize=6.4, va='bottom', ha='left', color='0.35', wrap=True)

    fig.savefig(out_path, dpi=230, bbox_inches='tight')
    plt.close(fig)
    if display is not None:
        display.close()
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
    colours, groups, hue_for = build_palette(spec['members'], spec['display'],
                                             base_order, unassigned)

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
                     .fillna(UNASSIGNED_LABEL),
                     colour=lambda d: d['region'].map(colours)))
    table['_o'] = table['region'].map({r: i for i, r in enumerate(base_order)})
    table = table.sort_values('_o').drop(columns='_o')
    io.write_table(table[['domain', 'region', 'colour', 'n_electrodes',
                          'n_subjects']],
                   out_dir / 'colour_source.csv', script=SCRIPT,
                   parents=[str(run_dir / 'provenance.json')],
                   params={'domain_hues': list(DOMAIN_HUES),
                           'control_greys': list(CONTROL_GREYS),
                           'unassigned_grey_range': list(UNASSIGNED_GREY_RANGE),
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
                'insula_threshold': params.get('insula_threshold')},
        parents=[str(run_dir / 'provenance.json')],
        subjects=sorted(el['subject_id'].unique()),
        extra={'status': DISCLAIMER, 'domain_caveat': domain_caveat(scheme),
               'palette': {'domain_hue': hue_for,
                           'region_colour': {r: colours[r] for r in colours},
                           'rule': 'hue = domain, lightness = region within '
                                   'it; Control grey because it is the '
                                   'reference domain; unassigned regions in '
                                   'light greys, kept light so they do not '
                                   'bury the four coloured systems'},
               'what_this_is': 'ANATOMY ONLY. No slope, no p-value, no scale. '
                               'It shows which electrodes the study has and '
                               'which domain each belongs to.'})

    io.log_analysis(
        f'glass brain: all {len(el)} study electrodes coloured by processing '
        'domain, shaded by region (ANATOMY, no result)', out_dir)
    glass(el, colours, groups, out_dir / 'fig_anatomy_glass.png',
          f'Electrode coverage by processing domain — {len(el)} bipolar pairs, '
          f'{el["subject_id"].nunique()} subjects',
          args, domain_caveat(scheme), counts)
    logger.info('done -> %s', out_dir)


if __name__ == '__main__':
    main()
