"""Region x band heatmap of one pain_change fixed effect, in the poster style.

    python -m ieeg_ehr.analysis.plot_pain_change_heatmap --run-dir <pain_change run> \\
        [--term dpain | dpain_x_cbar | dpain_x_dc | dc | cbar | dhr | dhrv]

Reads `cells.csv` and `provenance.json` from a `pain_change` run and writes
`<run>/figures/heatmap_<term token>.png` (docs/run_tokens.md): `heatmap_dpain.png`
for `d_pain`, `heatmap_dpain-x-cbar.png` for `d_pain:Cbar`,
`heatmap_dpain-x-dc.png` for `d_pain:dC`, `heatmap_dc.png` and
`heatmap_cbar.png` for the opioid main effects `dC` and `Cbar`. Drawn by
`plot_band_map_domains.render_map` in its poster mode, so it matches the
band-power poster map: rows grouped by domain, colour = the term's fixed effect,
outlines = BH across all fitted cells (`<term>_bh_reject`), stars = the same BH
q-values (`<term>_bh`) on STAR_CUTS: * q < 0.05, ** q < 0.01, *** q < 0.001. Every
star therefore sits in an outline. Region names take `roi_schemes.ROI_RENAMES`.
`--domain-scheme none` draws the rows ungrouped, in the run's own region order,
for a region set no domain scheme covers (an hcpex_paper ROI mapping).
`--domain-scheme anatomy` groups an hcpexv5 run's rows by lobe, in the order and
hues of the anatomy glass brain (`plot_pain_change_anatomy.GROUPS`), and refuses
an ROI that GROUPS does not place.

The colour scale is capped at max |beta| over the term's BH-significant converged
cells (`colour_cap`), so a large non-significant cell (M1 delta in the 2026-10-22
run) cannot wash out the significant ones: cells beyond the cap saturate, and the
colorbar's arrow marks the clipped side. A term with no significant cell is
capped at max |beta| over every converged cell. The glass brain uses the same cap.

`--scratch` writes `heatmap_<term token>_<grouping>.png` to
`analysis/scratch/pain_change_heatmap/` instead of the run, for trying a look
before it replaces the run's figure.
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from ieeg_ehr import config, io
from ieeg_ehr.analysis.pain_change import FIGURES_SUBDIR
from ieeg_ehr.analysis.plot_band_map_domains import UNASSIGNED, domain_rows, render_map
from ieeg_ehr.analysis.run_bandpower_mixed import BAND_SETS
from ieeg_ehr.config import roi_schemes

logger = logging.getLogger(__name__)

#: cells.csv prefix -> (figure token, title, colourbar label).
TERMS = {
    'dpain': ('dpain', 'Δpain–power slopes by region', 'Δ z per pain point'),
    'dpain_x_cbar': ('dpain-x-cbar', 'Opioid moderation of the pain-change slope',
                     'Δ slope per typical dose'),
    'dc': ('dc', 'Opioid exposure change by region', 'Δ z per typical-dose change'),
    'cbar': ('cbar', 'Mean opioid exposure by region', 'Δ z per typical dose'),
    'dpain_x_dc': ('dpain-x-dc', 'Opioid exposure change and the pain-change slope',
                   'Δ slope per typical-dose change'),
    'dhr': ('dhr', 'Heart-rate change by region', 'Δ z per bpm'),
    'dhrv': ('dhrv', 'HRV change by region', 'Δ z per Δ ln RMSSD'),
}
#: The domain colours of the band-power poster map (Sensory, Affective,
#: Cognitive, Modulatory, Other).
POSTER_COLOURS = ('#d95f02', '#1b9e77', '#7570b3', '#447eae', '0.55')
#: The one block's label under `--domain-scheme none`.
UNGROUPED = 'ROIs'
ANATOMY = 'anatomy'
#: Under `--domain-scheme anatomy` the lobe labels and the longest hcpexv5 ROI
#: names ("Dorsal/medial thalamus") need a wider left margin than the default.
ANATOMY_MARGINS_IN = (3.2, 1.05, 0.50, 0.72)
ANATOMY_BRACKET_IN = (1.95, 2.05)
SCRATCH_SUBDIR = 'pain_change_heatmap'


def colour_cap(cells, term):
    """max |beta| over the term's BH-significant converged cells, else all converged."""
    ok = cells['converged'].eq(True)
    sig = ok & cells[f'{term}_bh_reject'].eq(True)
    return float(cells.loc[sig if sig.any() else ok, f'{term}_beta'].abs().max())


def anatomy_rows(regions):
    """(groups, colours) in the anatomy glass brain's lobe order and hues."""
    from ieeg_ehr.analysis.plot_pain_change_anatomy import GROUP_COLOURS, GROUPS
    placed = {r for rois in GROUPS.values() for r in rois}
    missing = sorted(set(regions) - placed)
    if missing:
        raise SystemExit(f'--domain-scheme {ANATOMY}: no lobe group for {missing}')
    label = {g: g.replace(' & ', ' &\n') for g in GROUPS}
    groups = [(label[g], [r for r in rois if r in set(regions)])
              for g, rois in GROUPS.items()]
    return ([g for g in groups if g[1]],
            {label[g]: c for g, c in GROUP_COLOURS.items()})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--term', choices=list(TERMS), default='dpain')
    ap.add_argument('--domain-scheme', default='pain_domains_v4',
                    help=f"a domain scheme, '{ANATOMY}' for the lobe groups, or 'none' "
                         'for one ungrouped block')
    ap.add_argument('--scratch', action='store_true',
                    help='write to analysis/scratch/ instead of the run')
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    run_dir = Path(args.run_dir)
    params = json.loads((run_dir / 'provenance.json').read_text())['params']
    bands = BAND_SETS[params['band_set']]
    cells = io.read_table(run_dir / 'cells.csv', on_stale='warn')
    cells = cells.assign(region=cells['region'].replace(roi_schemes.ROI_RENAMES))
    regions = list(dict.fromkeys(cells['region']))
    layout = {}
    if args.domain_scheme == 'none':
        groups = [(UNGROUPED, regions)]
        colours = {UNGROUPED: POSTER_COLOURS[-1]}
    elif args.domain_scheme == ANATOMY:
        groups, colours = anatomy_rows(regions)
        layout = {'margins_in': ANATOMY_MARGINS_IN, 'bracket_in': ANATOMY_BRACKET_IN}
    else:
        groups = domain_rows(regions, args.domain_scheme)
        display = roi_schemes.domain_scheme(args.domain_scheme)['display']
        colours = dict(zip(display + [UNASSIGNED], POSTER_COLOURS))

    token, title, cb_label = TERMS[args.term]
    if args.scratch:
        out = (config.PLOTS_ROOT / SCRATCH_SUBDIR
               / f"{run_dir.name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
               / f'heatmap_{token}_{args.domain_scheme}.png')
    else:
        out = run_dir / FIGURES_SUBDIR / f'heatmap_{token}.png'
    render_map(cells, groups, bands, out, poster=True, colours=colours,
               title=title, cb_label=cb_label, value=f'{args.term}_beta',
               p_col=f'{args.term}_bh', reject=f'{args.term}_bh_reject', band_col='freq',
               cap=colour_cap(cells, args.term), **layout)
    io.log_analysis(f'pain_change {args.term} heatmap, rows grouped by '
                    f'{args.domain_scheme}', out.parent)
    logger.info('wrote %s', out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
