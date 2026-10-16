"""Region x band heatmap of one pain_change fixed effect, in the poster style.

    python -m ieeg_ehr.analysis.plot_pain_change_heatmap --run-dir <pain_change run> \\
        [--term dpain | dpain_x_cbar | dpain_x_dc | dc | cbar]

Reads `cells.csv` and `provenance.json` from a `pain_change` run and writes
`<run>/figures/heatmap_<term token>.png` (docs/run_tokens.md): `heatmap_dpain.png`
for `d_pain`, `heatmap_dpain-x-cbar.png` for `d_pain:Cbar`,
`heatmap_dpain-x-dc.png` for `d_pain:dC`, `heatmap_dc.png` and
`heatmap_cbar.png` for the opioid main effects `dC` and `Cbar`. Drawn by
`plot_band_map_domains.render_map` in its poster mode, so it matches the
band-power poster map: rows grouped by domain, colour = the term's fixed effect,
outlines = BH across all fitted cells, stars = uncorrected p. A star without an
outline is not BH-significant. Region names take `roi_schemes.ROI_RENAMES`.
"""

import argparse
import json
import logging
from pathlib import Path

from ieeg_ehr import io
from ieeg_ehr.analysis.pain_change import FIGURES_SUBDIR
from ieeg_ehr.analysis.plot_band_map_domains import UNASSIGNED, domain_rows, render_map
from ieeg_ehr.analysis.run_bandpower_mixed import BAND_SETS
from ieeg_ehr.config import roi_schemes

logger = logging.getLogger(__name__)

#: cells.csv prefix -> (figure token, title, colourbar label).
TERMS = {
    'dpain': ('dpain', 'Pain-change slopes by region', 'Δ z per pain point'),
    'dpain_x_cbar': ('dpain-x-cbar', 'Opioid moderation of the pain-change slope',
                     'Δ slope per typical dose'),
    'dc': ('dc', 'Opioid exposure change by region', 'Δ z per typical-dose change'),
    'cbar': ('cbar', 'Mean opioid exposure by region', 'Δ z per typical dose'),
    'dpain_x_dc': ('dpain-x-dc', 'Opioid exposure change and the pain-change slope',
                   'Δ slope per typical-dose change'),
}
#: The domain colours of the band-power poster map (Sensory, Affective,
#: Cognitive, Modulatory, Other).
POSTER_COLOURS = ('#d95f02', '#1b9e77', '#7570b3', '#447eae', '0.55')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--term', choices=list(TERMS), default='dpain')
    ap.add_argument('--domain-scheme', default='pain_domains_v4')
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    run_dir = Path(args.run_dir)
    params = json.loads((run_dir / 'provenance.json').read_text())['params']
    bands = BAND_SETS[params['band_set']]
    cells = io.read_table(run_dir / 'cells.csv', on_stale='warn')
    cells = cells.assign(region=cells['region'].replace(roi_schemes.ROI_RENAMES))
    groups = domain_rows(list(dict.fromkeys(cells['region'])), args.domain_scheme)
    display = roi_schemes.domain_scheme(args.domain_scheme)['display']
    colours = dict(zip(display + [UNASSIGNED], POSTER_COLOURS))

    token, title, cb_label = TERMS[args.term]
    out = run_dir / FIGURES_SUBDIR / f'heatmap_{token}.png'
    render_map(cells, groups, bands, out, poster=True, colours=colours,
               title=title, cb_label=cb_label, value=f'{args.term}_beta',
               p_col=f'{args.term}_p', reject=f'{args.term}_bh_reject', band_col='freq')
    io.log_analysis(f'pain_change {args.term} heatmap, rows grouped by '
                    f'{args.domain_scheme}', out.parent)
    logger.info('wrote %s', out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
