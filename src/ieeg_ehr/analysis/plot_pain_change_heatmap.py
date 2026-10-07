"""Region x band heatmap of the pain_change `d_pain` fixed effect, in the poster style.

    python -m ieeg_ehr.analysis.plot_pain_change_heatmap --run-dir <pain_change run>

Reads `cells.csv` and `provenance.json` from a `pain_change` run and writes
`<run>/figures/pain_change_heatmap.png`. Drawn by
`plot_band_map_domains.render_map` in its poster mode, so it matches the
band-power poster map: rows grouped by domain, colour = the `d_pain` fixed
effect (change in z per NRS point), outlines = BH across all fitted cells,
stars = uncorrected p. A star without an outline is not BH-significant.
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

TITLE = 'Pain-change slopes by region'
CB_LABEL = 'Δ z per pain point'
#: The domain colours of the band-power poster map (Sensory, Affective,
#: Cognitive, Modulatory, Other).
POSTER_COLOURS = ('#d95f02', '#1b9e77', '#7570b3', '#447eae', '0.55')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--domain-scheme', default='pain_domains_v4')
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    run_dir = Path(args.run_dir)
    params = json.loads((run_dir / 'provenance.json').read_text())['params']
    bands = BAND_SETS[params['band_set']]
    cells = io.read_table(run_dir / 'cells.csv', on_stale='warn')
    groups = domain_rows(list(dict.fromkeys(cells['region'])), args.domain_scheme)
    display = roi_schemes.domain_scheme(args.domain_scheme)['display']
    colours = dict(zip(display + [UNASSIGNED], POSTER_COLOURS))

    out = run_dir / FIGURES_SUBDIR / 'pain_change_heatmap.png'
    render_map(cells, groups, bands, out, poster=True, colours=colours,
               title=TITLE, cb_label=CB_LABEL, value='dpain_beta', p_col='dpain_p',
               reject='dpain_bh_reject', band_col='freq')
    io.log_analysis('pain_change d_pain heatmap, rows grouped by '
                    f'{args.domain_scheme}', out.parent)
    logger.info('wrote %s', out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
