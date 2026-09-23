"""Re-plot a bandpower mixed-effects run's region x band map, rows grouped by domain.

    <bandpower run>/domain_map/<label>_<timestamp>/fig_band_map_domains.png

Reads `band_cells.parquet` from a `run_bandpower_mixed` run and redraws
`fig_band_map.png` with the ROI rows reordered into processing domains
(`config.roi_schemes.DOMAIN_SCHEMES`), a coloured domain bracket on the left and
a rule between domains. Nothing is refitted and NOTHING IS RE-CORRECTED: the
outlines are the run's own `p_bh_reject`, i.e. BH over every cell of the pain
family (all regions x all bands), exactly as in the original map. Regrouping
rows does not change the family, so it must not change which cells pass.

ROI NAME RECONCILIATION. The domain schemes are defined on `roi_v2_ins`, where
OFC is split into mOFC/lOFC; the `_ofc` schemes merge them into one `OFC` row.
A merged `OFC` row is placed in whichever domain both halves belong to (and the
script refuses if they disagree). ROIs no domain claims are kept in a final
`Unassigned` block rather than dropped -- they are fitted, BH-counted cells, and
hiding them would misstate how many of the family passed.

No title and no caption by default (poster/slide use); the provenance sidecar
carries what the caption used to say.
"""
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from ieeg_ehr import io
from ieeg_ehr.analysis.run_bandpower_mixed import BAND_SETS
from ieeg_ehr.analysis.run_domain_model import DOMAIN_COLOURS
from ieeg_ehr.config import roi_schemes
from ieeg_ehr.features import common

logger = logging.getLogger(__name__)

SCRIPT = 'ieeg_ehr/analysis/plot_band_map_domains.py'
SUBDIR = 'domain_map'
UNASSIGNED = 'Unassigned'
MERGED = {'OFC': ('mOFC', 'lOFC')}


def domain_rows(regions, scheme):
    """[(domain, [roi, ...]), ...] in the scheme's display order, Unassigned last."""
    ds = roi_schemes.domain_scheme(scheme)
    r2d = dict(ds['roi_to_domain'])
    for merged, parts in MERGED.items():
        doms = {r2d.get(p) for p in parts}
        if len(doms) != 1:
            raise SystemExit(f'{merged} merges {parts}, which sit in different '
                             f'domains {doms}; cannot place the merged row')
        dom = doms.pop()
        if dom is not None:
            r2d[merged] = dom
    groups = []
    for dom in ds['display']:
        # member order from the scheme, so the within-domain order is the
        # framework's (e.g. S1, S2/PO, Thalamus, pIns), not the input's
        members = [m for m in ds['members'][dom] if m in regions]
        for merged, parts in MERGED.items():
            if merged in regions and r2d.get(merged) == dom and merged not in members:
                idx = min([i for i, m in enumerate(ds['members'][dom]) if m in parts],
                          default=len(members))
                members.insert(min(idx, len(members)), merged)
        if members:
            groups.append((dom, members))
    placed = {r for _, ms in groups for r in ms}
    rest = [r for r in regions if r not in placed]
    if rest:
        groups.append((UNASSIGNED, rest))
    return groups


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-dir', required=True,
                    help='a run_bandpower_mixed run holding band_cells.parquet')
    ap.add_argument('--domain-scheme', default='pain_domains_v3')
    ap.add_argument('--no-unassigned', action='store_true',
                    help='drop ROIs no domain claims (the outline count then '
                         'no longer matches the BH family size)')
    ap.add_argument('--title', action='store_true', help='draw the old title')
    ap.add_argument('--label', default='v3')
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

    run_dir = Path(args.run_dir)
    prov = json.loads((run_dir / 'provenance.json').read_text())
    params = prov['params']
    band_set = params['band_set']
    bands = list(BAND_SETS[band_set])
    fdr_q = params.get('fdr_q', 0.05)

    cells = io.read_table(run_dir / 'band_cells.parquet', on_stale='warn')
    if 'model' in cells.columns:
        cells = cells[cells['model'] == 'pain']
    regions = list(dict.fromkeys(cells['region']))
    groups = domain_rows(regions, args.domain_scheme)   # list: Unassigned keeps run order
    if args.no_unassigned:
        groups = [g for g in groups if g[0] != UNASSIGNED]
    order = [r for _, ms in groups for r in ms]

    beta = (cells.pivot_table(index='region', columns='band', values='beta_nrs_within')
            .reindex(index=order, columns=bands))
    sig = (cells.assign(rej=cells['p_bh_reject'].fillna(False).astype(bool))
           .pivot_table(index='region', columns='band', values='rej')
           .reindex(index=order, columns=bands).fillna(0).astype(bool))
    n_sig_all = int(cells['p_bh_reject'].fillna(False).astype(bool).sum())
    logger.info('BH family: %d cells, %d rejected at q=%s; %d of them drawn',
                len(cells), n_sig_all, fdr_q, int(sig.to_numpy().sum()))

    # same symmetric cap as the original: max |beta| over the whole family
    cap = float(np.nanmax(np.abs(cells['beta_nrs_within'].to_numpy(dtype=float))))
    cm = plt.get_cmap('RdBu_r').copy()
    cm.set_bad('0.85')
    fig, ax = plt.subplots(figsize=(7.8, 0.42 * len(order) + 1.2))
    im = ax.imshow(beta.to_numpy(dtype=float), aspect='auto', cmap=cm,
                   vmin=-cap, vmax=cap, interpolation='nearest')
    common.draw_mask_outline(ax, sig.to_numpy())
    ax.set_xticks(range(len(bands)))
    ax.set_xticklabels([f'{b}\n{BAND_SETS[band_set][b][0]}-'
                        f'{BAND_SETS[band_set][b][1]} Hz' for b in bands], fontsize=8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=8)

    # domain brackets in axes-x / data-y coordinates, left of the tick labels
    tr = matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData)
    y0 = 0
    for k, (dom, ms) in enumerate(groups):
        y1 = y0 + len(ms)
        colour = DOMAIN_COLOURS.get(dom, '0.55')
        ax.plot([-0.215, -0.215], [y0 - 0.4, y1 - 0.6], transform=tr,
                color=colour, lw=3, clip_on=False, solid_capstyle='butt')
        # horizontal, not rotated: one-row domains (M1) have no height to
        # hold a vertical label
        ax.text(-0.235, (y0 + y1 - 1) / 2, dom, transform=tr, ha='right',
                va='center', fontsize=9, color=colour, fontweight='bold')
        for lab in ax.get_yticklabels()[y0:y1]:
            lab.set_color(colour if dom != UNASSIGNED else '0.25')
        if k < len(groups) - 1:
            ax.axhline(y1 - 0.5, color='white', lw=3)
        y0 = y1

    if args.title:
        ax.set_title(f'Band power vs pain, mixed-effects beta\n{n_sig_all} of '
                     f'{len(cells)} cells BH-significant at q={fdr_q}', fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03,
                 label='d log10(band power) per pain point')

    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    out_dir = run_dir / SUBDIR / f'{args.label}_{stamp}'
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / 'fig_band_map_domains.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    io.write_run_provenance(
        out_dir, script=SCRIPT,
        params={**vars(args), 'band_set': band_set, 'fdr_q': fdr_q, 'cap': cap,
                'source_run': str(run_dir), 'roi_scheme': params.get('roi_scheme')},
        parents=[str(run_dir / 'band_cells.parquet')],
        subjects=prov.get('subjects', []),
        extra={
            'row_groups': {d: ms for d, ms in groups},
            'outlines': f'the source run p_bh_reject: BH over all {len(cells)} '
                        f'cells of the pain family at q={fdr_q}; {n_sig_all} '
                        'rejected. Not recomputed -- row grouping does not '
                        'change the family.',
            'p_caveat': 'p is the parametric Wald z on NRS_within (see source '
                        'run METHODS.md for calibration notes)',
            'status': 'EXPLORATORY -- discovery cohort, NOMINATIONS NOT FINDINGS.',
        })
    io.log_analysis(
        f'band x region pain-slope map, rows grouped by {args.domain_scheme}, '
        f'BH outlines from source run ({n_sig_all}/{len(cells)}) (EXPLORATORY)',
        out_dir)
    logger.info('done -> %s', out_dir)


if __name__ == '__main__':
    main()
