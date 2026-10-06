"""Re-plot a bandpower mixed-effects run's region x band map, rows grouped by domain.

    <bandpower run>/domain_map/<label>_<timestamp>/fig_band_map_domains.png

Reads `band_cells.parquet` from a `run_bandpower_mixed` run and redraws
`fig_band_map.png` with the ROI rows reordered into processing domains
(`config.roi_schemes.DOMAIN_SCHEMES`), a coloured domain bracket on the left and
a rule between domains. Nothing is refitted and NOTHING IS RE-CORRECTED: the
outlines are the run's own `p_bh_reject`, i.e. BH over every cell of the pain
family (all regions x all bands), exactly as in the original map. Regrouping
rows does not change the family, so it must not change which cells pass.

ROI NAME RECONCILIATION. A run fitted under an older scheme is shown with the
current labels via `roi_schemes.ROI_RENAMES` (pure renames, e.g. `dmPFC/SMA` ->
`dmPFC`). The domain schemes split OFC into mOFC/lOFC; the `_ofc` schemes merge
them into one `OFC` row, which is placed in whichever domain both halves belong
to (and the script refuses if they disagree). ROIs no domain claims are kept in
a final `Other` block rather than dropped -- they are fitted, BH-counted cells,
and hiding them would misstate how many of the family passed.

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
UNASSIGNED = 'Other'
MERGED = {'OFC': ('mOFC', 'lOFC')}

#: `--poster`: the type of the other poster cuts
#: (`plot_domain_lmer_consistency.POSTER_FS`, `plot_domain_glass.POSTER_TYPE`)
#: -- ROI names 10, domain labels 11 bold, band ticks 9.5 with the unit once as
#: a 10 pt axis label, colorbar 9 / 8.5 -- plus bold significance stars, and
#: no frame round the heatmap (the colorbar keeps its outline).
POSTER_FS = dict(roi=10, domain=11, band=9.5, xlabel=10, cb_label=9,
                 cb_tick=8.5, star=12)
#: `--poster` page: 8 x 7 in -- the height of the heatmap + glass-brain poster
#: (`plot_domain_glass`, --panel-size 7 7), 1 in wider. Margins (in): left,
#: right, top, bottom; the bracket and domain label sit these inches left of
#: the heatmap.
POSTER_SIZE = (8.0, 7.0)
POSTER_MARGINS_IN = (2.55, 1.05, 0.50, 0.72)
POSTER_TITLE = 'Pain–power slopes by region'
POSTER_BRACKET_IN = (1.38, 1.48)
BAND_SYMBOL = {'delta': 'δ', 'theta': 'θ', 'alpha': 'α', 'beta': 'β',
               'gamma': 'γ', 'high_gamma': 'hγ'}
#: The stars of `plot_domain_lmer.heatmap`: the UNCORRECTED p. The outlines
#: stay the BH decision, so a star without an outline is not BH-significant.
STAR_CUTS = ((1e-3, '***'), (1e-2, '**'), (5e-2, '*'))


def _stars(p):
    if not np.isfinite(p):
        return ''
    return next((s for cut, s in STAR_CUTS if p < cut), '')


def domain_rows(regions, scheme):
    """[(domain, [roi, ...]), ...] in the scheme's display order, Other last."""
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
    ap.add_argument('--domain-scheme', default='pain_domains_v4')
    ap.add_argument('--no-other', action='store_true',
                    help='drop ROIs no domain claims (the outline count then '
                         'no longer matches the BH family size)')
    ap.add_argument('--title', action='store_true', help='draw the old title')
    ap.add_argument('--label', default='v4')
    ap.add_argument('--poster', action='store_true',
                    help='POSTER_FS type, bold uncorrected-p stars, no frame on '
                         'the heatmap; into <run>/poster/<label>_<ts>/')
    ap.add_argument('--colours', nargs='+', default=None,
                    help="one colour per domain in the scheme's display order, "
                         "optionally one more for Other (default: DOMAIN_COLOURS)")
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
    # current labels for a run fitted under an older scheme; pure renames only
    renamed = {k: v for k, v in roi_schemes.ROI_RENAMES.items()
               if k in set(cells['region'])}
    cells = cells.assign(region=cells['region'].replace(renamed))
    if renamed:
        logger.info('relabelled %s', renamed)
    regions = list(dict.fromkeys(cells['region']))
    groups = domain_rows(regions, args.domain_scheme)   # list: Other keeps run order
    if args.no_other:
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
    if args.poster:
        # A fixed page (POSTER_SIZE), margins in INCHES: left holds the domain
        # label + bracket + ROI names, bottom the two-line band ticks + label,
        # right the colorbar and its label.
        W, H = POSTER_SIZE
        L_in, R_in, T_in, B_in = POSTER_MARGINS_IN
        fig = plt.figure(figsize=(W, H))
        ax = fig.add_axes((L_in / W, B_in / H, 1 - (L_in + R_in) / W,
                           1 - (T_in + B_in) / H))
    else:
        fig, ax = plt.subplots(figsize=(7.8, 0.42 * len(order) + 1.2))
    im = ax.imshow(beta.to_numpy(dtype=float), aspect='auto', cmap=cm,
                   vmin=-cap, vmax=cap, interpolation='nearest')
    common.draw_mask_outline(ax, sig.to_numpy())
    ax.set_xticks(range(len(bands)))
    ax.set_yticks(range(len(order)))
    if args.poster:
        fs = POSTER_FS
        ax.set_xticklabels([f'{BAND_SYMBOL.get(b, b)}\n{BAND_SETS[band_set][b][0]}–'
                            f'{BAND_SETS[band_set][b][1]}' for b in bands],
                           fontsize=fs['band'])
        ax.set_xlabel('Frequency Band (Hz)', fontsize=fs['xlabel'])
        ax.set_yticklabels(order, fontsize=fs['roi'])
        for s in ax.spines.values():
            s.set_visible(False)
        # stars: uncorrected p, white on the darkest cells as the domain map
        pval = (cells.pivot_table(index='region', columns='band', values='p')
                .reindex(index=order, columns=bands).to_numpy(dtype=float))
        b_ = beta.to_numpy(dtype=float)
        for i in range(len(order)):
            for j in range(len(bands)):
                st = _stars(pval[i, j])
                if st:
                    ax.text(j, i, st, ha='center', va='center',
                            fontsize=fs['star'], fontweight='bold',
                            color='white' if abs(b_[i, j]) > 0.62 * cap
                            else '0.1')
    else:
        ax.set_xticklabels([f'{b}\n{BAND_SETS[band_set][b][0]}-'
                            f'{BAND_SETS[band_set][b][1]} Hz' for b in bands],
                           fontsize=8)
        ax.set_yticklabels(order, fontsize=8)

    # domain brackets in axes-x / data-y coordinates, left of the tick labels
    tr = matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData)
    colours = dict(DOMAIN_COLOURS)
    if args.colours:
        display = roi_schemes.domain_scheme(args.domain_scheme)['display']
        if len(args.colours) not in (len(display), len(display) + 1):
            raise SystemExit(f'--colours needs {len(display)} values ({display}), '
                             f'or {len(display) + 1} with Other last')
        colours.update(zip(display + [UNASSIGNED], args.colours))
    y0 = 0
    # bracket / domain-label x (axes fraction): the poster's 10 pt ROI names
    # ("Lateral Temporal") reach the default bracket, so it moves left
    if args.poster:
        # inches left of the heatmap, converted to axes fraction
        aw = ax.get_position().width * POSTER_SIZE[0]
        bx, lx = -POSTER_BRACKET_IN[0] / aw, -POSTER_BRACKET_IN[1] / aw
    else:
        bx, lx = -0.215, -0.235
    for k, (dom, ms) in enumerate(groups):
        y1 = y0 + len(ms)
        colour = colours.get(dom, '0.55')
        ax.plot([bx, bx], [y0 - 0.4, y1 - 0.6], transform=tr,
                color=colour, lw=3, clip_on=False, solid_capstyle='butt')
        # horizontal, not rotated: one-row domains (M1) have no height to
        # hold a vertical label
        ax.text(lx, (y0 + y1 - 1) / 2, dom, transform=tr, ha='right',
                va='center', fontsize=POSTER_FS['domain'] if args.poster else 9,
                color=colour, fontweight='bold')
        for lab in ax.get_yticklabels()[y0:y1]:
            lab.set_color(colour if dom != UNASSIGNED else '0.25')
        if k < len(groups) - 1:
            ax.axhline(y1 - 0.5, color='white', lw=3)
        y0 = y1

    if args.title:
        ax.set_title(f'Band power vs pain, mixed-effects beta\n{n_sig_all} of '
                     f'{len(cells)} cells BH-significant at q={fdr_q}', fontsize=11)
    if args.poster:
        # its own axes on the fixed page: 60% of the heatmap's height, centred
        p = ax.get_position()
        cax = fig.add_axes((p.x1 + 0.15 / POSTER_SIZE[0],
                            p.y0 + 0.2 * p.height, 0.16 / POSTER_SIZE[0],
                            0.6 * p.height))
        cb = fig.colorbar(im, cax=cax)
        # bold 12 pt, centred over the heatmap, as on the other poster cuts
        fig.text((p.x0 + p.x1) / 2, 1 - 0.10 / POSTER_SIZE[1], POSTER_TITLE,
                 ha='center', va='top', fontsize=12, fontweight='bold',
                 color='0.1')
    else:
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03,
                          label='d log10(band power) per pain point')
    if args.poster:
        cb.set_label('Δ log10 power per pain point', fontsize=POSTER_FS['cb_label'])
        cb.ax.tick_params(labelsize=POSTER_FS['cb_tick'])

    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    out_dir = (run_dir / ('poster' if args.poster else SUBDIR)
               / f'{args.label}_{stamp}')
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.poster:
        # exactly POSTER_SIZE: no bbox_inches='tight', which would re-crop it
        fig.savefig(out_dir / 'fig_band_map_domains.png', dpi=300,
                    facecolor='white')
    else:
        fig.savefig(out_dir / 'fig_band_map_domains.png', dpi=200,
                    bbox_inches='tight')
    plt.close(fig)

    io.write_run_provenance(
        out_dir, script=SCRIPT,
        params={**vars(args), 'band_set': band_set, 'fdr_q': fdr_q, 'cap': cap,
                'source_run': str(run_dir), 'roi_scheme': params.get('roi_scheme')},
        parents=[str(run_dir / 'band_cells.parquet')],
        subjects=prov.get('subjects', []),
        extra={
            'row_groups': {d: ms for d, ms in groups},
            'roi_relabels': renamed,
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
