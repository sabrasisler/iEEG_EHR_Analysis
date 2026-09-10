"""Look at the DATA behind a medication coefficient, for a handful of named cells.

Everything produced for the medication models so far has been a coefficient map.
A map cannot tell you whether a group effect is a coherent shift across patients
or an average over a bimodal set, whether between-subject slope spread is real or
three patients with interictal spikes, or whether a linear slope is even the right
summary of an 11-point ordinal scale that people use in lumpy ways. These three
figures answer those, per cell:

`spaghetti`  per-subject fitted lines of log10 power on NRS_within, SPLIT BY
             medication state, group fixed effect in bold for each state, and a
             rug of the actual NRS values along the x-axis. The rug matters: this
             cohort puts a third of its unmedicated epochs at NRS=0, and a line
             drawn across a range nobody occupies is a extrapolation dressed as a
             fit.

`caterpillar` per-subject pain slopes, sorted -- the model's BLUP beside that
             subject's own unpooled fit with a 95% CI. Distinguishes genuine
             between-subject spread from a few extreme subjects, which matters
             because the remedy differs: real spread wants a moderator, three bad
             subjects want artifact rejection.

`partial`    partial residual for NRS_within against NRS_within, with a loess
             overlay. The model assumes a straight line. If the real relationship
             is a step at zero-vs-nonzero, or only appears above 5, or saturates,
             then the linear slope is a bad summary and any interaction on it is
             testing a change in the wrong quantity.

The grid runs deliberately save no model objects -- 900 of them are not worth
keeping -- so each named cell is REFITTED here. That costs seconds.

    python -m ieeg_ehr.analysis.plot_med_cell_diagnostics --run-dir <medstrata run>
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ieeg_ehr import io
from ieeg_ehr.analysis import mixed_model as mm, reference_run
from ieeg_ehr.analysis.run_mixed_model_grid import resolve_cohort
from ieeg_ehr.analysis.run_mixed_model_pilot import (load_cell_frames, resolve_view_dir,
                                                     roi_maps, view_subject_paths)

logger = logging.getLogger(__name__)

DISCLAIMER = ('EXPLORATORY -- discovery cohort, NOMINATIONS NOT FINDINGS. '
              'Parametric Wald p. Not confirmed out of sample.')

#: Chosen to span the patterns the maps showed, not at random:
#: M1/S1 ~10 Hz carry the band-limited medication deflection; lOFC at 1 Hz sits in
#: the broadband-positive region; Insula ~28 Hz was a pre-committed null in the
#: Phase 1 pilot and is the control.
DEFAULT_CELLS = (('M1', 21), ('S1', 21), ('lOFC', 3), ('Insula', 30))

MED_COLOR = '#c1442f'
UNMED_COLOR = '#2c6fad'


def _grid(n, ncol, panel_w, panel_h):
    """(fig, flat axes list). One row per `ncol` cells, unused panels hidden.

    A single row stops being readable somewhere around six cells, and the point
    of running 50 is to see how a shape CHANGES across frequency within a region
    -- which needs them adjacent, not strung out.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ncol = max(1, min(ncol, n))
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(panel_w * ncol, panel_h * nrow),
                             squeeze=False)
    flat = [ax for row in axes for ax in row]
    for ax in flat[n:]:
        ax.set_visible(False)
    return fig, flat[:n]


def load_cells(run_dir, cells, ref):
    """{(region, bin): model frame with med_state attached}. Refit-ready."""
    import json
    run_dir = Path(run_dir)
    state = io.read_table(run_dir / 'epoch_med_state.parquet', on_stale='warn')
    cohort = set(json.loads((run_dir / 'provenance.json').read_text())['subjects'])

    # READ the view the run recorded, do not re-resolve it. Re-deriving the hash
    # from the reference run's params reproduces it only while every input to
    # `io.config_hash` is unchanged, and one of them drifted -- the recomputed
    # hash pointed at a directory that does not exist. The run wrote down which
    # view it used; that is the answer, and it cannot go stale.
    prov = json.loads((run_dir / 'provenance.json').read_text())
    view_dir = prov.get('params', {}).get('view_dir')
    if not view_dir or not Path(view_dir).exists():
        logger.warning('run provenance has no usable view_dir (%s); falling back '
                       'to re-resolving it by hash', view_dir)
        view_dir = resolve_view_dir(
            None, mask_label=ref.view_params.get('mask_label'),
            roi_scheme=ref.view_params.get('roi_scheme', 'roi_v2'))
    logger.info('per-channel view: %s', view_dir)
    paths = view_subject_paths(view_dir)
    roi_by_subject, _ = roi_maps(paths, cohort,
                                 ref.view_params.get('roi_scheme', 'roi_v2'))
    frames = load_cell_frames(paths, cohort, set(cells), roi_by_subject)

    lookup = state.assign(subject='sub-' + state['subject'])[
        ['subject', 'epoch_id', 'med_state']]
    out = {}
    for cell, df in frames.items():
        if df is None or df.empty:
            continue
        d = df.merge(lookup, on=['subject', 'epoch_id'], how='inner')
        d['med_state'] = d['med_state'].astype(float)
        out[cell] = d
    return out


def epoch_level(df):
    """One point per (subject, epoch): channels averaged, power centred per subject."""
    e = (df.assign(log10_power=df['log10_power'].astype('float64'))
         .groupby(['subject', 'epoch_id'], as_index=False)
         .agg(y=('log10_power', 'mean'), NRS_within=('NRS_within', 'first'),
              NRS=('NRS', 'first'), med=('med_state', 'first')))
    e['yc'] = e['y'] - e.groupby('subject')['y'].transform('mean')
    return e


def _subject_lines(ax, e, med_value, colour):
    """Per-subject OLS fits within one medication state."""
    n = 0
    for _, g in e[e['med'] == med_value].groupby('subject'):
        x = g['NRS_within'].to_numpy(dtype=float)
        y = g['yc'].to_numpy(dtype=float)
        if len(np.unique(x)) < 2:
            continue
        s, b = np.polyfit(x, y, 1)
        xs = np.array([x.min(), x.max()])
        ax.plot(xs, s * xs + b, color=colour, lw=0.7, alpha=0.35, zorder=2)
        n += 1
    return n


def fig_spaghetti(cells_data, records, out_path, ncol=4):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    items = list(cells_data.items())
    fig, flat = _grid(len(items), ncol, 4.2, 3.6)
    small = len(items) > 8
    for i, (cell, df) in enumerate(items):
        ax = flat[i]
        e = epoch_level(df)
        n_un = _subject_lines(ax, e, 0.0, UNMED_COLOR)
        n_med = _subject_lines(ax, e, 1.0, MED_COLOR)

        rec = records.get(cell, {})
        base = rec.get('beta_nrs_within', np.nan)
        delta = rec.get('med_ix_beta', np.nan)
        xlim = float(np.nanmax(np.abs(e['NRS_within']))) * 1.05
        xs = np.array([-xlim, xlim])
        if np.isfinite(base):
            ax.plot(xs, base * xs, color=UNMED_COLOR, lw=3, zorder=5,
                    label=f'group, unmedicated ({base:+.4f})')
        if np.isfinite(base) and np.isfinite(delta):
            ax.plot(xs, (base + delta) * xs, color=MED_COLOR, lw=3, zorder=5,
                    label=f'group, medicated ({base + delta:+.4f})')
        if not np.isfinite(base):
            # SAY SO. The subject lines still draw from the cell frame, so a
            # panel with no group effect looks identical to one whose effect is
            # zero. A cell can be absent from the fitted table because it fell
            # below the coverage floor -- M1 does exactly this in the opioid run,
            # where the cohort is 28 -- and that must not be silent.
            ax.text(0.5, 0.5, 'NO FITTED GROUP EFFECT\ncell absent from the '
                              'model table\n(below the coverage floor?)',
                    transform=ax.transAxes, ha='center', va='center', fontsize=8,
                    color=MED_COLOR, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.4', fc='white',
                              ec=MED_COLOR, alpha=0.9))

        # THE RUG. Where the data actually is -- a fitted line across a range
        # nobody occupies is an extrapolation wearing a fit's clothes.
        lo = ax.get_ylim()[0]
        for val, colour, off in ((0.0, UNMED_COLOR, 0.0), (1.0, MED_COLOR, 0.03)):
            xr = e.loc[e['med'] == val, 'NRS_within'].to_numpy()
            ax.plot(xr, np.full_like(xr, lo + off * abs(lo)), '|', color=colour,
                    ms=6, alpha=0.25, zorder=1)

        ax.axhline(0, color='0.85', lw=0.7)
        ax.axvline(0, color='0.85', lw=0.7)
        ax.set_xlim(-xlim, xlim)
        ax.set_title(f'{cell[0]}  bin {cell[1]}\n'
                     f'{n_un} unmedicated / {n_med} medicated subject lines',
                     fontsize=10)
        if not small or i % ncol == 0:
            ax.set_ylabel('log10 power rel. subject mean', fontsize=8)
        if not small:
            ax.set_xlabel('pain relative to subject mean', fontsize=9)
            ax.legend(fontsize=7, loc='upper left')
        ax.tick_params(labelsize=7 if small else 8)

    fig.suptitle('Per-subject pain-power lines, split by medication state', fontsize=13)
    fig.tight_layout(rect=(0, 0.09, 1, 0.94))
    fig.text(0.01, 0.01,
             'Thin lines are each subject\'s own unpooled fit within one medication '
             'state; bold lines are the model\'s group effect for that state. The '
             'RUG along the bottom shows where the pain values actually are -- this '
             'cohort puts a third of its unmedicated epochs at NRS=0, so a slope '
             'drawn across the full range is partly extrapolation. Read the thin '
             'lines for whether the group effect is a coherent shift or an average '
             'over a bimodal set.\n' + DISCLAIMER,
             fontsize=7, va='bottom', color='0.35', wrap=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig_caterpillar(cells_data, fits, out_path, ncol=4):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy import stats

    items = list(cells_data.items())
    fig, flat = _grid(len(items), ncol, 3.8, 4.0)
    small = len(items) > 8
    for i, (cell, df) in enumerate(items):
        ax = flat[i]
        e = epoch_level(df)
        rows = []
        for subject, g in e.groupby('subject'):
            x = g['NRS_within'].to_numpy(dtype=float)
            y = g['yc'].to_numpy(dtype=float)
            if len(np.unique(x)) < 2 or len(x) < 3:
                continue
            s, b = np.polyfit(x, y, 1)
            resid = y - (s * x + b)
            sxx = float(((x - x.mean()) ** 2).sum())
            se = float(np.sqrt((resid @ resid) / (len(x) - 2) / sxx)) if sxx else np.nan
            rows.append({'subject': subject, 'raw': s, 'se': se, 'n': len(x)})
        r = pd.DataFrame(rows)

        res = fits.get(cell)
        blup = {}
        if res is not None:
            try:
                for grp, s in res.random_effects.items():
                    for k, v in s.items():
                        if 'subj_slope' in k:
                            blup[grp] = float(v)
            except Exception:                                # noqa: BLE001
                blup = {}
        beta = float(res.fe_params['NRS_within']) if res is not None else np.nan
        r['blup'] = r['subject'].map(lambda s: beta + blup.get(s, np.nan))
        r = r.sort_values('blup' if r['blup'].notna().any() else 'raw')
        y0 = np.arange(len(r))

        ax.hlines(y0, r['raw'] - 1.96 * r['se'], r['raw'] + 1.96 * r['se'],
                  color='0.6', lw=0.8, alpha=0.7)
        ax.scatter(r['raw'], y0, s=14, facecolors='none', edgecolors='0.4',
                   linewidths=0.7, label='own unpooled fit (95% CI)')
        if r['blup'].notna().any():
            ax.scatter(r['blup'], y0, s=18, color=MED_COLOR, zorder=4,
                       label='model estimate (beta + BLUP)')
        if np.isfinite(beta):
            ax.axvline(beta, color='black', lw=1.8, zorder=3,
                       label=f'group fixed effect ({beta:+.4f})')
        ax.axvline(0, color='0.75', lw=0.8, ls='--')
        ax.set_yticks([])
        ax.set_title(f'{cell[0]}  bin {cell[1]}\n{len(r)} subjects', fontsize=10)
        if not small:
            ax.set_xlabel('pain slope', fontsize=9)
            ax.legend(fontsize=7, loc='lower right')
        ax.tick_params(labelsize=7 if small else 8)

    fig.suptitle('Per-subject pain slopes, sorted: is between-subject spread real '
                 'or a few extreme subjects?', fontsize=12)
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    fig.text(0.01, 0.01,
             'Open markers with intervals are each subject\'s OWN fit; filled '
             'markers are the model\'s shrunk estimate. BLUPs deliberately carry '
             'no interval -- they are predictions, not fitted parameters, and an '
             'interval on one invites reading it as a per-subject test. A smooth '
             'gradient across subjects is genuine spread; a flat body with two or '
             'three outliers is an artifact problem, and the two want different '
             'remedies.\n' + DISCLAIMER,
             fontsize=7, va='bottom', color='0.35', wrap=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig_partial(cells_data, fits, out_path, ncol=4):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from statsmodels.nonparametric.smoothers_lowess import lowess

    items = list(cells_data.items())
    fig, flat = _grid(len(items), ncol, 4.2, 3.4)
    small = len(items) > 8
    for i, (cell, df) in enumerate(items):
        ax = flat[i]
        res = fits.get(cell)
        if res is None:
            ax.set_visible(False)
            continue
        d = df.copy()
        beta = float(res.fe_params['NRS_within'])
        # Partial residual: the residual with this term's fitted contribution added
        # back, so the plot shows what the model has left to explain PLUS what it
        # attributes to pain. A straight cloud means the linear term fits.
        d['pres'] = np.asarray(res.resid) + beta * d['NRS_within'].to_numpy()

        for val, colour, lab in ((0.0, UNMED_COLOR, 'unmedicated'),
                                 (1.0, MED_COLOR, 'medicated')):
            s = d[d['med_state'] == val]
            ax.scatter(s['NRS_within'], s['pres'], s=2, alpha=0.05, color=colour,
                       linewidths=0, zorder=1)
        lo = lowess(d['pres'].to_numpy(dtype=float),
                    d['NRS_within'].to_numpy(dtype=float), frac=0.4, return_sorted=True)
        ax.plot(lo[:, 0], lo[:, 1], color='black', lw=2.2, zorder=5, label='loess')
        xs = np.array([d['NRS_within'].min(), d['NRS_within'].max()])
        ax.plot(xs, beta * xs, color='0.35', lw=1.8, ls='--', zorder=4,
                label=f'linear term ({beta:+.4f})')

        # Where the data is, by pain level -- the lumpiness is the point.
        cnt = d.groupby('NRS')['pres'].size()
        ax2 = ax.twinx()
        ax2.bar(d.groupby('NRS')['NRS_within'].first(), cnt / cnt.sum(),
                width=0.25, color='0.8', alpha=0.5, zorder=0)
        ax2.set_ylim(0, 1.0)
        ax2.set_yticks([])

        ax.axhline(0, color='0.85', lw=0.7)
        ax.set_title(f'{cell[0]}  bin {cell[1]}', fontsize=10)
        if not small:
            ax.set_xlabel('pain relative to subject mean', fontsize=9)
            ax.legend(fontsize=7, loc='upper left')
        if not small or i % ncol == 0:
            ax.set_ylabel('partial residual', fontsize=8)
        ax.tick_params(labelsize=7 if small else 8)

    fig.suptitle('Partial residual vs pain, with loess: is a straight line the '
                 'right summary?', fontsize=12)
    fig.tight_layout(rect=(0, 0.09, 1, 0.93))
    fig.text(0.01, 0.01,
             'Dashed line is what the model fitted; black is a loess through the '
             'partial residuals. Grey bars show the share of data at each pain '
             'level. NRS is an 11-point ordinal scale used in lumpy ways -- if the '
             'loess shows a STEP at zero-vs-nonzero, or a relationship that only '
             'appears above 5, or saturation at the top, then the linear slope is '
             'a poor summary and a medication interaction ON that slope is testing '
             'a change in the wrong quantity.\n' + DISCLAIMER,
             fontsize=7, va='bottom', color='0.35', wrap=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-dir', required=True, help='A medstrata run directory.')
    ap.add_argument('--cells', default=None,
                    help='Semicolon-separated "Region:bin" pairs. Default: '
                         + '; '.join(f'{r}:{b}' for r, b in DEFAULT_CELLS))
    ap.add_argument('--regions', default=None,
                    help='Comma-separated regions, crossed with --bins. The way to '
                         'ask for 50 cells without naming 50 pairs.')
    ap.add_argument('--bins', default=None,
                    help='Comma-separated freq bin indices, crossed with --regions.')
    ap.add_argument('--ncol', type=int, default=4,
                    help='Panels per row. Raise it for large cell counts.')
    ap.add_argument('--out-dir', default=None,
                    help='Default: <run-dir>/cell_diagnostics')
    ap.add_argument('--reference-run', default=str(reference_run.CONTPAIN_HEATMAP))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    io.warn_if_dirty()

    cells = DEFAULT_CELLS
    if args.regions and args.bins:
        # Region x bin cross product, ordered region-major so a row of the figure
        # is one region swept across frequency -- which is the comparison the
        # layout exists to make.
        regions = [r.strip() for r in args.regions.split(',') if r.strip()]
        bins = [int(b) for b in args.bins.split(',') if b.strip()]
        cells = tuple((r, b) for r in regions for b in bins)
    elif args.cells:
        cells = tuple((c.rsplit(':', 1)[0], int(c.rsplit(':', 1)[1]))
                      for c in args.cells.split(';') if c.strip())

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / 'cell_diagnostics'
    out_dir.mkdir(parents=True, exist_ok=True)

    ref = reference_run.load(args.reference_run)
    data = load_cells(run_dir, cells, ref)
    if not data:
        raise SystemExit(f'no data for any of {cells}')
    # Re-key in the ORDER REQUESTED. load_cell_frames returns whatever order the
    # groupby produced, which would scramble a region-major sweep into noise.
    data = {c: data[c] for c in cells if c in data}
    logger.info('loaded %d cell(s): %s', len(data), list(data))

    # Refit: the grid saves no model objects, and the BLUPs and partial residuals
    # both need one.
    fits, records = {}, {}
    ixcells = io.read_table(run_dir / 'interaction' / 'grid_cells.parquet',
                            on_stale='ignore')
    for cell, df in data.items():
        row = ixcells[(ixcells['region'] == cell[0])
                      & (ixcells['freq_bin_index'] == cell[1])]
        if len(row):
            records[cell] = row.iloc[0].to_dict()
        try:
            res, _ = mm.fit_cell(df, mm.VC_FULL,
                                 formula=mm.FORMULA_MED_INTERACTION)
            fits[cell] = res
            logger.info('%s bin %d refitted', cell[0], cell[1])
        except mm.CellFitError as exc:
            logger.error('%s bin %d refit FAILED: %s', cell[0], cell[1], exc)

    fig_spaghetti(data, records, out_dir / 'fig_spaghetti.png', ncol=args.ncol)
    logger.info('wrote %s', out_dir / 'fig_spaghetti.png')
    fig_caterpillar(data, fits, out_dir / 'fig_caterpillar.png', ncol=args.ncol)
    logger.info('wrote %s', out_dir / 'fig_caterpillar.png')
    fig_partial(data, fits, out_dir / 'fig_partial_residual.png', ncol=args.ncol)
    logger.info('wrote %s', out_dir / 'fig_partial_residual.png')

    io.log_analysis('medication cell diagnostics: spaghetti, caterpillar and '
                    'partial-residual plots for named cells (EXPLORATORY)', run_dir)


if __name__ == '__main__':
    main()
