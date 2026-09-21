"""Three figures for a domain model, one per question it can answer.

    python -m ieeg_ehr.analysis.plot_domain_model --run-dir <run> [--figure F]

The scheme is deliberately THREE figures and not seven, because the questions
collapse further than the panels do. Written down so the next person does not
re-derive it:

    A  result         Does pain move power, does medication move power, and does
                      medication change the pain->power relationship? All three
                      are "a coefficient with an interval", so they are ONE
                      figure with three term blocks -- not three figures.
                      Includes the omnibus per band: do the domains differ.
    B  heterogeneity  Does the average describe anyone? The per-subject slope
                      distribution beside the variance decomposition, because a
                      domain mean is only a summary if the subjects agree, and
                      the heterogeneity LRT fires in nearly every cell of the
                      band-level runs.
    C  diagnostics    Can the p-values be trusted? Conditional residuals against
                      fitted, their QQ, and their autocorrelation across
                      assessments in real time.

WHY A DOT-AND-INTERVAL AND NOT A HEATMAP. Every earlier figure encoded the
estimate as COLOUR, which forces uncertainty to arrive separately as stars or an
outline -- so a cell with beta +0.020 +/- 0.002 and one with +0.020 +/- 0.019
looked identical. Printing the numbers in the cells was a patch over that, not a
fix. Here the estimate and its interval are the primary encoding and colour only
groups, so it carries no information that is lost when it is misread.

READING THE TERMS. Both predictors are subject-mean-centred, so each coefficient
is read at the OTHER one's patient-specific mean:

    pain        d log10 power per pain point, AT THAT PATIENT'S OWN AVERAGE
                MEDICATION LEVEL -- not unmedicated, not marginal over dosing
    med         d log10 power when recently dosed, AT THAT PATIENT'S OWN AVERAGE
                PAIN -- not at zero pain, and not at the cohort's mean pain
    pain_x_med  how much the pain slope CHANGES per unit of medication state

The three blocks get SEPARATE x-scales because the units genuinely differ: pain
is per one point of a 0-10 scale, medication is per a 0->1 dose-state switch
(the whole range of its predictor), and the interaction is per point per switch.
A shared scale would imply a comparison that is not available.

EXPLORATORY. Discovery cohort only. Nominations, not findings.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ieeg_ehr import io
from ieeg_ehr.analysis import view_tables

logger = logging.getLogger(__name__)

SCRIPT = 'ieeg_ehr/analysis/plot_domain_model.py'
FIGURES = ('result', 'heterogeneity', 'diagnostics')

DISCLAIMER = ('EXPLORATORY -- discovery cohort, NOMINATIONS NOT FINDINGS. '
              'Not confirmed out of sample.')

#: One colour per domain, used ONLY to group rows. Nothing is encoded in it that
#: is lost if a reader cannot separate two hues -- which matters, because
#: Cognitive and Affective are ~1.5 dE apart under deuteranopia. Imported from
#: the fitting module rather than copied: the copy that used to live here drifted
#: to the v1 domain names and painted Modulatory the same grey as Control.
from ieeg_ehr.analysis.run_domain_model import DOMAIN_COLOURS  # noqa: E402

TERM_LABELS = {
    'pain': ('PAIN\nd log10 power per pain point',
             "at that patient's own average medication level"),
    'med': ('MEDICATION\nd log10 power when recently dosed',
            "at that patient's own average pain"),
    'pain_x_med': ('PAIN x MEDICATION\nchange in the pain slope when dosed',
                   'per pain point per dose-state switch'),
}


def load_run(run_dir):
    run_dir = Path(run_dir)
    out = {'run_dir': run_dir,
           'cells': io.read_table(run_dir / 'domain_bands.parquet', on_stale='warn'),
           'slopes': io.read_table(run_dir / 'domain_slopes.parquet',
                                   on_stale='warn')}
    cov = run_dir / 'parcel_coverage.parquet'
    out['coverage'] = (io.read_table(cov, on_stale='ignore') if cov.exists()
                       else None)
    return out


def run_params(run_dir):
    import json
    try:
        return json.loads((Path(run_dir) / 'provenance.json').read_text()).get(
            'params', {})
    except (OSError, ValueError):
        return {}


def domain_order(slopes, roi_scheme):
    present = set(slopes['domain'])
    ordered = [d for d in view_tables.roi_regions_for({'roi_scheme': roi_scheme})
               if d in present]
    return ordered or sorted(present)


def _footnote(fig, text, size=6.4):
    fig.text(0.01, 0.005, text, fontsize=size, va='bottom', ha='left',
             color='0.35', wrap=True)


# ============================================================================
# A. THE RESULT
# ============================================================================

def fig_result(data, args):
    """Dot-and-interval matrix: rows are band x domain, one panel per term.

    Rows are grouped by BAND with domains inside, because the comparison the
    figure exists for is between domains WITHIN a band -- grouping the other way
    would put the comparison across a panel break.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cells, slopes = data['cells'], data['slopes']
    params = run_params(data['run_dir'])
    roi_scheme = params.get('roi_scheme', 'pain_domains_v2')
    domains = domain_order(slopes, roi_scheme)
    bands = list(cells['band'])

    if 'term' not in slopes.columns:            # pre-2026-09-18 schema
        slopes = slopes.assign(term='pain',
                               beta=slopes.get('beta_pain', slopes.get('beta')))
    terms = [t for t in ('pain', 'med', 'pain_x_med') if t in set(slopes['term'])]

    # One row per (band, domain), bands in band order, a gap between bands.
    rows, ylabels, yband = [], [], []
    for band in bands:
        for dom in domains:
            rows.append((band, dom))
            ylabels.append(dom)
            yband.append(band)
        rows.append((None, None))               # spacer
        ylabels.append('')
        yband.append(None)
    rows, ylabels = rows[:-1], ylabels[:-1]
    yband = yband[:-1]
    y = np.arange(len(rows))

    fig, axes = plt.subplots(
        1, len(terms), figsize=(4.6 * len(terms) + 2.8, 0.30 * len(rows) + 3.4),
        sharey=True, squeeze=False)

    for j, term in enumerate(terms):
        ax = axes[0][j]
        sub = slopes[slopes['term'] == term].set_index(['band', 'domain'])
        for i, key in enumerate(rows):
            if key[0] is None:
                continue
            if key not in sub.index:
                continue
            r = sub.loc[key]
            sig = bool(r.get('p_bh_reject', False))
            colour = DOMAIN_COLOURS.get(key[1], '0.4')
            ax.errorbar(r['beta'], i, xerr=1.96 * r['se'], fmt='o',
                        ms=7.5 if sig else 5.0, lw=2.0 if sig else 1.0,
                        capsize=2.5, color=colour,
                        markerfacecolor=colour if sig else 'white', zorder=3)
        ax.axvline(0, color='0.45', lw=1.0, ls='--', zorder=1)
        # Band separators, so the grouping is structural rather than implied by
        # the label column alone.
        for i, key in enumerate(rows):
            if key[0] is None:
                ax.axhline(i, color='0.88', lw=0.8, zorder=0)
        head, unit = TERM_LABELS.get(term, (term, ''))
        ax.set_title(head, fontsize=10)
        ax.set_xlabel(unit, fontsize=8)
        ax.tick_params(labelsize=7.5)
        ax.set_ylim(len(rows) - 0.5, -0.5)

    ax = axes[0][0]
    ax.set_yticks(y)
    ax.set_yticklabels(ylabels, fontsize=7.5)
    # Band names in the margin, once per block.
    for band in bands:
        idx = [i for i, b in enumerate(yband) if b == band]
        if idx:
            ax.text(-0.30, float(np.mean(idx)), band, transform=ax.get_yaxis_transform(),
                    rotation=90, ha='center', va='center', fontsize=9,
                    fontweight='bold')

    # The omnibus: one number per band per term, reported as text rather than
    # plotted, because it is a property of the BLOCK and not of any single row.
    lines = []
    for band in bands:
        c = cells[cells['band'] == band].iloc[0]
        bits = []
        for term in terms:
            col = f'p_omnibus_{term}' if f'p_omnibus_{term}' in cells.columns else (
                'p_omnibus' if term == 'pain' else None)
            if col and np.isfinite(c.get(col, np.nan)):
                bits.append(f'{term} p={c[col]:.3g}')
        lines.append(f'{band}: ' + ', '.join(bits))

    drug = params.get('drug_set')
    fig.suptitle('Domain model: pain, medication, and their interaction\n'
                 + (f'{drug}, {params.get("med_window_hours", "?")} h before the '
                    'score · ' if drug else '')
                 + 'filled = BH-significant within that term',
                 fontsize=12.5)
    fig.tight_layout(rect=(0, 0.15, 1, 0.92))
    _footnote(fig,
              'ESTIMATE AND 95% CI ARE THE PRIMARY ENCODING, not colour -- colour '
              'only groups the domains, so nothing is lost if two hues are hard '
              'to separate. THE THREE PANELS HAVE SEPARATE x-SCALES because the '
              'units differ: pain is per ONE point of a 0-10 scale, medication is '
              'per a 0->1 dose-state switch (the whole range of its predictor), '
              'and the interaction is per point per switch. BOTH PREDICTORS ARE '
              "SUBJECT-MEAN-CENTRED, so each coefficient is read at the OTHER's "
              "patient-specific mean: the pain slope is at that patient's own "
              'average medication level (NOT unmedicated), and the medication '
              "effect is at that patient's own average pain (NOT zero pain, and "
              'NOT the cohort mean). BH runs WITHIN each term, never pooled '
              'across them. OMNIBUS -- whether the domains differ from each '
              'other, a property of the whole block rather than of any row: '
              + ' · '.join(lines) + f'.\n{DISCLAIMER}')
    out = data['run_dir'] / 'fig_A_result.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info('wrote %s', out)
    return out


# ============================================================================
# B. HETEROGENEITY
# ============================================================================

def subject_domain_slopes(data, args, band):
    """Per-(subject, domain) unpooled OLS slope for one band.

    Computed POST HOC from the view rather than read from the run, because the
    domain fits do not save per-subject estimates and refitting six models to get
    them would cost hours. Unpooled on purpose: partial pooling drags every
    subject toward the group, so BLUP spread systematically understates real
    between-subject variation -- which is the very thing this figure is for.
    """
    from ieeg_ehr.analysis import fullres_cells, reference_run
    from ieeg_ehr.analysis.run_bandpower_mixed import BAND_SETS, band_table, aggregate
    from ieeg_ehr.analysis.run_domain_model import parcel_domain_maps
    from ieeg_ehr.analysis.run_fullres_grid import resolve_cohort
    from ieeg_ehr.analysis.plot_mixed_model_subject_lines import (epoch_level,
                                                                  subject_slopes)
    from ieeg_ehr.analysis import mixed_model as mm

    params = run_params(data['run_dir'])
    roi_scheme = params.get('roi_scheme', 'pain_domains_v2')
    band_set = params.get('band_set', 'paper_bands_6_hg200')
    em = params.get('epoch_minutes')

    ref = reference_run.load(args.reference_run)
    view_dir = fullres_cells.resolve_view_dir(
        mask_label=ref.view_params.get('mask_label'),
        max_excluded_frac=ref.view_params.get('max_excluded_frac'),
        epoch_minutes=em)
    paths, _, _, subjects, _, _ = resolve_cohort(ref, view_dir,
                                                 roi_scheme=roi_scheme)
    parcel_of, parcel_to_domain, _ = parcel_domain_maps(paths, subjects, roi_scheme)
    kept, bands, _ = band_table(band_set, params.get('notch_half_width_hz'), em)
    index, values = fullres_cells.load_all_parcels(paths, subjects, parcel_of,
                                                   list(kept.index),
                                                   epoch_minutes=em)
    band_values, names = aggregate(values, kept, bands)
    del values
    bi = list(names).index(band)

    frame = pd.DataFrame({
        'subject_id': index['subject_id'].to_numpy(),
        'channel': index['channel'].to_numpy(),
        'epoch_id': index['epoch_id'].to_numpy(),
        'pain_score': index['pain_score'].to_numpy(),
        'value': band_values[:, bi],
        'parcel': index['parcel'].to_numpy()})
    frame['domain'] = frame['parcel'].map(parcel_to_domain)

    out = []
    for dom, grp in frame.groupby('domain'):
        df = mm.build_cell_frame(grp)
        if not len(df):
            continue
        s = subject_slopes(epoch_level(df))
        s['domain'] = dom
        out.append(s)
    return pd.concat(out, ignore_index=True), band


def fig_heterogeneity(data, args):
    """Does the average describe anyone? Subject slopes + where the variance is."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cells, slopes = data['cells'], data['slopes']
    params = run_params(data['run_dir'])
    domains = domain_order(slopes, params.get('roi_scheme', 'pain_domains_v2'))
    bands = list(cells['band'])

    band = args.band
    if band is None:
        # The band whose domains differ most -- the one a reader will ask about.
        col = ('p_omnibus_pain' if 'p_omnibus_pain' in cells.columns
               else 'p_omnibus')
        band = str(cells.loc[cells[col].idxmin(), 'band'])
    logger.info('caterpillar band: %s', band)

    subj, band = subject_domain_slopes(data, args, band)

    if 'term' not in slopes.columns:
        slopes = slopes.assign(term='pain',
                               beta=slopes.get('beta_pain', slopes.get('beta')))
    pain = slopes[(slopes['term'] == 'pain') & (slopes['band'] == band)] \
        .set_index('domain')

    fig = plt.figure(figsize=(5.0 + 2.5 * len(domains), 7.6))
    gs = fig.add_gridspec(1, len(domains) + 1, width_ratios=[1.5] + [1] * len(domains))

    # ---- variance decomposition, per band ---------------------------------
    ax = fig.add_subplot(gs[0, 0])
    comps = [('var_subj_int', 'subject intercept', '#4C78A8'),
             ('var_subj_slope', 'subject slope', '#E45756'),
             ('var_subj_parcel_slope', 'parcel slope', '#F58518'),
             ('var_channel', 'channel', '#72B7B2'),
             ('var_resid', 'residual', '#BAB0AC')]
    present = [c for c in comps if c[0] in cells.columns]
    mat = np.array([[float(cells.loc[cells['band'] == b, c[0]].iloc[0])
                     for c in present] for b in bands])
    # The SLOPE variances enter the linear predictor scaled by var(predictor);
    # a raw slope variance is in different units and looks negligible for
    # arithmetic reasons alone. Reported as a share of the total so the bars are
    # comparable across bands.
    share = mat / mat.sum(axis=1, keepdims=True) * 100
    left = np.zeros(len(bands))
    ypos = np.arange(len(bands))
    for k, c in enumerate(present):
        ax.barh(ypos, share[:, k], left=left, color=c[2], label=c[1], height=0.68)
        left += share[:, k]
    ax.set_yticks(ypos)
    ax.set_yticklabels(bands, fontsize=8)
    ax.set_ylim(len(bands) - 0.5, -0.5)
    ax.set_xlabel('% of total variance', fontsize=8.5)
    ax.set_title('Where the variance lives', fontsize=10)
    ax.legend(fontsize=6.5, loc='lower right')
    ax.tick_params(labelsize=7.5)

    # ---- one caterpillar per domain ---------------------------------------
    for k, dom in enumerate(domains):
        ax = fig.add_subplot(gs[0, k + 1])
        s = subj[(subj['domain'] == dom) & subj['slope'].notna()] \
            .sort_values('slope').reset_index(drop=True)
        if not len(s):
            ax.set_visible(False)
            continue
        beta = float(pain.loc[dom, 'beta']) if dom in pain.index else np.nan
        se = float(pain.loc[dom, 'se']) if dom in pain.index else np.nan
        yy = np.arange(len(s))
        agree = np.sign(s['slope'].to_numpy()) == np.sign(beta)
        colour = DOMAIN_COLOURS.get(dom, '0.4')
        if np.isfinite(beta):
            ax.axvspan(beta - 1.96 * se, beta + 1.96 * se, color=colour, alpha=0.16,
                       lw=0, zorder=0)
            ax.axvline(beta, color=colour, lw=2.0, zorder=1)
        ax.axvline(0, color='0.5', lw=0.9, ls='--', zorder=1)
        ax.errorbar(s['slope'][agree], yy[agree], xerr=1.96 * s['se'][agree],
                    fmt='o', ms=2.8, lw=0.6, color='0.25', zorder=3)
        ax.errorbar(s['slope'][~agree], yy[~agree], xerr=1.96 * s['se'][~agree],
                    fmt='o', ms=2.8, lw=0.6, color='#b03a2e', zorder=3)
        ax.set_ylim(-1, len(s))
        ax.set_yticks([])
        ax.set_xlabel('subject slope', fontsize=8)
        frac = float(agree.mean()) if len(s) else np.nan
        ax.set_title(f'{dom}\n{int(agree.sum())}/{len(s)} share the sign '
                     f'({frac:.0%})', fontsize=8.5)
        ax.tick_params(labelsize=7)

    fig.suptitle(f'Does the average describe anyone?  ({band} band)', fontsize=12.5)
    fig.tight_layout(rect=(0, 0.13, 1, 0.93))
    _footnote(fig,
              'LEFT: the variance components as a share of the total. This is the '
              'panel that says whether the modelling choices matter -- if channel '
              'and residual are nearly everything, the subject and parcel '
              'structure is doing no work and the pooling decisions were moot. '
              'The two SLOPE components are variances of a slope, so they are in '
              'different units from the intercept-like ones and look small for '
              'arithmetic reasons; read them against each other across bands, not '
              'against the channel term. RIGHT: every subject\'s OWN unpooled OLS '
              'slope with its own SE, sorted, against the domain fixed effect and '
              'its interval. Unpooled on purpose -- partial pooling drags each '
              'subject toward the group, so BLUP spread understates real '
              'between-subject variation, which is the thing this panel exists to '
              'show. RED points OPPOSE the group direction. A domain whose column '
              'straddles zero has a mean that is a group average over disagreeing '
              f'patients, not a description of a typical one.\n{DISCLAIMER}')
    out = data['run_dir'] / f'fig_B_heterogeneity_{band}.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info('wrote %s', out)
    return out


# ============================================================================

BUILDERS = {'result': fig_result, 'heterogeneity': fig_heterogeneity}


def main():
    from ieeg_ehr.analysis import reference_run

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--figure', nargs='*', choices=list(FIGURES) + ['all'],
                    default=['result'])
    ap.add_argument('--band', default=None,
                    help='Band for the per-subject panels. Default: the band '
                         'whose domains differ most.')
    ap.add_argument('--reference-run', default=str(reference_run.CONTPAIN_HEATMAP))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    io.warn_if_dirty()

    data = load_run(args.run_dir)
    wanted = ('result', 'heterogeneity') if 'all' in args.figure else tuple(
        args.figure)
    written = []
    for name in wanted:
        if name not in BUILDERS:
            logger.warning('figure %r is not implemented yet', name)
            continue
        try:
            out = BUILDERS[name](data, args)
        except Exception as exc:                        # noqa: BLE001
            logger.exception('figure %r failed: %s', name, exc)
            continue
        if out is not None:
            written.append(out)
    if written:
        io.log_analysis(f'domain-model figures: {len(written)} (EXPLORATORY)',
                        Path(args.run_dir))
    logger.info('%d figure(s) written', len(written))


if __name__ == '__main__':
    main()
