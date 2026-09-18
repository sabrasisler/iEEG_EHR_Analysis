"""ONE mixed-effects model per band, with PROCESSING DOMAIN as a fixed effect.

    analysis/pain/bandpower/domain_model/<view_scheme>/<run>_<timestamp>/

The question: do the pain-matrix processing domains differ in how their power
tracks pain? That is a contrast BETWEEN domains, so it has to be one model
containing all of them -- not a domain-by-domain map a reader compares by eye.
Two stratified estimates cannot support a claim that they differ.

    log10_power ~ NRS_within * C(domain, Treatment('Control')) + NRS_submean
                  + (NRS_within || subject)
                  + (NRS_within || subject:parcel)
                  + (1 | subject:channel)

ONE FIT PER BAND. Six fits over the whole cohort's rows, rather than one fit per
(region, band) cell.

THE PARCEL LEVEL IS THE POINT, AND IT IS WHY THE ROI LAYER IS GONE
------------------------------------------------------------------
The band-power runs used hand-built ROIs (S1, S2/PO, dlPFC...) that already
collapse several atlas parcels each. Stacking a domain on top of those would
average twice, and the second average would hide exactly what a domain claim
needs to survive: whether every parcel in the domain agrees, or one
well-sampled parcel is carrying it. So the intermediate layer is removed. The
unit is the ATLAS PARCEL (Desikan-Killiany, hemispheres collapsed), the domain is
the fixed effect, and parcel enters as a random SLOPE so each parcel deviates
around its domain's mean instead of being averaged into it.

WHY `subject:parcel` AND NOT `domain:region`
--------------------------------------------
A parcel-level term shared across subjects is a CROSSED random effect: parcels
appear in many patients, patients contribute many parcels. `statsmodels.MixedLM`
takes ONE grouping variable and evaluates every `vc_formula` term WITHIN it --
verified on synthetic data where parcel crosses subject: the term came back with
one variance and a per-subject design matrix, i.e. silently NESTED, with no error
and no warning. So the crossed form cannot be fitted here and, worse, would look
like it had been. Parcel is therefore nested in subject, which is what the
library can honestly fit and is a slightly different claim: each subject-parcel
deviates, so the domain effect is not driven by any one parcel, but the domain
standard error does not account for a parcel deviating CONSISTENTLY across
patients. The exact-crossing counterpart -- per-(subject, parcel) OLS slopes with
subject- AND parcel-clustered standard errors -- is not implemented here yet; it
is the robustness check this model wants and it is in TASKS.md.

WHY A WALD TEST AND NOT AN LRT FOR THE OMNIBUS
----------------------------------------------
"Do domains differ at all in this band" is a joint test on the interaction
block. The obvious route -- fit with and without the interaction and compare
likelihoods -- is INVALID under REML, because REML likelihoods are not comparable
across different fixed-effects designs; `mixed_model.lrt` refuses exactly that
comparison for exactly that reason. Refitting under ML to enable the LRT would
change every variance component, so the omnibus is a Wald chi2 on the
interaction coefficients instead, from the fitted covariance.

EVERY DOMAIN'S SLOPE IS A LINEAR COMBINATION, NOT A COEFFICIENT. With treatment
coding the reference domain's slope is `NRS_within` and domain d's is
`NRS_within + NRS_within:domain[T.d]`, whose SE needs the covariance of the two
-- var(a) + var(b) + 2cov(a,b). Reading the interaction coefficient alone as
"domain d's effect" is the standard misreading and it is not what is reported.

`Control` is the reference level, so every contrast reads as "differs from the
quasi-control domain (Occipital + Auditory)".

    python -m ieeg_ehr.analysis.run_domain_model

EXPLORATORY. Discovery cohort only. Nominations, not findings.
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ieeg_ehr import config, io
from ieeg_ehr.analysis import cluster_permutation as cp
from ieeg_ehr.analysis import fullres_cells, mixed_model as mm
from ieeg_ehr.analysis import reference_run, view_tables
from ieeg_ehr.analysis.run_bandpower_mixed import BAND_SETS, band_table, aggregate
from ieeg_ehr.analysis.run_fullres_grid import CONFOUND_CAVEAT, resolve_cohort
from ieeg_ehr.config import roi_schemes
from ieeg_ehr.views import channel_meta

logger = logging.getLogger(__name__)

SCRIPT = 'ieeg_ehr/analysis/run_domain_model.py'
QUESTION = 'bandpower'
OUTPUT_TYPE = 'domain_model'
RUN_NAME = 'domain_mixedlm'

#: The domain that every contrast is read against.
REFERENCE_DOMAIN = 'Control'

#: Random effects. `subj_parcel_slope` is the parcel term, nested in subject --
#: see the module docstring for why it cannot be crossed here.
VC_DOMAIN = {
    'subj_int': '1',
    'subj_slope': '0 + NRS_within',
    'subj_parcel_slope': '0 + C(parcel):NRS_within',
    'channel': '0 + C(channel_uid)',
}

DISCLAIMER = ('EXPLORATORY -- discovery cohort, NOMINATIONS NOT FINDINGS. '
              'Not confirmed out of sample.')

MODULATORY_CAVEAT = (
    'THE MODULATORY DOMAIN IS M1 ALONE. Brainstem has zero contacts in this '
    'cohort -- measured over 7,068 contacts in 51 subjects, no brain-stem, pons, '
    'medulla, midbrain or periaqueductal label at all -- so the descending '
    'modulatory arm of the framework is represented by one parcel, 82 contacts '
    'and 20 subjects. It is the weakest row in every figure and the only domain '
    'whose internal consistency cannot be checked, because one parcel has no '
    'parcel-to-parcel agreement to inspect.')

DOMAIN_CAVEAT = (
    'INSULA is not in any domain, pending the Destrieux/a2009s anterior-'
    'posterior split: the framework puts anterior insula in the affective '
    'pathway and posterior insula in the sensory one, and Desikan-Killiany gives '
    'one parcel, so assigning it either way would fabricate the distinction the '
    'domains rest on. THALAMUS *is* in Sensory by assignment, and the same class '
    'of caveat applies to it and is accepted rather than avoided: DK gives one '
    'thalamus parcel, the ascending pathway is VPL/VPM specifically, so a Sensory '
    'effect here partly reflects medial and dorsal nuclei that sit in the '
    'affective pathway. Basal Ganglia, Hippocampus, PCC, Parietal (other), MTL '
    '(other) and Lateral Temporal are UNASSIGNED and therefore dropped; PCC '
    'carries one of the larger low-frequency effects in this cohort, so that is a '
    'real and deliberate loss.')


# ============================================================================
# THE PARCEL -> DOMAIN MAP
# ============================================================================

def parcel_domain_maps(paths, subjects, scheme, collapse_hemisphere=True):
    """({subject: {channel: parcel}}, {parcel: domain}, coverage frame).

    The parcel is the ATLAS label, hemispheres collapsed by default -- matching
    how the scheme's patterns already substring-match, and keeping each parcel
    better sampled. The domain comes from the same scheme, so one mapping decides
    both and they cannot disagree.

    A channel whose parcel has no domain in `scheme['display']` is DROPPED here,
    which is how insula, thalamus, PCC, parietal and lateral temporal leave.
    """
    resolved = roi_schemes.resolve_roi_scheme(scheme)
    domains = set(resolved['display'])
    patterns = resolved['patterns']

    #: parcel string -> domain, built from the scheme's own patterns so the two
    #: levels are always consistent.
    parcel_to_domain = {pat: dom for dom, pats in patterns.items()
                        for pat in pats if dom in domains}

    def parcel_of_label(label):
        if not isinstance(label, str):
            return None
        low = label.lower()
        for pat, dom in parcel_to_domain.items():
            if pat in low:
                return (pat if collapse_hemisphere else label), dom
        return None

    per_subject, rows = {}, []
    for p in paths:
        subject, session = fullres_cells.subject_session_of(p)
        sid = f'sub-{subject}'
        if sid not in subjects:
            continue
        try:
            meta = channel_meta.build(subject, session, [])
        except FileNotFoundError:
            logger.warning('%s: no channel_meta, skipped', sid)
            continue
        mapping = {}
        for ch, lab in (meta[['channel', 'dk_anode']].drop_duplicates('channel')
                        .itertuples(index=False)):
            hit = parcel_of_label(lab)
            if hit is not None:
                mapping[ch] = hit[0]
        if mapping:
            per_subject.setdefault(sid, {}).update(mapping)
            for ch, parcel in mapping.items():
                rows.append({'subject_id': sid, 'channel': ch, 'parcel': parcel})

    coverage = pd.DataFrame(rows)
    if coverage.empty:
        raise SystemExit('no channel mapped to any parcel in a displayed domain')
    coverage['domain'] = coverage['parcel'].map(
        {pat: dom for pat, dom in parcel_to_domain.items()})
    return per_subject, parcel_to_domain, coverage


# ============================================================================
# THE MODEL
# ============================================================================

def domain_formula(domains):
    """The fixed-effects formula, with `Control` as the reference if present."""
    ref = REFERENCE_DOMAIN if REFERENCE_DOMAIN in domains else sorted(domains)[0]
    return (f"log10_power ~ NRS_within * C(domain, Treatment('{ref}')) "
            '+ NRS_submean'), ref


def marginal_slopes(res, domains, ref):
    """Each domain's pain slope as a LINEAR COMBINATION, with its own SE.

    The interaction coefficient is NOT domain d's slope -- it is the DIFFERENCE
    from the reference. Reading it as the slope is the standard misreading of a
    treatment-coded interaction, so the contrast is formed explicitly and its SE
    comes from the fitted covariance (var(a) + var(b) + 2cov(a,b)).
    """
    from scipy import stats

    names = list(res.fe_params.index)
    cov = np.asarray(res.cov_params())[:len(names), :len(names)]
    beta = res.fe_params.to_numpy()

    def ix(name):
        return names.index(name) if name in names else None

    base = ix('NRS_within')
    rows = []
    for dom in domains:
        contrast = np.zeros(len(names))
        contrast[base] = 1.0
        term = None
        if dom != ref:
            # patsy spells it C(domain, Treatment('Control'))[T.Sensory]
            term = next((n for n in names
                         if n.startswith('NRS_within:') and n.endswith(f'[T.{dom}]')),
                        None)
            if term is None:
                logger.warning('no interaction term found for domain %r', dom)
                continue
            contrast[ix(term)] = 1.0
        est = float(contrast @ beta)
        se = float(np.sqrt(contrast @ cov @ contrast))
        z = est / se if se > 0 else np.nan
        rows.append({'domain': dom, 'is_reference': dom == ref,
                     'beta_pain': est, 'se': se, 'z': z,
                     'p': float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else np.nan,
                     'ci_lo': est - 1.96 * se, 'ci_hi': est + 1.96 * se,
                     'diff_from_ref': (float(beta[ix(term)]) if term else 0.0),
                     'diff_se': (float(np.sqrt(cov[ix(term), ix(term)])) if term
                                 else np.nan),
                     'diff_p': (float(res.pvalues[term]) if term else np.nan)})
    return pd.DataFrame(rows)


def omnibus_wald(res, domains, ref):
    """(chi2, df, p) for "do domains differ at all", as a joint Wald test.

    NOT an LRT: REML likelihoods are not comparable across different
    fixed-effects designs, and refitting under ML to permit one would change
    every variance component in the model being tested.
    """
    names = list(res.fe_params.index)
    terms = [n for n in names
             if n.startswith('NRS_within:') and any(n.endswith(f'[T.{d}]')
                                                    for d in domains if d != ref)]
    if not terms:
        return np.nan, 0, np.nan
    # THE CONSTRAINT IS OVER `res.params`, NOT `fe_params`. For MixedLM the
    # parameter vector is the fixed effects FOLLOWED BY the variance components,
    # so an R matrix sized to the fixed effects alone is silently the wrong shape
    # and patsy rejects it ("wrong shape for coefs"). Pad with zeros: the test
    # constrains only the interaction block.
    n_params = len(np.asarray(res.params))
    R = np.zeros((len(terms), n_params))
    for i, term in enumerate(terms):
        R[i, names.index(term)] = 1.0
    test = res.wald_test(R, scalar=False)
    stat = float(np.squeeze(test.statistic))
    return stat, len(terms), float(test.pvalue)


def fit_band(df, domains, band):
    """(record, marginal slope frame) for one band."""
    formula, ref = domain_formula(domains)
    t0 = time.time()
    res, warn = mm.fit_cell(df, VC_DOMAIN, formula=formula)
    elapsed = time.time() - t0

    chi2, ddf, p_omni = omnibus_wald(res, domains, ref)
    slopes = marginal_slopes(res, domains, ref)
    slopes.insert(0, 'band', band)

    vc = mm.vcomp_by_name(res)
    rec = {
        'band': band, 'reference_domain': ref,
        'n_rows': int(len(df)), 'n_subjects': int(df['subject'].nunique()),
        'n_parcels': int(df['parcel'].nunique()),
        'n_channels': int(df['channel_uid'].nunique()),
        'omnibus_chi2': chi2, 'omnibus_df': ddf, 'p_omnibus': p_omni,
        'var_subj_int': float(vc.get('subj_int', np.nan)),
        'var_subj_slope': float(vc.get('subj_slope', np.nan)),
        'var_subj_parcel_slope': float(vc.get('subj_parcel_slope', np.nan)),
        'var_channel': float(vc.get('channel', np.nan)),
        'var_resid': float(res.scale),
        'converged': bool(res.converged),
        'fit_seconds': elapsed, 'n_warnings': len(warn),
        'warnings': ' | '.join(sorted(set(warn)))[:400],
    }
    logger.info('%-11s | %6d rows %2d parcels %4d chan | omnibus chi2 %7.2f '
                'df %d p %.4g | %.0fs%s', band, rec['n_rows'], rec['n_parcels'],
                rec['n_channels'], chi2, ddf, p_omni, elapsed,
                '' if res.converged else '  NOT CONVERGED')
    for r in slopes.itertuples():
        logger.info('    %-10s beta %+.5f (SE %.5f) z %+5.2f p %.4g%s',
                    r.domain, r.beta_pain, r.se, r.z, r.p,
                    '   [reference]' if r.is_reference
                    else f'   vs ref {r.diff_from_ref:+.5f} p {r.diff_p:.3g}')
    return rec, slopes


# ============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--band-set', choices=list(BAND_SETS),
                    default='paper_bands_6_hg200')
    ap.add_argument('--roi-scheme', default='pain_domains_v2',
                    help='Scheme whose DISPLAY names are the domains and whose '
                         'patterns are the atlas parcels.')
    ap.add_argument('--hemisphere-separate', action='store_true',
                    help='Keep left and right parcels distinct. Doubles the '
                         'parcel count and halves the contacts behind each.')
    ap.add_argument('--bands', nargs='*', default=None,
                    help='Subset of bands, for a timing test.')
    ap.add_argument('--view-dir', default=None)
    ap.add_argument('--reference-run', default=str(reference_run.CONTPAIN_HEATMAP))
    ap.add_argument('--mask-label', default=None)
    ap.add_argument('--cohort', choices=['reference', 'eligible-discovery'],
                    default='reference')
    ap.add_argument('--allow-cohort-drift', action='store_true')
    ap.add_argument('--notch-half-width-hz', type=float, default=None)
    ap.add_argument('--fdr-q', type=float, default=0.05)
    ap.add_argument('--run-name', default=RUN_NAME)
    ap.add_argument('--replot', default=None,
                    help='Re-render the figures from an existing run directory '
                         'and exit. Nothing is refitted.')
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    io.warn_if_dirty()

    if args.replot:
        run_dir = Path(args.replot)
        cells = io.read_table(run_dir / 'domain_bands.parquet', on_stale='warn')
        slopes = io.read_table(run_dir / 'domain_slopes.parquet', on_stale='warn')
        coverage = io.read_table(run_dir / 'parcel_coverage.parquet',
                                 on_stale='ignore')
        domains = [d for d in view_tables.roi_regions_for(
            {'roi_scheme': args.roi_scheme}) if d in set(slopes['domain'])]
        per_domain = (coverage.groupby('domain')
                      .agg(n_parcels=('parcel', 'nunique'),
                           n_contacts=('channel', 'size'),
                           n_subjects=('subject_id', 'nunique'))
                      .reindex(domains))
        summary_figure(run_dir, cells, slopes, per_domain, domains, args)
        figure(run_dir, cells, slopes, per_domain, domains, args)
        return

    ref_run = reference_run.load(args.reference_run)
    ref_run.describe()
    epoch_minutes = ref_run.view_params.get('epoch_minutes')

    view_dir = fullres_cells.resolve_view_dir(
        args.view_dir,
        mask_label=args.mask_label or ref_run.view_params.get('mask_label'),
        max_excluded_frac=ref_run.view_params.get('max_excluded_frac'),
        epoch_minutes=epoch_minutes)
    paths, _, _, subjects, _, _ = resolve_cohort(ref_run, view_dir,
                                                 cohort=args.cohort,
                                                 roi_scheme=args.roi_scheme)
    ref_run.assert_cohort_matches(subjects,
                                  allow_drift=args.allow_cohort_drift
                                  or args.cohort != 'reference')

    parcel_of, parcel_to_domain, coverage = parcel_domain_maps(
        paths, subjects, args.roi_scheme,
        collapse_hemisphere=not args.hemisphere_separate)
    domains = [d for d in view_tables.roi_regions_for({'roi_scheme': args.roi_scheme})
               if d in set(coverage['domain'])]
    logger.info('%d domains, %d parcels, %d contacts',
                len(domains), coverage['parcel'].nunique(), len(coverage))
    per_domain = (coverage.groupby('domain')
                  .agg(n_parcels=('parcel', 'nunique'),
                       n_contacts=('channel', 'size'),
                       n_subjects=('subject_id', 'nunique'))
                  .reindex(domains))
    logger.info('\n%s', per_domain.to_string())
    logger.info('parcels per domain:\n%s',
                coverage.groupby(['domain', 'parcel']).size().to_string())

    kept, bands, notched = band_table(args.band_set, args.notch_half_width_hz,
                                      epoch_minutes)
    want = list(bands) if args.bands is None else [b for b in bands if b in args.bands]

    t0 = time.time()
    index, values = fullres_cells.load_all_parcels(
        paths, subjects, parcel_of, list(kept.index), epoch_minutes=epoch_minutes)
    band_values, names = aggregate(values, kept, bands)
    del values
    logger.info('loaded and aggregated in %.0fs', time.time() - t0)

    records, slope_parts = [], []
    for bi, band in enumerate(names):
        if band not in want:
            continue
        frame = pd.DataFrame({
            'subject_id': index['subject_id'].to_numpy(),
            'channel': index['channel'].to_numpy(),
            'epoch_id': index['epoch_id'].to_numpy(),
            'pain_score': index['pain_score'].to_numpy(),
            'value': band_values[:, bi],
            'parcel': index['parcel'].to_numpy()})
        frame['domain'] = frame['parcel'].map(parcel_to_domain)
        df = mm.build_cell_frame(frame, extra_columns=('parcel', 'domain'))
        rec, slopes = fit_band(df, domains, band)
        records.append(rec)
        slope_parts.append(slopes)

    cells = pd.DataFrame(records)
    slopes = pd.concat(slope_parts, ignore_index=True)

    # BH over the per-domain slopes (one family) and over the omnibus tests
    # (another). Two questions, two families.
    for frame, col, out in ((slopes, 'p', 'p_bh'), (cells, 'p_omnibus', 'p_omnibus_bh')):
        m = frame[col].notna()
        frame[out] = np.nan
        if m.any():
            _, adj = cp.bh_fdr(frame.loc[m, col].to_numpy(), q=args.fdr_q)
            frame.loc[m, out] = adj
            frame[f'{out}_reject'] = frame[out] <= args.fdr_q

    from ieeg_ehr.views.view_config import ROI_SCHEME_CODES
    scheme_code = ROI_SCHEME_CODES.get(args.roi_scheme, 'paindomains')
    run_dir = config.analysis_run_dir(
        question=QUESTION, output_type=OUTPUT_TYPE,
        view_scheme=f'{args.band_set.replace("_", "")}-{scheme_code}',
        run_name=args.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info('run dir: %s', run_dir)

    params = {'formula': domain_formula(domains)[0],
              'variance_components': VC_DOMAIN,
              'parcel_level': 'Desikan-Killiany, hemispheres '
                              + ('separate' if args.hemisphere_separate
                                 else 'collapsed'),
              'reference_domain': domain_formula(domains)[1],
              'band_set': args.band_set, 'roi_scheme': args.roi_scheme,
              'roi_scheme_contents': roi_schemes.scheme_provenance(args.roi_scheme),
              'notched_bins_excluded': notched, 'fdr_q': args.fdr_q,
              'epoch_minutes': epoch_minutes,
              'omnibus': 'joint Wald chi2 on the interaction block; NOT an LRT, '
                         'because REML likelihoods are not comparable across '
                         'fixed-effects designs'}

    io.write_table(cells, run_dir / 'domain_bands.parquet', params=params,
                   parents=[str(Path(args.reference_run) / 'provenance.json'),
                            str(view_dir)],
                   subjects=sorted(subjects), script=SCRIPT,
                   extra={'status': DISCLAIMER, 'domain_caveat': DOMAIN_CAVEAT})
    io.write_table(slopes, run_dir / 'domain_slopes.parquet', params=params,
                   subjects=sorted(subjects), script=SCRIPT,
                   extra={'status': DISCLAIMER, 'domain_caveat': DOMAIN_CAVEAT,
                          'reading': 'beta_pain is each domain\'s marginal slope, a '
                                     'LINEAR COMBINATION of the reference slope and '
                                     'that domain\'s interaction term. diff_from_ref '
                                     'is the interaction coefficient itself.'})
    io.write_table(coverage, run_dir / 'parcel_coverage.parquet', params=params,
                   script=SCRIPT)
    io.write_run_provenance(run_dir, script=SCRIPT, params=params,
                            parents=[str(view_dir)], subjects=sorted(subjects),
                            extra={'status': DISCLAIMER,
                                   'domain_caveat': DOMAIN_CAVEAT,
                                   'mask_content': CONFOUND_CAVEAT})
    summary_figure(run_dir, cells, slopes, per_domain, domains, args)
    figure(run_dir, cells, slopes, per_domain, domains, args)
    io.log_analysis('domain-level mixed models: pain x processing domain, one fit '
                    'per band, parcel as a nested random slope (EXPLORATORY)',
                    run_dir)
    print(run_dir)


def summary_figure(run_dir, cells, slopes, per_domain, domains, args):
    """WHICH domains change, in WHICH bands, and which way -- the headline grid.

    Deliberately NOT the omnibus. That test answers "do the domains differ from
    each other", which is one number per band and a question about contrasts; this
    answers "does this domain's power track pain at all in this band", which is
    what a reader wants first. Each cell is the domain's MARGINAL slope tested
    against ZERO, so a significant Control cell is not a contradiction -- it is
    the quasi-control firing, and it argues for a global or artifactual driver
    rather than nociception.

    30 cells, so the numbers are printed. A heatmap with a colourbar and no
    values would make a reader estimate a beta off a colour ramp when the exact
    value fits in the cell.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from ieeg_ehr.features import common

    bands = list(cells['band'])
    piv = (slopes.pivot_table(index='domain', columns='band', values='beta_pain')
           .reindex(index=domains, columns=bands))
    rej = (slopes.assign(_r=slopes['p_bh_reject'].fillna(False).astype(bool))
           .pivot_table(index='domain', columns='band', values='_r')
           .reindex(index=domains, columns=bands).fillna(0).astype(bool))
    pval = (slopes.pivot_table(index='domain', columns='band', values='p')
            .reindex(index=domains, columns=bands))

    arr = piv.to_numpy(dtype=float)
    cap = float(np.nanmax(np.abs(arr)))
    cm = plt.get_cmap('RdBu_r').copy()
    cm.set_bad('0.85')

    fig, ax = plt.subplots(figsize=(1.55 * len(bands) + 4.2,
                                    0.78 * len(domains) + 4.4))
    im = ax.imshow(arr, aspect='auto', cmap=cm, vmin=-cap, vmax=cap,
                   interpolation='nearest')
    common.draw_mask_outline(ax, rej.to_numpy(), linewidth=2.4)

    for i in range(len(domains)):
        for j in range(len(bands)):
            v, p, sig = arr[i, j], pval.to_numpy()[i, j], rej.to_numpy()[i, j]
            if not np.isfinite(v):
                continue
            # White text on the saturated ends, where black would vanish.
            shade = 'white' if abs(v) > 0.62 * cap else 'black'
            stars = ('***' if p < 1e-3 else '**' if p < 1e-2 else
                     '*' if p < 0.05 else '')
            ax.text(j, i - 0.13, f'{v:+.4f}', ha='center', va='center',
                    fontsize=9.5, color=shade,
                    fontweight='bold' if sig else 'normal')
            ax.text(j, i + 0.20, stars if sig else ('(ns)' if not stars else stars),
                    ha='center', va='center', fontsize=8, color=shade,
                    fontweight='bold' if sig else 'normal')

    ax.set_xticks(range(len(bands)))
    ax.set_xticklabels([f'{b}\n{BAND_SETS[args.band_set][b][0]}-'
                        f'{BAND_SETS[args.band_set][b][1]} Hz' for b in bands],
                       fontsize=9)
    ax.set_yticks(range(len(domains)))
    ax.set_yticklabels([f'{d}\n{int(per_domain.loc[d, "n_parcels"])} parcels, '
                        f'{int(per_domain.loc[d, "n_contacts"])} contacts, '
                        f'{int(per_domain.loc[d, "n_subjects"])} subj'
                        for d in domains], fontsize=8.5)
    ax.set_xlabel('')
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label('d log10 band power per pain point', fontsize=9)
    cb.ax.tick_params(labelsize=8)

    n_sig = int(rej.to_numpy().sum())
    ctrl = (int(rej.loc['Control'].sum()) if 'Control' in rej.index else 0)
    fig.suptitle('Which processing domains track pain, and in which bands\n'
                 f'{n_sig} of {arr.size} domain x band cells significant '
                 f'(BH q={args.fdr_q:g}); bold + outlined = significant',
                 fontsize=12.5)
    fig.tight_layout(rect=(0, 0.16, 1, 0.90))

    omni = ' · '.join(f'{r.band} p={r.p_omnibus:.3g}' for r in cells.itertuples())
    fig.text(0.01, 0.005,
             'EACH CELL IS THAT DOMAIN\'S MARGINAL PAIN SLOPE TESTED AGAINST '
             'ZERO -- not against the Control domain. A significant Control cell '
             'is therefore not a contradiction: Occipital and Auditory are the '
             'quasi-controls, and an effect there argues for a global or '
             'artifactual driver rather than nociception, so read that row first. '
             f'CONTROL ROW: {ctrl} of {len(bands)} cells significant. Stars are '
             'the uncorrected p (* <0.05, ** <0.01, *** <0.001); bold and the '
             f'outline are BH at q={args.fdr_q:g} over all {arr.size} cells. '
             'Slopes come from ONE mixed model per band over every parcel at '
             'once, with domain as a fixed effect and the atlas parcel as a '
             'random slope nested in subject, so no domain is carried by a single '
             'well-sampled parcel; each value is a linear combination of the '
             'reference slope and that domain\'s interaction term, with its SE '
             'from the fitted covariance. THE OMNIBUS -- whether the domains '
             f'differ FROM EACH OTHER -- is a separate question: {omni}. '
             f'{MODULATORY_CAVEAT} {DOMAIN_CAVEAT}\n{DISCLAIMER}',
             fontsize=6.4, va='bottom', ha='left', color='0.35', wrap=True)
    out = run_dir / 'fig_domain_summary.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info('wrote %s', out)
    return out


def figure(run_dir, cells, slopes, per_domain, domains, args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    bands = [b for b in cells['band']]
    fig, axes = plt.subplots(1, len(bands), figsize=(2.9 * len(bands) + 1.6, 5.4),
                             sharey=True, squeeze=False)
    xmax = float(np.nanmax(np.abs(np.concatenate(
        [slopes['ci_lo'].to_numpy(), slopes['ci_hi'].to_numpy()])))) * 1.08
    colours = {'Sensory': '#b03a2e', 'Affective': '#8e44ad',
               'Cognitive': '#2b6ca3', 'Memory': '#e08214', 'Control': '0.45'}
    y = np.arange(len(domains))

    for j, band in enumerate(bands):
        ax = axes[0][j]
        d = slopes[slopes['band'] == band].set_index('domain').reindex(domains)
        for i, dom in enumerate(domains):
            r = d.loc[dom]
            sig = bool(r.get('p_bh_reject', False))
            ax.errorbar(r['beta_pain'], i, xerr=1.96 * r['se'], fmt='o',
                        ms=8 if sig else 5.5, lw=1.8 if sig else 1.1, capsize=3,
                        color=colours.get(dom, '0.4'),
                        markerfacecolor=colours.get(dom, '0.4') if sig else 'white',
                        zorder=3)
        ax.axvline(0, color='0.5', lw=1.0, ls='--')
        omni = cells[cells['band'] == band].iloc[0]
        ax.set_title(f'{band}\nomnibus p = {omni["p_omnibus"]:.3g}', fontsize=9)
        ax.set_xlim(-xmax, xmax)
        ax.set_xlabel('d log10 power / pain point', fontsize=8)
        ax.tick_params(labelsize=7)
        if j == 0:
            ax.set_yticks(y)
            ax.set_yticklabels([f'{dom}\n{int(per_domain.loc[dom, "n_parcels"])} parc, '
                                f'{int(per_domain.loc[dom, "n_contacts"])} chan'
                                for dom in domains], fontsize=7.5)
            ax.set_ylim(len(domains) - 0.5, -0.5)

    ref = cells['reference_domain'].iloc[0]
    fig.suptitle('Pain-power slope by PROCESSING DOMAIN, one mixed model per band\n'
                 'filled = BH-significant slope; omnibus tests whether the domains '
                 f'differ at all (reference domain: {ref})', fontsize=11.5)
    fig.tight_layout(rect=(0, 0.17, 1, 0.90))
    fig.text(0.01, 0.005,
             'ONE MODEL PER BAND over every parcel at once: '
             'log10_power ~ NRS_within * C(domain) + NRS_submean + '
             '(NRS_within || subject) + (NRS_within || subject:parcel) + '
             '(1 | subject:channel). THE UNIT IS THE ATLAS PARCEL, not a '
             'hand-built ROI -- parcel enters as a random SLOPE so each parcel '
             'deviates around its domain rather than being averaged into it, '
             'which is what keeps one well-sampled parcel from carrying a domain. '
             'Each marker is a MARGINAL slope, a linear combination of the '
             'reference slope and that domain\'s interaction term, with the SE '
             'from the fitted covariance -- not the interaction coefficient, '
             'which is only the DIFFERENCE from the reference. The omnibus is a '
             'joint Wald chi2 on the interaction block, not an LRT: REML '
             'likelihoods are not comparable across fixed-effects designs. '
             'Parcel is NESTED in subject because statsmodels cannot fit a '
             'crossed parcel term -- it silently nests it instead -- so the '
             'domain SE does not account for a parcel deviating consistently '
             f'across patients. {DOMAIN_CAVEAT}\n{DISCLAIMER}',
             fontsize=6.3, va='bottom', ha='left', color='0.35', wrap=True)
    out = run_dir / 'fig_domain_slopes.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info('wrote %s', out)


if __name__ == '__main__':
    main()
