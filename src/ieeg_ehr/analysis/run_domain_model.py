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

TWO UNITS, `--unit parcel` (the original) AND `--unit roi`
----------------------------------------------------------
`--unit parcel` is the design this script was written for. The band-power runs
used hand-built ROIs (S1, S2/PO, dlPFC...) that already collapse several atlas
parcels each; stacking a domain on top of those averages twice, and the second
average hides exactly what a domain claim needs to survive -- whether every
parcel in the domain agrees, or one well-sampled parcel is carrying it. So the
intermediate layer was removed: the unit is the ATLAS PARCEL (Desikan-Killiany,
hemispheres collapsed), the domain is the fixed effect, and parcel enters as a
random SLOPE so each parcel deviates around its domain's mean.

`--unit roi` (2026-09-21) puts the ROI layer back and takes the region out of
the model entirely. One row is still ONE CHANNEL x ONE EPOCH -- nothing is
averaged, which is the difference from the design the ROI layer was originally
rejected for -- but a channel's region is now the ROI, and the ROI's only job is
to say which domain the channel is in:

    log10_power ~ NRS_within * C(domain) + NRS_submean
                  + (NRS_within || subject) + (1 | subject:channel)

WHY. `subj_parcel_slope` is the smallest variance component in every band of
every run measured (ratios to residual 1.6e-3 to 4.3e-2, against a boundary
tolerance of 1e-3), which flattens the likelihood in that direction and is the
documented cause of the non-positive-definite Hessians. Dropping it while
keeping the ATLAS PARCEL as the unit would be the worst combination -- a unit
finer than the claim needs, with nothing in the model acknowledging it. Moving
to the ROI makes the unit match the level the domains are defined at.

WHAT THAT GIVES UP, and where it went. With no region term, nothing in the fit
guards against one well-sampled ROI carrying its domain. That guard does not
disappear, it MOVES: the region-level consistency map
(`run_bandpower_mixed` + `plot_bandpower_consistency`) fits every ROI separately
and reports how many subjects share each one's sign, including the ROIs no
domain contains. Read the two together; neither alone is the answer.

`--unit roi` also needs a DOMAIN scheme (`pain_domains_v3`), which keeps the
ROI->domain membership visible. `pain_domains_v2` fuses the ROI layer away into
a label->domain map, which is the right shape for the parcel unit and the wrong
one here.

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
from ieeg_ehr.analysis import dx_state, fullres_cells, med_state, mixed_model as mm
from ieeg_ehr.analysis import reference_run, view_tables
from ieeg_ehr.analysis.run_bandpower_mixed import (BAND_SETS, DX_CAVEAT,
                                                     band_table, aggregate)
from ieeg_ehr.analysis.run_fullres_grid import CONFOUND_CAVEAT, resolve_cohort
from ieeg_ehr.config import roi_schemes
from ieeg_ehr.views import channel_meta

logger = logging.getLogger(__name__)

SCRIPT = 'ieeg_ehr/analysis/run_domain_model.py'
QUESTION = 'bandpower'
OUTPUT_TYPE = 'domain_model'
RUN_NAME = 'domain_mixedlm'

# The domain that every contrast is read against. `Control` here is a PROCESSING
# DOMAIN -- Auditory + Occipital, the regions not expected to encode pain -- and
# it is the reference precisely because it is the scheme's negative control.
#
# IT HAS NOTHING TO DO WITH THE NON-MDD GROUP. The diagnosis strata are named
# `non-MDD` and `MDD` for exactly this reason: an earlier version called them
# `control` and `case`, which put a "Control" circuit row and a "control"
# stratum colour on the same figure and made it unreadable.
REFERENCE_DOMAIN = 'Control'

#: Diagnosis stratum labels. These are the DATA VALUES in the slopes table:
#: ASCII, stable, and already written into stored artifacts, so they do not
#: change for presentation reasons.
DX_LABEL = 'MDD'
NON_DX_LABEL = 'non-MDD'

#: What a FIGURE calls them, following `plot_dx_pain.arm_labels`: the negative
#: arm is not a matched control group, it is everyone in the cohort without the
#: code, and `MDD+` / `MDD-` says that and nothing more. Kept separate from the
#: data values above so a stored table stays ASCII and greppable while the
#: figures use the typographic minus.
DX_DISPLAY = 'MDD+'
NON_DX_DISPLAY = 'MDD−'

#: Runs written before 2026-09-21 stored 'case'/'control'. `--replot` has to
#: keep working on them, so the old values are mapped on READ rather than the
#: files being rewritten -- the stored tables are immutable artifacts.
_LEGACY_STRATA = {'case': DX_LABEL, 'control': NON_DX_LABEL}


def norm_strata(slopes):
    """Map any legacy stratum values to the current labels, on a copy."""
    if 'stratum' not in slopes.columns:
        return slopes
    out = slopes.copy()
    out['stratum'] = out['stratum'].replace(_LEGACY_STRATA)
    return out

#: Random effects. `subj_parcel_slope` is the parcel term, nested in subject --
#: see the module docstring for why it cannot be crossed here.
VC_DOMAIN = {
    'subj_int': '1',
    'subj_slope': '0 + NRS_within',
    'subj_parcel_slope': '0 + C(parcel):NRS_within',
    'channel': '0 + C(channel_uid)',
}

#: `VC_DOMAIN` without the parcel-nested slope, for `--drop-parcel-term`.
#:
#: WHY THIS EXISTS. `subj_parcel_slope` is the smallest component in every band
#: of every run so far — measured ratios to the residual variance run 1.6e-3 to
#: 4.3e-2, against a `BOUNDARY_TOL` of 1e-3 — so it sits within an order of
#: magnitude of the boundary throughout. `VC_CHANGE` already documents what that
#: costs: asking for a variance the design has set to near zero flattens the
#: likelihood in that direction, the optimiser never settles, and the fit reports
#: trouble even though the fixed effects are fine. On the change-score grids
#: dropping the analogous component took convergence from 3/10 to 10/10 at 1/24th
#: the fit time.
#:
#: IT IS A HYPOTHESIS, NOT A FIX, and the comparison is the point: if the four
#: well-conditioned bands' fixed effects barely move, the term is not
#: load-bearing and dropping it is a safe simplification; if they do move, the
#: term is carrying real parcel-level heterogeneity and the badly-conditioned
#: bands need a different remedy. Never swap the default on one band's behaviour.
VC_DOMAIN_NO_PARCEL = {k: v for k, v in VC_DOMAIN.items()
                       if k != 'subj_parcel_slope'}

DISCLAIMER = ('EXPLORATORY -- discovery cohort, NOMINATIONS NOT FINDINGS. '
              'Not confirmed out of sample.')

# One palette for every domain figure in the project. Kept here, and imported by
# plot_domain_model, so the two cannot drift: this module's private copy had gone
# stale at the v1 domain names and still said 'Memory', which meant Modulatory
# fell through to the '0.4' default and rendered in the SAME grey as Control.
# Grey for Control is deliberate -- the reference domain should recede.
#
# Known limitation, deliberately not repainted: Cognitive and Affective are
# ~1.5 dE apart under deuteranopia (validated 2026-09-18), i.e. indistinguishable
# to a deuteranopic reader. That is tolerable ONLY because no domain figure uses
# colour as its identity channel -- every domain carries a text label, as a panel
# title in the by-domain figure and a row label in the by-band one. Do not add a
# figure that relies on these hues alone to tell domains apart.
DOMAIN_COLOURS = {'Sensory': '#b03a2e', 'Affective': '#8e44ad',
                  'Cognitive': '#2b6ca3', 'Modulatory': '#e08214',
                  'Control': '0.45'}

MODULATORY_CAVEAT = (
    'THE MODULATORY DOMAIN IS M1 ALONE. Brainstem has zero contacts in this '
    'cohort -- measured over 7,068 contacts in 51 subjects, no brain-stem, pons, '
    'medulla, midbrain or periaqueductal label at all -- so the descending '
    'modulatory arm of the framework is represented by one parcel, 82 contacts '
    'and 20 subjects. It is the weakest row in every figure and the only domain '
    'whose internal consistency cannot be checked, because one parcel has no '
    'parcel-to-parcel agreement to inspect.')

#: What `pain_domains_v3` has to say for itself instead. The v2 text below is
#: NOT edited: it is stamped into artifacts that already exist, and a caveat
#: that silently changes meaning is worse than two caveats.
DOMAIN_CAVEAT_V3 = (
    'INSULA IS IN, SPLIT BY COORDINATE RATHER THAN BY LABEL: aIns joins '
    'Affective and pIns joins Sensory, which is the anterior/posterior '
    'dissociation the framework rests on. The split is a MEDIAN CUT ON MNI y '
    'over this cohort\'s insular contacts -- see analysis/insula_ap.py and the '
    '`insula_split` block of this provenance for the threshold. It is coarse: '
    'the threshold is this cohort\'s median and not an anatomical landmark, the '
    'insula is folded so a plane normal to y is not its anterior-posterior '
    'axis, and contacts near the line are near-arbitrary. It supports "on '
    'average more anterior", not "this contact is in aIns"; the Destrieux/a2009s '
    'assignment remains the correct fix. THALAMUS is in Sensory by assignment '
    'and carries the same class of caveat, accepted rather than avoided: DK '
    'gives one thalamus parcel while the ascending pathway is VPL/VPM '
    'specifically, so a Sensory effect here partly reflects medial and dorsal '
    'nuclei that sit in the affective pathway. Basal Ganglia, Hippocampus, PCC, '
    'Parietal (other), MTL (other) and Lateral Temporal are UNASSIGNED and '
    'therefore dropped from the model; PCC carries one of the larger '
    'low-frequency effects in this cohort, so that is a real and deliberate '
    'loss. Every one of them IS in the region-level consistency map, which is '
    'where to look for them.')

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




def unit_count(args, n, short=False):
    """'4 ROIs' / '1 ROI'. Modulatory is one region and prints on every figure."""
    return f'{int(n)} {unit_word(args, plural=int(n) != 1, short=short)}'


def unit_word(args, plural=True, short=False):
    """What one row's region is CALLED, on a figure. Not cosmetic.

    A `--unit roi` run's domains are made of ROIs, not atlas parcels, and a tick
    label reading "4 parcels" under Sensory would name a unit the run did not
    use -- the exact confusion the ROI/parcel distinction exists to prevent.
    """
    word = 'ROI' if getattr(args, 'unit', 'parcel') == 'roi' else 'parcel'
    if short:
        word = 'ROI' if word == 'ROI' else 'parc'
    if plural and word != 'parc':
        word += 's'
    return word


def domain_caveat(roi_scheme):
    """The caveat text this SCHEME earns, not a constant.

    v3 puts insula in and has to say how; v2 leaves it out and has to say why.
    Stamping v2's text onto a v3 artifact -- or onto a v3 FIGURE -- would
    describe a model the run did not fit.
    """
    return DOMAIN_CAVEAT_V3 if roi_scheme == 'pain_domains_v3' else DOMAIN_CAVEAT


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


def roi_domain_maps(roi_by_subject, subjects, domain_scheme):
    """({subject: {channel: ROI}}, {ROI: domain}, coverage, unassigned counts).

    THE ROI-LEVEL UNIT. Same three return values as `parcel_domain_maps` and the
    same column names, so everything downstream -- the model frame, the coverage
    table, the figures -- is unchanged; only what a 'parcel' IS changes, from an
    atlas label to the hand-built ROI that several labels collapse into.

    WHY BOTH EXIST. The parcel version was written because stacking a domain on
    top of ROIs averages twice and hides whether one well-sampled parcel carries
    a domain -- so it kept the parcel and let it deviate as a random slope. That
    random slope is the model's smallest variance component in every band
    measured, sits within an order of magnitude of the boundary, and is the
    documented cause of the ill-conditioned fits. Dropping it while KEEPING the
    atlas parcel as the unit would be the worst of both: a finer unit than the
    claim needs, with nothing in the model acknowledging it. Going to the ROI
    makes the unit match the level the domains are actually defined at, and the
    ROI then never enters the model at all -- it is only the lookup that says
    which domain a channel is in.

    WHAT IS GIVEN UP, stated rather than hidden: with no ROI term, nothing in
    the fit guards against one well-sampled ROI carrying its domain. That guard
    moves OUT of the model and into the region-level consistency map, which
    shows every ROI's own slope and how many subjects share its sign --
    including the ROIs no domain contains.

    TAKES THE ROI MAP IT IS GIVEN rather than building one. `resolve_cohort`
    already builds `{subject: {channel: ROI}}` for the base scheme -- and its
    `no_roi` result already decided the cohort -- so rebuilding here would read
    every channel_meta a second time, run the insula split a second time, and
    leave two maps that could disagree. This function is pure: it filters that
    map to the ROIs a domain claims.

    The insula split therefore happens upstream, in `roi_maps`, driven by the
    base scheme's `coordinate_regions` -- which is why it does not appear here.
    """
    spec = roi_schemes.domain_scheme(domain_scheme)
    roi_to_domain = spec['roi_to_domain']

    per_subject, rows = {}, []
    unassigned = {}
    for sid, mapping in roi_by_subject.items():
        if sid not in subjects:
            continue
        keep = {}
        for ch, roi in mapping.items():
            if roi in roi_to_domain:
                keep[ch] = roi
            else:
                unassigned[roi] = unassigned.get(roi, 0) + 1
        if keep:
            per_subject[sid] = keep
            rows.extend({'subject_id': sid, 'channel': ch, 'parcel': roi}
                        for ch, roi in keep.items())

    if unassigned:
        # These are REAL regions of the base scheme that no domain claims --
        # PCC, Hippocampus, Basal Ganglia and friends. Named with counts rather
        # than dropped quietly, because "the domains do not cover the cohort's
        # coverage" is a result about the framework, not a detail of the code.
        logger.warning('%d contact(s) are in a region NO domain claims and are '
                       'dropped from the model: %s',
                       sum(unassigned.values()),
                       ', '.join(f'{k} {v}' for k, v in
                                 sorted(unassigned.items(), key=lambda kv: -kv[1])))

    coverage = pd.DataFrame(rows)
    if coverage.empty:
        raise SystemExit('no channel mapped to an ROI in a displayed domain')
    coverage['domain'] = coverage['parcel'].map(roi_to_domain)
    return per_subject, roi_to_domain, coverage, unassigned


# ============================================================================
# THE MODEL
# ============================================================================

def domain_formula(domains, med=False, dx=False):
    """The fixed-effects formula, with `Control` as the reference if present.

    With `med=True` the design is the FULL THREE-WAY, pain x domain x medication:

        NRS_within * C(domain) * med_within + NRS_submean + med_submean

    which nests every weaker version and lets all four questions be asked of one
    fit -- does the pain slope differ by domain, does medication shift power, does
    medication change the pain slope, and does THAT differ by domain. 22 fixed
    effects at five domains, which is cheap: the cost of these fits is the
    variance components, not the design matrix.

    `med_within` is subject-mean-centred, so a coefficient read at med_within = 0
    is the estimate AT THE PATIENT'S OWN AVERAGE MEDICATION LEVEL -- not
    marginally over dosing, and not in an unmedicated state. `med_submean` is the
    between-patient nuisance term, present for the same reason NRS_submean is:
    subject mean pain correlates +0.685 with subject proportion medicated, so
    without it the within-patient medication effect absorbs a pain difference
    wearing a medication label.
    """
    ref = REFERENCE_DOMAIN if REFERENCE_DOMAIN in domains else sorted(domains)[0]
    dom = f"C(domain, Treatment('{ref}'))"
    if med and dx:
        raise ValueError('--med-model and --dx-model together would be a FOUR-way '
                         'design (pain x domain x medication x diagnosis). At 17 '
                         'cases the three-way cells are already thin; a four-way '
                         'is not estimable on this cohort. Run them separately.')
    if med:
        return (f'log10_power ~ NRS_within * {dom} * med_within '
                '+ NRS_submean + med_submean'), ref
    if dx:
        # THE FULL THREE-WAY, pain x domain x diagnosis. Same shape as the
        # medication version and the same reason for it: one fit answers all
        # four questions -- does the pain slope differ by domain, is mean power
        # different in cases, is the pain slope different in cases, and does
        # THAT difference itself differ by domain, which is the circuit-level
        # question ("are different circuits changed by depression?").
        #
        # `dx_state` enters RAW, with no within/between split, which is the one
        # real difference from the medication design. A diagnosis is a
        # subject-level constant, so it has no within-patient part to separate;
        # `med_within`/`med_submean` exist because dosing varies inside a
        # patient and correlates with their mean pain, and neither is true here.
        # There is correspondingly no "at the patient's own average" reading: a
        # coefficient read at dx_state = 0 is simply the control stratum.
        return (f'log10_power ~ NRS_within * {dom} * dx_state '
                '+ NRS_submean'), ref
    return f'log10_power ~ NRS_within * {dom} + NRS_submean', ref


def marginal_contrast(res, domains, ref, base_term, term_label):
    """Each domain's marginal value of `base_term`, as a linear combination.

    Generalises `marginal_slopes` to any term that is crossed with domain:
    pass 'NRS_within' for the pain slope, 'med_within' for the medication effect,
    'NRS_within:med_within' for the pain x medication interaction. For domain d
    the value is `base + base:domain[T.d]`, and its SE needs the covariance of
    the two, not the square root of the sum of variances.

    patsy spells the interaction with the domain factor inline, and the ORDER of
    the factors in a term name is patsy's, not ours -- so the term is found by
    matching its PARTS rather than by string-building a name that would silently
    fail to match and hand back the reference value for every domain.
    """
    from scipy import stats

    names = list(res.fe_params.index)
    cov = np.asarray(res.cov_params())[:len(names), :len(names)]
    beta = res.fe_params.to_numpy()
    base_parts = set(base_term.split(':'))

    def find_base():
        for i, n in enumerate(names):
            if set(n.split(':')) == base_parts:
                return i
        return None

    def find_interaction(dom):
        for i, n in enumerate(names):
            parts = n.split(':')
            dom_part = [q for q in parts if q.endswith(f'[T.{dom}]')]
            rest = {q for q in parts if not q.endswith(f'[T.{dom}]')}
            if dom_part and rest == base_parts:
                return i
        return None

    bi = find_base()
    if bi is None:
        logger.warning('no base term %r in the fit; skipping %s', base_term,
                       term_label)
        return pd.DataFrame()

    rows = []
    for dom in domains:
        c = np.zeros(len(names))
        c[bi] = 1.0
        ii = None
        if dom != ref:
            ii = find_interaction(dom)
            if ii is None:
                logger.warning('no %s x %s interaction term found', base_term, dom)
                continue
            c[ii] = 1.0
        est = float(c @ beta)
        se = float(np.sqrt(c @ cov @ c))
        z = est / se if se > 0 else np.nan
        rows.append({'term': term_label, 'domain': dom, 'is_reference': dom == ref,
                     'beta': est, 'se': se, 'z': z,
                     'p': float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else np.nan,
                     'ci_lo': est - 1.96 * se, 'ci_hi': est + 1.96 * se,
                     'diff_from_ref': float(beta[ii]) if ii is not None else 0.0,
                     'diff_se': (float(np.sqrt(cov[ii, ii])) if ii is not None
                                 else np.nan),
                     'diff_p': (float(res.pvalues.iloc[ii]) if ii is not None
                                else np.nan)})
    return pd.DataFrame(rows)


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


def domain_simple_slopes(res, domains, ref, moderator='dx_state'):
    """Each domain's pain slope IN EACH STRATUM, as explicit linear combinations.

    For domain d the control slope is `NRS_within + NRS_within:domain[T.d]` and
    the case slope adds `NRS_within:moderator` and the three-way
    `NRS_within:domain[T.d]:moderator`. Four terms, so the SE needs the full
    quadratic form `c' V c` -- summing variances would be wrong by the three
    covariance pairs, and they are not small.

    This exists so the circuit figure can draw both strata WITHOUT refitting
    either arm on its own. Refitting arms separately is the thing the whole
    interaction design is there to avoid.

    Returns one row per (domain, stratum). `marginal_contrast` already gives the
    control slope and the difference; this gives the CASE slope with an SE,
    which neither of those does.
    """
    from scipy import stats

    names = list(res.fe_params.index)
    cov = np.asarray(res.cov_params())[:len(names), :len(names)]
    beta = res.fe_params.to_numpy()

    def find(*required, domain=None):
        """Index of the term whose factor set is exactly `required` (+ domain)."""
        want = set(required)
        for i, n in enumerate(names):
            parts = n.split(':')
            dom_parts = [q for q in parts if '[T.' in q]
            rest = {q for q in parts if '[T.' not in q}
            if rest != want:
                continue
            if domain is None:
                if not dom_parts:
                    return i
            else:
                if len(dom_parts) == 1 and dom_parts[0].endswith(f'[T.{domain}]'):
                    return i
        return None

    rows = []
    for dom in domains:
        d = None if dom == ref else dom
        # (index, required) for the four pieces; a missing non-reference term is
        # fatal for that domain rather than silently treated as zero.
        base = find('NRS_within')
        ix_dom = None if d is None else find('NRS_within', domain=d)
        ix_mod = find('NRS_within', moderator)
        ix_three = None if d is None else find('NRS_within', moderator, domain=d)
        if base is None or ix_mod is None:
            logger.warning('domain simple slopes: missing base terms, skipped')
            return pd.DataFrame()
        if d is not None and (ix_dom is None or ix_three is None):
            logger.warning('domain simple slopes: missing %s terms for %r', d, dom)
            continue

        for stratum, use_mod in ((NON_DX_LABEL, False), (DX_LABEL, True)):
            c = np.zeros(len(names))
            c[base] = 1.0
            if ix_dom is not None:
                c[ix_dom] = 1.0
            if use_mod:
                c[ix_mod] = 1.0
                if ix_three is not None:
                    c[ix_three] = 1.0
            est = float(c @ beta)
            se = float(np.sqrt(c @ cov @ c))
            z = est / se if se > 0 else np.nan
            rows.append({'term': 'pain_slope_by_stratum', 'domain': dom,
                         'stratum': stratum, 'is_reference': dom == ref,
                         'beta': est, 'se': se, 'z': z,
                         'p': (float(2 * stats.norm.sf(abs(z)))
                               if np.isfinite(z) else np.nan),
                         'ci_lo': est - 1.96 * se, 'ci_hi': est + 1.96 * se})
    return pd.DataFrame(rows)


def omnibus_block(res, domains, ref, base_term, label):
    """(chi2, df, p) for "does `base_term` differ across domains".

    Joint Wald over that term's domain interactions. NOT an LRT: REML
    likelihoods are not comparable across fixed-effects designs.
    """
    names = list(res.fe_params.index)
    base_parts = set(base_term.split(':'))
    idx = []
    for i, n in enumerate(names):
        parts = n.split(':')
        dom_part = [q for q in parts if any(q.endswith(f'[T.{d}]')
                                            for d in domains if d != ref)]
        rest = {q for q in parts if q not in dom_part}
        if dom_part and rest == base_parts:
            idx.append(i)
    if not idx:
        return {'block': label, 'chi2': np.nan, 'df': 0, 'p': np.nan}
    R = np.zeros((len(idx), len(np.asarray(res.params))))
    for r, i in enumerate(idx):
        R[r, i] = 1.0
    test = res.wald_test(R, scalar=False)
    return {'block': label, 'chi2': float(np.squeeze(test.statistic)),
            'df': len(idx), 'p': float(test.pvalue)}


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


def fit_band(df, domains, band, med=False, dx=False, out_dir=None,
             drop_parcel=False):
    """(record, contrast frame) for one band, written to disk as it completes.

    `out_dir` makes the run INCREMENTAL. The first version of this script created
    its run directory only after all six fits, so a job that died at band five
    left nothing at all -- no folder, no partial result, 40 minutes unaccounted
    for -- and there was nothing to look at while it ran. Each band now lands the
    moment it is finished.
    """
    formula, ref = domain_formula(domains, med=med, dx=dx)
    vc = VC_DOMAIN_NO_PARCEL if drop_parcel else VC_DOMAIN
    t0 = time.time()
    res, warn = mm.fit_cell(df, vc, formula=formula)
    elapsed = time.time() - t0

    chi2, ddf, p_omni = omnibus_wald(res, domains, ref)

    # Every term that is crossed with domain gets the same treatment: a marginal
    # value per domain from a linear combination, never the raw interaction
    # coefficient (which is only the difference from the reference).
    wanted = [('NRS_within', 'pain')]
    if med:
        wanted += [('med_within', 'med'),
                   ('NRS_within:med_within', 'pain_x_med')]
    if dx:
        # `pain_x_dx` per domain IS the circuit-level question: within this
        # circuit, how much does the pain slope differ between diagnosis
        # strata? `dx` alone is the nuisance level difference in mean power.
        wanted += [('dx_state', 'dx'),
                   ('NRS_within:dx_state', 'pain_x_dx')]
    parts = [marginal_contrast(res, domains, ref, base, label)
             for base, label in wanted]
    if dx:
        # Both strata's slopes per domain, from THIS fit. Appended to the same
        # long frame so the figure has one table to read, distinguished by the
        # `stratum` column (NaN for every non-dx term).
        parts.append(domain_simple_slopes(res, domains, ref))
    slopes = pd.concat([x for x in parts if len(x)], ignore_index=True)
    slopes.insert(0, 'band', band)

    blocks = [omnibus_block(res, domains, ref, base, label)
              for base, label in wanted]

    vc = mm.vcomp_by_name(res)
    rec = {
        'band': band, 'reference_domain': ref, 'med_model': bool(med),
        'dx_model': bool(dx), 'drop_parcel_term': bool(drop_parcel),
        'variance_components': ','.join(sorted(vc)),
        **({'n_subjects_case':
            int(df.loc[df['dx_state'] == 1, 'subject'].nunique()),
            'n_subjects_control':
            int(df.loc[df['dx_state'] == 0, 'subject'].nunique())} if dx else {}),
        **{f'omnibus_{b["block"]}_chi2': b['chi2'] for b in blocks},
        **{f'omnibus_{b["block"]}_df': b['df'] for b in blocks},
        **{f'p_omnibus_{b["block"]}': b['p'] for b in blocks},
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
    for b in blocks:
        logger.info('    omnibus %-11s chi2 %7.2f df %d p %.4g', b['block'],
                    b['chi2'], b['df'], b['p'])
    for r in slopes.itertuples():
        logger.info('    %-10s %-10s beta %+.5f (SE %.5f) z %+5.2f p %.4g%s',
                    r.term, r.domain, r.beta, r.se, r.z, r.p,
                    '   [ref]' if r.is_reference
                    else f'   vs ref {r.diff_from_ref:+.5f} p {r.diff_p:.3g}')

    if out_dir is not None:
        (out_dir / 'bands').mkdir(parents=True, exist_ok=True)
        io.write_table(pd.DataFrame([rec]), out_dir / 'bands' / f'{band}.parquet',
                       params={'band': band}, script=SCRIPT)
        io.write_table(slopes, out_dir / 'bands' / f'{band}_slopes.parquet',
                       params={'band': band}, script=SCRIPT)

        # RESIDUALS, so the diagnostics figure needs no refit. A domain refit is
        # 5-20 minutes, unlike a band cell's 2 seconds, so saving the frame to
        # refit from would not have helped -- the fitted values themselves are
        # what is expensive, and they are ~2 MB a band.
        try:
            from ieeg_ehr.analysis.plot_bandpower_checks import conditional_parts
            fitted, resid = conditional_parts(res, df)
            io.write_table(
                pd.DataFrame({'subject': df['subject'].to_numpy(),
                              'epoch_id': df['epoch_id'].to_numpy(),
                              'domain': df['domain'].to_numpy(),
                              'fitted': fitted.astype(np.float32),
                              'resid': resid.astype(np.float32)}),
                out_dir / 'bands' / f'{band}_residuals.parquet',
                params={'band': band,
                        'residual': 'conditional (y - Xb - Zu)'},
                script=SCRIPT, float_dtype=np.float32)
        except Exception as exc:                        # noqa: BLE001
            logger.warning('could not save residuals for %s: %s', band, exc)
    return rec, slopes


# ============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--band-set', choices=list(BAND_SETS),
                    default='paper_bands_6_hg200')
    ap.add_argument('--roi-scheme', default='pain_domains_v2',
                    help='Scheme whose DISPLAY names are the domains. With '
                         "--unit parcel its patterns are the atlas parcels; with "
                         '--unit roi it must name a DOMAIN scheme '
                         '(pain_domains_v3), whose members are ROIs.')
    ap.add_argument('--unit', choices=['parcel', 'roi'], default='parcel',
                    help="What one channel's region IS. 'parcel' (the original) "
                         'is the Desikan-Killiany label and enters the model as '
                         'a nested random slope. "roi" is the hand-built region '
                         '(S1, dACC, aIns...) and enters the model NOT AT ALL -- '
                         'it is only the lookup that says which domain a channel '
                         'is in, so --drop-parcel-term is implied. Use it with a '
                         'DOMAIN scheme, which keeps the ROI layer visible; '
                         'pain_domains_v2 fuses it away and cannot serve.')
    ap.add_argument('--insula-threshold', type=float, default=None,
                    help='Pin the anterior/posterior insula cut (MNI y, mm) '
                         "instead of re-deriving this cohort's median. Only used "
                         'by a scheme whose base declares coordinate regions '
                         '(pain_domains_v3). PASS THE THRESHOLD FROM THE '
                         'plot_insula_split RUN THAT WAS REVIEWED, so the model '
                         'uses the split a human actually looked at.')
    ap.add_argument('--hemisphere-separate', action='store_true',
                    help='Keep left and right parcels distinct. Doubles the '
                         'parcel count and halves the contacts behind each. '
                         '--unit roi ROIs are hemisphere-collapsed by '
                         'construction, so this is refused there.')
    ap.add_argument('--bands', nargs='*', default=None,
                    help='Subset of bands, for a timing test.')
    ap.add_argument('--view-dir', default=None)
    ap.add_argument('--reference-run', default=str(reference_run.CONTPAIN_HEATMAP))
    ap.add_argument('--mask-label', default=None)
    ap.add_argument('--cohort', choices=['reference', 'eligible-discovery'],
                    default='reference')
    ap.add_argument('--allow-cohort-drift', action='store_true')
    ap.add_argument('--notch-half-width-hz', type=float, default=None)
    ap.add_argument('--med-model', choices=['none', 'interaction'], default='none',
                    help="'interaction' fits the FULL THREE-WAY design, "
                         'pain x domain x medication, which nests every weaker '
                         'version: the pain slope per domain adjusted for '
                         'medication, the medication effect per domain, the '
                         'pain x medication interaction, and whether THAT differs '
                         'by domain.')
    ap.add_argument('--drug-set', choices=list(med_state.DRUG_SETS),
                    default='opioids')
    ap.add_argument('--exclude-drug-set', choices=list(med_state.DRUG_SETS),
                    default=None,
                    help='DROP epochs dosed with this set but NOT with '
                         '--drug-set. With --drug-set opioids '
                         '--exclude-drug-set non_opioid_analgesics the contrast '
                         'becomes opioid-dosed vs ANALGESIC-FREE, instead of '
                         'opioid-dosed vs (nothing OR acetaminophen/NSAID). The '
                         'dropped epochs are counted, never silently reassigned.')
    ap.add_argument('--drop-random-unmedicated', type=int, default=None,
                    metavar='N',
                    help='NEGATIVE CONTROL for --exclude-drug-set. Drop N '
                         'unmedicated epochs chosen AT RANDOM instead of the '
                         'ones dosed with the excluded drug. Matches the '
                         'exclusion on sample size, on the medicated FRACTION '
                         '(and so on the binary predictor variance), and on the '
                         'fact that both predictors get re-centred on a smaller '
                         'set -- everything except WHICH epochs leave. If the '
                         'interaction still moves, the move is about resampling '
                         'and re-centring, not about analgesics.')
    ap.add_argument('--drop-seed', type=int, default=0,
                    help='Seed for --drop-random-unmedicated.')
    ap.add_argument('--med-window-hours', type=float,
                    default=med_state.DEFAULT_WINDOW_HOURS)
    ap.add_argument('--dx-model', choices=['none', 'interaction'], default='none',
                    help='Add a subject-level diagnosis as a FULL THREE-WAY '
                         'pain x domain x diagnosis interaction. The '
                         'circuit-level question -- "are different circuits '
                         'changed by depression?" -- is the omnibus over the '
                         'three-way block, `p_omnibus_pain_x_dx`, with the '
                         'per-domain differences in the `pain_x_dx` rows. '
                         'Cannot be combined with --med-model: that would be a '
                         'four-way design, which 17 cases cannot support.')
    ap.add_argument('--hide-omnibus', action='store_true',
                    help='Leave the per-band omnibus p out of the diagnosis '
                         'figure titles. The omnibus tests whether circuits '
                         'differ FROM EACH OTHER in how the diagnosis changes '
                         'pain encoding, which is a different question from '
                         '"does this circuit differ between groups" -- hide it '
                         'when only the latter is being asked.')
    ap.add_argument('--drop-parcel-term', action='store_true',
                    help='Fit WITHOUT the parcel-nested random slope '
                         '(subj_parcel_slope). It is the smallest variance '
                         'component in every band measured so far and sits '
                         'within an order of magnitude of the boundary, which '
                         'is a known cause of a flat likelihood direction and '
                         'a non-positive-definite Hessian. Run it BESIDE the '
                         'full spec and compare: if the well-conditioned bands '
                         'barely move, the term is not load-bearing.')
    ap.add_argument('--question', default=QUESTION,
                    help="Level-2 folder. Pass 'mdd' to keep the diagnosis runs "
                         'out of the pain-physiology tree.')
    ap.add_argument('--dx', choices=list(dx_state.CONDITIONS), default='mdd')
    ap.add_argument('--dx-window-days', type=int,
                    default=dx_state.DEFAULT_WINDOW_DAYS,
                    help='Days before session_start in which a code counts '
                         '(default 90). 0 means EVER.')
    ap.add_argument('--dx-sources', choices=list(dx_state.SOURCE_SETS),
                    default='any')
    ap.add_argument('--fdr-q', type=float, default=0.05)
    ap.add_argument('--run-name', default=RUN_NAME)
    ap.add_argument('--replot', default=None,
                    help='Re-render the figures from an existing run directory '
                         'and exit. Nothing is refitted.')
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    io.warn_if_dirty()

    if args.unit == 'roi':
        # Checked BEFORE the view load, which is minutes of work: a scheme name
        # that cannot serve should cost a second, not a coffee.
        if args.roi_scheme not in roi_schemes.DOMAIN_SCHEMES:
            raise SystemExit(
                f'--unit roi needs a DOMAIN scheme, whose members are ROIs. '
                f'{args.roi_scheme!r} is not one. Known: '
                f'{sorted(roi_schemes.DOMAIN_SCHEMES)}. (pain_domains_v2 is '
                'registered as a fused ROI scheme too, but fusing is what '
                'removes the ROI layer this unit needs.)')
        roi_schemes.domain_scheme(args.roi_scheme)     # validates membership
        if args.hemisphere_separate:
            raise SystemExit('--hemisphere-separate is meaningless with '
                             '--unit roi: ROIs are hemisphere-collapsed by '
                             'construction.')
        if not args.drop_parcel_term:
            # Not an error, because it is what the flag combination MEANS rather
            # than a mistake -- but it is set loudly, because a provenance file
            # saying drop_parcel_term=false while the fit had no parcel term
            # would be a lie about the model.
            logger.info('--unit roi: the ROI does not enter the model, so '
                        'the parcel-nested random slope is dropped '
                        '(drop_parcel_term set to True).')
            args.drop_parcel_term = True

    # Refused HERE rather than at the first fit: the view load ahead of it is
    # minutes of work, and failing after it would waste all of them.
    if args.med_model != 'none' and args.dx_model != 'none':
        raise SystemExit(
            '--med-model and --dx-model together would be a FOUR-way design '
            '(pain x domain x medication x diagnosis). At 17 diagnosis cases '
            'the three-way cells are already thin and a four-way is not '
            'estimable on this cohort. Run them separately.')

    if args.replot:
        run_dir = Path(args.replot)
        # Take the run's OWN parameters from its provenance, not from argparse
        # defaults. A replot invoked without --drug-set would otherwise inherit
        # the default 'opioids' and caption an analgesics figure as an opioid
        # one -- a wrong label on a real figure, not a cosmetic slip.
        prov_path = run_dir / 'provenance.json'
        if prov_path.exists():
            import json
            prov = json.loads(prov_path.read_text())
            prov = prov.get('params', prov)
            for key in ('drug_set', 'exclude_drug_set', 'med_window_hours',
                        'med_model', 'roi_scheme', 'band_set',
                        'dx_model', 'dx', 'dx_window_days', 'dx_sources',
                        'drop_parcel_term', 'unit'):
                if key in prov and prov[key] is not None:
                    if getattr(args, key, None) != prov[key]:
                        logger.info('replot: %s = %r (from provenance, '
                                    'overriding %r)', key, prov[key],
                                    getattr(args, key, None))
                    setattr(args, key, prov[key])
        else:
            logger.warning('no provenance.json in %s -- captions will use '
                           'argparse defaults and may name the wrong drug set',
                           run_dir)
        cells = io.read_table(run_dir / 'domain_bands.parquet', on_stale='warn')
        slopes = io.read_table(run_dir / 'domain_slopes.parquet', on_stale='warn')
        coverage = io.read_table(run_dir / 'parcel_coverage.parquet',
                                 on_stale='ignore')
        display = (roi_schemes.domain_scheme(args.roi_scheme)['display']
                   if getattr(args, 'unit', 'parcel') == 'roi'
                   else view_tables.roi_regions_for(
                       {'roi_scheme': args.roi_scheme}))
        domains = [d for d in display if d in set(slopes['domain'])]
        per_domain = (coverage.groupby('domain')
                      .agg(n_parcels=('parcel', 'nunique'),
                           n_contacts=('channel', 'size'),
                           n_subjects=('subject_id', 'nunique'))
                      .reindex(domains))
        summary_figure(run_dir, cells, slopes, per_domain, domains, args)
        figure(run_dir, cells, slopes, per_domain, domains, args)
        if args.dx_model != 'none':
            dx_domain_figure(run_dir, cells, slopes, domains, args)
            dx_circuit_figure(run_dir, cells, slopes, domains, args)
        for _t in SPECTRA_TERMS:
            figure_by_domain(run_dir, cells, slopes, per_domain,
                             domains, args, term=_t)
        return

    ref_run = reference_run.load(args.reference_run)
    ref_run.describe()
    epoch_minutes = ref_run.view_params.get('epoch_minutes')

    view_dir = fullres_cells.resolve_view_dir(
        args.view_dir,
        mask_label=args.mask_label or ref_run.view_params.get('mask_label'),
        max_excluded_frac=ref_run.view_params.get('max_excluded_frac'),
        epoch_minutes=epoch_minutes)
    # `--unit roi` names a DOMAIN scheme, which `resolve_cohort` cannot resolve
    # -- it wants the ROI scheme whose regions the domains are made of. The
    # cohort gate ("does this subject contribute any labelled channel at all")
    # belongs at that base level anyway.
    split_report = {}
    cohort_scheme = (roi_schemes.domain_scheme(args.roi_scheme)['base']
                     if args.unit == 'roi' else args.roi_scheme)
    paths, _, _, subjects, roi_by_subject, _ = resolve_cohort(
        ref_run, view_dir, cohort=args.cohort, roi_scheme=cohort_scheme,
        insula_threshold=args.insula_threshold, report=split_report)
    ref_run.assert_cohort_matches(subjects,
                                  allow_drift=args.allow_cohort_drift
                                  or args.cohort != 'reference')

    if args.unit == 'roi':
        parcel_of, parcel_to_domain, coverage, unassigned = roi_domain_maps(
            roi_by_subject, subjects, args.roi_scheme)
        split_report['rois_unassigned_dropped'] = unassigned
        domain_display = roi_schemes.domain_scheme(args.roi_scheme)['display']
        # A subject who passed the cohort gate on a region NO DOMAIN CLAIMS
        # contributes nothing to this fit. Leaving them in `subjects` would put
        # them in provenance as a participant in a run they are absent from --
        # and `subjects[]` is the only sanctioned answer to "who was in this
        # run", so it has to be the subjects who were.
        contributing = set(coverage['subject_id'])
        absent = sorted(set(subjects) - contributing)
        if absent:
            logger.warning('%d cohort subject(s) have no channel in ANY domain '
                           'and are not in this fit: %s', len(absent), absent)
            split_report['subjects_without_a_domain_channel'] = absent
            subjects = {s for s in subjects if s in contributing}
    else:
        parcel_of, parcel_to_domain, coverage = parcel_domain_maps(
            paths, subjects, args.roi_scheme,
            collapse_hemisphere=not args.hemisphere_separate)
        domain_display = view_tables.roi_regions_for(
            {'roi_scheme': args.roi_scheme})
    domains = [d for d in domain_display if d in set(coverage['domain'])]
    logger.info('%d domains, %d %ss, %d contacts',
                len(domains), coverage['parcel'].nunique(), args.unit,
                len(coverage))
    per_domain = (coverage.groupby('domain')
                  .agg(n_parcels=('parcel', 'nunique'),
                       n_contacts=('channel', 'size'),
                       n_subjects=('subject_id', 'nunique'))
                  .reindex(domains))
    logger.info('\n%s', per_domain.to_string())
    logger.info('parcels per domain:\n%s',
                coverage.groupby(['domain', 'parcel']).size().to_string())

    med_lookup, med_summary = None, None
    if args.med_model != 'none':
        subclasses = med_state.DRUG_SETS[args.drug_set]
        admin = med_state.load_admin_table(subclasses=subclasses)
        bare = sorted(s.replace('sub-', '') for s in subjects)
        mdefs = med_state.load_epoch_defs(epoch_minutes=epoch_minutes,
                                          subjects=bare)
        state = med_state.epoch_med_state(mdefs, admin,
                                          hours=args.med_window_hours)
        missing = sorted(set(mdefs['subject']) - set(admin['subject']))
        if missing:
            logger.warning('%d subject(s) have NO %s administrations at all, so '
                           'every one of their epochs reads as unmedicated: %s',
                           len(missing), args.drug_set, missing)
        med_summary = med_state.stratum_summary(state)
        logger.info('\n%s', med_summary.to_string(index=False))
        med_lookup = state.assign(
            subject_id='sub-' + state['subject'].astype(str),
            med_state=state['med_state'].astype(float))[
                ['subject_id', 'session', 'epoch_id', 'med_state']]

        if args.exclude_drug_set:
            # The comparison stratum has to be free of the OTHER drug too, or
            # "unmedicated" silently means "no opioid, but possibly paracetamol".
            # An epoch dosed with BOTH stays in the medicated stratum: it did
            # receive the drug under study, and dropping it would select against
            # the patients who get combination analgesia.
            other = med_state.epoch_med_state(
                mdefs, med_state.load_admin_table(
                    subclasses=med_state.DRUG_SETS[args.exclude_drug_set]),
                hours=args.med_window_hours)
            other = other.assign(
                subject_id='sub-' + other['subject'].astype(str))[
                    ['subject_id', 'session', 'epoch_id', 'med_state']].rename(
                        columns={'med_state': 'other_state'})
            before = len(med_lookup)
            med_lookup = med_lookup.merge(
                other, on=['subject_id', 'session', 'epoch_id'], how='left')
            drop = (med_lookup['med_state'] == 0) & (
                med_lookup['other_state'].fillna(False).astype(bool))
            n_drop = int(drop.sum())
            med_lookup = med_lookup.loc[~drop, ['subject_id', 'session',
                                                'epoch_id', 'med_state']]
            logger.warning('EXCLUSION: dropped %d of %d epochs dosed with %s but '
                           'NOT with %s, so the comparison stratum is free of '
                           'both. Remaining: %d medicated, %d unmedicated.',
                           n_drop, before, args.exclude_drug_set, args.drug_set,
                           int((med_lookup['med_state'] == 1).sum()),
                           int((med_lookup['med_state'] == 0).sum()))
            med_summary = med_summary.assign(
                note=f'before excluding {args.exclude_drug_set}')

        if args.drop_random_unmedicated:
            # The negative control for the exclusion above. Same number of
            # unmedicated epochs leave, so the medicated fraction, the binary
            # predictor's variance and the re-centring of BOTH predictors on a
            # smaller set all move the same way -- only the SELECTION differs.
            # Positional, not by index label: med_lookup reaches here with a
            # different index depending on whether --exclude-drug-set ran.
            unmed = np.flatnonzero((med_lookup['med_state'] == 0).to_numpy())
            n_drop = min(args.drop_random_unmedicated, len(unmed))
            rng = np.random.default_rng(args.drop_seed)
            keep = np.ones(len(med_lookup), dtype=bool)
            keep[rng.choice(unmed, size=n_drop, replace=False)] = False
            med_lookup = med_lookup[keep]
            logger.warning('RANDOM-DROP CONTROL: dropped %d of %d unmedicated '
                           'epochs at random (seed %d). Remaining: %d medicated, '
                           '%d unmedicated.',
                           n_drop, len(unmed), args.drop_seed,
                           int((med_lookup['med_state'] == 1).sum()),
                           int((med_lookup['med_state'] == 0).sum()))

    # ---- subject-level diagnosis label ---------------------------------
    dx_lookup, dx_labels = None, None
    if args.dx_model != 'none':
        bare = sorted(s.replace('sub-', '') for s in subjects)
        dx_all = dx_state.load_diagnoses(subjects=bare)
        dx_labels, _ = dx_state.subject_labels(
            dx_all, condition=args.dx, window_days=args.dx_window_days,
            sources=args.dx_sources)
        missing = sorted(set(subjects) - set(dx_labels['subject_id']))
        if missing:
            raise SystemExit(
                f'{len(missing)} cohort subject(s) have no diagnoses table, so '
                f'their status is unknown, not negative: {missing}.')
        dx_labels = dx_labels[dx_labels['subject_id'].isin(subjects)]
        logger.info('\n%s', dx_state.stratum_summary(
            dx_labels, dx_state.pain_by_subject(subjects=bare)).to_string(index=False))
        dx_lookup = dx_state.epoch_lookup(dx_labels)

    kept, bands, notched = band_table(args.band_set, args.notch_half_width_hz,
                                      epoch_minutes)
    want = list(bands) if args.bands is None else [b for b in bands if b in args.bands]

    from ieeg_ehr.views.view_config import ROI_SCHEME_CODES
    scheme_code = ROI_SCHEME_CODES.get(args.roi_scheme, 'paindomains')
    run_dir = config.analysis_run_dir(
        question=args.question, output_type=OUTPUT_TYPE,
        view_scheme='-'.join(
            [args.band_set.replace('_', ''), scheme_code]
            + (['roiunit'] if args.unit == 'roi' else [])
            + ([args.drug_set] if args.med_model != 'none' else [])
            + ([f'{args.dx}{args.dx_window_days}d']
               if args.dx_model != 'none' else [])
            + (['noparcel'] if args.drop_parcel_term else [])
            + ([f'excl{args.exclude_drug_set}'] if args.exclude_drug_set else [])
            + ([f'randdrop{args.drop_random_unmedicated}s{args.drop_seed}']
               if args.drop_random_unmedicated else [])),
        run_name=args.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info('run dir (created BEFORE fitting, results append per band): %s',
                run_dir)
    # A PROVISIONAL provenance, so a job that dies mid-run still leaves a record
    # of what it was trying to do. The final write at the end overwrites it with
    # the complete params, including everything only known after fitting.
    io.write_run_provenance(
        run_dir, script=SCRIPT, params={**vars(args), 'status': 'IN PROGRESS'},
        subjects=sorted(subjects),
        extra={'status': 'started, not finished -- if this text survives, the '
                         'run did not complete and its tables are partial'})

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
            'session': index['session'].to_numpy(),
            'channel': index['channel'].to_numpy(),
            'epoch_id': index['epoch_id'].to_numpy(),
            'pain_score': index['pain_score'].to_numpy(),
            'value': band_values[:, bi],
            'parcel': index['parcel'].to_numpy()})
        frame['domain'] = frame['parcel'].map(parcel_to_domain)
        extra = ['parcel', 'domain']
        if med_lookup is not None:
            frame = frame.merge(med_lookup,
                                on=['subject_id', 'session', 'epoch_id'],
                                how='inner')
            extra.append('med_state')
        if dx_lookup is not None:
            # Subject-level: there is no session or epoch key, because the
            # label does not vary across either.
            frame = frame.merge(dx_lookup, on='subject_id', how='inner')
            extra.append('dx_state')
        df = mm.build_cell_frame(frame, extra_columns=tuple(extra))
        if med_lookup is not None:
            df = mm.add_med_components(df)
        rec, slopes = fit_band(df, domains, band, med=med_lookup is not None,
                               dx=dx_lookup is not None, out_dir=run_dir,
                               drop_parcel=args.drop_parcel_term)
        records.append(rec)
        slope_parts.append(slopes)

    cells = pd.DataFrame(records)
    slopes = pd.concat(slope_parts, ignore_index=True)

    # BH over the per-domain slopes (one family) and over the omnibus tests
    # (another). Two questions, two families.
    # BH WITHIN each term, never pooled across them: the pain slope, the
    # medication effect and their interaction are three different questions asked
    # of the same fit.
    slopes['p_bh'] = np.nan
    for term, idx in slopes.groupby('term').groups.items():
        sub = slopes.loc[idx]
        m = sub['p'].notna()
        if not m.any():
            continue
        _, adj = cp.bh_fdr(sub.loc[m, 'p'].to_numpy(), q=args.fdr_q)
        slopes.loc[sub.index[m], 'p_bh'] = adj
        logger.info('BH family %r: %d cells, %d rejected', term, int(m.sum()),
                    int((adj <= args.fdr_q).sum()))
    slopes['p_bh_reject'] = slopes['p_bh'] <= args.fdr_q

    m = cells['p_omnibus'].notna()
    cells['p_omnibus_bh'] = np.nan
    if m.any():
        _, adj = cp.bh_fdr(cells.loc[m, 'p_omnibus'].to_numpy(), q=args.fdr_q)
        cells.loc[m, 'p_omnibus_bh'] = adj
        cells['p_omnibus_bh_reject'] = cells['p_omnibus_bh'] <= args.fdr_q

    # THE FORMULA THAT ACTUALLY RAN, with the same med/dx flags the fits used.
    # It was previously recorded as the pain-only formula unconditionally, so a
    # medication or diagnosis run wrote provenance naming a model it had not
    # fitted -- the one kind of provenance error that cannot be caught later,
    # because the file looks complete and self-consistent.
    params = {'formula': domain_formula(
                  domains, med=args.med_model != 'none',
                  dx=args.dx_model != 'none')[0],
              'variance_components': (VC_DOMAIN_NO_PARCEL
                                      if args.drop_parcel_term else VC_DOMAIN),
              'drop_parcel_term': args.drop_parcel_term,
              'dx_model': args.dx_model,
              'dx': args.dx if args.dx_model != 'none' else None,
              'dx_window_days': (args.dx_window_days
                                 if args.dx_model != 'none' else None),
              'dx_sources': (args.dx_sources if args.dx_model != 'none'
                             else None),
              'dx_n_case': (int(dx_labels['dx_state'].sum())
                            if dx_labels is not None else None),
              'dx_n_control': (int((~dx_labels['dx_state']).sum())
                               if dx_labels is not None else None),
              'unit': args.unit,
              'parcel_level': (
                  f'ROI ({roi_schemes.domain_scheme(args.roi_scheme)["base"]}), '
                  'hemispheres collapsed; the ROI does NOT enter the model, it '
                  'is only the domain lookup'
                  if args.unit == 'roi' else
                  'Desikan-Killiany, hemispheres '
                  + ('separate' if args.hemisphere_separate else 'collapsed')),
              'insula_threshold': args.insula_threshold,
              **split_report,
              'reference_domain': domain_formula(domains)[1],
              'band_set': args.band_set, 'roi_scheme': args.roi_scheme,
              'med_model': args.med_model,
              'drug_set': args.drug_set if args.med_model != 'none' else None,
              'med_window_hours': (args.med_window_hours
                                   if args.med_model != 'none' else None),
              'exclude_drug_set': args.exclude_drug_set,
              'drop_random_unmedicated': args.drop_random_unmedicated,
              'drop_seed': args.drop_seed,
              'roi_scheme_contents': (
                  roi_schemes.domain_scheme_provenance(args.roi_scheme)
                  if args.unit == 'roi'
                  else roi_schemes.scheme_provenance(args.roi_scheme)),
              'notched_bins_excluded': notched, 'fdr_q': args.fdr_q,
              'epoch_minutes': epoch_minutes,
              'omnibus': 'joint Wald chi2 on the interaction block; NOT an LRT, '
                         'because REML likelihoods are not comparable across '
                         'fixed-effects designs'}

    io.write_table(cells, run_dir / 'domain_bands.parquet', params=params,
                   parents=[str(Path(args.reference_run) / 'provenance.json'),
                            str(view_dir)],
                   subjects=sorted(subjects), script=SCRIPT,
                   extra={'status': DISCLAIMER,
                          'domain_caveat': domain_caveat(args.roi_scheme)})
    io.write_table(slopes, run_dir / 'domain_slopes.parquet', params=params,
                   subjects=sorted(subjects), script=SCRIPT,
                   extra={'status': DISCLAIMER,
                          'domain_caveat': domain_caveat(args.roi_scheme),
                          'reading': 'beta_pain is each domain\'s marginal slope, a '
                                     'LINEAR COMBINATION of the reference slope and '
                                     'that domain\'s interaction term. diff_from_ref '
                                     'is the interaction coefficient itself.'})
    io.write_table(coverage, run_dir / 'parcel_coverage.parquet', params=params,
                   script=SCRIPT)
    if med_summary is not None:
        io.write_table(med_summary, run_dir / 'med_stratum_summary.parquet',
                       params=params, parents=[str(med_state.ADMIN_TABLE)],
                       script=SCRIPT)
    io.write_run_provenance(run_dir, script=SCRIPT, params=params,
                            parents=[str(view_dir)], subjects=sorted(subjects),
                            extra={'status': DISCLAIMER,
                                   'domain_caveat': domain_caveat(args.roi_scheme),
                                   'mask_content': CONFOUND_CAVEAT,
                                   **({'dx_caveat': DX_CAVEAT}
                                      if args.dx_model != 'none' else {})})
    summary_figure(run_dir, cells, slopes, per_domain, domains, args)
    figure(run_dir, cells, slopes, per_domain, domains, args)
    for _t in SPECTRA_TERMS:
        figure_by_domain(run_dir, cells, slopes, per_domain, domains,
                         args, term=_t)
    if args.dx_model != 'none':
        dx_domain_figure(run_dir, cells, slopes, domains, args)
        dx_circuit_figure(run_dir, cells, slopes, domains, args)
    # The index line has to name the model that ran. It was hard-coded to the
    # parcel spec, so the first ROI-unit run indexed itself as having a
    # "parcel as a nested random slope" it did not fit -- and analyses_run.md is
    # append-only, so a wrong line there is permanent.
    io.log_analysis(
        'domain-level mixed models: pain x processing domain, one fit per band, '
        + ('ROI unit, region NOT in the model' if args.unit == 'roi'
           else 'parcel as a nested random slope')
        + f' [{args.roi_scheme}] (EXPLORATORY)', run_dir)
    print(run_dir)


def dx_circuit_figure(run_dir, cells, slopes, domains, args):
    """ONE PANEL PER CIRCUIT: the pain slope in MDD vs non-MDD, across bands.

    The same numbers as `fig_dx_domain.png` read the other way round. That
    figure puts one panel per BAND with circuits down the axis, which answers
    "within this band, which circuit differs". This one puts one panel per
    CIRCUIT with bands down the axis, which answers "within this circuit, at
    which frequency do the strata diverge" -- the spectral profile of the
    diagnosis effect, per network. Both come from the identical fit; neither is
    a re-analysis.

    THE X SCALE IS SHARED ACROSS PANELS, deliberately, and this is the opposite
    choice from `fig_dx_domain.png`. There, panels were bands and the badly
    conditioned low-frequency fits forced per-band scales or nothing else was
    readable. Here panels are CIRCUITS and comparing circuits is the entire
    point of the layout, so a per-panel scale would make a weak circuit look
    like a strong one. Any band whose fit did not produce a positive-definite
    Hessian is therefore DROPPED rather than plotted, because a single 0.08 SE
    would otherwise flatten every honest interval in the figure -- and plotting
    it would imply its interval means something.

    NO PER-STRATUM SIGNIFICANCE MARKS, same as every other figure in this
    analysis. What is marked is the DIFFERENCE, from the `pain_x_dx` rows and
    their own BH family: a filled band label means the strata differ there.
    Whether one arm's interval happens to exclude zero is not that test.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    slopes = norm_strata(slopes)
    ss = slopes[slopes['term'] == 'pain_slope_by_stratum']
    diff = slopes[slopes['term'] == 'pain_x_dx']
    if not len(ss):
        logger.info('no per-stratum slope rows -- skipping fig_dx_circuit.png')
        return

    # Bands whose fit was ill-conditioned are excluded BY NAME, and the caption
    # says which. `cells.warnings` carries the statsmodels text; a non-positive-
    # definite Hessian means the SEs came out of a non-invertible curvature
    # estimate and are not interval estimates at all.
    bad = sorted(cells.loc[cells['warnings'].astype(str)
                 .str.contains('not positive definite'), 'band'])
    bands = [b for b in BAND_SETS[args.band_set]
             if b in set(ss['band']) and b not in bad]
    if not bands:
        logger.warning('every band ill-conditioned -- skipping fig_dx_circuit.png')
        return
    doms = [d for d in domains if d in set(ss['domain'])]
    n_case = int(cells['n_subjects_case'].max())
    n_ctrl = int(cells['n_subjects_control'].max())

    keep = ss[ss['band'].isin(bands)]
    xmax = float(np.nanmax(np.abs(np.concatenate(
        [keep['ci_lo'].to_numpy(dtype=float),
         keep['ci_hi'].to_numpy(dtype=float)])))) * 1.10

    fig, axes = plt.subplots(1, len(doms), figsize=(2.9 * len(doms) + 1.4, 5.6),
                             sharey=True, sharex=True, squeeze=False)
    y = np.arange(len(bands))
    colours = {NON_DX_LABEL: '#4a7fb5', DX_LABEL: '#b03a2e'}

    for j, dom in enumerate(doms):
        ax = axes[0][j]
        for stratum, off in ((NON_DX_LABEL, -0.16), (DX_LABEL, +0.16)):
            d = (keep[(keep['domain'] == dom) & (keep['stratum'] == stratum)]
                 .set_index('band').reindex(bands))
            b = d['beta'].to_numpy(dtype=float)
            se = d['se'].to_numpy(dtype=float)
            ax.plot(b, y + off, '-', color=colours[stratum], lw=1.0, alpha=0.30)
            ax.errorbar(b, y + off, xerr=1.96 * se, fmt='o', ms=4.6, lw=1.3,
                        capsize=2, color=colours[stratum],
                        label=(f'{DX_DISPLAY if stratum == DX_LABEL else NON_DX_DISPLAY}'
                               f' (n={n_case if stratum == DX_LABEL else n_ctrl})')
                        if j == 0 else None)
        ax.axvline(0, color='0.4', lw=0.9, ls='--')
        ax.set_xlim(-xmax, xmax)
        ax.set_title(dom, fontsize=10.5,
                     color=DOMAIN_COLOURS.get(dom, '0.2'))
        ax.tick_params(labelsize=7.5)
        ax.spines[['top', 'right']].set_visible(False)
        if j == 0:
            ax.set_yticks(y)
            ax.set_yticklabels(
                [f'{b}\n{BAND_SETS[args.band_set][b][0]}-'
                 f'{BAND_SETS[args.band_set][b][1]} Hz' for b in bands],
                fontsize=8)
            ax.set_ylim(len(bands) - 0.5, -0.5)
            ax.legend(fontsize=7.5, loc='lower right', frameon=False)

        # Mark the DIFFERENCE, not either arm. BH-corrected within its own
        # family across the whole grid, which is why the flag is read off the
        # `pain_x_dx` rows rather than recomputed here.
        dd = diff[(diff['domain'] == dom) & (diff['band'].isin(bands))]
        rej = dd.get('p_bh_reject')
        if rej is not None:
            for band in dd.loc[rej.fillna(False).astype(bool), 'band']:
                if band in bands:
                    ax.annotate('*', xy=(0.965, (bands.index(band) + 0.5)
                                         / len(bands)),
                                xycoords=('axes fraction', 'axes fraction'),
                                ha='right', va='center', fontsize=17,
                                color='#7d3c98')

    dropped = (f'  Bands EXCLUDED for an ill-conditioned fit: {", ".join(bad)}.'
               if bad else '')
    fig.suptitle(
        f'Pain slope in {DX_DISPLAY} vs {NON_DX_DISPLAY}, one panel per '
        f'circuit  ({n_case} {DX_DISPLAY}, {n_ctrl} {NON_DX_DISPLAY})\n'
        'both strata from ONE three-way fit; * = BH-significant DIFFERENCE '
        f'at q={args.fdr_q}', fontsize=12)
    # ONE x label, on the CENTRE panel's own axis. Five copies overlap each
    # other, and a figure-level supxlabel lands in the caption block below the
    # tight_layout rect -- attaching it to an axis keeps it clear of both.
    axes[0][len(doms) // 2].set_xlabel('d log10 power / pain point', fontsize=9)
    fig.tight_layout(rect=(0, 0.17, 1, 0.90))
    fig.text(0.01, 0.005,
             'SAME FIT AND SAME NUMBERS as fig_dx_domain.png, transposed: '
             'panels are circuits and the axis is frequency, so each panel is '
             'one network\'s spectral profile. ' + formula_note(args)
             + 'Each stratum\'s slope is a linear combination of that single '
             'fit with its SE from the fitted covariance, NOT a separate fit '
             'per arm. Neither arm carries significance marks, deliberately: '
             'the MDD arm is the smaller one and has wider intervals '
             'everywhere from power alone, so "one arm excludes zero and the '
             'other does not" is the difference-of-significance fallacy. Only '
             'the starred DIFFERENCE is a test.' + dropped + ' The connecting '
             'line joins ordinal band categories of unequal width and is a '
             'reading aid, NOT an interpolated spectrum.\n'
             f'{DX_CAVEAT}\n{domain_caveat(args.roi_scheme)}\n{MODULATORY_CAVEAT}\n'
             f'{DISCLAIMER}',
             fontsize=6.2, va='bottom', ha='left', color='0.35', wrap=True)
    out = run_dir / 'fig_dx_circuit.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info('wrote %s', out.name)
    return out


def dx_domain_figure(run_dir, cells, slopes, domains, args):
    """Are different CIRCUITS changed by depression?

    Two panels, because there are two distinct questions and conflating them is
    the usual error:

      LEFT   each circuit's pain slope in each stratum, as paired intervals.
             This is the descriptive picture, and it carries NO significance
             marks on the individual arms -- for the same reason the band
             figure does not. Whether one arm's interval excludes zero is not a
             test that the arms differ.
      RIGHT  the per-circuit DIFFERENCE (case - control) with its own BH family.
             This is the only panel that licenses a claim.

    The title carries `p_omnibus_pain_x_dx`, the joint Wald over the whole
    three-way block. THAT is the literal answer to "do different circuits
    change differently in depression" -- a single test per band, asked before
    any individual circuit is inspected, so reading the per-circuit panel after
    a non-significant omnibus is exploration and is labelled as such.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    slopes = norm_strata(slopes)
    ss = slopes[slopes['term'] == 'pain_slope_by_stratum']
    diff = slopes[slopes['term'] == 'pain_x_dx']
    if not len(ss) or not len(diff):
        logger.warning('no diagnosis contrasts in the slope table, no figure')
        return

    bands = [b for b in BAND_SETS[args.band_set] if b in set(ss['band'])]
    doms = [d for d in domains if d in set(ss['domain'])]
    n_case = int(cells['n_subjects_case'].max())
    n_ctrl = int(cells['n_subjects_control'].max())

    fig, axs = plt.subplots(2, len(bands), figsize=(2.9 * len(bands), 8.6),
                            squeeze=False, sharey='row')  # x NOT shared: see band_limit
    y = np.arange(len(doms))
    colors = {NON_DX_LABEL: '#4a7fb5', DX_LABEL: '#b03a2e'}

    def band_limit(frame, band):
        """x-limit from THIS band only.

        PER BAND, not shared. delta and theta are badly conditioned on this
        cohort -- their interaction SEs run ~20x the other four bands -- and a
        shared scale set by them squashes alpha through high_gamma into a
        vertical line, hiding the only bands with usable precision. The cost is
        that panels are no longer visually comparable across bands, so each
        carries its own axis and the reader is told here rather than left to
        assume otherwise.
        """
        d = frame[frame['band'] == band]
        if not len(d):
            return 1.0
        hi = np.nanmax(np.abs(np.concatenate(
            [(d['beta'] + 1.96 * d['se']).to_numpy(dtype=float),
             (d['beta'] - 1.96 * d['se']).to_numpy(dtype=float)])))
        return float(hi) * 1.08 if np.isfinite(hi) and hi > 0 else 1.0

    for j, band in enumerate(bands):
        # --- row 0: both strata, paired
        ax = axs[0][j]
        for stratum, off in ((NON_DX_LABEL, -0.16), (DX_LABEL, +0.16)):
            d = (ss[(ss['band'] == band) & (ss['stratum'] == stratum)]
                 .set_index('domain').reindex(doms))
            ax.errorbar(d['beta'].to_numpy(dtype=float), y + off,
                        xerr=1.96 * d['se'].to_numpy(dtype=float), fmt='o',
                        ms=4.5, lw=1.2, capsize=2, color=colors[stratum],
                        label=(f'{DX_DISPLAY if stratum == DX_LABEL else NON_DX_DISPLAY}'
                               f' (n={n_case if stratum == DX_LABEL else n_ctrl})')
                        if j == 0 else None)
        ax.axvline(0, color='0.4', lw=0.9, ls='--')
        ax.set_xlim(-band_limit(ss, band), band_limit(ss, band))
        row = cells[cells['band'] == band]
        p_omni = (float(row['p_omnibus_pain_x_dx'].iloc[0])
                  if len(row) and 'p_omnibus_pain_x_dx' in row else np.nan)
        # SUPPRESSIBLE, not deleted. The omnibus is the only test of "do
        # circuits differ FROM EACH OTHER in how MDD changes pain encoding",
        # so hiding it invites reading each per-circuit row as its own test.
        # Opt-out, and the caption records that it is gone.
        ax.set_title(band if getattr(args, 'hide_omnibus', False)
                     else f'{band}\nomnibus pain x dx p={p_omni:.3g}',
                     fontsize=8.5)
        ax.tick_params(labelsize=7)
        if j == 0:
            ax.set_yticks(y)
            ax.set_yticklabels(doms, fontsize=8)
            ax.set_ylim(len(doms) - 0.5, -0.5)
            ax.legend(fontsize=7, loc='lower right', frameon=False)

        # --- row 1: the difference, BH-outlined
        ax = axs[1][j]
        d = diff[diff['band'] == band].set_index('domain').reindex(doms)
        b = d['beta'].to_numpy(dtype=float)
        se = d['se'].to_numpy(dtype=float)
        rej = d.get('p_bh_reject', pd.Series(False, index=d.index)) \
               .fillna(False).to_numpy(dtype=bool)
        ax.errorbar(b[~rej], y[~rej], xerr=1.96 * se[~rej], fmt='o', ms=4.5,
                    lw=1.2, capsize=2, color='0.55')
        ax.errorbar(b[rej], y[rej], xerr=1.96 * se[rej], fmt='o', ms=6,
                    lw=1.6, capsize=2, color='#7d3c98')
        ax.axvline(0, color='0.4', lw=0.9, ls='--')
        ax.set_xlim(-band_limit(diff, band), band_limit(diff, band))
        ax.set_title(f'{band}: {DX_DISPLAY} - {NON_DX_DISPLAY}', fontsize=8.5)
        ax.tick_params(labelsize=7)
        if j == 0:
            ax.set_yticks(y)
            ax.set_yticklabels(doms, fontsize=8)
            ax.set_ylim(len(doms) - 0.5, -0.5)

    n_rej = int(diff.get('p_bh_reject',
                         pd.Series(dtype=bool)).fillna(False).sum())
    fig.suptitle(
        f'Are different circuits changed by {args.dx.upper()}?  '
        f'{n_case} {DX_DISPLAY} vs {n_ctrl} {NON_DX_DISPLAY}\n'
        'TOP: each circuit\'s pain slope per stratum.   '
        f'BOTTOM: the difference, {n_rej} BH-significant at q={args.fdr_q}',
        fontsize=12)
    fig.tight_layout(rect=(0, 0.11, 1, 0.92))
    # The omnibus sentence has to track whether the omnibus is actually shown.
    # Left unconditional it described a number that --hide-omnibus had removed.
    omni_note = (
        'THE OMNIBUS IS NOT SHOWN on this figure (--hide-omnibus). It is the '
        'joint Wald over the whole three-way block and is the only test of "do '
        'circuits differ FROM EACH OTHER in how MDD changes pain encoding"; '
        'without it, each per-circuit row should be read as its own contrast '
        'and nothing here speaks to differences between circuits. '
        if getattr(args, 'hide_omnibus', False) else
        'The omnibus in each top title is the joint Wald over the whole '
        'three-way block and is the actual test of "do circuits differ FROM '
        'EACH OTHER in how MDD changes pain encoding"; per-circuit rows read '
        'after a non-significant omnibus are exploratory. ')
    fig.text(0.01, 0.005,
             'One fit per band: log10_power ~ NRS_within * domain * dx_state + '
             'NRS_submean, with the parcel-nested random slope unchanged. Both '
             'groups come from that ONE fit as linear combinations with SEs '
             'from the fitted covariance -- neither group was refitted alone, '
             'which is the point: comparing which group reaches significance '
             f'is the difference-of-significance fallacy, and {DX_DISPLAY} is '
             'the smaller group here so its intervals are wider everywhere '
             'regardless. AN INTERVAL CLEAR OF ZERO IS NOT THE TEST either -- '
             'that is the uncorrected p < 0.05 boundary, while the outlines '
             'are BH-corrected over all 30 circuit x band cells, which is why '
             'a difference can sit off zero and still not be outlined. '
             + omni_note + '\n'
             f'{domain_caveat(args.roi_scheme)}\n{MODULATORY_CAVEAT}\n{DISCLAIMER}',
             fontsize=6.2, va='bottom', ha='left', color='0.35', wrap=True)
    out = run_dir / 'fig_dx_domain.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info('wrote %s', out.name)
    return out


#: Per-term specification for the by-domain spectral figures. Each term is its
#: OWN figure with its OWN shared x scale, never a shared one: the units differ
#: (per pain point / per dose-state switch / per pain point per dose-state
#: switch), so one scale across terms would be the dual-axis mistake wearing a
#: different hat. The omnibus column differs per term too -- it is a separate
#: joint Wald on that term's interaction block.
def random_effects_note(args):
    """The random-effects tail THE RUN ACTUALLY FITTED, as caption text.

    Built rather than written out, because there are four places a formula
    appears in a caption and `--drop-parcel-term` (which `--unit roi` implies)
    removes a term from all of them. A hard-coded string gets one of the four
    updated and leaves three captions naming a variance component the fit did
    not estimate.
    """
    parts = ['(NRS_within || subject)']
    if not getattr(args, 'drop_parcel_term', False):
        parts.append('(NRS_within || subject:parcel)')
    parts.append('(1 | subject:channel)')
    return ' + '.join(parts) + '. '


def med_formula_note(args):
    return ('log10_power ~ NRS_within * C(domain) * med_within + NRS_submean + '
            'med_submean + ' + random_effects_note(args)
            + 'BOTH predictors are SUBJECT-MEAN-CENTRED, so every '
            "coefficient is read at the OTHER one's patient-specific mean. ")


def dx_formula_note(args):
    return ('log10_power ~ NRS_within * C(domain) * dx_state + NRS_submean + '
            + random_effects_note(args)
            + '`dx_state` is a SUBJECT-LEVEL 0/1 constant and is '
            'NOT subject-mean-centred -- unlike `med_within` it has no '
            'within-patient part to centre -- so every coefficient here is read '
            "IN THE CONTROL STRATUM, not at a patient's own average diagnosis "
            'state, which would be meaningless. ')


def formula_note(args):
    """The formula THE RUN ACTUALLY FITTED, for a figure caption.

    Keyed on the run's flags rather than on which term is being drawn. The
    earlier version picked the medication note for medication terms and
    otherwise fell back to the pain-only formula, which mislabelled two things:
    a diagnosis run's captions claimed a model with no `dx_state` in it, and
    even the `pain` panel of a med or dx run names a conditional slope whose
    conditioning the pain-only string does not mention.
    """
    if args.dx_model != 'none':
        return dx_formula_note(args)
    if args.med_model != 'none':
        return med_formula_note(args)
    return ('log10_power ~ NRS_within * C(domain) + NRS_submean + '
            + random_effects_note(args))


SPECTRA_TERMS = {
    'pain': dict(
        out='fig_domain_spectra.png',
        omnibus='p_omnibus_pain',
        title='Pain-power SPECTRAL PROFILE by processing domain',
        xlabel='d log10 power / pain point',
        reading='Each marker is the WITHIN-PATIENT slope of log10 power on pain, '
                'read at that patient\'s own average medication state -- NOT at '
                'unmedicated.'),
    'med': dict(
        out='fig_domain_spectra_med.png',
        omnibus='p_omnibus_med',
        title='MEDICATION effect on power by processing domain',
        xlabel='d log10 power when recently dosed',
        reading='Each marker is the change in log10 power for a 0->1 dose-state '
                'switch, read AT THAT PATIENT\'S OWN MEAN PAIN. Because '
                'NRS_within is subject-mean-centred, that is the patient\'s '
                'average NRS, which is NOT zero pain, NOT a pain-free baseline '
                'and NOT the cohort mean: a patient averaging NRS 6 has their '
                'medication effect evaluated at 6. Medication is also not '
                'randomised -- it is given BECAUSE of pain -- so this is an '
                'association with the dosed state, not a drug effect.'),
    # The diagnosis pair. Deliberately NOT sharing a figure with the medication
    # pair even though the shapes match: `med_within` is subject-mean-centred
    # and `dx_state` is a raw subject-level 0/1, so a `med` coefficient is read
    # at the patient's own average dosing while a `dx` coefficient is read in
    # the control stratum. Same layout, different conditioning, so separate
    # figures with separate captions.
    #
    # `pain_slope_by_stratum` is deliberately absent: it carries TWO rows per
    # (band, domain) and this layout draws one marker per band, so it would
    # silently plot whichever stratum happened to sort first. Both strata are in
    # `fig_dx_domain.png` instead, which is built for the pairing.
    'dx': dict(
        out='fig_domain_spectra_dx.png',
        omnibus='p_omnibus_dx',
        title='MDD effect on power LEVEL by processing domain',
        xlabel='d log10 power, MDD - control',
        reading='Each marker is the difference in mean log10 power between '
                'diagnosis strata, read AT THAT PATIENT\'S OWN MEAN PAIN '
                '(NRS_within is subject-mean-centred, so NRS_within = 0 is the '
                'patient\'s average NRS, not zero pain). TREAT THIS PANEL AS A '
                'NUISANCE TERM, NOT A RESULT: absolute log power level is set '
                'partly by electrode impedance, amplifier gain and where a '
                'contact sits inside a parcel, all of which are multiplicative '
                'in power and therefore ADDITIVE in log space -- the same place '
                'this coefficient lives. A between-patient LEVEL contrast is '
                'confounded with hardware in a way a within-patient SLOPE is '
                'not, which is why the slope terms are the interpretable ones '
                'and this one gets no BH family.'),
    'pain_x_dx': dict(
        out='fig_domain_spectra_dx_interaction.png',
        omnibus='p_omnibus_pain_x_dx',
        title='PAIN x MDD interaction by processing domain',
        xlabel='change in the pain slope in MDD',
        reading='Each marker is how much the within-patient pain slope DIFFERS '
                'in MDD patients, in units of d log10 power per pain point. '
                'Positive means the pain slope is more positive in MDD. This is '
                'THE ESTIMAND of the diagnosis analysis -- the only term here '
                'that tests a difference between strata rather than describing '
                'one -- and it is a BETWEEN-PATIENT contrast, so its standard '
                'errors are about twice the pain slope\'s on this cohort and it '
                'is underpowered by construction. A null with a wide interval '
                'is "not resolved at this n", not "no difference".'),
    'pain_x_med': dict(
        out='fig_domain_spectra_interaction.png',
        omnibus='p_omnibus_pain_x_med',
        title='PAIN x MEDICATION interaction by processing domain',
        xlabel='change in the pain slope when dosed',
        reading='Each marker is how much the within-patient pain slope CHANGES '
                'for a 0->1 dose-state switch, in units of d log10 power per '
                'pain point per switch. Positive means the pain slope becomes '
                'more positive when dosed. This is the only one of the three '
                'terms that is a contrast BETWEEN medication strata, so it is '
                'the one most sensitive to what the comparison stratum contains '
                '-- see --exclude-drug-set.'),
}


def term_slopes_frame(slopes, term):
    """The rows for one term in a uniform shape, old schema or new.

    Generalises `pain_slopes_frame`. Old runs carry only a pain slope and no
    `term` column, so anything but 'pain' comes back EMPTY rather than raising --
    the caller skips that figure and says so.
    """
    if 'beta_pain' in slopes.columns:
        if term != 'pain':
            return slopes.iloc[0:0].rename(columns={'beta_pain': 'beta'})
        return slopes.rename(columns={'beta_pain': 'beta'})
    return slopes[slopes['term'] == term].copy()


def pain_slopes_frame(slopes):
    """The pain-slope rows in a uniform shape, old schema or new.

    Runs before 2026-09-18 wrote one row per (band, domain) with the column
    `beta_pain`; runs after carry a `term` column and a generic `beta`, because
    the medication design produces three contrasts per cell. Normalising here
    keeps `--replot` working on both rather than silently drawing nothing.
    """
    if 'beta_pain' in slopes.columns:
        return slopes.rename(columns={'beta_pain': 'beta'})
    return slopes[slopes['term'] == 'pain'].copy()


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
    slopes = pain_slopes_frame(slopes)
    piv = (slopes.pivot_table(index='domain', columns='band', values='beta')
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
    ax.set_yticklabels([f'{d}\n{unit_count(args, per_domain.loc[d, "n_parcels"])}, '
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

    col = 'p_omnibus' if 'p_omnibus' in cells.columns else 'p_omnibus_pain'
    omni = ' · '.join(f'{r.band} p={getattr(r, col):.3g}' for r in cells.itertuples())
    fig.text(0.01, 0.005,
             'EACH CELL IS THAT DOMAIN\'S MARGINAL PAIN SLOPE TESTED AGAINST '
             'ZERO -- not against the Control domain. A significant Control cell '
             'is therefore not a contradiction: Occipital and Auditory are the '
             'quasi-controls, and an effect there argues for a global or '
             'artifactual driver rather than nociception, so read that row first. '
             f'CONTROL ROW: {ctrl} of {len(bands)} cells significant. Stars are '
             'the uncorrected p (* <0.05, ** <0.01, *** <0.001); bold and the '
             f'outline are BH at q={args.fdr_q:g} over all {arr.size} cells. '
             f'Slopes come from ONE mixed model per band over every '
             f'{unit_word(args, plural=False)} at once, with domain as a fixed '
             + ('effect and the ROI NOT IN THE MODEL AT ALL -- it is only the '
                'lookup that says which domain a channel is in, so nothing here '
                'guards against one well-sampled ROI carrying its domain; that '
                'guard is the region-level consistency map, which fits every '
                'ROI separately. '
                if getattr(args, 'unit', 'parcel') == 'roi' else
                'effect and the atlas parcel as a random slope nested in '
                'subject, so no domain is carried by a single well-sampled '
                'parcel. ')
             + 'Each value is a linear combination of the '
             'reference slope and that domain\'s interaction term, with its SE '
             'from the fitted covariance. THE OMNIBUS -- whether the domains '
             f'differ FROM EACH OTHER -- is a separate question: {omni}. '
             f'{MODULATORY_CAVEAT} {domain_caveat(args.roi_scheme)}\n{DISCLAIMER}',
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
    slopes = pain_slopes_frame(slopes)
    fig, axes = plt.subplots(1, len(bands), figsize=(2.9 * len(bands) + 1.6, 5.4),
                             sharey=True, squeeze=False)
    xmax = float(np.nanmax(np.abs(np.concatenate(
        [slopes['ci_lo'].to_numpy(), slopes['ci_hi'].to_numpy()])))) * 1.08
    colours = DOMAIN_COLOURS
    y = np.arange(len(domains))

    for j, band in enumerate(bands):
        ax = axes[0][j]
        d = slopes[slopes['band'] == band].set_index('domain').reindex(domains)
        for i, dom in enumerate(domains):
            r = d.loc[dom]
            sig = bool(r.get('p_bh_reject', False))
            ax.errorbar(r['beta'], i, xerr=1.96 * r['se'], fmt='o',
                        ms=8 if sig else 5.5, lw=1.8 if sig else 1.1, capsize=3,
                        color=colours.get(dom, '0.4'),
                        markerfacecolor=colours.get(dom, '0.4') if sig else 'white',
                        zorder=3)
        ax.axvline(0, color='0.5', lw=1.0, ls='--')
        omni = cells[cells['band'] == band].iloc[0]
        p_omni = omni.get('p_omnibus', omni.get('p_omnibus_pain', np.nan))
        ax.set_title(f'{band}\nomnibus p = {p_omni:.3g}', fontsize=9)
        ax.set_xlim(-xmax, xmax)
        ax.set_xlabel('d log10 power / pain point', fontsize=8)
        ax.tick_params(labelsize=7)
        if j == 0:
            ax.set_yticks(y)
            ax.set_yticklabels([f'{dom}\n'
                                f'{unit_count(args, per_domain.loc[dom, "n_parcels"], short=True)}, '
                                f'{int(per_domain.loc[dom, "n_contacts"])} chan'
                                for dom in domains], fontsize=7.5)
            ax.set_ylim(len(domains) - 0.5, -0.5)

    ref = cells['reference_domain'].iloc[0]
    fig.suptitle('Pain-power slope by PROCESSING DOMAIN, one mixed model per band\n'
                 'filled = BH-significant slope; omnibus tests whether the domains '
                 f'differ at all (reference domain: {ref})', fontsize=11.5)
    fig.tight_layout(rect=(0, 0.17, 1, 0.90))
    fig.text(0.01, 0.005,
             f'ONE MODEL PER BAND over every {unit_word(args, plural=False)} '
             'at once: log10_power ~ NRS_within * C(domain) + NRS_submean + '
             + random_effects_note(args)
             + ('THE UNIT IS THE ROI, and the ROI is NOT IN THE MODEL -- it is '
                'only the lookup that says which domain a channel is in. '
                'Nothing here keeps one well-sampled ROI from carrying a '
                'domain; the region-level consistency map is what does. '
                if getattr(args, 'unit', 'parcel') == 'roi' else
                'THE UNIT IS THE ATLAS PARCEL, not a hand-built ROI -- parcel '
                'enters as a random SLOPE so each parcel deviates around its '
                'domain rather than being averaged into it, which is what keeps '
                'one well-sampled parcel from carrying a domain. ')
             + 'Each marker is a MARGINAL slope, a linear combination of the '
             'reference slope and that domain\'s interaction term, with the SE '
             'from the fitted covariance -- not the interaction coefficient, '
             'which is only the DIFFERENCE from the reference. The omnibus is a '
             'joint Wald chi2 on the interaction block, not an LRT: REML '
             'likelihoods are not comparable across fixed-effects designs. '
             'Parcel is NESTED in subject because statsmodels cannot fit a '
             'crossed parcel term -- it silently nests it instead -- so the '
             'domain SE does not account for a parcel deviating consistently '
             f'across patients. {domain_caveat(args.roi_scheme)}\n{DISCLAIMER}',
             fontsize=6.3, va='bottom', ha='left', color='0.35', wrap=True)
    out = run_dir / 'fig_domain_slopes.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info('wrote %s', out)


def figure_by_domain(run_dir, cells, slopes, per_domain, domains, args,
                     term='pain'):
    """`fig_domain_slopes.png` transposed: ONE PANEL PER DOMAIN, bands as rows.

    Same numbers, same model, different question. The by-band figure asks "at
    this frequency, which domains respond?" and is the right shape for the
    omnibus, which is a within-band contrast ACROSS domains. This one asks
    "what is this network's SPECTRAL PROFILE?" -- a question the by-band layout
    can only answer by eye-hopping across six panels.

    Drawn once per TERM of the medication design (`SPECTRA_TERMS`), each to its
    own file with its own shared x scale. The three are never put on one scale:
    the units are per pain point, per dose-state switch, and per pain point per
    dose-state switch respectively.

    Three consequences of the transpose, all deliberate:

    - Rows are ordered by frequency, so a panel reads as a spectrum. A faint
      line connects the estimates to make that profile legible. It joins ORDINAL
      band categories, not a continuous axis -- the bands are unequal in width
      (delta 1-4 Hz, high_gamma 70-200 Hz) and separated by gaps, so the line is
      a reading aid and never an interpolation. Kept thin and pale for exactly
      that reason.
    - The omnibus is a property of a BAND, not of a domain, so it cannot sit in
      a panel title here. It moves to the shared y axis, beside the band name,
      where it annotates the row it actually describes and appears once. It is
      the omnibus FOR THIS TERM -- three different joint Wald tests.
    - The x scale is shared across panels. Comparing networks is the entire
      point of this layout, and per-panel scales would make a weak domain look
      like a strong one.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    spec = SPECTRA_TERMS[term]
    bands = [b for b in cells['band']]
    slopes = term_slopes_frame(slopes, term)
    if slopes.empty:
        logger.info('no %r rows in this run -- skipping %s', term, spec['out'])
        return
    fig, axes = plt.subplots(1, len(domains),
                             figsize=(2.65 * len(domains) + 1.8, 5.2),
                             sharey=True, sharex=True, squeeze=False)
    xmax = float(np.nanmax(np.abs(np.concatenate(
        [slopes['ci_lo'].to_numpy(), slopes['ci_hi'].to_numpy()])))) * 1.08
    y = np.arange(len(bands))

    for j, dom in enumerate(domains):
        ax = axes[0][j]
        colour = DOMAIN_COLOURS.get(dom, '0.4')
        d = slopes[slopes['domain'] == dom].set_index('band').reindex(bands)
        ax.plot(d['beta'].to_numpy(), y, '-', color=colour, lw=1.0, alpha=0.30,
                zorder=2)
        for i, band in enumerate(bands):
            r = d.loc[band]
            sig = bool(r.get('p_bh_reject', False))
            ax.errorbar(r['beta'], i, xerr=1.96 * r['se'], fmt='o',
                        ms=8 if sig else 5.5, lw=1.8 if sig else 1.1, capsize=3,
                        color=colour,
                        markerfacecolor=colour if sig else 'white', zorder=3)
        ax.axvline(0, color='0.5', lw=1.0, ls='--', zorder=1)
        n_p = int(per_domain.loc[dom, 'n_parcels'])
        n_c = int(per_domain.loc[dom, 'n_contacts'])
        n_s = int(per_domain.loc[dom, 'n_subjects'])
        ax.set_title(f'{dom}\n{unit_count(args, n_p, short=True)}, {n_c} chan, '
                     f'{n_s} subj', fontsize=9,
                     color=colour)
        ax.set_xlim(-xmax, xmax)
        ax.set_xlabel(spec['xlabel'], fontsize=8)
        ax.tick_params(labelsize=7)
        if j == 0:
            labels = []
            for band in bands:
                omni = cells[cells['band'] == band].iloc[0]
                p_omni = omni.get(spec['omnibus'],
                                  omni.get('p_omnibus', np.nan))
                labels.append(f'{band}\nomnibus p = {p_omni:.3g}')
            ax.set_yticks(y)
            ax.set_yticklabels(labels, fontsize=7.5)
            ax.set_ylim(len(bands) - 0.5, -0.5)

    ref = cells['reference_domain'].iloc[0]
    drug = getattr(args, 'drug_set', None)
    excl = getattr(args, 'exclude_drug_set', None)
    strat = ''
    if term in ('med', 'pain_x_med') and drug:
        strat = (f'\n{drug} within '
                 f'{getattr(args, "med_window_hours", "?")} h of the score'
                 + (f', compared against epochs free of {excl} too' if excl
                    else ''))
    fig.suptitle(f'{spec["title"]} (the by-band figure, transposed)\n'
                 'filled = BH-significant within this term; the omnibus on each '
                 'row tests whether the domains differ IN THAT BAND '
                 f'(reference domain: {ref}){strat}', fontsize=11.5)
    fig.tight_layout(rect=(0, 0.17, 1, 0.90))
    fig.text(0.01, 0.005,
             'SAME MODEL AND SAME NUMBERS as the by-band figure -- one fit '
             'per band, read down instead of across: '
             + formula_note(args)
             + spec['reading'] + ' The connecting line joins ORDINAL band '
             'categories of unequal width (delta 1-4 Hz, high_gamma 70-200 Hz) '
             'with gaps between them, and with the line-noise bins removed; it '
             'is a reading aid for the profile, NOT an interpolated spectrum. '
             'THE OMNIBUS IS A PROPERTY OF THE ROW, not of a panel: it asks '
             'whether domains differ within that band FOR THIS TERM, so it is '
             'printed once on the shared y axis -- the three terms have three '
             'different omnibus tests. BH runs WITHIN this term, never pooled '
             'across the three. A panel with no significant marker is not '
             'evidence of no effect -- see the per-domain n. Each marker is a '
             'MARGINAL estimate with the SE from the fitted covariance, not the '
             'interaction coefficient, which is only the DIFFERENCE from the '
             'reference. X SCALE IS SHARED ACROSS PANELS so domains are '
             'comparable by eye, but NOT across the three term figures -- their '
             f'units differ. {domain_caveat(args.roi_scheme)}\n{DISCLAIMER}',
             fontsize=6.3, va='bottom', ha='left', color='0.35', wrap=True)
    out = run_dir / spec['out']
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info('wrote %s', out)


if __name__ == '__main__':
    main()
