"""Mixed-effects models of BAND POWER on pain score, one fit per region x band.

    analysis/pain/bandpower/mixed_effects/<view_scheme>/<run>_<timestamp>/

A deliberate new level-2 question (`bandpower`), opened because this asks
something the `psd_physiology` grid cannot: not "which frequency" but "which
BAND", with one estimate per band that carries its own standard error. 20 regions
x 6 bands = 120 fits, against the grid's 9,723 cells.

THE REGION SET IS `roi_v2_ofc`, NOT `roi_v2`: mOFC and lOFC are fused into one
OFC (2026-09-16). roi_v2 measured 94 contacts in mOFC against 191 in lOFC and 504
in Insula, and the pain literature usually reports orbitofrontal cortex
undivided. Pass `--roi-scheme roi_v2` to keep them split; the scheme's full
CONTENTS go into provenance either way, so a run is never ambiguous about which
region set produced it.

WHY THIS EXISTS RATHER THAN AVERAGING THE GRID'S BETAS
------------------------------------------------------
Averaging the per-frequency betas inside a band gives a defensible POINT
ESTIMATE -- measured on 6 region x band cells, it agrees with a refit to under 1%
when the aggregation is a mean of logs -- but it cannot give an interval. The SE
of a mean of 463 estimates depends on their cross-bin covariance, and every cell
of the grid was fitted independently, so that covariance was never computed.
Averaging the per-cell SEs is not a substitute: measured against a refit it ran
from 5% to 67% too large. A band-level p-value therefore has to come from a
band-level fit.

The aggregation rule is also not free. `log10(mean(10**x))` (registry AXIS 5,
what a band POWER actually is) gave betas 12-31% LARGER in magnitude than the
average of the per-bin betas on those same 6 cells, and moved one from z=1.74 to
z=2.07. So the band value is built with `axes.aggregate_bands`, the one sanctioned
implementation, and never by averaging betas.

WHAT IS FITTED
--------------
One row = ONE CHANNEL x ONE 5-minute pre-report epoch. The outcome is that
channel's log10 band power in that epoch: the bins whose geometric centre falls
in the band, line-noise bins EXCLUDED, combined linear-then-log. A coefficient
reads as d log10(band power) per pain point.

    log10_power ~ NRS_within + NRS_submean
                  + (NRS_within || subject) + (1 | subject:channel)

Identical to `run_mixed_model_grid` -- same cohort, same ROI scheme, same
coverage floor, same REML fit. Only the outcome's frequency support changed. No
normalization and no baseline: contact amplitude differences are multiplicative,
so in log space they are additive and the channel random intercept absorbs them
exactly.

BAND SETS. The default is `paper_bands_6_hg200`: the target paper's six bands
with high_gamma extended to 70-200 Hz, because 170 Hz was their ceiling and not
this dataset's -- the native-resolution map shows Thalamus high-frequency
structure running past it. The published `paper_bands_6` stays selectable and
UNEDITED, since the decoding replication's edges must remain the paper's. Both
keep the paper's GAP AT 12-15 HZ, reproduced rather than closed. Bands that cross
a 60 Hz harmonic have the notched bins removed before aggregation; `band_caveat()`
states each set's own gaps and crossings, and it is computed rather than written
down so it cannot misdescribe a variant.

INFERENCE, IN TWO STAGES BECAUSE THEY COST DIFFERENT AMOUNTS
------------------------------------------------------------
    fit + collect   Wald z, BH over the 120-cell family. Seconds. Assumes
                    z ~ N(0,1) under the null, which needs the SE to be right,
                    which needs the random-effect structure approximately right
                    and enough subjects for the asymptotics.
    perm + collect  Within-subject shuffle of the epoch -> pain-score pairing,
                    refitting each cell. ~1.6 s a fit, so 120 cells x 1,000
                    shuffles is ~53 CPU-hours -- affordable here precisely
                    because there are 120 cells and not 9,723. Assumes only
                    exchangeability of that pairing.

NEITHER FIXES THE CONFOUND THEY SHARE: epochs within a subject are exchangeable
only if nothing else drifts with pain over a hospital stay. No time-of-day term,
no time-since-admission term, and the QC mask is signal quality only -- opioid
administrations and post-ictal periods are not excluded. A permutation test
replaces the reference distribution, not the design.

    python -m ieeg_ehr.analysis.run_bandpower_mixed --stage fit
    python -m ieeg_ehr.analysis.run_bandpower_mixed --stage collect --run-dir R

EXPLORATORY. Discovery cohort only. Nominations, not findings.
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ieeg_ehr import config, io
from ieeg_ehr.config import roi_schemes
from ieeg_ehr.analysis import cluster_permutation as cp
from ieeg_ehr.analysis import dx_state, fullres_cells, med_state, mixed_model as mm
from ieeg_ehr.analysis import reference_run, view_tables
# The medication fits are REUSED, not reimplemented: these are the same two
# models `run_mixed_model_med_strata` runs on the 50-log-bin grid, so a band
# result and a frequency result mean the same thing by construction.
from ieeg_ehr.analysis.run_mixed_model_med_strata import fit_decomposed, fit_matched
from ieeg_ehr.analysis.plot_mixed_model_subject_lines import epoch_level, subject_slopes
from ieeg_ehr.analysis.run_fullres_grid import (CONFOUND_CAVEAT, resolve_cohort)
from ieeg_ehr.views import axes, fullres_reader

logger = logging.getLogger(__name__)

SCRIPT = 'ieeg_ehr/analysis/run_bandpower_mixed.py'

QUESTION = 'bandpower'
OUTPUT_TYPE = 'mixed_effects'
VIEW_SCHEME = 'paperbands6-roiv2ofc'
RUN_NAME = 'paperbands6_mixedlm'

FDR_Q = 0.05

#: Subjects required IN EACH ARM before a diagnosis interaction is fitted for a
#: cell. Deliberately the same number as `mm.MIN_SUBJECTS`, applied per stratum
#: rather than in total: an interaction is a comparison of two groups, so the
#: smaller group is what the estimate rests on and a total count hides it. On
#: the reference cohort this refuses dACC, Occipital, rACC, Auditory and M1,
#: whose MDD arms hold 4-7 subjects. Those cells become ROWS with a reason, not
#: silent gaps -- and they are still fitted by the pain-only model beside them.
DX_MIN_SUBJECTS_PER_ARM = mm.MIN_SUBJECTS

#: What the diagnosis contrast does NOT control for, stated once and carried
#: into provenance, METHODS and every figure this run writes.
DX_CAVEAT = (
    'THE DIAGNOSIS LABEL IS AN ADMINISTRATIVE ICD CODE, not a structured '
    'clinical assessment, and most cases are labelled by a professional '
    'billing code alone. ANTIDEPRESSANT EXPOSURE IS NOT CONTROLLED: SSRIs and '
    'SNRIs alter cortical oscillatory power directly, and treated MDD patients '
    'are by definition more likely to be on them, so a stratum difference '
    'confounds diagnosis with medication. The recency window also decides the '
    'label -- subjects carrying an MDD code outside it fall into the MDD- '
    'arm and dilute the contrast toward the null.')

#: `paper_bands_6_hg200` is the DEFAULT rather than the published
#: `paper_bands_6`: 170 Hz was the target paper's ceiling, not this dataset's,
#: and the native-resolution map shows Thalamus high-frequency structure running
#: past it. The published set stays selectable, unedited, because the decoding
#: replication's edges must remain the paper's. The chosen set is in the run's
#: folder name, so a path never misstates which edges produced it.
BAND_SETS = {'paper_bands_6_hg200': config.PAPER_BANDS_6_HG200_HZ,
             'paper_bands_6': config.PAPER_BANDS_6_HZ,
             'canonical': config.CANONICAL_BANDS_HZ}

#: Level-4 folder per band set. The ROI scheme's code is appended by
#: `view_scheme_for`, so a path can never claim a region set the run did not use.
VIEW_SCHEMES = {'paper_bands_6_hg200': 'paperbands6hg200',
                'paper_bands_6': 'paperbands6',
                'canonical': 'canonicalbands'}


def view_scheme_for(band_set, roi_scheme, drug_set=None):
    """The level-4 folder: band edges, region set, and drug set if medicated.

    All three change what every coefficient in the run MEANS, so all three are in
    the path. Built in one place because the alternative -- a hardcoded string per
    band set -- silently kept saying `roiv2ofc` after `--roi-scheme` was added.
    """
    from ieeg_ehr.views.view_config import ROI_SCHEME_CODES
    from pathlib import Path as _Path
    roi = ROI_SCHEME_CODES.get(roi_scheme)
    if roi is None:
        roi = _Path(str(roi_scheme)).stem.replace('_', '').replace('-', '')
    parts = [VIEW_SCHEMES.get(band_set, band_set), roi or 'roidefault']
    if drug_set:
        parts.append(drug_set)
    return '-'.join(parts)

DISCLAIMER = ('EXPLORATORY -- discovery cohort, NOMINATIONS NOT FINDINGS. '
              'Not confirmed out of sample.')

def band_caveat(band_set):
    """The band set's own gaps and harmonic crossings, in words.

    Computed rather than written down, because the two paper variants differ in
    exactly the property a reader needs (what the top of high_gamma is), and a
    fixed string would misdescribe one of them.
    """
    bands = BAND_SETS[band_set]
    top = max(hi for _, hi in bands.values())
    parts = []
    if band_set.startswith('paper_bands_6'):
        parts.append(
            'The band edges are the target paper\'s (Huang et al. 2025), which '
            'leaves a GAP AT 12-15 HZ -- alpha ends at 12 and beta starts at 15 '
            '-- reproduced rather than closed.')
    if band_set == 'paper_bands_6':
        parts.append(
            'Nothing covers above 170 Hz, so the pain-related high-frequency '
            'increase the native-resolution map shows in Thalamus above ~170 Hz '
            'has no band to appear in. `--band-set paper_bands_6_hg200` extends '
            'high_gamma to 200 Hz for that reason.')
    elif band_set == 'paper_bands_6_hg200':
        parts.append(
            'high_gamma is EXTENDED to 70-200 Hz, not the paper\'s 70-170: 170 Hz '
            'was their ceiling, not this dataset\'s, and the native-resolution map '
            'shows Thalamus structure running past it. The published set is '
            'unedited and still selectable as `--band-set paper_bands_6`. Nothing '
            f'covers {top:g}-250 Hz.')
    parts.append(
        'Bands that CROSS a 60 Hz harmonic have the notched bins removed before '
        'aggregation -- mandatory, not optional: without it gamma would absorb '
        'the 58-62 Hz residue and high_gamma the 118-122 and 178-182 Hz ones.')
    return ' '.join(parts)


MED_CAVEAT = (
    'CONFOUNDING BY INDICATION IS THE FIRST-ORDER PROBLEM HERE, not a footnote. '
    'Patients are dosed BECAUSE they are in pain: on this cohort opioid-medicated '
    'epochs average NRS 4.67 against 2.22 unmedicated, so medication state and '
    'pain are entangled by design and no term in a regression undoes that. The '
    'arrow is also not identifiable from this table -- a scheduled dose is given '
    'whatever the score, and an assessment is often charted precisely BECAUSE a '
    'PRN dose was requested (see med_state and med_analysis/pain_link). What the '
    'models below do is separate the WITHIN-patient medication contrast from the '
    'BETWEEN-patient one, and condition on the reported score; what they cannot '
    'do is make either contrast causal.')

WALD_CAVEAT = (
    'p and p_bh are PARAMETRIC Wald. They assume z ~ N(0,1) under the null, which '
    'requires the standard error to be correct and therefore the random-effect '
    'structure to be approximately right with enough subjects for the '
    'asymptotics. The Phase-1 pilot measured the null Wald z SD at ~1.03 on 19 '
    'cells -- near-calibrated, slightly anticonservative. `--stage perm` replaces '
    'them with a permutation p that assumes only exchangeability of the '
    'epoch -> pain-score pairing.')


# ============================================================================
# THE BAND OUTCOME
# ============================================================================

def band_table(band_set, notch_half_width_hz=None, epoch_minutes=None):
    """(bin_table with the notch REMOVED, band edge dict).

    The notch is removed from the table handed to `axes.aggregate_bands`, which
    is what excludes those frequencies from every band: that function does not
    drop flagged bins itself, it relies on them being absent or already NaN.
    Mandatory for paper_bands_6, whose gamma and high_gamma cross 60 and 120 Hz.
    """
    table = fullres_reader.freq_table(epoch_minutes).set_index('freq_bin_index')
    notched = [int(b) for b in fullres_reader.notch_freqs(
        half_width_hz=notch_half_width_hz, epoch_minutes=epoch_minutes)]
    kept = table.drop(index=[b for b in notched if b in table.index])
    bands = BAND_SETS[band_set]
    centres = np.sqrt(kept['bin_low_hz'].to_numpy() * kept['bin_high_hz'].to_numpy())
    for name, (lo, hi) in bands.items():
        n = int(((centres >= lo) & (centres < hi)).sum())
        logger.info('  %-12s %3g-%3g Hz : %3d native bins', name, lo, hi, n)
    return kept, bands, notched


def aggregate(values, kept_table, bands):
    """(n_rows, n_bins) log10 power -> (n_rows, n_bands) log10 BAND power.

    Straight through `axes.aggregate_bands` with `is_difference=False,
    domain='log'`, i.e. log10(mean(10**x)) -- linear-then-log, because raw log
    power does not average arithmetically (a mean of logs is a geometric mean).
    On this axis every bin is the same width, so uniform and width weighting are
    numerically identical and the default stands.
    """
    return axes.aggregate_bands(np.asarray(values, dtype=np.float64), kept_table,
                                bands=bands, is_difference=False, domain='log',
                                weighting='uniform')


# ============================================================================
# STAGE: fit
# ============================================================================

def fit_one_cell(df, meta):
    """(record, per-subject unpooled slopes) for one region x band cell."""
    ok, reason = mm.cell_is_fittable(df)
    if not ok:
        rec = mm.failed_record(meta['region'], meta['band_index'], meta['band_lo_hz'],
                               meta['band_hi_hz'], reason, df=df)
        rec.update({k: meta[k] for k in ('band', 'cell_index')})
        return rec, None, None

    t0 = time.time()
    try:
        res, warn = mm.fit_cell(df, mm.VC_FULL)
    except mm.CellFitError as exc:
        rec = mm.failed_record(meta['region'], meta['band_index'], meta['band_lo_hz'],
                               meta['band_hi_hz'], f'full: {exc}', df=df,
                               fit_seconds=time.time() - t0)
        rec.update({k: meta[k] for k in ('band', 'cell_index')})
        return rec, None, None

    # The reduced model IS fitted here, unlike the 9,723-cell grid. At 120 cells
    # the second fit is affordable, and the heterogeneity LRT is a question worth
    # answering per band: "subjects respond, but not in a consistent direction"
    # is a different claim from "no effect", and only the LRT separates them.
    try:
        res_red, warn_red = mm.fit_cell(df, mm.VC_REDUCED)
    except mm.CellFitError as exc:
        logger.warning('%s %s reduced model failed: %s', meta['region'], meta['band'],
                       exc)
        res_red, warn_red = None, [f'reduced failed: {exc}']

    rec = mm.cell_record(res, res_red, df, region=meta['region'],
                         freq_bin_index=meta['band_index'],
                         bin_low_hz=meta['band_lo_hz'],
                         bin_high_hz=meta['band_hi_hz'],
                         fit_seconds=time.time() - t0,
                         warnings_full=warn, warnings_reduced=warn_red)
    rec.update({k: meta[k] for k in ('band', 'cell_index')})
    rec['nrs_within_var'] = float(np.var(df['NRS_within'].to_numpy(), ddof=0))

    slopes = subject_slopes(epoch_level(df))
    slopes['region'] = meta['region']
    slopes['band'] = meta['band']

    # BLUPs as well as the unpooled slopes. They are NOT interchangeable -- partial
    # pooling drags every subject toward the group, which is exactly why the
    # caterpillar plot wants both: the gap between them IS the shrinkage, and a
    # cell whose spread only exists before pooling is telling you something.
    blups = pd.DataFrame(mm.blup_rows(res, df, region=meta['region'],
                                      freq_bin_index=meta['band_index']))
    if len(blups):
        blups['band'] = meta['band']
    logger.info('%-18s %-11s | n=%2d subj %4d chan %6d rows | beta %+.5f z %+5.2f '
                'p %.4g | LRT p %.4g | %.1fs',
                meta['region'], meta['band'], rec['n_subjects'], rec['n_channels'],
                rec['n_rows'], rec['beta_nrs_within'], rec['z'], rec['p'],
                rec['p_lrt_mixture'], rec['fit_seconds'])
    return rec, slopes, blups


def fit_dx_interaction(df, meta):
    """The diagnosis-moderator fit for one cell: does the pain slope differ?

    Same variance components as the pain-only fit (`mm.VC_FULL`), so the two are
    directly comparable and the ONLY thing that changed is the fixed-effects
    design. See `mm.FORMULA_DX_INTERACTION` for why there is no within/between
    split of `dx_state` and why the random effects are left alone.

    The record carries BOTH simple slopes as well as the interaction, computed
    from this one fit via `mm.simple_slopes`. Nothing downstream should ever
    refit an arm on its own to get them.
    """
    t0 = time.time()

    # Structural refusals FIRST, so an unfittable cell becomes a row rather than
    # a fit that "worked" on one stratum. A cell where every subject is a case
    # (or none is) has no between-stratum contrast at all, and patsy would
    # happily drop the aliased column and hand back a pain-only fit wearing an
    # interaction model's name -- which is the failure mode worth spending an
    # explicit branch on.
    n_case = int(df.loc[df['dx_state'] == 1, 'subject'].nunique())
    n_ctrl = int(df.loc[df['dx_state'] == 0, 'subject'].nunique())
    if min(n_case, n_ctrl) < DX_MIN_SUBJECTS_PER_ARM:
        rec = mm.failed_record(
            meta['region'], meta['band_index'], meta['band_lo_hz'],
            meta['band_hi_hz'],
            f'dx strata too small: {n_case} case / {n_ctrl} control subjects, '
            f'need {DX_MIN_SUBJECTS_PER_ARM} in each', df=df)
        rec.update({k: meta[k] for k in ('band', 'cell_index')})
        rec.update({'n_subjects_case': n_case, 'n_subjects_control': n_ctrl})
        return rec

    try:
        res, warn = mm.fit_cell(df, mm.VC_FULL,
                                formula=mm.FORMULA_DX_INTERACTION)
    except mm.CellFitError as exc:
        rec = mm.failed_record(meta['region'], meta['band_index'],
                               meta['band_lo_hz'], meta['band_hi_hz'],
                               f'dx interaction: {exc}', df=df,
                               fit_seconds=time.time() - t0)
        rec.update({k: meta[k] for k in ('band', 'cell_index')})
        rec.update({'n_subjects_case': n_case, 'n_subjects_control': n_ctrl})
        return rec
    t_full = time.time() - t0

    try:
        res_red, warn_red = mm.fit_cell(df, mm.VC_REDUCED,
                                        formula=mm.FORMULA_DX_INTERACTION)
    except mm.CellFitError as exc:
        res_red, warn_red = None, [f'reduced failed: {exc}']

    rec = mm.cell_record(res, res_red, df, region=meta['region'],
                         freq_bin_index=meta['band_index'],
                         bin_low_hz=meta['band_lo_hz'],
                         bin_high_hz=meta['band_hi_hz'], fit_seconds=t_full,
                         warnings_full=warn, warnings_reduced=warn_red,
                         extra_terms=mm.DX_INTERACTION_TERMS)
    rec.update({k: meta[k] for k in ('band', 'cell_index')})
    rec.update(mm.simple_slopes(res))
    rec.update({'n_subjects_case': n_case, 'n_subjects_control': n_ctrl,
                'n_epochs_case': int(df.loc[df['dx_state'] == 1]
                                     .groupby('subject')['epoch_id'].nunique().sum()),
                'n_epochs_control': int(df.loc[df['dx_state'] == 0]
                                        .groupby('subject')['epoch_id'].nunique().sum())})

    logger.info('%-18s %-11s | DX  non-MDD %+.5f  MDD %+.5f  | diff %+.5f '
                '(p %.3g) | %d vs %d subj', meta['region'], meta['band'],
                rec.get('slope_ref', np.nan), rec.get('slope_mod', np.nan),
                rec.get('dx_ix_beta', np.nan), rec.get('dx_ix_p', np.nan),
                n_case, n_ctrl)
    return rec


def stage_fit(args):
    ref = reference_run.load(args.reference_run)
    ref.describe()
    epoch_minutes = ref.view_params.get('epoch_minutes')

    view_dir = fullres_cells.resolve_view_dir(
        args.view_dir, mask_label=args.mask_label or ref.view_params.get('mask_label'),
        max_excluded_frac=ref.view_params.get('max_excluded_frac'),
        epoch_minutes=epoch_minutes)
    logger.info('epoch-mean full-res view: %s', view_dir)

    # A scheme with COORDINATE REGIONS (roi_v2_ofc_ins) has its insula split into
    # aIns/pIns inside resolve_cohort; `split_report` carries what that did so
    # provenance records the threshold, not just the scheme name.
    split_report = {}
    paths, scores, diagnostics, subjects, roi_by_subject, no_roi = resolve_cohort(
        ref, view_dir, cohort=args.cohort, roi_scheme=args.roi_scheme,
        insula_threshold=args.insula_threshold, report=split_report)
    ref.assert_cohort_matches(subjects, allow_drift=args.allow_cohort_drift
                              or args.cohort != 'reference')

    roi_scheme = args.roi_scheme or ref.view_params.get('roi_scheme', 'roi_v2')
    regions = view_tables.roi_regions_for({'roi_scheme': roi_scheme})

    # DOMAINS AS THE ROI. `roi_by_subject` is {subject: {channel: ROI}}, and
    # everything downstream -- load_region_matrix, the cell loop, the tables --
    # only ever asks it "which unit is this channel in". So collapsing ROIs
    # into domains is a VALUE REMAP on that dict plus a swap of the region
    # list; no other code path changes and the model is untouched.
    #
    # It has to happen HERE rather than via a flat ROI scheme, because
    # pain_domains_v3's aIns/pIns are COORDINATE-derived: no set of label
    # patterns can produce them, so there is no fused label scheme to resolve.
    # The split runs at the base level inside resolve_cohort, and this folds
    # the resulting ROIs up afterwards.
    #
    # A channel whose ROI NO DOMAIN CLAIMS is dropped (Hippocampus, PCC, Basal
    # Ganglia, Parietal/MTL other, Lateral Temporal in v3). That is the domain
    # scheme's own membership decision, recorded in provenance, not a silent
    # loss -- the dropped ROIs and their channel counts are logged and stored.
    domain_drop = {}
    if args.domain_scheme:
        ds = roi_schemes.domain_scheme(args.domain_scheme)
        r2d = ds['roi_to_domain']
        for sub, chmap in roi_by_subject.items():
            for ch, r in chmap.items():
                if r not in r2d:
                    domain_drop[r] = domain_drop.get(r, 0) + 1
        roi_by_subject = {sub: {ch: r2d[r] for ch, r in chmap.items()
                                if r in r2d}
                          for sub, chmap in roi_by_subject.items()}
        regions = list(ds['display'])
        logger.info('DOMAINS ARE THE UNIT (%s): %d domain(s) %s',
                    args.domain_scheme, len(regions), regions)
        if domain_drop:
            logger.warning('ROIs claimed by NO domain in %s, dropped: %s',
                           args.domain_scheme,
                           {k: v for k, v in sorted(domain_drop.items())})
    if args.exclude_regions:
        unknown = [r for r in args.exclude_regions if r not in regions]
        if unknown:
            raise SystemExit(
                f'--exclude-regions {unknown} are not regions of {roi_scheme!r}. '
                f'Known: {regions}. Refusing rather than silently excluding '
                'nothing, which would look like the exclusion worked.')
        regions = [r for r in regions if r not in set(args.exclude_regions)]
        logger.warning('EXCLUDED from this run entirely (not fitted, not in the BH '
                       'family, not plotted): %s -- %s', args.exclude_regions,
                       args.exclude_reason or 'no reason recorded')
    logger.info('ROI scheme %r -> %d region(s): %s', roi_scheme, len(regions), regions)
    logger.info('band set %r:', args.band_set)
    kept, bands, notched = band_table(args.band_set, args.notch_half_width_hz,
                                      epoch_minutes)
    band_names = list(bands)

    # A NAMED CELL SUBSET. Only these region x band cells are fitted, so they --
    # and nothing else -- are every BH family in collect. That is the point: a
    # follow-up on cells chosen from an earlier run corrects over the cells it
    # asks about. The choice was made on that earlier run's results, so this
    # run's p-values are CONDITIONAL on the selection, and `cell_selection` in
    # provenance records the reason verbatim. `cell_index` keeps its full-grid
    # value, so a cell here joins to the same cell of the full run.
    wanted = None
    if args.cells:
        wanted = set()
        for spec in args.cells:
            region, sep, band = spec.rpartition(':')
            if not sep or region not in regions or band not in band_names:
                raise SystemExit(
                    f'--cells {spec!r} is not REGION:BAND of this run. Regions: '
                    f'{regions}. Bands: {band_names}.')
            wanted.add((region, band))
        regions_in_order = [r for r in regions if any(r == w[0] for w in wanted)]
        logger.warning('CELL SUBSET: %d cell(s) fitted, and they alone are the BH '
                       'family: %s -- %s', len(wanted),
                       sorted(wanted), args.cell_selection or 'no reason recorded')

    # ---- per-epoch medication state ------------------------------------
    med_lookup, med_summary, med_missing = None, None, []
    if args.med_model != 'none':
        subclasses = med_state.DRUG_SETS[args.drug_set]
        admin = med_state.load_admin_table(subclasses=subclasses)
        bare = sorted(s.replace('sub-', '') for s in subjects)
        defs = med_state.load_epoch_defs(epoch_minutes=epoch_minutes, subjects=bare)
        state = med_state.epoch_med_state(defs, admin, hours=args.med_window_hours)
        med_missing = sorted(set(defs['subject']) - set(admin['subject']))
        if med_missing:
            # NOT fatal: "no opioid charted" is a real observation. But a session
            # absent from the MAR export looks identical to a patient who was
            # never dosed, so it is named rather than absorbed into the
            # unmedicated stratum.
            logger.warning('%d cohort subject(s) have NO %s administrations at '
                           'all, so every one of their epochs reads as '
                           'unmedicated: %s', len(med_missing), args.drug_set,
                           med_missing)
        med_summary = med_state.stratum_summary(state)
        logger.info('\n%s', med_summary.to_string(index=False))
        med_lookup = state.assign(
            subject_id='sub-' + state['subject'].astype(str),
            med_state=state['med_state'].astype(float))[
                ['subject_id', 'session', 'epoch_id', 'med_state']]

    # ---- subject-level diagnosis label ---------------------------------
    dx_lookup, dx_labels, dx_detail, dx_summary = None, None, None, None
    if args.dx_model != 'none':
        bare = sorted(s.replace('sub-', '') for s in subjects)
        dx_all = dx_state.load_diagnoses(subjects=bare)
        dx_labels, dx_detail = dx_state.subject_labels(
            dx_all, condition=args.dx, window_days=args.dx_window_days,
            sources=args.dx_sources)
        # Cohort coverage, named rather than absorbed. A subject with no
        # diagnoses table at all would otherwise read as MDD-negative, which is the
        # same silent-default failure `med_state` refuses for dosing.
        missing = sorted(set(subjects) - set(dx_labels['subject_id']))
        if missing:
            raise SystemExit(
                f'{len(missing)} cohort subject(s) have no diagnoses table, so '
                f'their MDD status is unknown, not negative: {missing}. Refusing '
                'rather than labelling them controls.')
        dx_labels = dx_labels[dx_labels['subject_id'].isin(subjects)]
        dx_summary = dx_state.stratum_summary(
            dx_labels, dx_state.pain_by_subject(subjects=bare))
        logger.info('\n%s', dx_summary.to_string(index=False))
        dx_lookup = dx_state.epoch_lookup(dx_labels)

    run_dir = (Path(args.run_dir) if args.run_dir else
               config.analysis_run_dir(question=args.question,
                                       output_type=OUTPUT_TYPE,
                                       view_scheme=args.view_scheme,
                                       run_name=args.run_name))
    (run_dir / 'cells').mkdir(parents=True, exist_ok=True)
    (run_dir / 'frames').mkdir(parents=True, exist_ok=True)

    todo = (regions if args.region_index is None
            else [regions[args.region_index]])
    if wanted is not None:
        todo = [r for r in todo if r in regions_in_order]
    if args.region_index is not None and not 0 <= args.region_index < len(regions):
        raise SystemExit(f'--region-index {args.region_index} outside '
                         f'0..{len(regions) - 1}')

    records, slope_parts, blup_parts = [], [], []
    for region in todo:
        ri = regions.index(region)
        t0 = time.time()
        index, values, stats = fullres_cells.load_region_matrix(
            paths, subjects, region, roi_by_subject, list(kept.index),
            epoch_minutes=epoch_minutes)
        if not len(index):
            logger.warning('%s: no rows after the ROI join, skipped', region)
            continue
        band_values, names = aggregate(values, kept, bands)
        logger.info('%s: %d rows x %d bins -> %d band(s) in %.1fs (%d subject files)',
                    region, len(index), values.shape[1], len(names),
                    time.time() - t0, stats['n_files'])

        for bi, band in enumerate(names):
            if wanted is not None and (region, band) not in wanted:
                continue
            lo, hi = bands[band]
            meta = {'region': region, 'band': band, 'band_index': band_names.index(band),
                    'band_lo_hz': float(lo), 'band_hi_hz': float(hi),
                    'cell_index': ri * len(band_names) + band_names.index(band)}
            frame = pd.DataFrame({
                'subject_id': index['subject_id'].to_numpy(),
                'session': index['session'].to_numpy(),
                'channel': index['channel'].to_numpy(),
                'epoch_id': index['epoch_id'].to_numpy(),
                'pain_score': index['pain_score'].to_numpy(),
                'value': band_values[:, bi]})
            extra = ()
            if med_lookup is not None:
                # Joined on subject, SESSION and epoch_id -- epoch_id is unique
                # only within a subject-session. An epoch with no med row is
                # dropped rather than defaulted: a silent False here would invent
                # an unmedicated observation.
                before = len(frame)
                frame = frame.merge(med_lookup,
                                    on=['subject_id', 'session', 'epoch_id'],
                                    how='inner')
                if len(frame) < before:
                    logger.info('%s %s: %d of %d rows had no medication record '
                                'and were dropped', region, band,
                                before - len(frame), before)
                extra = ('med_state',)
            if dx_lookup is not None:
                # Joined on SUBJECT only -- the label is a subject-level
                # constant, so there is no session or epoch key to match on.
                # `how='inner'` and the cohort check above together guarantee no
                # row survives with an invented label.
                before = len(frame)
                frame = frame.merge(dx_lookup, on='subject_id', how='inner')
                if len(frame) < before:
                    logger.warning('%s %s: %d of %d rows had no diagnosis label '
                                   'and were dropped', region, band,
                                   before - len(frame), before)
                extra = tuple(extra) + ('dx_state',)
            df = mm.build_cell_frame(frame, region=region,
                                     freq_bin_index=meta['band_index'],
                                     extra_columns=extra)
            rec, slopes, blups = fit_one_cell(df, meta)
            rec['model'] = 'pain'
            records.append(rec)

            if args.dx_model == 'interaction':
                dx_rec = fit_dx_interaction(df, meta)
                dx_rec['model'] = f'dx_{args.dx}'
                records.append(dx_rec)

            if args.med_model in ('decomposed', 'both'):
                dec, _ = fit_decomposed(df, {**meta,
                                             'freq_bin_index': meta['band_index'],
                                             'bin_low_hz': meta['band_lo_hz'],
                                             'bin_high_hz': meta['band_hi_hz']})
                dec.update({'band': band, 'model': 'med_decomposed'})
                records.append(dec)
                logger.info('%-18s %-11s | DECOMPOSED  med_within %+.5f (p %.3g)  '
                            'pain x med %+.5f (p %.3g)', region, band,
                            dec.get('medw_beta', np.nan), dec.get('medw_p', np.nan),
                            dec.get('med_ix_beta', np.nan),
                            dec.get('med_ix_p', np.nan))
            if args.med_model in ('matched', 'both'):
                mat = fit_matched(df, {**meta,
                                       'freq_bin_index': meta['band_index'],
                                       'bin_low_hz': meta['band_lo_hz'],
                                       'bin_high_hz': meta['band_hi_hz']})
                mat.update({'band': band, 'model': 'med_matched'})
                records.append(mat)
                logger.info('%-18s %-11s | MATCHED     med at same NRS %+.5f '
                            '(p %.3g)  nonparam %+.5f over %s subj', region, band,
                            mat.get('beta_med', np.nan), mat.get('p', np.nan),
                            mat.get('nonparam_diff', np.nan),
                            mat.get('n_subjects_matched', 0))
            if slopes is not None:
                slope_parts.append(slopes)
            if blups is not None and len(blups):
                blup_parts.append(blups)
            # The frame, so `--stage perm` refits without re-reading the view.
            if len(df):
                io.write_table(df, run_dir / 'frames' /
                               f"cell_{meta['cell_index']:03d}.parquet",
                               params={'region': region, 'band': band},
                               script=SCRIPT)

    tag = 'all' if args.region_index is None else f'{args.region_index:03d}'
    io.write_table(pd.DataFrame(records), run_dir / 'cells' / f'region_{tag}.parquet',
                   params={'band_set': args.band_set, 'n_cells': len(records)},
                   script=SCRIPT)
    io.write_table(pd.concat(slope_parts, ignore_index=True) if slope_parts
                   else pd.DataFrame(columns=['subject', 'slope', 'region', 'band']),
                   run_dir / 'cells' / f'slopes_{tag}.parquet',
                   params={'source': 'unpooled per-subject OLS, not BLUPs'},
                   script=SCRIPT)
    io.write_table(pd.concat(blup_parts, ignore_index=True) if blup_parts
                   else pd.DataFrame(columns=['subject', 'region', 'band']),
                   run_dir / 'cells' / f'blups_{tag}.parquet',
                   params={'source': 'model BLUPs -- SHRUNK toward the group, keep '
                                     'beside the unpooled slopes, never instead'},
                   script=SCRIPT)

    if dx_labels is not None:
        # CSV, not Parquet: these are small, terminal, read-by-eye tables under
        # analysis/ (CLAUDE.md, io_conventions).
        dx_params = {'condition': args.dx, 'window_days': args.dx_window_days,
                     'source_set': args.dx_sources,
                     'icd10_prefixes': list(dx_state.CONDITIONS[args.dx]['icd10']),
                     'icd9_prefixes': list(dx_state.CONDITIONS[args.dx]['icd9'])}
        io.write_table(dx_labels, run_dir / 'dx_subject_labels.csv',
                       params=dx_params, script=SCRIPT,
                       extra={'caveat': DX_CAVEAT,
                              'definition': dx_state.CONDITIONS[args.dx]['description']})
        io.write_table(dx_detail, run_dir / 'dx_session_detail.csv',
                       params=dx_params, script=SCRIPT)
        io.write_table(dx_summary, run_dir / 'dx_stratum_summary.csv',
                       params=dx_params, script=SCRIPT,
                       extra={'caveat': DX_CAVEAT})

    if med_summary is not None:
        io.write_table(med_summary, run_dir / 'med_stratum_summary.parquet',
                       params={'drug_set': args.drug_set,
                               'window_hours': args.med_window_hours},
                       parents=[str(med_state.ADMIN_TABLE)], script=SCRIPT,
                       extra={'caveat': MED_CAVEAT})
        io.write_table(med_lookup, run_dir / 'epoch_med_state.parquet',
                       params={'drug_set': args.drug_set,
                               'window_hours': args.med_window_hours,
                               'subclasses': list(med_state.DRUG_SETS[args.drug_set])},
                       parents=[str(med_state.ADMIN_TABLE)], script=SCRIPT,
                       extra={'caveat': MED_CAVEAT,
                              'subjects_without_administrations': med_missing})

    if args.region_index is None:
        io.write_table(inventory(scores, diagnostics), run_dir / 'inventory_subjects.parquet',
                       script=SCRIPT)
        io.write_run_provenance(
            run_dir, script=SCRIPT,
            params={'stage': 'fit', 'band_set': args.band_set,
                    'bands': {k: list(v) for k, v in bands.items()},
                    'view_dir': str(view_dir), 'view_params': ref.view_params,
                    'criteria': ref.criteria, 'roi_scheme': roi_scheme,
                    'roi_scheme_contents': __import__(
                        'ieeg_ehr.config.roi_schemes', fromlist=['x']
                    ).scheme_provenance(roi_scheme),
                    **split_report,
                    'cohort': args.cohort, 'epoch_minutes': epoch_minutes,
                    'domain_scheme': args.domain_scheme,
                    'unit': 'domain' if args.domain_scheme else 'roi',
                    'rois_dropped_no_domain': domain_drop,
                    'notched_bins_excluded': notched,
                    'med_model': args.med_model,
                    'drug_set': args.drug_set if args.med_model != 'none' else None,
                    'med_window_hours': (args.med_window_hours
                                         if args.med_model != 'none' else None),
                    'subjects_without_administrations': med_missing,
                    'dx_model': args.dx_model,
                    'dx_condition': args.dx if args.dx_model != 'none' else None,
                    'dx_window_days': (args.dx_window_days
                                       if args.dx_model != 'none' else None),
                    'dx_sources': (args.dx_sources if args.dx_model != 'none'
                                   else None),
                    'dx_n_case': (int(dx_labels['dx_state'].sum())
                                  if dx_labels is not None else None),
                    'dx_n_control': (int((~dx_labels['dx_state']).sum())
                                     if dx_labels is not None else None),
                    'dx_min_subjects_per_arm': (DX_MIN_SUBJECTS_PER_ARM
                                                if args.dx_model != 'none' else None),
                    'excluded_regions': list(args.exclude_regions),
                    'excluded_regions_reason': args.exclude_reason,
                    'cells_selected': sorted(args.cells) if args.cells else None,
                    'cell_selection': args.cell_selection,
                    'aggregation': 'linear_then_log via axes.aggregate_bands',
                    'n_cells': len(records)},
            parents=[str(Path(args.reference_run) / 'provenance.json'), str(view_dir)],
            subjects=sorted(subjects),
            extra={'status': DISCLAIMER,
                   'band_caveat': band_caveat(args.band_set),
                   'inference_caveat': WALD_CAVEAT,
                   'mask_content': CONFOUND_CAVEAT,
                   'subjects_without_roi': sorted(no_roi),
                   **({'dx_caveat': DX_CAVEAT} if args.dx_model != 'none' else {})})
    print(run_dir)
    return run_dir


def inventory(scores, diagnostics):
    g = scores.groupby('subject_id')['pain_score']
    inv = pd.DataFrame({'n_reports': g.size(), 'nrs_mean': g.mean(),
                        'nrs_sd': g.std(ddof=1), 'nrs_min': g.min(),
                        'nrs_max': g.max(), 'n_distinct': g.nunique()}).reset_index()
    return inv.merge(diagnostics[['subject_id', 'included', 'excluded_because']],
                     on='subject_id', how='left')


# ============================================================================
# STAGE: perm  (one array task per cell)
# ============================================================================

def stage_perm(args):
    run_dir = Path(args.run_dir)
    frame_path = run_dir / 'frames' / f'cell_{args.cell_index:03d}.parquet'
    (run_dir / 'perm').mkdir(exist_ok=True)
    out_path = run_dir / 'perm' / f'perm_{args.cell_index:03d}.parquet'

    if not frame_path.exists():
        logger.warning('cell %d has no saved frame (not fittable); empty shard',
                       args.cell_index)
        io.write_table(pd.DataFrame(columns=['perm', 'beta', 'z', 'converged']),
                       out_path, params={'cell_index': args.cell_index},
                       script=SCRIPT)
        return

    df = io.read_table(frame_path, on_stale='ignore')
    res, _ = mm.fit_cell(df, mm.VC_FULL)
    logger.info('cell %d: observed beta %+.6f z %+.3f p %.4g over %d rows',
                args.cell_index, res.fe_params['NRS_within'],
                res.tvalues['NRS_within'], res.pvalues['NRS_within'], len(df))

    t0 = time.time()
    # Warm-started from the observed fit: the variance components barely move
    # under a permutation of the predictor, and the starting point is identical
    # for every shuffle so it cannot bias the null.
    out = mm.permutation_null(df, args.n_perm, seed=args.seed,
                              start_params=res.params_object, n_jobs=args.n_jobs)
    logger.info('%d shuffles in %.0fs (%.2fs each, %d jobs), %d failed',
                args.n_perm, time.time() - t0,
                (time.time() - t0) / max(args.n_perm, 1), args.n_jobs,
                int((~out['converged']).sum()))
    out.insert(0, 'cell_index', args.cell_index)
    io.write_table(out, out_path,
                   params={'n_perm': args.n_perm, 'seed': args.seed,
                           'cell_index': args.cell_index,
                           'exchangeability': 'epoch -> pain-score pairing, within '
                                              'subject, epoch relabelled as a whole'},
                   script=SCRIPT)


# ============================================================================
# STAGE: collect
# ============================================================================

def stage_collect(args):
    run_dir = Path(args.run_dir)
    parts = [io.read_table(p, on_stale='ignore')
             for p in sorted((run_dir / 'cells').glob('region_*.parquet'))]
    if not parts:
        raise SystemExit(f'no cell tables in {run_dir / "cells"}')
    cells = pd.concat(parts, ignore_index=True).sort_values('cell_index')
    cells = cells.reset_index(drop=True)

    if 'model' not in cells.columns:
        cells['model'] = 'pain'

    # BH RUNS WITHIN EACH MODEL FAMILY, never pooled across them. The pain slope,
    # the within-patient medication effect and the matched medication contrast are
    # three different questions asked of the same cells; correcting each for the
    # others' tests would make every one of them harder to detect for no
    # inferential reason.
    cells['p_bh'] = np.nan
    cells['p_bh_reject'] = pd.NA
    for model, idx in cells.groupby('model').groups.items():
        sub = cells.loc[idx]
        usable = sub['p'].notna()
        if not usable.any():
            continue
        _, adj = cp.bh_fdr(sub.loc[usable, 'p'].to_numpy(), q=args.fdr_q)
        cells.loc[sub.index[usable], 'p_bh'] = adj
        cells.loc[sub.index[usable], 'p_bh_reject'] = adj <= args.fdr_q
        logger.info('BH family %r: %d cells, %d rejected at q=%.2f', model,
                    int(usable.sum()), int((adj <= args.fdr_q).sum()), args.fdr_q)

    # The decomposed model's medication terms get their OWN families, for the same
    # reason: `medw` and `med_ix` are separate hypotheses from the pain slope that
    # shares their fit.
    # `dx_ix` -- "does the pain slope differ between diagnosis strata" -- is a
    # THIRD question asked of the same cells, so it gets its own family too. It
    # is emphatically NOT corrected against the pain slope's family: the pain
    # slope being significant somewhere says nothing about whether the strata
    # differ there, and pooling them would penalise each for the other's tests.
    #
    # `dx` (the main effect) deliberately gets NO family. It is a nuisance term
    # -- a between-subject difference in mean power, which the subject random
    # intercept exists to absorb -- and BH-correcting a nuisance term invites it
    # to be read as a result.
    for prefix in ('medw', 'med_ix', 'dx_ix'):
        col = f'{prefix}_p'
        if col not in cells.columns:
            continue
        m = cells['p'].notna() if col not in cells else cells[col].notna()
        cells[f'{prefix}_p_bh'] = np.nan
        cells[f'{prefix}_p_bh_reject'] = pd.NA
        if m.any():
            _, adj = cp.bh_fdr(cells.loc[m, col].to_numpy(), q=args.fdr_q)
            cells.loc[m, f'{prefix}_p_bh'] = adj
            cells.loc[m, f'{prefix}_p_bh_reject'] = adj <= args.fdr_q
            logger.info('BH family %r: %d cells, %d rejected', prefix,
                        int(m.sum()), int((adj <= args.fdr_q).sum()))

    # The heterogeneity LRT gets its own family. Two questions, two corrections;
    # pooling them would correct each for the other's tests.
    u2 = cells['p_lrt_mixture'].notna()
    cells['p_lrt_bh'] = np.nan
    cells['p_lrt_bh_reject'] = pd.NA
    if u2.any():
        _, adj = cp.bh_fdr(cells.loc[u2, 'p_lrt_mixture'].to_numpy(), q=args.fdr_q)
        cells.loc[u2, 'p_lrt_bh'] = adj
        cells.loc[u2, 'p_lrt_bh_reject'] = adj <= args.fdr_q

    # Permutation shards, if `--stage perm` has run.
    shards = sorted((run_dir / 'perm').glob('perm_*.parquet')) if (
        run_dir / 'perm').exists() else []
    if shards:
        nulls = pd.concat([io.read_table(p, on_stale='ignore') for p in shards],
                          ignore_index=True)
        rows = []
        for ci, grp in nulls.groupby('cell_index'):
            obs = cells.loc[cells['cell_index'] == ci, 'beta_nrs_within']
            obs = float(obs.iloc[0]) if len(obs) else np.nan
            p_perm, n_used = mm.permutation_p(obs, grp['beta'].to_numpy())
            rows.append({'cell_index': int(ci), 'p_perm': p_perm,
                         'n_perm_used': n_used})
        cells = cells.merge(pd.DataFrame(rows), on='cell_index', how='left')
        up = cells['p_perm'].notna()
        cells['p_perm_bh'] = np.nan
        cells['p_perm_bh_reject'] = pd.NA
        if up.any():
            _, adj = cp.bh_fdr(cells.loc[up, 'p_perm'].to_numpy(), q=args.fdr_q)
            cells.loc[up, 'p_perm_bh'] = adj
            cells.loc[up, 'p_perm_bh_reject'] = adj <= args.fdr_q
        logger.info('permutation p present for %d/%d cells', int(up.sum()), len(cells))
        io.write_table(nulls, run_dir / 'permutation_null.parquet',
                       params={'n_shards': len(shards)}, script=SCRIPT)

    blup_parts = [io.read_table(p, on_stale='ignore')
                  for p in sorted((run_dir / 'cells').glob('blups_*.parquet'))]
    blup_parts = [b for b in blup_parts if len(b)]
    if blup_parts:
        io.write_table(pd.concat(blup_parts, ignore_index=True),
                       run_dir / 'blups.parquet',
                       params={'source': 'model BLUPs, shrunk toward the group'},
                       script=SCRIPT)

    slopes = [io.read_table(p, on_stale='ignore')
              for p in sorted((run_dir / 'cells').glob('slopes_*.parquet'))]
    cons = pd.DataFrame()
    if slopes:
        s = pd.concat(slopes, ignore_index=True)
        merged = s.merge(cells[['region', 'band', 'beta_nrs_within']],
                         on=['region', 'band'], how='inner')
        ok = merged[merged['slope'].notna() & merged['beta_nrs_within'].notna()].copy()
        ok['agrees'] = np.sign(ok['slope']) == np.sign(ok['beta_nrs_within'])
        cons = (ok.groupby(['region', 'band'])
                .agg(frac_sign_consistent=('agrees', 'mean'),
                     n_with_slope=('agrees', 'size')).reset_index())
        cells = cells.merge(cons, on=['region', 'band'], how='left')
        io.write_table(s, run_dir / 'subject_slopes.parquet',
                       params={'source': 'unpooled per-subject OLS'}, script=SCRIPT)

    io.write_table(cells, run_dir / 'band_cells.parquet',
                   params={'fdr_q': args.fdr_q,
                           'families': 'NRS_within across all fitted cells; the '
                                       'heterogeneity LRT separately; the '
                                       'permutation p separately'},
                   script=SCRIPT,
                   extra={'status': DISCLAIMER, 'inference_caveat': WALD_CAVEAT,
                          'band_caveat': band_caveat(args.band_set)})

    report(cells[cells['model'] == 'pain'], args.fdr_q)
    # The TABLES and the BH correction are the analysis; the figures are a
    # separate act. --no-figures exists so a run can be collected, corrected
    # and reported without committing to a plot design.
    if not args.no_figures:
        figures(run_dir, cells[cells['model'] == 'pain'], args)
        if cells['model'].isin(('med_decomposed', 'med_matched')).any():
            med_figure(run_dir, cells, args)
        if cells['model'].str.startswith('dx_').any():
            dx_figure(run_dir, cells, args)
    else:
        logger.info('--no-figures: tables and BH written, no plots drawn')
    if cells['model'].str.startswith('dx_').any():
        dx_report(cells, args.fdr_q)
    write_methods(run_dir, cells, args)
    io.log_analysis(f'band-power mixed-effects models, {len(cells)} region x band '
                    'cells, BH-corrected (EXPLORATORY)', run_dir)
    print(run_dir)


def report(cells, q):
    conv = int(cells['converged'].fillna(False).sum())
    logger.info('=' * 74)
    logger.info('BAND-POWER MIXED MODELS: %d cells, %d converged', len(cells), conv)
    sig = cells[cells['p_bh_reject'] == True]                    # noqa: E712
    logger.info('  BH-significant (Wald, q=%.2f): %d', q, len(sig))
    for r in sig.sort_values('p').itertuples():
        extra = (f"  p_perm {r.p_perm:.4g}" if 'p_perm' in cells.columns
                 and np.isfinite(getattr(r, 'p_perm', np.nan)) else '')
        logger.info('    %-18s %-11s beta %+.5f  z %+5.2f  p_bh %.4g  '
                    'sign-consistent %.2f%s', r.region, r.band, r.beta_nrs_within,
                    r.z, r.p_bh, getattr(r, 'frac_sign_consistent', np.nan), extra)
    if 'p_perm' in cells.columns and cells['p_perm'].notna().any():
        d = cells.dropna(subset=['p', 'p_perm'])
        n_anti = int((d['p_perm'] > d['p']).sum())
        logger.info('  permutation vs Wald: p_perm LARGER (Wald anticonservative) '
                    'in %d/%d cells; median log10 ratio %+.3f', n_anti, len(d),
                    float(np.median(np.log10(d['p_perm'] / d['p']))))
    het = cells[cells['p_lrt_bh_reject'] == True]                # noqa: E712
    logger.info('  heterogeneity LRT BH-significant: %d (subjects differ in their '
                'slope)', len(het))
    logger.info('  %s', WALD_CAVEAT)
    logger.info('=' * 74)


def figures(run_dir, cells, args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from ieeg_ehr.features import common

    bands = list(BAND_SETS[args.band_set])
    regions = [r for r in view_tables.roi_regions_for({'roi_scheme': args.roi_scheme})
               if r in set(cells['region'])]
    if not regions:
        raise SystemExit(
            f'--roi-scheme {args.roi_scheme!r} names none of the regions in this '
            f'run, which holds {sorted(set(cells["region"]))}. The collect stage '
            'derives its region list from the flag, so it must be given the SAME '
            '--roi-scheme the fit stage used.')
    beta = (cells.pivot_table(index='region', columns='band', values='beta_nrs_within')
            .reindex(index=regions, columns=bands))
    sig = (cells.assign(rej=cells['p_bh_reject'].fillna(False).astype(bool))
           .pivot_table(index='region', columns='band', values='rej')
           .reindex(index=regions, columns=bands).fillna(0).astype(bool))

    # --- 1. the map: 6 columns, so imshow over categorical cells is right here
    cap = float(np.nanmax(np.abs(beta.to_numpy(dtype=float))))
    cm = plt.get_cmap('RdBu_r').copy()
    cm.set_bad('0.85')
    fig, ax = plt.subplots(figsize=(7.4, 0.42 * len(regions) + 3.0))
    im = ax.imshow(beta.to_numpy(dtype=float), aspect='auto', cmap=cm,
                   vmin=-cap, vmax=cap, interpolation='nearest')
    common.draw_mask_outline(ax, sig.to_numpy())
    ax.set_xticks(range(len(bands)))
    ax.set_xticklabels([f'{b}\n{BAND_SETS[args.band_set][b][0]}-'
                        f'{BAND_SETS[args.band_set][b][1]} Hz' for b in bands],
                       fontsize=8)
    ax.set_yticks(range(len(regions)))
    ax.set_yticklabels(regions, fontsize=8)
    ax.set_title(f'Band power vs pain, mixed-effects beta\n'
                 f'{int(sig.to_numpy().sum())} of {len(cells)} cells BH-significant '
                 f'at q={args.fdr_q}', fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03,
                 label='d log10(band power) per pain point')
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.text(0.01, 0.005,
             'Outlines mark BH-significant cells across all fitted cells at '
             f'q={args.fdr_q}. {WALD_CAVEAT} {band_caveat(args.band_set)}\n'
             f'{DISCLAIMER}',
             fontsize=6.5, va='bottom', ha='left', color='0.35', wrap=True)
    p1 = run_dir / 'fig_band_map.png'
    fig.savefig(p1, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # --- 2. the forest plot: 120 estimates with intervals, which a heatmap hides
    fig, axs = plt.subplots(1, len(bands), figsize=(3.0 * len(bands), 0.30 * len(regions) + 3.0),
                            sharey=True, squeeze=False)
    y = np.arange(len(regions))
    xmax = float(np.nanmax(np.abs(np.concatenate(
        [(cells['beta_nrs_within'] + 1.96 * cells['se']).to_numpy(),
         (cells['beta_nrs_within'] - 1.96 * cells['se']).to_numpy()])))) * 1.05
    for j, band in enumerate(bands):
        ax = axs[0][j]
        d = cells[cells['band'] == band].set_index('region').reindex(regions)
        b = d['beta_nrs_within'].to_numpy(dtype=float)
        se = d['se'].to_numpy(dtype=float)
        rej = d['p_bh_reject'].fillna(False).to_numpy(dtype=bool)
        ax.errorbar(b[~rej], y[~rej], xerr=1.96 * se[~rej], fmt='o', ms=4, lw=1,
                    capsize=2, color='0.6')
        ax.errorbar(b[rej], y[rej], xerr=1.96 * se[rej], fmt='o', ms=5.5, lw=1.4,
                    capsize=2, color='#b03a2e')
        ax.axvline(0, color='0.4', lw=0.9, ls='--')
        ax.set_xlim(-xmax, xmax)
        ax.set_title(f'{band}\n{BAND_SETS[args.band_set][band][0]}-'
                     f'{BAND_SETS[args.band_set][band][1]} Hz', fontsize=9)
        ax.tick_params(labelsize=7)
        if j == 0:
            ax.set_yticks(y)
            ax.set_yticklabels(regions, fontsize=8)
            ax.set_ylim(len(regions) - 0.5, -0.5)
    fig.suptitle('Band power vs pain: beta with 95% CI, per region\n'
                 'red = BH-significant', fontsize=12)
    fig.tight_layout(rect=(0, 0.09, 1, 0.92))
    fig.text(0.01, 0.005,
             'Intervals are Wald 95% CIs from the mixed model, so they inherit the '
             f'assumption in the caveat. {WALD_CAVEAT}\n{DISCLAIMER}',
             fontsize=6.5, va='bottom', ha='left', color='0.35', wrap=True)
    p2 = run_dir / 'fig_band_forest.png'
    fig.savefig(p2, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info('wrote %s and %s', p1.name, p2.name)


def dx_report(cells, q):
    """Log the interaction cells that survive their own BH family."""
    d = cells[cells['model'].str.startswith('dx_')]
    if not len(d):
        return
    logger.info('=' * 74)
    n_fit = int(d['dx_ix_beta'].notna().sum())
    logger.info('DIAGNOSIS MODERATION: %d of %d cells fitted (%d refused for a '
                'stratum under %d subjects)', n_fit, len(d), len(d) - n_fit,
                DX_MIN_SUBJECTS_PER_ARM)
    sig = d[d.get('dx_ix_p_bh_reject', pd.Series(dtype=object)) == True]  # noqa: E712
    logger.info('  interaction BH-significant at q=%.2f: %d', q, len(sig))
    for r in sig.sort_values('dx_ix_p').itertuples():
        logger.info('    %-18s %-11s  non-MDD %+.5f  MDD %+.5f  diff %+.5f  '
                    'p_bh %.4g', r.region, r.band, r.slope_ref, r.slope_mod,
                    r.dx_ix_beta, r.dx_ix_p_bh)
    if not len(sig):
        logger.info('    none. With %d vs %d subjects this is the expected '
                    'outcome for anything but a large slope difference -- read '
                    'it as "not resolved at this n", not as "no difference".',
                    int(d['n_subjects_case'].max() or 0),
                    int(d['n_subjects_control'].max() or 0))
    logger.info('  %s', DX_CAVEAT)
    logger.info('=' * 74)


def dx_figure(run_dir, cells, args):
    """Three panels: the pain slope in each stratum, and the DIFFERENCE.

    THE TWO STRATUM PANELS CARRY NO SIGNIFICANCE OUTLINES, and that is the whole
    design of this figure rather than an omission. Outlining each arm separately
    would invite exactly the inference the interaction model exists to prevent:
    "significant in cases, not in controls" is not evidence that the two differ,
    and with unequal arms (here 17 vs 34 subjects) the smaller one carries wider
    standard errors everywhere and will look weaker from power alone. Only the
    third panel -- the interaction, with its own BH family -- licenses a claim
    about a difference, so only the third panel is outlined.

    The two stratum panels also SHARE a colour scale. They are the same quantity
    in two groups; giving each its own scale would manufacture a visual contrast
    out of a scale difference.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from ieeg_ehr.features import common

    d = cells[cells['model'].str.startswith('dx_')]
    if not len(d):
        return
    bands = list(BAND_SETS[args.band_set])
    regions = [r for r in view_tables.roi_regions_for({'roi_scheme': args.roi_scheme})
               if r in set(d['region'])]

    def grid(value):
        return (d.pivot_table(index='region', columns='band', values=value,
                              dropna=False)
                .reindex(index=regions, columns=bands).to_numpy(dtype=float))

    ctrl, case, ix = grid('slope_ref'), grid('slope_mod'), grid('dx_ix_beta')
    rej = (d.assign(r=d.get('dx_ix_p_bh_reject',
                            pd.Series(False, index=d.index)).fillna(False).astype(bool))
           .pivot_table(index='region', columns='band', values='r', dropna=False)
           .reindex(index=regions, columns=bands).fillna(0).to_numpy(dtype=bool))

    n_case = int(np.nanmax(d['n_subjects_case'].to_numpy(dtype=float)))
    n_ctrl = int(np.nanmax(d['n_subjects_control'].to_numpy(dtype=float)))

    # One scale for the two slope panels, a separate one for the difference.
    slope_cap = float(np.nanmax(np.abs(np.concatenate([ctrl.ravel(), case.ravel()]))))
    ix_cap = float(np.nanmax(np.abs(ix)))
    cm = plt.get_cmap('RdBu_r').copy()
    cm.set_bad('0.85')

    panels = [
        # 'non-MDD', never 'control'. The domain scheme has a PROCESSING
        # DOMAIN called Control (Auditory + Occipital, the negative-control
        # circuit), so 'control' for the undiagnosed group put two unrelated
        # meanings of the word on neighbouring figures.
        (ctrl, slope_cap, None, f'non-{args.dx.upper()}\n{n_ctrl} subjects'),
        (case, slope_cap, None, f'{args.dx.upper()}\n{n_case} subjects'),
        (ix, ix_cap, rej, f'DIFFERENCE ({args.dx.upper()} - '
                          f'non-{args.dx.upper()})\n'
                          f'outlined: BH-significant at q={args.fdr_q}'),
    ]
    fig, axs = plt.subplots(1, 3, figsize=(16.5, 0.42 * len(regions) + 3.4),
                            squeeze=False)
    for ax, (arr, cap, outline, title) in zip(axs[0], panels):
        im = ax.imshow(arr, aspect='auto', cmap=cm, vmin=-cap, vmax=cap,
                       interpolation='nearest')
        if outline is not None:
            common.draw_mask_outline(ax, outline)
        ax.set_xticks(range(len(bands)))
        ax.set_xticklabels([f'{b}\n{BAND_SETS[args.band_set][b][0]}-'
                            f'{BAND_SETS[args.band_set][b][1]} Hz' for b in bands],
                           fontsize=7.5)
        ax.set_yticks(range(len(regions)))
        ax.set_yticklabels(regions, fontsize=7.5)
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03,
                     label='d log10(power) per pain point')

    n_fit = int(np.isfinite(ix).sum())
    fig.suptitle(
        f'Pain encoding by {args.dx.upper()} status: both groups from ONE '
        f'interaction fit\n{n_fit} of {ix.size} cells fitted; '
        f'{int(rej.sum())} BH-significant differences at q={args.fdr_q}',
        fontsize=12.5)
    fig.tight_layout(rect=(0, 0.13, 1, 0.93))
    fig.text(0.01, 0.005,
             'The two left panels are SIMPLE SLOPES from the single pooled fit '
             'log10_power ~ NRS_within * dx_state + NRS_submean, with the '
             'random effects of the pain-only model unchanged. They are '
             'deliberately NOT outlined for significance: comparing which arm '
             'reaches significance is the difference-of-significance fallacy, '
             f'and the {args.dx.upper()} arm ({n_case} subjects) has wider SEs '
             f'than the non-{args.dx.upper()} arm ({n_ctrl}) everywhere from '
             'power alone. '
             'Only the right panel tests a difference. Grey = not fitted '
             f'(a stratum under {DX_MIN_SUBJECTS_PER_ARM} subjects).\n'
             f'{DX_CAVEAT}\n{WALD_CAVEAT} {DISCLAIMER}',
             fontsize=6.3, va='bottom', ha='left', color='0.35', wrap=True)
    out = run_dir / 'fig_dx_effects.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info('wrote %s', out.name)
    return out


def med_figure(run_dir, cells, args):
    """One heat panel per medication term: region x band, significance outlined.

    THREE TERMS, THREE DIFFERENT QUESTIONS, which is why they are not one panel:

      med_within   within a patient, is power different when recently dosed?
      pain x med   does being dosed CHANGE the pain slope?
      matched      at the SAME reported score, is power different when dosed?

    The pain slope from the medication-free fit is drawn beside them, on its own
    colour scale, because the medication terms are only interpretable against the
    effect they are supposed to be confounding.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from ieeg_ehr.features import common

    bands = [b for b in BAND_SETS[args.band_set] if b in set(cells['band'])]
    regions = [r for r in view_tables.roi_regions_for({'roi_scheme': args.roi_scheme})
               if r in set(cells['region'])]

    def grid(sub, value, reject):
        v = (sub.pivot_table(index='region', columns='band', values=value)
             .reindex(index=regions, columns=bands))
        if reject in sub.columns:
            r = (sub.assign(_r=sub[reject].fillna(False).astype(bool))
                 .pivot_table(index='region', columns='band', values='_r')
                 .reindex(index=regions, columns=bands).fillna(0).astype(bool))
        else:
            r = pd.DataFrame(False, index=regions, columns=bands)
        return v, r

    dec = cells[cells['model'] == 'med_decomposed']
    mat = cells[cells['model'] == 'med_matched']
    pain = cells[cells['model'] == 'pain']

    panels = []
    if len(pain):
        panels.append((*grid(pain, 'beta_nrs_within', 'p_bh_reject'),
                       'PAIN SLOPE (no medication term)\nd log10 power per pain point'))
    if len(dec):
        panels.append((*grid(dec, 'medw_beta', 'medw_p_bh_reject'),
                       'MED_WITHIN\nd log10 power when recently dosed'))
        panels.append((*grid(dec, 'med_ix_beta', 'med_ix_p_bh_reject'),
                       'PAIN x MED interaction\nchange in the pain slope when dosed'))
    if len(mat):
        panels.append((*grid(mat, 'beta_med', 'p_bh_reject'),
                       'MATCHED at the same NRS\nd log10 power when dosed'))
        panels.append((*grid(mat, 'nonparam_diff', None),
                       'MATCHED, NON-PARAMETRIC\nmean within-subject level difference'))

    fig, axes = plt.subplots(1, len(panels),
                             figsize=(3.3 * len(panels) + 1.2,
                                      0.40 * len(regions) + 3.6), squeeze=False)
    for i, (val, rej, title) in enumerate(panels):
        ax = axes[0][i]
        arr = val.to_numpy(dtype=float)
        cap = float(np.nanmax(np.abs(arr))) or 1.0
        cm = plt.get_cmap('RdBu_r').copy()
        cm.set_bad('0.85')
        im = ax.imshow(arr, aspect='auto', cmap=cm, vmin=-cap, vmax=cap,
                       interpolation='nearest')
        common.draw_mask_outline(ax, rej.to_numpy())
        ax.set_xticks(range(len(bands)))
        ax.set_xticklabels(bands, fontsize=7, rotation=45, ha='right')
        ax.set_yticks(range(len(regions)))
        ax.set_yticklabels(regions if i == 0 else [], fontsize=7.5)
        ax.set_title(title, fontsize=8.5)
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03).ax.tick_params(labelsize=6.5)

    fig.suptitle(f'Band power vs pain WITH {args.drug_set} in the model '
                 f'({args.med_window_hours:g} h before the score)', fontsize=12.5)
    fig.tight_layout(rect=(0, 0.13, 1, 0.93))
    fig.text(0.01, 0.005,
             f'EVERY PANEL HAS ITS OWN COLOUR SCALE -- they are different '
             f'quantities, and a shared one would imply a comparison that is not '
             f'available. Outlines are BH at q={args.fdr_q:g} WITHIN that term\'s '
             f'own family, never pooled across terms. {MED_CAVEAT} '
             f'{WALD_CAVEAT}\n{DISCLAIMER}',
             fontsize=6.4, va='bottom', ha='left', color='0.35', wrap=True)
    out = run_dir / 'fig_med_effects.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info('wrote %s', out.name)
    return out


def write_methods(run_dir, cells, args):
    bands = BAND_SETS[args.band_set]
    if 'model' in cells.columns:
        med = cells[cells['model'].isin(('med_decomposed', 'med_matched'))]
        dx = cells[cells['model'].str.startswith('dx_')]
        cells = cells[cells['model'] == 'pain']
    else:
        med = dx = cells.iloc[0:0]
    sig = cells[cells['p_bh_reject'] == True]                    # noqa: E712
    lines = [f"""# Band-power mixed-effects models

{DISCLAIMER}

## What is fitted

One row = ONE CHANNEL x ONE 5-minute pre-report epoch. The outcome is that
channel's log10 BAND power for that epoch: the native 0.5 Hz bins whose geometric
centre falls in the band, line-noise bins excluded, combined LINEAR-THEN-LOG via
`views.axes.aggregate_bands` -- log10(mean(10**x)), because raw log power does not
average arithmetically. A coefficient is d log10(band power) per pain point.

    log10_power ~ NRS_within + NRS_submean
                  + (NRS_within || subject) + (1 | subject:channel)

`NRS_within` is the subject-mean-centred pain score and is the effect of
interest. `NRS_submean` is a nuisance term that keeps the between-subject
contrast out of the within-subject slope. The by-subject random slope lets
patients differ in how strongly power tracks their pain; the channel random
intercept absorbs each contact's own amplitude, which is why no normalization is
applied. REML, statsmodels.

Band set `{args.band_set}`: {', '.join(f'{k} {v[0]}-{v[1]} Hz' for k, v in bands.items())}.

{band_caveat(args.band_set)}

## Why a refit rather than averaging the grid's betas

Averaging the per-frequency betas within a band is a fine point estimate -- it
agreed with a refit to under 1% on 6 test cells -- but it has no usable standard
error: the SE of a mean of 463 estimates needs their cross-bin covariance, which
independent per-cell fits never produce, and the mean of the per-cell SEs ran 5%
to 67% away from the refit's. The aggregation rule also matters on its own:
linear-then-log gave betas 12-31% larger in magnitude than the averaged betas on
those cells, and moved one cell from z=1.74 to z=2.07.

## Significance

`p` is the parametric Wald z on `NRS_within`; `p_bh` is Benjamini-Hochberg over
all fitted cells at q={args.fdr_q}. {WALD_CAVEAT}

The heterogeneity LRT (`p_lrt_mixture`, referenced to the 50:50 mixture of a
point mass at 0 and chi2(1) because the variance sits at the parameter-space
boundary) gets its OWN BH family: "do subjects differ in their slope" is a
separate question from "is the mean slope non-zero", and correcting each for the
other's tests would be wrong.

`frac_sign_consistent` is the fraction of subjects whose UNPOOLED OLS slope shares
the sign of the group fixed effect -- not BLUPs, which partial pooling drags
toward the group.

## Known limitations

- {CONFOUND_CAVEAT}
- **No covariates**: no time of day, no time since admission, no temporal term.
  Epochs within a subject are treated as exchangeable, and BOTH the Wald and the
  permutation p depend on that. A permutation test replaces the reference
  distribution, not the design.
- Channels are kept as channels (not averaged into the ROI), so a region's
  estimate is precision-weighted toward subjects with more contacts there.

## Results: {len(sig)} of {len(cells)} cells BH-significant

"""]
    if len(sig):
        lines.append('| region | band | beta | SE | z | p_bh | sign consistency |\n'
                     '|---|---|---|---|---|---|---|\n')
        for r in sig.sort_values('p').itertuples():
            lines.append('| {} | {} | {:+.5f} | {:.5f} | {:+.2f} | {:.4g} | {:.2f} |\n'
                         .format(r.region, r.band, r.beta_nrs_within, r.se, r.z,
                                 r.p_bh, getattr(r, 'frac_sign_consistent', np.nan)))

    if len(med):
        lines.append(f"""
## Medication in the model: `{args.med_model}`, drug set `{args.drug_set}`

An epoch counts as medicated when any administration in that drug set falls in
the {args.med_window_hours:g} h before its pain score. The window is anchored on
the ASSESSMENT, not the epoch start, because the assessment is the clinical event
a dose is timed against.

**{MED_CAVEAT}**

Three terms, three questions, each with its OWN BH family -- pooling them would
correct each hypothesis for the others' tests:

| term | question |
|---|---|
| `medw_beta` | within a patient, is power different when recently dosed? |
| `med_ix_beta` | does being dosed CHANGE the pain slope? |
| `beta_med` (matched) | at the SAME reported score, is power different when dosed? |
| `nonparam_diff` | the same contrast without a model: per subject, the mean difference between dosed and undosed epochs AT EQUAL NRS, averaged unweighted over the levels present in both |

`med_submean` (`medb_beta`) is the BETWEEN-patient part and is a nuisance term
here. It is not evidence about medication: subject mean pain correlates +0.685
with subject proportion medicated, so most of what it carries is a pain
difference wearing a medication label.

""")
        for model, label in (('med_decomposed', 'decomposed'),
                             ('med_matched', 'matched')):
            sub = med[med['model'] == model]
            if not len(sub):
                continue
            rej = sub[sub['p_bh_reject'] == True]                # noqa: E712
            lines.append(f'\n### {label}: {len(rej)} of {len(sub)} cells '
                         f'BH-significant on its primary term\n')

    if len(dx):
        spec = dx_state.CONDITIONS[args.dx]
        n_case = int(np.nanmax(dx['n_subjects_case'].to_numpy(dtype=float)))
        n_ctrl = int(np.nanmax(dx['n_subjects_control'].to_numpy(dtype=float)))
        n_fit = int(dx['dx_ix_beta'].notna().sum())
        rej = dx[dx.get('dx_ix_p_bh_reject',
                        pd.Series(dtype=object)) == True]        # noqa: E712
        window = (f'the {args.dx_window_days} days before `session_start`'
                  if args.dx_window_days else 'the record at any time up to '
                                              '`session_start`')
        lines.append(f"""
## {spec['label']} as a moderator of the pain slope

A subject is a CASE when a diagnosis code for {spec['label']} appears in
{window}. Codes dated DURING the admission never count -- letting them in would
let the encounter that produced the iEEG define its own predictor. Source set:
`{args.dx_sources}`.

{spec['description']}
ICD-10 prefixes `{list(spec['icd10'])}`, ICD-9 prefixes `{list(spec['icd9'])}`,
matched on the code with its dot stripped and NEVER on the free-text
description ('depression' also appears in 'ST segment depression' and
'respiratory depression').

**{n_case} cases, {n_ctrl} controls.**

    log10_power ~ NRS_within * dx_state + NRS_submean
                  + (NRS_within || subject) + (1 | subject:channel)

THE RANDOM EFFECTS ARE UNCHANGED from the pain-only model above. That is
load-bearing rather than incidental: `dx_state` is constant within a subject, so
the by-subject random intercept is what its standard error is judged against,
and dropping it would turn a {n_case + n_ctrl}-subject contrast into a
several-thousand-channel pseudo-replicated one.

`dx_state` is a subject-level 0/1 constant, so unlike `med_state` it gets NO
within/between decomposition -- a subject-constant predictor has no
within-subject part, and `dx_within` would be a column of exact zeros.

| term | question |
|---|---|
| `slope_ref` | the pain slope in the non-MDD group (this is `NRS_within`) |
| `slope_mod` | the pain slope in the MDD group (`NRS_within + NRS_within:dx_state`, SE from the fitted covariance, NOT the sum of variances) |
| `dx_ix_beta` | **the estimand**: how much the pain slope DIFFERS in cases |
| `dx_beta` | a nuisance between-subject difference in mean power; not a result, and deliberately given no BH family |

`dx_ix` gets its OWN BH family over the fitted cells. It is not corrected
against the pain slope's family: whether the strata differ somewhere is a
different question from whether the slope is non-zero there.

### Why one interaction fit and not two stratified maps

Fitting each arm separately gives two maps and no test that they differ. The
arms are unequal ({n_case} vs {n_ctrl}), so the smaller one carries wider
standard errors everywhere and looks weaker from power alone; reading that as a
group difference is the difference-of-significance fallacy. A stratified fit
also estimates its own variance components per arm, which costs every region
whose smaller arm falls under {DX_MIN_SUBJECTS_PER_ARM} subjects. Here
{n_fit} of {len(dx)} cells were fitted and {len(dx) - n_fit} were refused on
that rule; the refused cells are ROWS carrying the reason, and they are still
fitted by the pain-only model above.

**{DX_CAVEAT}**

### Result: {len(rej)} of {n_fit} fitted cells show a BH-significant difference
""")
        if len(rej):
            lines.append('\n| region | band | non-MDD slope | MDD slope | '
                         'difference | p_bh |\n|---|---|---|---|---|---|\n')
            for r in rej.sort_values('dx_ix_p').itertuples():
                lines.append(f'| {r.region} | {r.band} | {r.slope_ref:+.5f} | '
                             f'{r.slope_mod:+.5f} | {r.dx_ix_beta:+.5f} | '
                             f'{r.dx_ix_p_bh:.4g} |\n')
        else:
            lines.append(
                f'\nNone. With {n_case} vs {n_ctrl} subjects this is the '
                'expected outcome for anything short of a large slope '
                'difference, so it reads as NOT RESOLVED AT THIS N rather than '
                'as evidence of no difference. The effect sizes and their '
                'intervals are in `band_cells.parquet`; a null interaction with '
                'a wide CI is not a null result.\n')

    (run_dir / 'METHODS.md').write_text(''.join(lines))


# ============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', choices=['fit', 'perm', 'collect'], default='fit')
    ap.add_argument('--run-dir', default=None)
    ap.add_argument('--region-index', type=int, default=None,
                    help='Fit one region only (array mode). Default: all of them, '
                         'which takes minutes at 120 cells.')
    ap.add_argument('--cell-index', type=int, default=None,
                    help='Which cell `--stage perm` shuffles.')
    ap.add_argument('--band-set', choices=list(BAND_SETS),
                    default='paper_bands_6_hg200')
    ap.add_argument('--med-model', choices=['none', 'decomposed', 'matched', 'both'],
                    default='none',
                    help="Add medication to the model. 'decomposed' splits "
                         'med_state into within- and between-patient parts and '
                         'adds its interaction with pain -- the split is needed '
                         'because subject mean pain correlates +0.685 with subject '
                         "proportion medicated. 'matched' conditions on the "
                         'reported score as a FACTOR and asks whether power '
                         'differs at the SAME pain level, which is the cleaner '
                         "answer to confounding by indication. 'both' fits both "
                         'and keeps the pain-only fit beside them.')
    ap.add_argument('--drug-set', choices=list(med_state.DRUG_SETS),
                    default='opioids',
                    help="Which administrations count. 'opioids' is the "
                         'pharmacologically cleanest set and costs about a third '
                         "of the medicated epochs; 'non_opioid_analgesics' is its "
                         'complement and the nearest thing to a negative control.')
    ap.add_argument('--med-window-hours', type=float,
                    default=med_state.DEFAULT_WINDOW_HOURS,
                    help='Hours before the ASSESSMENT that count as recently '
                         'dosed (default 2.0). Anchored on the score, not the '
                         'epoch start.')
    ap.add_argument('--domain-scheme', default=None,
                    choices=sorted(roi_schemes.DOMAIN_SCHEMES),
                    help='Make the PROCESSING DOMAIN the unit instead of the '
                         'ROI: one fit per domain x band, the model itself '
                         'unchanged. --roi-scheme must be the domain scheme\'s '
                         'base (it is set automatically if left alone), because '
                         'the insula split runs at the base level before the '
                         'ROIs are folded up.')
    ap.add_argument('--no-figures', action='store_true',
                    help='Collect, BH-correct and write the tables, but draw '
                         'nothing. For when the plot design is still open.')
    ap.add_argument('--dx-model', choices=['none', 'interaction'], default='none',
                    help='Add a SUBJECT-LEVEL diagnosis as a moderator of the '
                         'pain slope: NRS_within * dx_state. The random effects '
                         'are unchanged. The estimand is the interaction -- does '
                         'the pain slope DIFFER between strata -- which fitting '
                         'the two arms separately cannot test. Both simple '
                         'slopes are still reported, from this one fit.')
    ap.add_argument('--dx', choices=list(dx_state.CONDITIONS), default='mdd',
                    help="Which condition. 'mdd' is F32/F33 and ICD-9 296.2x/"
                         "296.3x, excluding bipolar and dysthymia; "
                         "'depression_broad' adds dysthymia and ICD-9 311.")
    ap.add_argument('--dx-window-days', type=int,
                    default=dx_state.DEFAULT_WINDOW_DAYS,
                    help='Days before SESSION_START in which a code counts '
                         '(default 90). Codes dated during the admission never '
                         'count. Pass 0 for EVER, which is the pre-specified '
                         'sensitivity analysis: 90 days labels 17 of 51 subjects '
                         'and `ever` labels 29, and the 12 in between sit in the '
                         'control arm diluting the contrast.')
    ap.add_argument('--dx-sources', choices=list(dx_state.SOURCE_SETS),
                    default='any',
                    help="Which EHR sources count. 'clinical' keeps only "
                         'Encounter/Problem/Admit codes and drops billing and '
                         'HL7-historical entries -- worth running, because most '
                         'cases are labelled by a billing code alone, but it '
                         'costs most of the arm.')
    ap.add_argument('--exclude-regions', nargs='*', default=[],
                    help='Regions to leave out of the run ENTIRELY -- not fitted, '
                         'not in the BH family, not on the figures. Use for a '
                         'region whose data is under review, and say why: the '
                         'reason goes into provenance. Excluding a region because '
                         'it was not significant is a different and illegitimate '
                         'act, which is why this takes names rather than a '
                         'threshold.')
    ap.add_argument('--exclude-reason', default=None,
                    help='Recorded verbatim in provenance beside --exclude-regions.')
    ap.add_argument('--cells', nargs='*', default=[],
                    help='Fit ONLY these cells, each REGION:BAND (e.g. M1:beta '
                         'IFG/vlPFC:beta). They alone form every BH family. For '
                         'a follow-up on cells selected from an earlier run -- '
                         'say how they were chosen in --cell-selection, because '
                         'the p-values are conditional on that choice.')
    ap.add_argument('--cell-selection', default=None,
                    help='Recorded verbatim in provenance beside --cells.')
    ap.add_argument('--insula-threshold', type=float, default=None,
                    help='Pin the anterior/posterior insula cut (MNI y, mm) '
                         'instead of re-deriving this cohort\'s median. Only '
                         'used by a scheme with coordinate regions '
                         '(roi_v2_ofc_ins, roi_v2_ins). Pass an earlier run\'s '
                         'threshold to reproduce its split exactly.')
    # Default applied AFTER the --domain-scheme check, not here: a domain
    # scheme dictates its own base, and a string default would be
    # indistinguishable from the user asking for roi_v2_ofc explicitly.
    ap.add_argument('--roi-scheme', default=None,
                    help="Region set. Default 'roi_v2_ofc' = roi_v2 with mOFC and "
                         'lOFC fused into one OFC (20 regions). Pass roi_v2 to keep '
                         'them split, or a path to a JSON scheme. Ignored (and '
                         'forced to the base) when --domain-scheme is given.')
    ap.add_argument('--view-dir', default=None)
    ap.add_argument('--reference-run', default=str(reference_run.CONTPAIN_HEATMAP))
    ap.add_argument('--mask-label', default=None)
    ap.add_argument('--cohort', choices=['reference', 'eligible-discovery'],
                    default='reference')
    ap.add_argument('--allow-cohort-drift', action='store_true')
    ap.add_argument('--notch-half-width-hz', type=float, default=None)
    ap.add_argument('--n-perm', type=int, default=1000)
    ap.add_argument('--n-jobs', type=int,
                    default=int(__import__('os').environ.get('SLURM_CPUS_PER_TASK', 1)))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--fdr-q', type=float, default=FDR_Q)
    ap.add_argument('--question', default=QUESTION)
    ap.add_argument('--view-scheme', default=None,
                    help='Level-4 folder. Default: derived from --band-set, so '
                         'the path cannot claim band edges the run did not use.')
    ap.add_argument('--run-name', default=RUN_NAME)
    args = ap.parse_args()

    # The domain scheme dictates its own base ROI scheme -- the members are
    # ROIs OF that scheme, so resolving the cohort against anything else would
    # map channels to units the membership does not mention.
    if args.domain_scheme:
        base = roi_schemes.domain_scheme(args.domain_scheme)['base']
        if args.roi_scheme not in (None, base):
            raise SystemExit(
                f'--domain-scheme {args.domain_scheme!r} is built from '
                f'{base!r} ROIs, so --roi-scheme must be {base!r} (or left '
                f'unset); got {args.roi_scheme!r}.')
        args.roi_scheme = base
    args.roi_scheme = args.roi_scheme or 'roi_v2_ofc'

    if args.view_scheme is None:
        # The DOMAIN scheme names the folder when there is one: two domain
        # schemes can share a base (v2 and v3 differ only by the insula), so
        # naming the folder after the base alone would let them collide.
        args.view_scheme = view_scheme_for(
            args.band_set, args.domain_scheme or args.roi_scheme,
            args.drug_set if args.med_model != 'none' else None)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    io.warn_if_dirty()

    if args.stage == 'fit':
        stage_fit(args)
    elif args.stage == 'perm':
        if not args.run_dir or args.cell_index is None:
            raise SystemExit('--stage perm needs --run-dir and --cell-index')
        stage_perm(args)
    else:
        if not args.run_dir:
            raise SystemExit('--stage collect needs --run-dir')
        stage_collect(args)


if __name__ == '__main__':
    main()
