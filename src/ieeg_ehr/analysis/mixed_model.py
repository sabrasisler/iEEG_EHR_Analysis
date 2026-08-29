"""
Per-cell linear mixed-effects model of epoch-mean log10 power on pain score.

One model per (ROI region, frequency bin) cell. A row is ONE CHANNEL x ONE
5-minute pre-report epoch, so the channel structure the two-stage pipeline
averages away before fitting is instead absorbed by a random intercept:

    log10_power ~ NRS_within + NRS_submean + (NRS_within || subject)
                                           + (1 | subject:channel)

WHAT THIS BUYS OVER `pain_coef` + `cluster_permutation`
-------------------------------------------------------
The two-stage design (within-subject OLS -> equal-weighted group mean -> sign-flip
cluster permutation) is deliberately conservative and stays the reference. It
cannot do three things this can:

  - WEIGHT BY PRECISION. A subject with 94 epochs and 30 contacts in a region and
    one with 11 epochs and 2 contacts currently count the same.
  - TEST HETEROGENEITY. `subj_slope` is a variance, so "do subjects differ in
    their pain effect" becomes a likelihood-ratio test rather than an eyeball on
    a violin. A cell where the LRT is significant and the fixed effect is not is
    a real result: subjects respond, but not in a consistent direction.
  - KEEP CHANNELS AS CHANNELS. The reference collapses an ROI's contacts by a
    linear-then-log mean BEFORE the subject enters the model, so the loudest
    contact dominates the region and per-contact variability is invisible.

WHY NO NORMALIZATION
--------------------
No z-scoring, no relative power, no dB baseline. Channel amplitude differences
are MULTIPLICATIVE, so in log space they are additive and the channel random
intercept absorbs them exactly. The resulting coefficient, d(log10 power)/d(NRS),
is a proportional change per pain point and is already comparable across regions
and frequencies. This is the same argument `pain_coef` makes for why per-subject
amplifier gain cancels out of a slope, applied one level further down.

WHY NRS IS SPLIT
----------------
`NRS_within` (subject-mean-centred) is the effect of interest and is the quantity
directly comparable to the existing two-stage map. `NRS_submean` is a nuisance
term whose only job is to keep the BETWEEN-subject contrast -- do subjects who
hurt more on average have different average power -- from leaking into the
within-subject slope. Centring is done over the ROWS ACTUALLY ENTERING THE CELL,
which is what makes `NRS_within` exactly orthogonal to the subject dummy space in
the fitted design. Doing it globally would leave a residual between-subject
component in `NRS_within` wherever a subject lost an epoch in one region because
all of its channels were masked out there.

NRS stays on its raw 0-10 scale, so a coefficient reads as "per pain point".

STATSMODELS SPECIFICS, VERIFIED AGAINST 0.14.6
----------------------------------------------
`||` in lme4 means an uncorrelated intercept/slope. statsmodels variance
components are independent BY CONSTRUCTION, so expressing the random effects as
`vc_formula` entries with `re_formula='0'` is the faithful translation -- there
is no correlation parameter to suppress.

Confirmed by probing the installed package rather than assumed:

  - `vc_formula` is evaluated PER GROUP, so `'0 + C(channel_uid)'` produces only
    that subject's channel columns (a 3-contact subject gets a 3-column VC
    design, not a cohort-wide one). `channel_uid` is still made globally unique,
    because relying on that behaviour silently would be a trap.
  - `res.vcomp` is an array aligned with `res.model.exog_vc.names`, which is NOT
    the order of the dict passed in -- it is sorted. Always map BY NAME.
  - `res.random_effects[group]` is a Series indexed 'subj_int[Intercept]',
    'subj_slope[NRS_within]', 'channel[C(channel_uid)[<uid>]]'.
  - `res.scale` is the residual variance.
  - statsmodels emits "The MLE may be on the boundary of the parameter space"
    very freely -- it fired on a well-identified fit and on a true null alike --
    so it is RECORDED but is not the boundary criterion. See `boundary_report`.

A NOTE ON WHAT subj_int AND channel SHARE
-----------------------------------------
The mean of a subject's channel effects IS a subject-level intercept, so
`subj_int` and `channel` are only weakly separated when a subject has few
contacts in a region. Both are nuisance terms here and the fixed effect is
unaffected, but do not read `var_subj_int` as "between-subject baseline power" on
its own.

NAMING. `beta` is the frequency band and `slope` is the 1/f aperiodic slope, as
everywhere else in this codebase. The fixed effect here is `beta_nrs_within`,
spelled out, and the per-subject deviations are BLUPs.
"""

import logging
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from ieeg_ehr import config

logger = logging.getLogger(__name__)

FORMULA = 'log10_power ~ NRS_within + NRS_submean'

# The full model. Keys become the names in res.model.exog_vc.names.
VC_FULL = {
    'subj_int': '1',
    'subj_slope': '0 + NRS_within',
    'channel': '0 + C(channel_uid)',
}
# The heterogeneity LRT's reduced model: subj_slope removed, everything else and
# ALL FIXED EFFECTS identical. That identity is what makes a REML comparison
# valid, and `lrt` asserts it rather than trusting it.
VC_REDUCED = {k: v for k, v in VC_FULL.items() if k != 'subj_slope'}
# Optional 4th component for the pilot's "is ROI v2 too coarse" question.
VC_CHANNEL_SLOPE = dict(VC_FULL, channel_slope='0 + C(channel_uid):NRS_within')

# A variance component counts as AT THE BOUNDARY when the variance it puts into
# the linear predictor is this small a fraction of the residual variance. There
# is no principled value -- lme4's `isSingular` uses a comparable ad-hoc
# tolerance -- so the raw ratios are written out alongside the flag and the
# threshold can be revisited from the pilot's distribution without a refit.
BOUNDARY_TOL = 1e-3

MIN_SUBJECTS = 8          # inherited from the reference analysis
MIN_ROWS = 20             # below this a 4-variance-component model is not a model


class CellFitError(RuntimeError):
    """A cell could not be fitted. Carries the reason so it becomes a ROW."""


# ============================================================================
# BUILDING A CELL'S FRAME
# ============================================================================

def build_cell_frame(cell_rows, *, region=None, freq_bin_index=None):
    """Tidy one cell's rows into the model frame.

    `cell_rows` is the per-channel view's long table already subset to one
    (roi, freq_bin_index): columns subject_id, channel (the bipolar pair name),
    epoch_id, pain_score, value.

    Returns columns: subject, channel_uid, epoch_id, NRS, log10_power.

    UPCAST ON THE WAY IN. The cache is float32 and the view inherits it; every
    reduction downstream (the variance components are reductions) accumulates in
    config.CACHE_ACCUMULATE_DTYPE. numpy will not do this for you.
    """
    df = pd.DataFrame({
        'subject': cell_rows['subject_id'].to_numpy(),
        # Globally unique so a channel name reused across subjects can never be
        # pooled. statsmodels evaluates the VC per group so this is belt and
        # braces -- which is the point.
        'channel_uid': (cell_rows['subject_id'].astype(str) + '|'
                        + cell_rows['channel'].astype(str)).to_numpy(),
        'epoch_id': cell_rows['epoch_id'].to_numpy(),
        'NRS': cell_rows['pain_score'].to_numpy(dtype=config.CACHE_ACCUMULATE_DTYPE),
        'log10_power': cell_rows['value'].to_numpy(dtype=config.CACHE_ACCUMULATE_DTYPE),
    })
    df = df[np.isfinite(df['log10_power']) & np.isfinite(df['NRS'])]
    if region is not None:
        df.attrs['region'] = region
    if freq_bin_index is not None:
        df.attrs['freq_bin_index'] = freq_bin_index
    return add_nrs_components(df.reset_index(drop=True))


def add_nrs_components(df):
    """Split NRS into its within- and between-subject parts, in place on a copy.

    Over the ROWS in `df`, deliberately -- see the module docstring. The mean is
    row-weighted rather than epoch-weighted, so an epoch that lost channels to
    masking contributes proportionally less to its subject's centre. That is the
    centring that makes NRS_within orthogonal to the subject dummies in THIS
    design matrix, which is the only thing the decomposition has to achieve.
    """
    df = df.copy()
    df['NRS_submean'] = df.groupby('subject')['NRS'].transform('mean')
    df['NRS_within'] = df['NRS'] - df['NRS_submean']
    return df


def cell_is_fittable(df, *, min_subjects=MIN_SUBJECTS, min_rows=MIN_ROWS):
    """(ok, reason). Structural refusals, checked BEFORE burning a fit."""
    if len(df) < min_rows:
        return False, f'{len(df)} rows < {min_rows}'
    n_subj = df['subject'].nunique()
    if n_subj < min_subjects:
        return False, f'{n_subj} subjects < {min_subjects}'
    if not np.isfinite(df['NRS_within'].to_numpy()).all():
        return False, 'non-finite NRS_within'
    if df['NRS_within'].std(ddof=0) <= 0:
        return False, 'no within-subject NRS variance'
    # A subject contributing no within-subject spread contributes nothing to the
    # effect of interest, but it does still inform the nuisance terms, so this is
    # only fatal when it is true of everyone.
    spread = df.groupby('subject')['NRS_within'].std(ddof=0)
    if int((spread > 0).sum()) < min_subjects:
        return False, (f'{int((spread > 0).sum())} subjects with within-subject '
                       f'NRS variance < {min_subjects}')
    return True, ''


# ============================================================================
# FITTING
# ============================================================================

def fit_cell(df, vc=None, *, reml=True, start_params=None, method=None,
             maxiter=200):
    """Fit one cell. Returns (results, captured_warning_messages).

    Raises CellFitError rather than returning junk, so the caller records a row
    with converged=False instead of a silently plausible number.
    """
    import statsmodels.formula.api as smf

    vc = VC_FULL if vc is None else vc
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            md = smf.mixedlm(FORMULA, data=df, groups=df['subject'],
                             re_formula='0', vc_formula=vc)
            kw = {'reml': reml, 'maxiter': maxiter}
            if start_params is not None:
                kw['start_params'] = start_params
            if method is not None:
                kw['method'] = method
            res = md.fit(**kw)
        except Exception as exc:                       # noqa: BLE001 - reported
            raise CellFitError(f'{type(exc).__name__}: {exc}') from exc
    msgs = sorted({str(w.message) for w in caught})
    if res.fe_params is None or not np.isfinite(res.fe_params).all():
        raise CellFitError('non-finite fixed effects')
    return res, msgs


def vcomp_by_name(res):
    """{variance component name: variance}. BY NAME -- res.vcomp is sorted."""
    return dict(zip(res.model.exog_vc.names, np.asarray(res.vcomp, dtype=float)))


def boundary_report(res, df, tol=BOUNDARY_TOL):
    """(boundary component names, {name: variance ratio}).

    A component's ratio is the variance it contributes to the LINEAR PREDICTOR,
    over the residual variance. For an intercept-like component that is just its
    variance; for `subj_slope` and `channel_slope` the VC design column is
    NRS_within, so the contribution is scaled by var(NRS_within). Comparing the
    raw variances instead would call every slope component "tiny" purely because
    NRS spans 0-10 while log power spans a fraction of a decade.
    """
    scale = float(res.scale)
    var_within = float(np.var(df['NRS_within'].to_numpy(), ddof=0))
    ratios = {}
    for name, v in vcomp_by_name(res).items():
        contribution = v * var_within if name.endswith('slope') else v
        ratios[name] = float(contribution / scale) if scale > 0 else np.nan
    at_boundary = sorted(n for n, r in ratios.items() if np.isfinite(r) and r < tol)
    return at_boundary, ratios


def lrt(res_full, res_reduced):
    """(statistic, p) for dropping `subj_slope`, on the 50:50 boundary mixture.

    REML log-likelihoods are comparable here ONLY because the fixed-effects
    design is identical between the two models; that is asserted, not assumed,
    because it is exactly the kind of thing that silently stops being true when
    someone adds a covariate to one formula.

    THE VARIANCE IS AT THE BOUNDARY of the parameter space (it cannot be
    negative), so the null distribution is not chi2(1) -- it is the 50:50 mixture
    of a point mass at 0 and chi2(1). A plain chi2(1) reference is CONSERVATIVE,
    which sounds safe but here means under-detecting exactly the heterogeneity
    the analysis exists to find. Reported as `p_lrt_mixture` so the choice
    travels with the number.
    """
    if not np.array_equal(np.asarray(res_full.model.exog),
                          np.asarray(res_reduced.model.exog)):
        raise CellFitError('LRT refused: the two models have different fixed-effects '
                           'designs, so their REML likelihoods are not comparable')
    stat = 2.0 * (float(res_full.llf) - float(res_reduced.llf))
    if not np.isfinite(stat) or stat <= 0:
        # The full model cannot fit worse than the nested reduced one except by
        # optimizer noise. Report the statistic as it came out and a p of 1.
        return float(stat) if np.isfinite(stat) else np.nan, 1.0
    return stat, float(0.5 * stats.chi2.sf(stat, df=1))


# ============================================================================
# EXTRACTION
# ============================================================================

def cell_record(res, res_reduced, df, *, region, freq_bin_index, bin_low_hz,
                bin_high_hz, fit_seconds, warnings_full=(), warnings_reduced=(),
                tol=BOUNDARY_TOL):
    """The flat per-cell row. Column names are the analysis's output schema."""
    at_boundary, ratios = boundary_report(res, df, tol=tol)
    vc = vcomp_by_name(res)
    stat, p_lrt = lrt(res, res_reduced) if res_reduced is not None else (np.nan, np.nan)
    boundary_msg = 'boundary of the parameter space'

    return {
        'region': region,
        'freq_bin_index': int(freq_bin_index),
        'freq_bin_low': float(bin_low_hz),
        'freq_bin_high': float(bin_high_hz),
        'n_subjects': int(df['subject'].nunique()),
        'n_channels': int(df['channel_uid'].nunique()),
        'n_epochs': int(df.groupby('subject')['epoch_id'].nunique().sum()),
        'n_rows': int(len(df)),

        'beta_nrs_within': float(res.fe_params['NRS_within']),
        'se': float(res.bse['NRS_within']),
        'z': float(res.tvalues['NRS_within']),
        'p': float(res.pvalues['NRS_within']),

        'beta_nrs_submean': float(res.fe_params['NRS_submean']),
        'se_submean': float(res.bse['NRS_submean']),
        'p_submean': float(res.pvalues['NRS_submean']),

        'var_subj_int': float(vc.get('subj_int', np.nan)),
        'var_subj_slope': float(vc.get('subj_slope', np.nan)),
        'var_channel': float(vc.get('channel', np.nan)),
        'var_resid': float(res.scale),

        'lrt_stat': float(stat),
        'p_lrt_mixture': float(p_lrt),

        'converged': bool(res.converged),
        'converged_reduced': (bool(res_reduced.converged)
                              if res_reduced is not None else None),
        'singular_flag': bool(at_boundary),
        'boundary_components': ','.join(at_boundary),
        'ratio_subj_int': ratios.get('subj_int', np.nan),
        'ratio_subj_slope': ratios.get('subj_slope', np.nan),
        'ratio_channel': ratios.get('channel', np.nan),
        # Recorded, NOT used as the boundary criterion -- it fires on
        # well-identified fits too.
        'sm_boundary_warning': any(boundary_msg in m for m in warnings_full),
        'n_warnings': len(set(warnings_full) | set(warnings_reduced)),
        'warnings': ' | '.join(sorted(set(warnings_full) | set(warnings_reduced)))[:500],
        'fit_seconds': float(fit_seconds),
        'error': '',
    }


def failed_record(region, freq_bin_index, bin_low_hz, bin_high_hz, reason, df=None,
                  fit_seconds=np.nan):
    """A cell that could not be fitted, as a ROW. Never a silent drop."""
    rec = {k: np.nan for k in (
        'beta_nrs_within', 'se', 'z', 'p', 'beta_nrs_submean', 'se_submean',
        'p_submean', 'var_subj_int', 'var_subj_slope', 'var_channel', 'var_resid',
        'lrt_stat', 'p_lrt_mixture', 'ratio_subj_int', 'ratio_subj_slope',
        'ratio_channel')}
    rec.update({
        'region': region, 'freq_bin_index': int(freq_bin_index),
        'freq_bin_low': float(bin_low_hz), 'freq_bin_high': float(bin_high_hz),
        'n_subjects': int(df['subject'].nunique()) if df is not None else 0,
        'n_channels': int(df['channel_uid'].nunique()) if df is not None else 0,
        'n_epochs': (int(df.groupby('subject')['epoch_id'].nunique().sum())
                     if df is not None else 0),
        'n_rows': int(len(df)) if df is not None else 0,
        'converged': False, 'converged_reduced': None, 'singular_flag': False,
        'boundary_components': '', 'sm_boundary_warning': False,
        'n_warnings': 0, 'warnings': '',
        'fit_seconds': float(fit_seconds), 'error': str(reason),
    })
    return rec


def blup_rows(res, df, *, region, freq_bin_index):
    """Long per-subject BLUPs for the subject-level components.

    `res.random_effects[group]` is a Series indexed by '<vc name>[<column>]'.
    The channel entries are also in there; they are deliberately NOT emitted --
    per-contact BLUPs would be one row per contact per cell, which is the size of
    the raw data, and nothing downstream asks for them.
    """
    rows = []
    counts = (df.groupby('subject')
              .agg(n_reports=('epoch_id', 'nunique'),
                   n_channels=('channel_uid', 'nunique')))
    beta = float(res.fe_params['NRS_within'])
    for subject, effects in res.random_effects.items():
        rows.append({
            'region': region,
            'freq_bin': int(freq_bin_index),
            'subject': subject,
            'blup_slope': float(effects.get('subj_slope[NRS_within]', np.nan)),
            'blup_intercept': float(effects.get('subj_int[Intercept]', np.nan)),
            # The subject's own implied slope, which is what a reader actually
            # wants to compare against the two-stage per-subject coefficient.
            'subject_slope': beta + float(effects.get('subj_slope[NRS_within]', np.nan)),
            'n_reports': int(counts.loc[subject, 'n_reports']),
            'n_channels': int(counts.loc[subject, 'n_channels']),
        })
    return rows


# ============================================================================
# THE PERMUTATION NULL
# ============================================================================

def permute_within_subject(df, rng):
    """Permute the EPOCH -> NRS assignment inside each subject.

    An epoch is relabelled AS A WHOLE: every channel row of that epoch gets the
    same new score. This is the exchangeability rule
    `cluster_permutation.predictor_shuffle_null` already uses, and it is the one
    that matches the scientific claim -- does pairing a pain score to an epoch
    carry information -- rather than merely asking whether a distribution is
    symmetric about zero.

    Both NRS components are RE-DERIVED afterwards, exactly as the observed fit
    derived them. `NRS_submean` is invariant under this permutation whenever the
    design is balanced (the multiset of scores per subject is unchanged), and
    shifts very slightly when it is not, because the centring is row-weighted.
    Re-deriving is correct either way: the null must be built by the same
    procedure as the observed statistic.
    """
    out = df.copy()
    new_nrs = np.empty(len(out), dtype=config.CACHE_ACCUMULATE_DTYPE)
    for _, idx in out.groupby('subject', sort=False).indices.items():
        block = out.iloc[idx]
        epochs = block['epoch_id'].to_numpy()
        uniq, inverse = np.unique(epochs, return_inverse=True)
        # One score per distinct epoch, then permuted across epochs.
        per_epoch = (block.drop_duplicates('epoch_id')
                     .set_index('epoch_id')['NRS'].reindex(uniq).to_numpy())
        new_nrs[idx] = rng.permutation(per_epoch)[inverse]
    out['NRS'] = new_nrs
    return add_nrs_components(out.drop(columns=['NRS_submean', 'NRS_within']))


def permutation_null(df, n_perm, *, seed=0, start_params=None, n_jobs=1,
                     vc=None):
    """Null distribution of the NRS_within fixed effect under within-subject
    shuffling of the pain score.

    Returns a DataFrame with one row per shuffle: perm, beta, z, converged.
    A shuffle whose fit fails becomes a row with NaN, not a gap -- so the
    denominator of the permutation p is honest about what was actually computed.

    `start_params` warm-starts every refit from the OBSERVED fit. The variance
    components barely move under a permutation of the predictor, so this cuts
    iterations substantially; it cannot bias the null, because the starting point
    is identical for every shuffle and does not depend on that shuffle's data.
    """
    vc = VC_FULL if vc is None else vc
    seeds = np.random.SeedSequence(seed).spawn(n_perm)

    def one(i, ss):
        rng = np.random.default_rng(ss)
        permuted = permute_within_subject(df, rng)
        try:
            res, _ = fit_cell(permuted, vc, start_params=start_params)
        except CellFitError as exc:
            return {'perm': i, 'beta': np.nan, 'z': np.nan, 'converged': False,
                    'error': str(exc)[:200]}
        return {'perm': i, 'beta': float(res.fe_params['NRS_within']),
                'z': float(res.tvalues['NRS_within']),
                'converged': bool(res.converged), 'error': ''}

    if n_jobs and n_jobs > 1:
        from joblib import Parallel, delayed
        from threadpoolctl import threadpool_limits
        # Pin BLAS to one thread per worker. Without this, n_jobs processes each
        # spawn a full thread pool and the node thrashes -- the fits get SLOWER
        # than serial.
        with threadpool_limits(limits=1):
            rows = Parallel(n_jobs=n_jobs, backend='loky')(
                delayed(one)(i, ss) for i, ss in enumerate(seeds))
    else:
        rows = [one(i, ss) for i, ss in enumerate(seeds)]
    return pd.DataFrame(rows)


def permutation_p(observed, null, *, two_sided=True):
    """(1 + #{null at least as extreme}) / (1 + n). Never exactly 0.

    Same estimator as `cluster_permutation.permutation_p`, kept identical so the
    two analyses' p-values mean the same thing. Two-sided by magnitude, because
    the fixed effect's Wald p that it is being compared against is two-sided.
    """
    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]
    if null.size == 0 or not np.isfinite(observed):
        return np.nan, 0
    extreme = (np.abs(null) >= abs(observed)) if two_sided else (null >= observed)
    return float((1 + int(extreme.sum())) / (1 + null.size)), int(null.size)
