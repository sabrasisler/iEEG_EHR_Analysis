"""
`pain_coef`: per-subject regression of log power on PAIN SCORE.

One coefficient per (subject, region, frequency bin) -- the change in
log10(V^2/Hz) per pain point. Averaged across subjects it is the quantity the
continuous-pain heatmap shades.

WHY A REGRESSION INSTEAD OF A 0-PAIN BASELINE
---------------------------------------------
Every other pain figure in this project references a subject to their own 0-pain
epochs, so the precision of that reference sets the precision of the result. That
is a measured problem, not a hypothetical one: ten of 56 discovery subjects carry a
0-pain mean whose SEM (0.083) EXCEEDS the effect being measured (0.052)
(TASKS.md, docs/labnotebook 2026-08-07). A regression has no baseline, so the
problem does not arise -- and it uses the whole pain range rather than collapsing
it into two or three bins.

It also disposes of two defects diagnosed on 2026-08-05, both of which were
consequences of having a baseline at all:
  - the `none` bin's CIRCULARITY (the epochs that define the reference cannot also
    test it), and
  - the epoch-weighting asymmetry (the baseline pools WINDOWS while a reported
    value averages EPOCHS, so the 0-pain bin cannot be exactly zero).

Between-subject scale cancels for free. A per-subject amplifier gain multiplies
power, which is an ADDITIVE shift in log space, and an additive shift does not
change a slope. So coefficients from different subjects are on the same scale and
can simply be averaged -- no second normalization, and none of the pooling costs
that come with one.

NAMING. `beta` is the frequency band and `slope` is the 1/f aperiodic slope
(views/aperiodic.py); both are live in this codebase and neither means this. This
quantity is `pain_coef`, always.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Inclusion defaults. These are ELIGIBILITY criteria for estimating a slope, which
# is a different question from the 0-pain-epoch floor that
# view_tables.exclude_thin_baseline_subjects applies -- see `eligible_subjects`.
MIN_EPOCHS = 10          # strictly greater than
MIN_RANGE = 4.0          # max - min pain score, inclusive
MIN_NON_MODAL = 5        # epochs whose score is not the subject's modal score


def subject_pain_scores(epoch_tables):
    """One row per (subject, epoch): the pain score. De-duplicated.

    The epoch tables are exploded over region x frequency bin, so a subject's
    epoch appears ~1000 times; counting rows instead of epochs would inflate every
    eligibility criterion by that factor.
    """
    return (epoch_tables[['subject_id', 'epoch_id', 'pain_score']]
            .drop_duplicates(['subject_id', 'epoch_id'])
            .astype({'pain_score': float}))


def eligible_subjects(epoch_tables, min_epochs=MIN_EPOCHS, min_range=MIN_RANGE,
                      min_non_modal=MIN_NON_MODAL):
    """(kept subject ids, diagnostics frame) under the three eligibility rules.

    A slope needs SPREAD IN THE PREDICTOR, and each rule guards a different way of
    not having it:

    - `min_epochs`   -- too few points to fit anything stable.
    - `min_range`    -- the scores span too little to identify a trend. Matches the
                        >4-point criterion architecture.md PART 6 already uses.
    - `min_non_modal` -- THE ONE RANGE MISSES. A subject can report 0 forty times
                        and 8 twice: range 8, but the entire slope rests on two
                        epochs, so it is an outlier statistic rather than a trend.
                        On the discovery cohort this rule is what excludes sub-085,
                        which SCRATCHPAD.md independently flags as "looks off".

    NOT `view_tables.exclude_thin_baseline_subjects`. That filter exists because a
    thin 0-PAIN baseline poisons a baseline-referenced quantity -- and this quantity
    has no baseline. Applying it here would re-import the exact problem the
    regression was chosen to avoid, and would drop subjects for a reason that does
    not apply to them. The two look interchangeable and are not.
    """
    scores = subject_pain_scores(epoch_tables)
    rows = []
    for subject, grp in scores.groupby('subject_id'):
        s = grp['pain_score']
        mode = s.mode()
        modal = float(mode.iloc[0]) if not mode.empty else np.nan
        n_non_modal = int((s != modal).sum())
        rng = float(s.max() - s.min()) if len(s) else 0.0
        reasons = []
        if not len(s) > min_epochs:
            reasons.append(f'<={min_epochs} epochs')
        if not rng >= min_range:
            reasons.append(f'range<{min_range:g}')
        if not n_non_modal >= min_non_modal:
            reasons.append(f'<{min_non_modal} non-modal scores')
        rows.append({'subject_id': subject, 'n_epochs': int(len(s)),
                     'pain_range': rng, 'modal_score': modal,
                     'n_non_modal': n_non_modal, 'n_distinct': int(s.nunique()),
                     'included': not reasons,
                     'excluded_because': '; '.join(reasons)})

    diagnostics = pd.DataFrame(rows).sort_values('subject_id', ignore_index=True)
    kept = diagnostics.loc[diagnostics['included'], 'subject_id'].tolist()

    dropped = diagnostics[~diagnostics['included']]
    for _, r in dropped.iterrows():
        logger.info('excluded %s: n=%d range=%.1f non_modal=%d -- %s',
                    r.subject_id, r.n_epochs, r.pain_range, r.n_non_modal,
                    r.excluded_because)
    logger.info('%d/%d subject(s) eligible for a pain_coef '
                '(>%d epochs, range>=%g, >=%d non-modal)',
                len(kept), len(diagnostics), min_epochs, min_range, min_non_modal)
    return kept, diagnostics


def regression_weights(x):
    """w such that `w @ y` is the OLS slope of y on x. NaN if x has no variance.

    THE IDENTITY THIS MODULE IS BUILT AROUND. The predictor is the subject's pain
    score, which is the SAME for every region and frequency bin, so the slope for
    all ~1000 cells is one matrix product rather than 1000 regressions:

        w = (x - mean(x)) / sum((x - mean(x))^2)
        slope = w @ Y                                Y is (n_epochs, n_cells)

    It is also what makes the permutation null affordable: a permuted slope map is
    the same matmul with a shuffled x, so 10,000 permutations stay cheap.
    """
    x = np.asarray(x, dtype=np.float64)
    centred = x - x.mean()
    denom = float((centred ** 2).sum())
    if denom <= 0:
        return None
    return centred / denom


def subject_coef_matrix(epoch_tables, regions, freq_bins, subjects=None,
                        min_epochs=MIN_EPOCHS, min_range=MIN_RANGE,
                        min_non_modal=MIN_NON_MODAL):
    """(coef, subjects, per_subject_epoch_values, diagnostics).

    coef is (n_subject, n_region, n_bin); a cell a subject has no coverage for is
    NaN, never 0 -- 0 is a real coefficient value meaning "no relationship", and
    conflating the two would put fabricated null results into the group mean.

    `per_subject_epoch_values` is kept because the permutation null needs the
    epoch-level matrices again: {subject: (Y, x)} with Y (n_epochs, n_region*n_bin).
    """
    kept, diagnostics = eligible_subjects(epoch_tables, min_epochs, min_range,
                                          min_non_modal)
    if subjects is not None:
        kept = [s for s in kept if s in set(subjects)]
    if not kept:
        raise SystemExit('no subject satisfies the pain_coef eligibility criteria')

    r_idx = {r: i for i, r in enumerate(regions)}
    b_idx = {b: i for i, b in enumerate(freq_bins)}
    n_cells = len(regions) * len(freq_bins)

    coef = np.full((len(kept), len(regions), len(freq_bins)), np.nan)
    per_subject = {}

    for si, subject in enumerate(kept):
        rows = epoch_tables[epoch_tables['subject_id'] == subject]
        rows = rows[rows['region'].isin(r_idx) & rows['freq_bin_index'].isin(b_idx)]
        if rows.empty:
            continue

        epochs = sorted(rows['epoch_id'].unique())
        e_idx = {e: i for i, e in enumerate(epochs)}
        Y = np.full((len(epochs), n_cells), np.nan)
        flat = (rows['region'].map(r_idx).to_numpy() * len(freq_bins)
                + rows['freq_bin_index'].map(b_idx).to_numpy())
        Y[rows['epoch_id'].map(e_idx).to_numpy(), flat] = rows['value'].to_numpy()

        x = (rows.drop_duplicates('epoch_id').set_index('epoch_id')
             .loc[epochs, 'pain_score'].to_numpy(dtype=np.float64))
        if regression_weights(x) is None:  # caught by min_range, but never assume
            logger.warning('%s: pain score has zero variance, skipping', subject)
            continue

        coef[si] = coef_from_predictor(x, Y).reshape(len(regions), len(freq_bins))
        per_subject[subject] = (Y, x)

    return coef, kept, per_subject, diagnostics


def coef_from_predictor(x, Y):
    """OLS slope of every column of Y on x. NaN-aware, NaN where unestimable.

    Works from the PREDICTOR rather than from precomputed weights on purpose. The
    weights identity in `regression_weights` is exact only when every row is used;
    for a column with missing epochs the predictor must be re-centred over exactly
    the surviving rows, and re-centring the weight vector instead silently drops a
    scale factor of sum((x - xbar)^2). That is the kind of error that produces
    plausible numbers, so the subset case recomputes from x.

    The fast path still applies to every fully-observed column, which is nearly all
    of them: one matmul for the lot.
    """
    x = np.asarray(x, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    out = np.full(Y.shape[1], np.nan)

    finite = np.isfinite(Y)
    full = np.flatnonzero(finite.all(axis=0))
    if full.size:
        w = regression_weights(x)
        if w is not None:
            out[full] = w @ Y[:, full]

    for j in np.flatnonzero(~finite.all(axis=0)):
        rows = finite[:, j]
        # 3 is the floor at which a slope is even nominally defined with a residual
        # left over; below it the estimate is noise wearing a number.
        if rows.sum() < 3:
            continue
        w_sub = regression_weights(x[rows])
        if w_sub is None:
            continue
        out[j] = float(w_sub @ Y[rows, j])
    return out
