"""Paired change-score design: consecutive pain assessments and what happened between.

The 2-hour-lookback stratification asks "was a dose given before this score".
This asks a sharper question: take two consecutive assessments, and treat a dose
administered BETWEEN them as the intervening event. The first assessment is its
own control.

WHY THIS IS BETTER THAN THE LOOKBACK, on this data specifically:

- The exposure is well defined. A lookback cannot tell an acute effect from a
  rebound at 2.1 hours; "a dose fell between these two measurements" can.
- Differencing removes each channel's level AND slow within-subject drift. Two
  scores hours apart share essentially the same electrode state, so electrode
  impedance drift -- a confound this project flagged and had no way to test --
  largely cancels.
- It never fits a slope across the raw 0-10 scale. That scale is badly behaved
  here: non-monotone, a third of unmedicated epochs at NRS=0, and the extreme
  scores that carry the most regression leverage present almost only in the
  medicated stratum. Working in differences sidesteps all of it.

THE MINIMUM GAP IS NOT OPTIONAL. Epochs are the 5 minutes BEFORE a score, so two
scores less than 5 minutes apart have physically overlapping windows and their
difference is compressed toward zero by shared samples -- a silent null bias
concentrated in the shortest-gap pairs. 41 pairs (2.2%) are affected. The default
30 minutes clears that with margin and also covers oral onset, which matters
because oral is 82% of administrations.

Applying a route-DEPENDENT minimum would be worse than useless: it is undefined
for unmedicated pairs, which have no route, and it would make the gap
distribution differ systematically between the groups being compared. Onset
belongs in the model as a covariate, not in the inclusion rule.

REGRESSION TO THE MEAN IS THE DOMINANT SIGNAL and any model of these pairs must
carry `pain_1`. Measured on this cohort, mean change in pain runs from +2.12 at
baseline 0 to -3.00 at baseline 10. Medication is given BECAUSE pain is high, so
exposure and baseline are entangled by construction; a model without the baseline
will report regression to the mean as a drug effect.

GAP IS CONFOUNDED WITH BASELINE TOO. Short-gap pairs start around NRS 4.2-4.6 and
long-gap pairs around 2.2-2.9, because reassessment is faster when pain is high.
So a difference between gap bands is not cleanly a timescale effect, and the
`med_between x gap` interaction inside one model is the better instrument.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Epochs are 5 minutes pre-score; below this two epochs overlap or are heavily
#: autocorrelated. 30 also covers oral onset (82% of administrations).
DEFAULT_MIN_GAP_MIN = 30.0

#: Beyond this, "a dose fell between" stops being specific -- multiple doses,
#: sleep, procedures -- and the drift-cancelling benefit of differencing is gone.
DEFAULT_MAX_GAP_MIN = 240.0

FAST_ROUTES = ('Intravenous', 'Sublingual')
SLOW_ROUTES = ('Oral', 'Feeding Tube')


def build_pairs(defs, admin, min_gap_min=DEFAULT_MIN_GAP_MIN,
                max_gap_min=DEFAULT_MAX_GAP_MIN, exclude_admin=None):
    """CONSECUTIVE assessment pairs within a session, with what fell between them.

    Consecutive, not all-pairs: a pair that skips an assessment has that
    assessment's dose history inside it while pretending to be a clean interval.

    `admin` is expected PRE-FILTERED to the drug set under test, which means
    `med_between` is blind to every OTHER drug class by construction. Left alone
    that silently poisons a subset arm: measured 2026-09-10 on the 30-240 min
    pairs, 22.3% of the `non_opioid_analgesics` control pairs had an OPIOID
    administered between the two assessments, so the weakest analgesic was being
    contrasted against a control group containing the strongest one.

    `exclude_admin` is the escape hatch: administrations that DISQUALIFY a pair
    outright if they fall between it. Pass the complementary classes for a subset
    arm. Note this drops from BOTH groups, not only the controls -- 74 pairs had
    an opioid AND a non-opioid between, and leaving those in the exposed group
    would let an opioid effect be reported as a non-opioid one. What remains is a
    true partition: "this class and nothing else" vs "no analgesic at all".

    Deliberately NOT applied symmetrically by default. Leaving acetaminophen and
    NSAIDs inside the opioid arm's controls is a weak drug contaminating a strong
    drug's control group, which biases toward zero rather than toward a false
    positive; the reverse is not, and only the reverse is corrected here.

    Returns one row per pair with `subject_id` in the prefixed form the view
    tables use, so it joins to a cell frame without further translation.
    """
    rows = []
    for (subject, session), g in defs.groupby(['subject', 'session'], sort=False):
        g = g.sort_values('pain_time')
        a = admin[(admin['subject'] == subject) & (admin['session'] == session)]
        x = (None if exclude_admin is None else
             exclude_admin[(exclude_admin['subject'] == subject)
                           & (exclude_admin['session'] == session)])
        t = g['pain_time'].to_numpy()
        p = g['pain_score'].to_numpy(dtype=float)
        e = g['epoch_id'].to_numpy()
        for i in range(len(g) - 1):
            gap = (t[i + 1] - t[i]) / np.timedelta64(1, 'm')
            if not (min_gap_min <= gap <= max_gap_min):
                continue
            between = a[(a['taken_dt'] > t[i]) & (a['taken_dt'] <= t[i + 1])]
            n_excl = (0 if x is None else
                      int(((x['taken_dt'] > t[i])
                           & (x['taken_dt'] <= t[i + 1])).sum()))
            # CARRYOVER. A dose given before this pair is still pharmacologically
            # active during it, so "no dose between" does NOT mean unmedicated.
            # Measured on this cohort it is not a minor contamination: 58% of
            # control pairs had a dose within an hour of their start, and the
            # median control pair begins 1.0 h after a dose against 3.4 h for the
            # exposed pairs -- the controls are the MORE recently dosed group.
            # Recorded so a model can adjust for it, a clean subset can be
            # defined, or the exposure can be respecified as time-since-dose.
            # Carryover is measured over the UNION of the tested and excluded
            # classes, not just the tested one. In a subset arm the nearest prior
            # dose is often from the other class, and a `h_since_prior` blind to
            # it would report a pair as long-unmedicated when it is not.
            before = a.loc[a['taken_dt'] <= t[i], 'taken_dt']
            if x is not None:
                before = pd.concat([before, x.loc[x['taken_dt'] <= t[i], 'taken_dt']])
            h_prior = (np.nan if before.empty
                       else (t[i] - before.max()) / np.timedelta64(1, 'h'))
            rows.append({
                'h_since_prior': float(h_prior),
                'subject': subject, 'subject_id': f'sub-{subject}',
                'session': session, 'e1': int(e[i]), 'e2': int(e[i + 1]),
                'gap_m': float(gap), 'gap_h': float(gap) / 60.0,
                'pain_1': float(p[i]), 'pain_2': float(p[i + 1]),
                'd_pain': float(p[i + 1] - p[i]),
                'n_med': int(len(between)),
                'med_between': float(len(between) > 0),
                'n_excl_between': n_excl,
                'n_fast': int(between['route'].isin(FAST_ROUTES).sum()),
                'n_slow': int(between['route'].isin(SLOW_ROUTES).sum()),
            })
    pairs = pd.DataFrame(rows)
    if pairs.empty:
        raise ValueError(f'no assessment pairs in [{min_gap_min}, {max_gap_min}] min')

    if exclude_admin is not None:
        drop = pairs['n_excl_between'] > 0
        logger.info('CO-EXPOSURE: dropping %d of %d pairs (%.1f%%) with an '
                    'excluded-class dose between -- %d of them were controls, '
                    '%d were also exposed to the tested class',
                    int(drop.sum()), len(pairs), 100 * drop.mean(),
                    int((drop & (pairs['med_between'] == 0)).sum()),
                    int((drop & (pairs['med_between'] == 1)).sum()))
        pairs = pairs[~drop].reset_index(drop=True)
        if pairs.empty:
            raise ValueError('every pair had an excluded-class dose between')

    pairs.insert(0, 'pair_id', range(len(pairs)))
    logger.info('pairs: %d in [%g, %g] min | %d subjects | %d with a dose between '
                '(%.1f%%) | %d with d_pain != 0',
                len(pairs), min_gap_min, max_gap_min, pairs['subject'].nunique(),
                int((pairs['n_med'] > 0).sum()),
                100 * (pairs['n_med'] > 0).mean(),
                int((pairs['d_pain'] != 0).sum()))
    return pairs


def build_change_frame(cell_rows, pairs):
    """One row per (pair x channel): the CHANGE in log power across the pair.

    A channel must be present in BOTH epochs of a pair or it is dropped -- a
    difference needs both terms, and silently treating a missing epoch as zero
    would manufacture a change. The channel level cancels in the subtraction,
    which is the point: what survives is the change, free of that contact's
    baseline power and of any drift slow enough to be common to both epochs.
    """
    keep = ['subject', 'channel_uid', 'epoch_id', 'log10_power']
    left = cell_rows[keep].rename(columns={'epoch_id': 'e1', 'log10_power': 'y1'})
    right = cell_rows[keep].rename(columns={'epoch_id': 'e2', 'log10_power': 'y2'})

    # The view's `subject` is the prefixed form; the pair index carries both.
    df = pairs.copy()
    df['subject'] = df['subject_id']
    df = df.merge(left, on=['subject', 'e1'], how='inner')
    df = df.merge(right, on=['subject', 'channel_uid', 'e2'], how='inner')
    df['d_log10_power'] = df['y2'] - df['y1']
    df = df[np.isfinite(df['d_log10_power'])]
    return df.reset_index(drop=True)


def pair_summary(pairs):
    """Per stratum: how many pairs, and the regression-to-the-mean picture.

    Exists so the confound is in the artifact rather than only in a docstring.
    """
    rows = []
    for med, g in pairs.groupby(pairs['n_med'] > 0):
        rows.append({
            'stratum': 'dose between' if med else 'no dose between',
            'n_pairs': len(g), 'n_subjects': g['subject'].nunique(),
            'gap_m_median': g['gap_m'].median(),
            'pain_1_mean': g['pain_1'].mean(),
            'd_pain_mean': g['d_pain'].mean(),
            'd_pain_sd': g['d_pain'].std(),
            'frac_d_pain_zero': float((g['d_pain'] == 0).mean()),
        })
    return pd.DataFrame(rows).sort_values('stratum').reset_index(drop=True)


# ============================================================================
# PER-SUBJECT COEFFICIENTS
# ============================================================================
# The group mixed-model grid treats 798 region x frequency cells as 798 tests,
# which they are not: log-spaced bins come from the same FFT, and regions share
# subjects and channels. The principled inference is a CLUSTER PERMUTATION test
# over frequency (docs/cluster_permutation.md), and that test wants ONE VALUE PER
# SUBJECT per region per bin -- not a group coefficient.
#
# The naive per-subject value, a dosed-minus-undosed difference of mean
# d_log10_power, would be WRONG here. It drops `pain_1`, and regression to the
# mean is the dominant signal in these pairs (mean change in pain runs +2.12 at
# baseline 0 to -3.00 at baseline 10) while medication is given BECAUSE pain is
# high. That difference would report regression to the mean as a drug effect --
# the same Lord's-paradox trap the group formula exists to avoid. So each subject
# gets their own small OLS carrying the same covariates, and what travels upward
# is their `med_between` coefficient.

#: Per-subject design. No random effects: within ONE subject there is no subject
#: grouping left, and the channel level already cancelled in the differencing.
SUBJECT_FORMULA = 'd_log10_power ~ d_pain + med_between + pain_1 + gap_h'

#: A subject needs enough pairs in BOTH exposure states for `med_between` to be
#: estimable at all. Below this the coefficient is noise dressed as data.
MIN_PAIRS_PER_STATE = 2


def subject_med_coefficients(df, formula=SUBJECT_FORMULA,
                             min_pairs_per_state=MIN_PAIRS_PER_STATE):
    """Per-subject `med_between` coefficient for ONE cell.

    `df` is a change frame (one row per pair x channel) for a single region and
    frequency bin. Returns one row per subject with the coefficient, its SE, and
    the counts behind it, so a caller can drop thin subjects without refitting.

    Channels are POOLED within a subject rather than averaged first: averaging
    would discard the unequal channel counts that make some subjects' estimates
    genuinely better than others, and this coefficient is a per-subject summary,
    not a test -- its uncertainty is handled upstream by the permutation.
    """
    import statsmodels.formula.api as smf

    rows = []
    for subject, g in df.groupby('subject', sort=True):
        n_dosed = int((g['med_between'] == 1).sum())
        n_undosed = int((g['med_between'] == 0).sum())
        n_pairs_dosed = int(g.loc[g['med_between'] == 1, 'pair_id'].nunique())
        n_pairs_undosed = int(g.loc[g['med_between'] == 0, 'pair_id'].nunique())
        rec = {'subject': subject, 'n_rows': len(g),
               'n_channels': int(g['channel_uid'].nunique()),
               'n_pairs_dosed': n_pairs_dosed,
               'n_pairs_undosed': n_pairs_undosed,
               'med_beta': np.nan, 'med_se': np.nan, 'ok': False, 'why': ''}
        if min(n_pairs_dosed, n_pairs_undosed) < min_pairs_per_state:
            rec['why'] = 'too few pairs in one exposure state'
            rows.append(rec)
            continue
        if n_dosed == 0 or n_undosed == 0 or g['d_log10_power'].std(ddof=0) == 0:
            rec['why'] = 'no exposure contrast or no outcome variance'
            rows.append(rec)
            continue
        try:
            res = smf.ols(formula, data=g).fit()
        except Exception as exc:                             # noqa: BLE001
            rec['why'] = f'{type(exc).__name__}: {exc}'[:80]
            rows.append(rec)
            continue
        if 'med_between' not in res.params.index:
            rec['why'] = 'med_between dropped from the design'
            rows.append(rec)
            continue
        beta = float(res.params['med_between'])
        rec.update(med_beta=beta, med_se=float(res.bse['med_between']),
                   ok=bool(np.isfinite(beta)))
        if not rec['ok']:
            rec['why'] = 'non-finite coefficient'
        rows.append(rec)
    return pd.DataFrame(rows)
