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
                max_gap_min=DEFAULT_MAX_GAP_MIN):
    """CONSECUTIVE assessment pairs within a session, with what fell between them.

    Consecutive, not all-pairs: a pair that skips an assessment has that
    assessment's dose history inside it while pretending to be a clean interval.

    Returns one row per pair with `subject_id` in the prefixed form the view
    tables use, so it joins to a cell frame without further translation.
    """
    rows = []
    for (subject, session), g in defs.groupby(['subject', 'session'], sort=False):
        g = g.sort_values('pain_time')
        a = admin[(admin['subject'] == subject) & (admin['session'] == session)]
        t = g['pain_time'].to_numpy()
        p = g['pain_score'].to_numpy(dtype=float)
        e = g['epoch_id'].to_numpy()
        for i in range(len(g) - 1):
            gap = (t[i + 1] - t[i]) / np.timedelta64(1, 'm')
            if not (min_gap_min <= gap <= max_gap_min):
                continue
            between = a[(a['taken_dt'] > t[i]) & (a['taken_dt'] <= t[i + 1])]
            # CARRYOVER. A dose given before this pair is still pharmacologically
            # active during it, so "no dose between" does NOT mean unmedicated.
            # Measured on this cohort it is not a minor contamination: 58% of
            # control pairs had a dose within an hour of their start, and the
            # median control pair begins 1.0 h after a dose against 3.4 h for the
            # exposed pairs -- the controls are the MORE recently dosed group.
            # Recorded so a model can adjust for it, a clean subset can be
            # defined, or the exposure can be respecified as time-since-dose.
            before = a.loc[a['taken_dt'] <= t[i], 'taken_dt']
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
                'n_fast': int(between['route'].isin(FAST_ROUTES).sum()),
                'n_slow': int(between['route'].isin(SLOW_ROUTES).sum()),
            })
    pairs = pd.DataFrame(rows)
    if pairs.empty:
        raise ValueError(f'no assessment pairs in [{min_gap_min}, {max_gap_min}] min')
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
