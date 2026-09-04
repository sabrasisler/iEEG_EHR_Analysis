"""
Link each medication administration to the pain score that preceded it.

The pain-score export (`sub-*_ses-*_pain-scores.csv`, sibling of the MAR file in
the same `ehr/` folder) is one row per assessment: `date` for the timestamp and
`max_pain`, an integer 0-10. 4,849 assessments across 99 sessions, and every one
of the 98 MAR sessions has a pain-score file — so file coverage is not the
limiting factor here. TIMING is.

WHY A WINDOW AT ALL. Matching an administration to "the most recent prior score"
with no cap attributes a dose to an assessment that can be most of a day old:
measured on this corpus the median gap is 1.3 h, but p90 is
4.9 h and the tail runs to 19 h. The cap is therefore part of the question
rather than a tuning knob, and it is 30 MINUTES by request. An administration
with no assessment in the preceding 30 minutes is DROPPED, not matched to a
stale score — which makes the window an inclusion criterion, so `link_to_prior_score`
returns a frame with those rows already removed rather than a column of NaNs for
somebody downstream to remember about.

WHY A SAME-MINUTE SCORE COUNTS AS PRIOR. 45% of administrations (679 of 1,509
across the four most-administered analgesics) carry a pain score stamped in the
SAME MINUTE as the dose. Charting is minute-resolution and the nursing sequence
is assess -> administer -> chart both, so a same-minute score is the assessment
that prompted the dose; a gap of zero is "within 30 minutes prior". Measured
both ways, excluding exact matches drops the matched sample from 1,113 to 496
and leaves every per-drug distribution and their ordering unchanged — so this
decision buys sample size, not a conclusion. It stays a flag (`allow_exact`)
rather than a hardcode, and the strict-reading count goes into every run's
provenance so the choice is auditable from the artifact.

WHAT THIS IS NOT. A score preceding a dose does not make the score the REASON
for the dose. Scheduled drugs are given at a fixed time whatever the assessment
says, and an assessment is often charted precisely BECAUSE a PRN dose was
requested — the arrow can point either way and this table cannot separate them.
It measures what the chart said just before a dose. Nothing here is causal, and
per CLAUDE.md it is a nomination, not a finding.
"""

import logging

import pandas as pd

from ieeg_ehr import config

logger = logging.getLogger(__name__)

#: Inclusion window: an administration needs an assessment this recent to enter
#: the analysis at all. Set by request; see the module docstring.
WINDOW_MINUTES = 30

#: The clinical scale. Scores outside this are a data problem, not a rating.
PAIN_SCORE_MIN, PAIN_SCORE_MAX = 0, 10

PAIN_SCORE_COLUMNS = ('sub_id', 'ses_id', 'date', 'max_pain')


def load_pain_scores(paths=None):
    """One row per assessment: subject, session, score_dt, pain_score.

    `max_pain` is the export's name for the score; it is renamed to
    `pain_score` here so no downstream figure has to know that a column called
    "max" holds a single assessment's rating.
    """
    paths = list(paths) if paths is not None else config.pain_score_files()
    if not paths:
        raise FileNotFoundError('no pain-score exports found')

    frames = []
    for path in paths:
        df = pd.read_csv(path)
        missing = set(PAIN_SCORE_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f'{path}: missing columns {sorted(missing)}')
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    df['subject'] = df['sub_id'].str.replace('sub-', '', regex=False)
    df['session'] = df['ses_id'].str.replace('ses-', '', regex=False)
    df['score_dt'] = pd.to_datetime(df['date'], errors='coerce')
    df['pain_score'] = pd.to_numeric(df['max_pain'], errors='coerce')

    n_before = len(df)
    df = df.dropna(subset=['score_dt', 'pain_score'])
    if len(df) < n_before:
        logger.warning('%d pain-score row(s) dropped for an unparseable '
                       'timestamp or score', n_before - len(df))

    out_of_range = ~df['pain_score'].between(PAIN_SCORE_MIN, PAIN_SCORE_MAX)
    if out_of_range.any():
        raise ValueError(
            f'{int(out_of_range.sum())} pain score(s) outside '
            f'{PAIN_SCORE_MIN}-{PAIN_SCORE_MAX}: '
            f'{sorted(df.loc[out_of_range, "pain_score"].unique())}. The scale '
            f'is not what this code assumes; do not silently clip.')

    logger.info('pain scores: %d files, %d assessments, %d sessions',
                len(paths), len(df),
                df.groupby(['subject', 'session']).ngroups)
    return (df[['subject', 'session', 'score_dt', 'pain_score']]
            .sort_values(['subject', 'session', 'score_dt'])
            .reset_index(drop=True))


def _asof(admin, scores, window_minutes, allow_exact):
    """The backward as-of join, per (subject, session).

    `by=` is what keeps a score from one subject matching another's dose, and
    it also keeps the two sessions of a repeat subject separate — `taken_dt`
    is only comparable within a session, because de-identification re-anchors
    each session's dates independently.
    """
    return pd.merge_asof(
        admin.sort_values('taken_dt').reset_index(drop=True),
        scores.sort_values('score_dt').reset_index(drop=True),
        left_on='taken_dt', right_on='score_dt',
        by=['subject', 'session'], direction='backward',
        allow_exact_matches=allow_exact,
        tolerance=pd.Timedelta(minutes=window_minutes))


def link_to_prior_score(admin_df, scores_df, window_minutes=WINDOW_MINUTES,
                        allow_exact=True):
    """Attach the most recent pain score at or before each administration.

    Returns `(linked, stats)`. `linked` is `admin_df` plus `pain_score`,
    `score_dt` and `gap_minutes`, with unmatched administrations REMOVED (see
    the module docstring: the window is an inclusion criterion). `stats` is the
    audit trail — how many rows went in, how many were dropped and why, and
    what the strict-prior reading would have kept — and belongs in the run
    provenance and on the figure.
    """
    n_input = len(admin_df)

    admin = admin_df.dropna(subset=['taken_dt'])
    n_no_timestamp = n_input - len(admin)
    if n_no_timestamp:
        logger.warning('%d administration(s) have no parseable timestamp and '
                       'cannot be placed relative to an assessment',
                       n_no_timestamp)

    merged = _asof(admin, scores_df, window_minutes, allow_exact)
    merged['gap_minutes'] = ((merged['taken_dt'] - merged['score_dt'])
                             .dt.total_seconds() / 60.0)

    linked = merged.dropna(subset=['pain_score']).copy()
    linked['pain_score'] = linked['pain_score'].astype(int)

    # The strict reading, for the provenance record only — it is not what gets
    # plotted unless the caller asked for it.
    n_strict = int(_asof(admin, scores_df, window_minutes,
                         allow_exact=False)['pain_score'].notna().sum())
    n_exact_ties = int((merged['gap_minutes'] == 0).sum())

    stats = {
        'window_minutes': window_minutes,
        'allow_exact_matches': bool(allow_exact),
        'n_administrations_in': int(n_input),
        'n_no_timestamp': int(n_no_timestamp),
        'n_linked': int(len(linked)),
        'n_dropped_no_recent_score': int(len(merged) - len(linked)),
        'n_exact_timestamp_ties': n_exact_ties,
        'n_linked_if_strictly_prior': n_strict,
        'median_gap_minutes': (round(float(linked['gap_minutes'].median()), 2)
                               if len(linked) else None),
        'note': ('an administration with no pain assessment within the window '
                 'is EXCLUDED, not imputed; a score stamped in the same minute '
                 'counts as prior (gap 0) unless allow_exact_matches is false'),
    }
    logger.info('linked %d/%d administrations to a pain score within %d min '
                '(%d exact-minute ties; strictly-prior would keep %d)',
                stats['n_linked'], n_input, window_minutes, n_exact_ties,
                n_strict)
    return linked, stats


def session_bounds(admin_df):
    """(subject, session) -> session_start / session_end, from the MAR export.

    Doubles as the OBSERVABILITY index. An assessment can only be scored for
    "was a drug given after it" in a session that has a MAR export at all;
    without one, "no drug" is unobserved rather than false. Note this must come
    from the export EXISTING, not from it containing analgesic rows — three of
    the four sessions with no analgesic rows do have an export, and those are
    genuine "no analgesic given", not missing data.
    """
    return (admin_df.groupby(['subject', 'session'])
            [['session_start', 'session_end']].first())


def response_by_assessment(scores_df, analgesic_admin, bounds,
                           window_minutes=WINDOW_MINUTES,
                           exclude_clustered=False):
    """One row per observable assessment: which analgesics followed it.

    The mirror of `link_to_prior_score`. That one asks, of a dose, what the
    score was before it; this asks, of an assessment, whether a dose followed.
    Both are built on the SAME pairing so the two figures cannot disagree.

    THE ATTRIBUTION RULE, and why it settles the two-assessments-in-30-minutes
    problem. Each administration is attributed to its NEAREST PRECEDING
    assessment. That is identical to truncating each assessment's window at the
    next assessment — "the dose whose closest earlier assessment is this one"
    and "a dose inside (t_i, min(t_i + window, t_i+1))" select the same rows —
    so no dose is ever counted for two assessments and the percentages are a
    real partition. The cost is explicit: an assessment followed five minutes
    later by another assessment and then a dose reads as "no analgesic", on the
    grounds that the dose answered the LATER assessment. Measured on this
    corpus only 7.8% of assessments have another within 30 min (median gap to
    the next is 120 min), so the rule rarely bites; `exclude_clustered` drops
    those assessments entirely as a sensitivity check.

    Two exclusions, both because absence would otherwise be unearned:
    assessments in sessions with no MAR export (unobservable), and assessments
    whose window runs past `session_end` (right-censored — 0.5% here).

    Returns `(per_assessment, stats)`. `per_assessment` carries `drugs` (a
    sorted tuple, empty for none), `n_drugs`, `clustered` and
    `next_gap_minutes`; turning that into plot categories is the figure's job,
    not this function's.
    """
    n_input = len(scores_df)

    observable = scores_df.set_index(['subject', 'session']).index.isin(bounds.index)
    s = (scores_df[observable].set_index(['subject', 'session'])
         .join(bounds).reset_index())
    n_unobservable = n_input - len(s)

    s = s.sort_values(['subject', 'session', 'score_dt']).reset_index(drop=True)
    s['next_gap_minutes'] = (s.groupby(['subject', 'session'])['score_dt']
                             .shift(-1).sub(s['score_dt'])
                             .dt.total_seconds() / 60.0)
    s['clustered'] = s['next_gap_minutes'] <= window_minutes

    to_end = (s['session_end'] - s['score_dt']).dt.total_seconds() / 60.0
    censored = to_end < window_minutes
    n_censored = int(censored.sum())
    s = s[~censored].copy()

    n_clustered = int(s['clustered'].sum())
    if exclude_clustered:
        s = s[~s['clustered']].copy()

    linked, link_stats = link_to_prior_score(
        analgesic_admin, scores_df, window_minutes=window_minutes,
        allow_exact=True)
    grouped = (linked.groupby(['subject', 'session', 'score_dt'])['drug']
               .agg(drugs=lambda col: tuple(sorted(set(col))),
                    n_administrations='size')
               .reset_index())

    per = s.merge(grouped, on=['subject', 'session', 'score_dt'], how='left')
    per['drugs'] = per['drugs'].apply(
        lambda v: v if isinstance(v, tuple) else ())
    per['n_drugs'] = per['drugs'].apply(len)
    per['n_administrations'] = per['n_administrations'].fillna(0).astype(int)

    n_with_any = int((per['n_drugs'] > 0).sum())
    stats = {
        'window_minutes': window_minutes,
        'exclude_clustered': bool(exclude_clustered),
        'n_assessments_input': int(n_input),
        'n_dropped_session_has_no_mar': int(n_unobservable),
        'n_dropped_right_censored': n_censored,
        'n_clustered_within_window': n_clustered,
        'n_assessments_scored': int(len(per)),
        'n_with_any_analgesic': n_with_any,
        'frac_with_any_analgesic': (round(n_with_any / len(per), 4)
                                    if len(per) else None),
        'n_with_multiple_drugs': int((per['n_drugs'] > 1).sum()),
        'n_analgesic_administrations_linked': link_stats['n_linked'],
        'n_linked_admin_on_dropped_assessment': int(
            len(grouped) - (per['n_drugs'] > 0).sum()),
        'attribution': ('each administration attributed to its nearest '
                        'preceding assessment, equivalent to truncating each '
                        'assessment window at the next assessment; no dose is '
                        'counted twice'),
    }
    logger.info('%d/%d observable assessments followed by an analgesic within '
                '%d min (%.1f%%); dropped %d unobservable + %d censored',
                n_with_any, len(per), window_minutes,
                100 * n_with_any / len(per) if len(per) else 0,
                n_unobservable, n_censored)
    return per, stats


def counts_by_score(linked, drugs):
    """pain_score x drug administration counts on the FULL 0-10 scale.

    Reindexed to every score in range, so a score nobody was dosed at reads as
    an empty slot rather than vanishing and silently closing the gap — score 1
    is nearly empty here and that is a real feature of the data.
    """
    counts = (linked.pivot_table(index='pain_score', columns='drug',
                                 values='taken_dt', aggfunc='count')
              .reindex(range(PAIN_SCORE_MIN, PAIN_SCORE_MAX + 1))
              .reindex(columns=list(drugs))
              .fillna(0).astype(int))
    counts.index.name = 'pain_score'
    return counts


def per_drug_summary(linked, drugs):
    """One row per drug: n, and where its prior-score distribution sits."""
    rows = []
    for drug in drugs:
        g = linked[linked['drug'] == drug]
        if g.empty:
            rows.append({'drug': drug, 'n_linked': 0})
            continue
        scores = g['pain_score']
        rows.append({
            'drug': drug,
            'n_linked': len(g),
            'n_subjects': g['subject'].nunique(),
            'score_median': float(scores.median()),
            'score_q1': float(scores.quantile(0.25)),
            'score_q3': float(scores.quantile(0.75)),
            'score_mean': round(float(scores.mean()), 2),
            'frac_at_zero': round(float((scores == 0).mean()), 4),
            'frac_at_7_plus': round(float((scores >= 7).mean()), 4),
            'median_gap_minutes': round(float(g['gap_minutes'].median()), 2),
        })
    return pd.DataFrame(rows)
