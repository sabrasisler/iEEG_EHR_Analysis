"""
Recorded iEEG hours, per subject-session and per hospital day.

This is the denominator for every rate in Fig 3. Getting it wrong does not
produce an error, it produces a plausible wrong number, so the provenance of
each session's hours is carried as a column rather than assumed.

THE PROBLEM WITH THE FILE REGISTRY
----------------------------------
`sherlock_file_registry.csv` has one row per run with `start_datetime` /
`end_datetime`, and those timestamps sit on the same de-identified epoch as the
MAR table (verified: sub-019's first run starts 2000-01-01 15:59:39, which is
exactly its MAR `session_start`). But the registry only populates timing for
runs that have a PREPROCESSED file — every one of the 2,136 null-timestamp rows
has `has_preprocessed == False`. So registry timing measures *preprocessed*
coverage, not *recorded* coverage.

Across the 98 medication-cohort sessions: 41 fully timestamped, 41 partial, 16
with no timestamped run at all. Gap-aware coverage over what is timestamped
totals 9,900 h against 13,048 h of session span. Where the registry IS complete,
the coverage/span ratio has a median of 0.952 — real recording gaps, but modest.

WHAT THIS MODULE DOES
---------------------
Per session, in order of preference:

1. `registry` — merge the timestamped runs into a gap-aware coverage union.
   Used when the session has at least `MIN_TIMESTAMPED_RUNS` timestamped runs.
2. `session_span` — fall back to MAR `session_start` -> `session_end`.

The fallback exists because the alternative is worse: a session with no
timestamped run contributes administrations but zero hours, which either drops
its subject from the denominator or divides by zero. Sixteen sessions and ~1,600
hours turn on this. The `method` column makes the mixture visible in the output
table, and `coverage_report()` prints the split so it can never be silent.

Neither is the ideal denominator. The correct one reads `session_start_time` +
`starting_time` + `rate` + data shape straight from the raw NWB files (all four
are present — checked), giving true gap-aware coverage for every session. That
is a separate extraction job over ~7,900 files; see TASKS.md. Until then a rate
computed here is accurate to a few percent, not exact, and the caption should
not claim otherwise.
"""

import logging

import pandas as pd

from ieeg_ehr import config

logger = logging.getLogger(__name__)

#: A session needs at least this many timestamped runs before its registry
#: coverage is trusted over the session span. One timestamped run out of eighty
#: is not "gap-aware coverage", it is a two-hour sliver.
MIN_TIMESTAMPED_RUNS = 2

#: Below this fraction of runs timestamped, the registry union is so incomplete
#: that the span is the better estimate even though it overcounts.
MIN_TIMESTAMPED_FRACTION = 0.5

METHOD_REGISTRY = 'registry'
METHOD_SPAN = 'session_span'


def _merge_intervals(intervals):
    """Union of [(start, end), ...]. Assumes nothing about input order."""
    merged = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _registry_intervals():
    """(subject, session) -> [(start, end), ...] for timestamped runs, plus the
    timestamped/total run counts per session."""
    reg = pd.read_csv(config.FILE_REGISTRY_CSV,
                      usecols=['sub_id', 'ses_id', 'run_id',
                               'start_datetime', 'end_datetime'])
    reg['subject'] = reg['sub_id'].str.replace('sub-', '', regex=False)
    reg['session'] = reg['ses_id'].str.replace('ses-', '', regex=False)
    reg['start'] = pd.to_datetime(reg['start_datetime'], errors='coerce')
    reg['end'] = pd.to_datetime(reg['end_datetime'], errors='coerce')

    n_runs = reg.groupby(['subject', 'session']).size()
    timed = reg.dropna(subset=['start', 'end'])
    n_timed = timed.groupby(['subject', 'session']).size()

    intervals = {key: list(zip(g['start'], g['end']))
                 for key, g in timed.groupby(['subject', 'session'])}
    return intervals, n_runs, n_timed


def session_coverage(admin_df):
    """One row per (subject, session): the recorded-time intervals and method.

    Args:
        admin_df: the tidy administration table, which supplies `session_start`
            and `session_end` for the fallback.

    Returns a DataFrame with columns subject, session, method, n_runs,
    n_runs_timestamped, hours, span_hours, and an `intervals` object column of
    [(start, end), ...] used by `hours_by_day`.
    """
    intervals_by_session, n_runs, n_timed = _registry_intervals()

    sessions = (admin_df.groupby(['subject', 'session'], as_index=False)
                .agg(session_start=('session_start', 'first'),
                     session_end=('session_end', 'first')))

    rows = []
    for t in sessions.itertuples():
        key = (t.subject, t.session)
        total = int(n_runs.get(key, 0))
        timed = int(n_timed.get(key, 0))
        span = [(t.session_start, t.session_end)]
        span_h = max((t.session_end - t.session_start).total_seconds() / 3600.0, 0.0)

        use_registry = (timed >= MIN_TIMESTAMPED_RUNS
                        and total > 0
                        and timed / total >= MIN_TIMESTAMPED_FRACTION)
        if use_registry:
            merged = _merge_intervals(intervals_by_session[key])
            method = METHOD_REGISTRY
        else:
            merged = _merge_intervals(span)
            method = METHOD_SPAN

        hours = sum((e - s).total_seconds() for s, e in merged) / 3600.0
        rows.append(dict(subject=t.subject, session=t.session, method=method,
                         n_runs=total, n_runs_timestamped=timed,
                         hours=hours, span_hours=span_h, intervals=merged))

    return pd.DataFrame(rows)


def coverage_report(coverage):
    """Log the method split and the span overcount. Never let this be silent."""
    by_method = coverage['method'].value_counts().to_dict()
    reg = coverage[coverage['method'] == METHOD_REGISTRY]
    ratio = (reg['hours'] / reg['span_hours']).median() if len(reg) else float('nan')
    logger.info('recorded hours: %d sessions | %s | total %.0f h',
                len(coverage), by_method, coverage['hours'].sum())
    logger.info('  of the %d registry-based sessions, gap-aware coverage is a '
                'median %.3f of session span; the %d span-based sessions '
                'therefore overstate monitoring by roughly that much',
                len(reg), ratio, int(by_method.get(METHOD_SPAN, 0)))
    return {'n_sessions': int(len(coverage)),
            'n_registry': int(by_method.get(METHOD_REGISTRY, 0)),
            'n_session_span': int(by_method.get(METHOD_SPAN, 0)),
            'total_hours': float(coverage['hours'].sum()),
            'total_span_hours': float(coverage['span_hours'].sum()),
            'median_coverage_over_span': None if pd.isna(ratio) else float(ratio)}


def _split_interval_by_day(start, end, anchor):
    """[(hospital_day, hours), ...] for one interval.

    `anchor` is midnight of the session's own start date, matching
    `load.hospital_day`. Walking midnight to midnight rather than dividing by 24
    is what makes a partial day 0 come out as a partial day rather than being
    charged a full one.
    """
    pieces = []
    cursor = start
    while cursor < end:
        day = (cursor.normalize() - anchor).days
        next_midnight = cursor.normalize() + pd.Timedelta(days=1)
        piece_end = min(end, next_midnight)
        pieces.append((day, (piece_end - cursor).total_seconds() / 3600.0))
        cursor = piece_end
    return pieces


def hours_by_day(coverage, admin_df):
    """Long table: subject, session, hospital_day, hours.

    A day appears only if it has recorded time. That distinction carries real
    weight downstream: a monitored day with no dose is a zero, an unmonitored day
    is absent entirely, and collapsing the two would turn missing data into
    evidence of no dosing.
    """
    anchors = (admin_df.groupby(['subject', 'session'])['session_start']
               .first().dt.normalize())

    rows = []
    for t in coverage.itertuples():
        anchor = anchors.get((t.subject, t.session))
        if anchor is None:
            continue
        per_day = {}
        for start, end in t.intervals:
            for day, hours in _split_interval_by_day(start, end, anchor):
                per_day[day] = per_day.get(day, 0.0) + hours
        for day, hours in sorted(per_day.items()):
            rows.append(dict(subject=t.subject, session=t.session,
                             hospital_day=day, hours=hours))

    return pd.DataFrame(rows, columns=['subject', 'session', 'hospital_day',
                                       'hours'])


def subject_hours_by_day(coverage, admin_df):
    """subject x hospital_day -> hours, summed over that subject's sessions."""
    long = hours_by_day(coverage, admin_df)
    if long.empty:
        return long
    return (long.groupby(['subject', 'hospital_day'], as_index=False)['hours']
            .sum())
