"""Per-epoch analgesic state: was a dose given in the N hours before this score?

The pain-epoch analyses inherit a QC mask that is SIGNAL QUALITY ONLY. Their own
METHODS.md names analgesia as the largest untested confound, because a drug given
shortly before a pain assessment plausibly changes low-frequency power directly.
This module is the join that makes that testable.

THE JOIN. Epoch definitions carry `pain_time`, the assessment timestamp, and the
medication table carries `taken_dt`. Both come from the same EHR export and share
the same de-identified anchor -- verified: pain_time spans 1999-12-31 17:39 ->
2000-01-15 07:40 and taken_dt spans 1999-12-31 18:22 -> 2000-01-15 05:45. So an
epoch's state is just "does any administration fall in (pain_time - hours,
pain_time]".

THE WINDOW IS ANCHORED ON THE ASSESSMENT, NOT THE EPOCH. The epoch is the 5
minutes of recording BEFORE the score, so its start is 5 minutes earlier than
`pain_time`. Anchoring on the assessment is the right choice because the
assessment is the clinical event the dose is timed against; a 5-minute shift on a
2-hour window would also be a false precision, since charting is
minute-resolution and the charted time is when it was written down.

WHAT THIS IS NOT -- carried from `med_analysis/pain_link.py`, which makes the
same point about the opposite direction of the join. A dose preceding an
assessment does not make the dose the CAUSE of anything the assessment records,
and an assessment following a dose is not a measurement of that dose's effect.
Scheduled drugs are given at fixed times whatever the score says, and an
assessment is often charted precisely BECAUSE a PRN dose was requested. The arrow
can point either way and this table cannot separate them.

AND THE BIG ONE. Patients are medicated BECAUSE they are in pain, so the two
strata differ by ~2.1 NRS points on this cohort (4.15 medicated vs 2.01 not).
Confounding by indication is not a caveat you can put in a footnote and move past
-- any difference between a medicated and unmedicated map confounds pharmacology
with pain severity and with the predictor range the slope was estimated over.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ieeg_ehr import config
from ieeg_ehr.config import med_taxonomy

logger = logging.getLogger(__name__)

#: The tidy administration table built 2026-09-03: 1754 administrations, 93
#: subjects, drugs already classified into level1/level2 by `med_taxonomy`.
ADMIN_TABLE = (config.ANALYSIS_DIR / 'meds' / 'administration_patterns'
               / 'admin_table' / 'admin_table_20260903-145332'
               / 'administrations.csv')

#: Hours before the assessment that count as "recently medicated".
DEFAULT_WINDOW_HOURS = 2.0

#: Named drug sets. `analgesics` is every class in the table; `opioids` is the
#: pharmacologically cleanest subset but costs a third of the medicated epochs.
#:
#: `non_opioid_analgesics` is the COMPLEMENT of `opioids` within `analgesics`,
#: and exists so that "opioid" and "non-opioid" can be compared. Running
#: `analgesics` beside `opioids` does NOT give that comparison: analgesics is a
#: superset that CONTAINS opioids, so the two differ by the presence of
#: acetaminophen and NSAIDs rather than by opioid exposure. Only this set makes
#: the contrast a partition. It is also the arm where a central mechanism is
#: least expected, which makes it the closer thing to a negative control for a
#: broadband amplitude artifact.
DRUG_SETS = {
    'analgesics': med_taxonomy.ANALGESIC_SUBCLASSES,
    'opioids': ('Opioids',),
    'non_opioid_analgesics': tuple(s for s in med_taxonomy.ANALGESIC_SUBCLASSES
                                   if s != 'Opioids'),
}


def load_admin_table(path=ADMIN_TABLE, subclasses=None):
    """Administrations, optionally restricted to `level2` subclasses.

    `subject`/`session` are read as strings: they are zero-padded identifiers
    ('019', '01'), and letting pandas infer them turns 019 into 19 and breaks
    every join downstream in a way that looks like missing data.
    """
    path = Path(path)
    admin = pd.read_csv(path, dtype={'subject': str, 'session': str},
                        parse_dates=['taken_dt'])
    if subclasses is not None:
        subclasses = tuple(subclasses)
        unknown = set(subclasses) - set(admin['level2'].unique())
        if unknown:
            raise ValueError(f'no administrations for subclass(es) {sorted(unknown)}; '
                             f'table has {sorted(admin["level2"].unique())}')
        admin = admin[admin['level2'].isin(subclasses)]
    logger.info('administrations: %d rows, %d subjects, subclasses %s',
                len(admin), admin['subject'].nunique(),
                sorted(admin['level2'].unique()))
    return admin.sort_values(['subject', 'session', 'taken_dt']).reset_index(drop=True)


def load_epoch_defs(epoch_minutes=None, subjects=None):
    """Every epoch definition, as one frame with bare subject/session ids.

    `subjects` accepts either form ('019' or 'sub-019') because both are in use
    across the project -- cohort files and provenance use the bare form, paths
    use the prefixed one.
    """
    defs_dir = config.pain_epoch_unit_dir(epoch_minutes) / config.EPOCH_DEFS_SUBDIR
    paths = sorted(defs_dir.glob('sub-*_defs.parquet'))
    if not paths:
        raise FileNotFoundError(f'no epoch definitions in {defs_dir}')

    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    df['subject'] = df['subject_id'].str.replace('sub-', '', regex=False)
    df['session'] = df['session_id'].str.replace('ses-', '', regex=False)

    if subjects is not None:
        want = {str(s).replace('sub-', '') for s in subjects}
        df = df[df['subject'].isin(want)]

    if df['pain_time'].isna().any():
        raise ValueError(f'{int(df["pain_time"].isna().sum())} epoch(s) with no '
                         'pain_time; the medication join has no anchor for them')

    logger.info('epoch definitions: %d files, %d epochs, %d subjects',
                len(paths), len(df), df['subject'].nunique())
    return df[['subject', 'session', 'epoch_id', 'pain_score', 'pain_time']]


def epoch_med_state(defs, admin, hours=DEFAULT_WINDOW_HOURS):
    """Per epoch: was anything in `admin` given in the `hours` before its score?

    Returns `subject, session, epoch_id, med_state, n_admins, minutes_since_last`.

    `minutes_since_last` is kept even though the split only needs the boolean:
    it makes a dose-timing sensitivity analysis (does the effect depend on how
    recent the dose was?) possible later without re-reading anything, and it is
    free to compute here.

    An epoch in a (subject, session) with NO administrations at all is
    `med_state=False`, not missing -- "no analgesic was given" is an observation,
    not an absence of one. A session missing from the MAR export entirely would
    be a different thing, and the caller checks cohort coverage rather than
    letting it silently become a stratum of unmedicated epochs.
    """
    window = pd.Timedelta(hours=hours)
    out = []

    for (subject, session), grp in defs.groupby(['subject', 'session'], sort=False):
        doses = admin.loc[(admin['subject'] == subject)
                          & (admin['session'] == session), 'taken_dt'].to_numpy()
        times = grp['pain_time'].to_numpy()
        n = np.zeros(len(grp), dtype=int)
        gap = np.full(len(grp), np.nan)

        if doses.size:
            lo = times - window.to_timedelta64()
            for i, (a, b) in enumerate(zip(lo, times)):
                # (lo, hi] -- a dose stamped in the same minute as the score
                # counts, matching pain_link's reasoning that minute-resolution
                # charting makes a zero gap "just before" rather than "after".
                hit = doses[(doses > a) & (doses <= b)]
                n[i] = hit.size
                if hit.size:
                    gap[i] = (b - hit.max()) / np.timedelta64(1, 'm')

        out.append(pd.DataFrame({
            'subject': subject, 'session': session,
            'epoch_id': grp['epoch_id'].to_numpy(),
            'pain_score': grp['pain_score'].to_numpy(),
            'med_state': n > 0, 'n_admins': n, 'minutes_since_last': gap,
        }))

    state = pd.concat(out, ignore_index=True)
    logger.info('med state over %.1f h: %d/%d epochs medicated (%.1f%%)',
                hours, int(state['med_state'].sum()), len(state),
                100 * state['med_state'].mean())
    return state


def stratum_summary(state):
    """Per stratum: epochs, subjects, and the NRS distribution.

    This table exists to make the confound impossible to overlook. Medication is
    given BECAUSE of pain, so the strata differ systematically in the predictor,
    and any downstream comparison inherits that difference.
    """
    rows = []
    for med, grp in state.groupby('med_state'):
        rows.append({
            'stratum': 'medicated' if med else 'unmedicated',
            'n_epochs': len(grp), 'n_subjects': grp['subject'].nunique(),
            'nrs_mean': grp['pain_score'].mean(), 'nrs_sd': grp['pain_score'].std(),
            'nrs_median': grp['pain_score'].median(),
            'nrs_min': grp['pain_score'].min(), 'nrs_max': grp['pain_score'].max(),
        })
    return pd.DataFrame(rows).sort_values('stratum').reset_index(drop=True)
