"""Subject-level coded diagnoses from the EHR: is this patient an MDD case?

The companion to `med_state`. That module answers a question about an EPOCH --
was a dose given in the hours before this score -- and this one answers a
question about a PATIENT: does their record carry a diagnosis code for some
condition in a window before the admission.

THE STRUCTURAL DIFFERENCE, AND WHY IT DECIDES THE MODEL. `med_state` varies
epoch to epoch within a subject, so it decomposes into a within- and a
between-patient part and `FORMULA_MED_DECOMPOSED` needs that split. A diagnosis
does NOT vary within a subject. It is constant across every epoch that subject
contributes, which means:

  - it lives ENTIRELY in the between-subject space;
  - it has no within-subject main effect to estimate, so there is nothing to
    decompose and no analogue of `med_within`/`med_submean`;
  - the only thing it can do to the pain slope is MODULATE it, which is an
    interaction and nothing else.

So the model is `NRS_within * dx_state`, and the estimand is the interaction.
See `mixed_model.FORMULA_DX_INTERACTION`.

WHY NOT FIT THE TWO STRATA SEPARATELY. Two fits give two maps and no test that
they differ, and the arms here are unequal (17 MDD vs 34 not, on the reference
cohort), so the smaller arm carries systematically wider standard errors and
will look "less significant" everywhere from power alone. Reading that as a
group difference is the difference-of-significance fallacy. A stratified fit
also estimates its own four variance components per arm, which costs the five
regions whose MDD arm falls under `mixed_model.MIN_SUBJECTS`. One interaction
fit keeps all 51 subjects in the variance components and tests the difference
directly.

THE LABEL IS A BILLING ARTEFACT, NOT A DIAGNOSIS. This cannot be said too
plainly. These are administrative ICD codes scraped from an EHR, not structured
clinical assessments. On the reference cohort 11 of the 17 MDD-labelled subjects
are labelled by a `Professional Billing Code` alone. A billing code is generated
to justify reimbursement; it is entered by a coder, not by the treating
psychiatrist, and its presence or absence is driven by what was billable at that
encounter. `--dx-sources clinical` restricts to `Encounter Diagnosis`, `Problem`
and `Admit Diagnosis` and exists so that sensitivity can be measured rather than
asserted -- but it costs most of the arm, so it is a check, not the primary.

AND THE RECENCY WINDOW IS DOING REAL WORK. 17 subjects carry an MDD code in the
90 days before admission; 29 carry one at some point in the record. MDD is
substantially recurrent and problem-list entries are not re-dated, so "no code
in the last 90 days" is not evidence of no depression -- those 12 subjects go
into the CONTROL arm and dilute the contrast toward the null. Whatever window is
chosen, run `--dx-window-days 0` (meaning "ever") beside it and report both.

TIMESTAMPS ARE OFFSET-ANCHORED (docs/data_sop.md §5.2). Every subject's timeline
is shifted by a per-subject offset, so absolute dates are fiction and intervals
are true. A window expressed in days before `session_start` is therefore exactly
the kind of quantity that survives de-identification, and an absolute date is
exactly the kind that does not. Historical problem-list entries legitimately
appear decades before the session; that is the offset, not bad data.
"""

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ieeg_ehr import config

logger = logging.getLogger(__name__)

#: Days before `session_start` that count as "recent". The project's first MDD
#: run used 90 -- three months -- which is a clinical convention for "current
#: episode" rather than anything this dataset measures.
DEFAULT_WINDOW_DAYS = 90

#: `source` values present in the EHR export, enumerated from the data rather
#: than assumed: Professional Billing Code, Encounter Diagnosis, Admit
#: Diagnosis, Problem, Historical - HL7, External Injury Code.
#:
#: `clinical` is the subset a clinician touched during an encounter. It drops
#: `Professional Billing Code` (a coder's reimbursement entry) and
#: `Historical - HL7` (an interface import of unknown provenance, and the source
#: whose dates are most often decades stale).
SOURCE_SETS = {
    'any': None,
    'clinical': ('Encounter Diagnosis', 'Problem', 'Admit Diagnosis'),
    'billing': ('Professional Billing Code',),
}

# ICD prefixes, matched against the code with its dot stripped, so that both
# 'F32.1' and 'F321' and both '296.20' and '29620' resolve the same way. Prefix
# matching on the dotted string would need every subcode enumerated and would
# silently miss ICD-10's newer F32.A.
#
# THE BOUNDARIES MATTER AND ARE DELIBERATE:
#   F31  is bipolar        -- NOT depression, excluded
#   F32  MDD, single episode
#   F33  MDD, recurrent
#   F34.1 dysthymia / persistent depressive disorder -- a DIFFERENT disorder,
#         so it is in the broad set only
#   F41  is anxiety        -- NOT depression, excluded
#   296.2x / 296.3x  ICD-9 MDD single / recurrent
#   296.4+ is ICD-9 bipolar -- excluded, which is why the prefix stops at 3
#   311   ICD-9 'depressive disorder NEC' -- broad set only
#   300.4 ICD-9 dysthymia -- broad set only
CONDITIONS = {
    'mdd': {
        'label': 'MDD',
        'icd10': ('F32', 'F33'),
        'icd9': ('2962', '2963'),
        'description': 'Major depressive disorder, single episode or recurrent. '
                       'Excludes bipolar (F31, 296.4+), dysthymia (F34.1, 300.4) '
                       'and depressive disorder NEC (311).',
    },
    'depression_broad': {
        'label': 'depression (broad)',
        'icd10': ('F32', 'F33', 'F341'),
        'icd9': ('2962', '2963', '3004', '311'),
        'description': 'MDD plus dysthymia/persistent depressive disorder and '
                       'ICD-9 depressive disorder NEC. Still excludes bipolar.',
    },
}

#: The de-identified EHR tables, three CSVs per subject/session (data_sop.md §6).
EHR_SUBDIR = 'ehr'


def _norm_code(s):
    """'F32.1' -> 'F321'. Codes arrive dotted, undotted, and padded."""
    if not isinstance(s, str):
        return ''
    return re.sub(r'[^A-Z0-9]', '', s.upper())


def diagnoses_path(subject, session):
    """The per subject/session diagnoses CSV. `subject`/`session` are prefixed."""
    return (Path(config.RAW_DIR) / subject / session / EHR_SUBDIR
            / f'{subject}_{session}_diagnoses.csv')


def load_diagnoses(subjects=None):
    """Every diagnosis row for `subjects`, as one frame.

    `subjects` accepts either identifier form ('019' or 'sub-019'), matching
    `med_state.load_epoch_defs`, because cohort files and provenance use the
    bare form while paths use the prefixed one.

    `date` is parsed with `errors='coerce'` and the failures are COUNTED and
    logged rather than dropped silently: an unparseable date would otherwise
    fall out of every window and read as "this patient does not have the
    condition", which is the most dangerous way for this join to fail.
    """
    root = Path(config.RAW_DIR)
    if subjects is None:
        want = None
    else:
        want = {f"sub-{str(s).replace('sub-', '')}" for s in subjects}

    frames, missing = [], []
    for sub_dir in sorted(root.glob('sub-*')):
        if want is not None and sub_dir.name not in want:
            continue
        for ses_dir in sorted(sub_dir.glob('ses-*')):
            path = diagnoses_path(sub_dir.name, ses_dir.name)
            if not path.exists():
                missing.append(f'{sub_dir.name}/{ses_dir.name}')
                continue
            df = pd.read_csv(path, dtype=str)
            if df.empty:
                missing.append(f'{sub_dir.name}/{ses_dir.name} (empty)')
                continue
            df['subject_id'] = sub_dir.name
            df['session_id'] = ses_dir.name
            frames.append(df)

    if missing:
        logger.warning('%d subject-session(s) with no usable diagnoses table: %s',
                       len(missing), missing[:10])
    if not frames:
        raise FileNotFoundError(f'no diagnoses tables found under {root}')

    dx = pd.concat(frames, ignore_index=True)
    for col in ('date', 'session_start', 'session_end'):
        parsed = pd.to_datetime(dx[col], errors='coerce')
        n_bad = int(parsed.isna().sum())
        if n_bad:
            logger.warning('%d/%d rows have an unparseable %s', n_bad, len(dx), col)
        dx[col] = parsed

    dx['icd9_norm'] = dx['icd9_code'].map(_norm_code)
    dx['icd10_norm'] = dx['icd10_code'].map(_norm_code)
    dx['source'] = dx['source'].fillna('<none>')

    logger.info('diagnoses: %d rows, %d subjects, %d subject-sessions',
                len(dx), dx['subject_id'].nunique(),
                dx.groupby(['subject_id', 'session_id']).ngroups)
    return dx


def matches_condition(dx, condition):
    """Boolean Series: does each row code for `condition`?

    Matched on the NORMALIZED code only, never on the free-text description.
    Description matching is tempting and wrong here -- 'depression' appears in
    'ST segment depression' and in 'respiratory depression', both of which are
    common in this cohort and neither of which is a mood disorder.
    """
    if condition not in CONDITIONS:
        raise ValueError(f'unknown condition {condition!r}; '
                         f'known: {sorted(CONDITIONS)}')
    spec = CONDITIONS[condition]
    hit10 = dx['icd10_norm'].str.startswith(tuple(spec['icd10']), na=False)
    hit9 = dx['icd9_norm'].str.startswith(tuple(spec['icd9']), na=False)
    # A row carries at most one of the two coding systems, so this is a union,
    # not a double count.
    return hit10 | hit9


def subject_labels(dx, condition='mdd', window_days=DEFAULT_WINDOW_DAYS,
                   sources='any'):
    """Per-subject condition label, plus the per-session detail behind it.

    Returns `(labels, detail)`.

    `labels` is one row per SUBJECT: `subject_id, dx_state, n_codes,
    n_clinical_codes, n_billing_codes, first_days_before, sources`.
    `dx_state` is a bool. The model needs a subject-level constant because the
    variable is a subject-level constant; see the module docstring.

    `detail` is one row per subject-SESSION, kept because the window is anchored
    on that session's own `session_start` and a subject with two admissions has
    two windows. The rollup to subject is `any`, and a subject whose sessions
    DISAGREE is logged by name -- that is a real ambiguity in the label, not a
    rounding decision to bury.

    THE WINDOW is `(session_start - window_days, session_start]`. Left-open so a
    code exactly `window_days` old does not flicker on a boundary; right-closed
    at `session_start`, which excludes anything coded DURING the admission. That
    exclusion is the point: codes entered during the EMU stay are generated by
    the encounter the iEEG comes from, so letting them set the label would let
    the admission define its own predictor.

    `window_days=0` or `None` means EVER -- any code dated at or before
    `session_start`, with no lower bound.
    """
    if sources not in SOURCE_SETS:
        raise ValueError(f'unknown source set {sources!r}; '
                         f'known: {sorted(SOURCE_SETS)}')
    keep_sources = SOURCE_SETS[sources]
    spec = CONDITIONS[condition]

    d = dx.copy()
    if keep_sources is not None:
        d = d[d['source'].isin(keep_sources)]
    d = d[matches_condition(d, condition)]

    rows = []
    for (sid, ses), grp in dx.groupby(['subject_id', 'session_id'], sort=True):
        start = grp['session_start'].dropna()
        if start.empty:
            logger.warning('%s %s: no parseable session_start, cannot window; '
                           'labelled False', sid, ses)
            rows.append({'subject_id': sid, 'session_id': ses, 'dx_state': False,
                         'n_codes': 0, 'n_clinical_codes': 0, 'n_billing_codes': 0,
                         'first_days_before': np.nan, 'sources': '',
                         'no_session_start': True})
            continue
        t0 = start.iloc[0]
        cand = d[(d['subject_id'] == sid) & (d['session_id'] == ses)]
        in_win = cand['date'].notna() & (cand['date'] <= t0)
        if window_days:
            in_win &= cand['date'] > t0 - pd.Timedelta(days=float(window_days))
        hit = cand[in_win]
        days_before = ((t0 - hit['date']).dt.total_seconds() / 86400.0
                       if len(hit) else pd.Series(dtype=float))
        rows.append({
            'subject_id': sid, 'session_id': ses,
            'dx_state': bool(len(hit)),
            'n_codes': int(len(hit)),
            'n_clinical_codes': int(hit['source'].isin(
                SOURCE_SETS['clinical']).sum()),
            'n_billing_codes': int(hit['source'].isin(
                SOURCE_SETS['billing']).sum()),
            'first_days_before': float(days_before.max()) if len(hit) else np.nan,
            'sources': ';'.join(sorted(set(hit['source']))),
            'no_session_start': False,
        })

    detail = pd.DataFrame(rows)

    disagree = (detail.groupby('subject_id')['dx_state'].nunique() > 1)
    if disagree.any():
        names = sorted(disagree[disagree].index)
        logger.warning('%d subject(s) whose sessions DISAGREE on %s; rolled up '
                       'with `any`, so they enter the model as cases: %s',
                       len(names), spec['label'], names)

    labels = (detail.groupby('subject_id')
              .agg(dx_state=('dx_state', 'any'),
                   n_codes=('n_codes', 'sum'),
                   n_clinical_codes=('n_clinical_codes', 'sum'),
                   n_billing_codes=('n_billing_codes', 'sum'),
                   first_days_before=('first_days_before', 'max'),
                   n_sessions=('session_id', 'size'))
              .reset_index())
    labels['sources'] = labels['subject_id'].map(
        detail.groupby('subject_id')['sources']
        .apply(lambda s: ';'.join(sorted({x for v in s for x in v.split(';') if x}))))
    labels['condition'] = condition
    labels['window_days'] = window_days or 0
    labels['source_set'] = sources

    n_case = int(labels['dx_state'].sum())
    billing_only = int(((labels['dx_state']) & (labels['n_clinical_codes'] == 0)).sum())
    logger.info('%s over %s, sources=%s: %d/%d subjects are cases '
                '(%d of them by a BILLING CODE ALONE)', spec['label'],
                f'{window_days} d before session_start' if window_days else 'ever',
                sources, n_case, len(labels), billing_only)
    return labels, detail


def stratum_summary(labels, pain_by_subject=None):
    """Per stratum: how many subjects, and how the label was earned.

    `pain_by_subject` is an optional frame with `subject_id` and any numeric
    columns (mean NRS, n reports, ...); its columns are summarised per stratum.
    This table exists for the same reason `med_state.stratum_summary` does --
    to put the thing that could confound the contrast in front of the reader
    before any map is looked at. Here the first-order worry is the opposite one
    from medication's: if MDD patients simply REPORT more pain, a slope
    difference could be a difference in the predictor's range rather than in
    neural encoding.
    """
    rows = []
    for state, grp in labels.groupby('dx_state'):
        row = {'stratum': 'case' if state else 'control',
               'n_subjects': len(grp),
               'n_billing_only': (int((grp['n_clinical_codes'] == 0).sum())
                                  if state else 0),
               'median_codes': float(grp['n_codes'].median()),
               # All-NaN in the control arm BY CONSTRUCTION -- a control has no
               # matching code, so it has no age for one. Guarded rather than
               # left to emit an "empty slice" warning that looks like a bug.
               'median_days_before': (float(grp['first_days_before'].median())
                                      if grp['first_days_before'].notna().any()
                                      else np.nan)}
        if pain_by_subject is not None:
            sub = pain_by_subject[pain_by_subject['subject_id'].isin(grp['subject_id'])]
            for col in sub.select_dtypes('number').columns:
                row[f'{col}_mean'] = float(sub[col].mean())
                row[f'{col}_median'] = float(sub[col].median())
        rows.append(row)
    return pd.DataFrame(rows).sort_values('stratum').reset_index(drop=True)


def pain_by_subject(subjects=None):
    """Per-subject pain-report summary straight from the EHR pain-scores CSVs.

    Read from the SAME de-identified tables the epoch definitions were built
    from, rather than from the epoch cache, so this can be run and looked at
    without a view, a mask or a model. The numbers are therefore over ALL
    charted ratings, not only the ones that survived QC into an epoch -- which
    is the right denominator for "does this patient report more pain", and the
    wrong one for anything about power. Kept distinct deliberately.
    """
    root = Path(config.RAW_DIR)
    want = (None if subjects is None
            else {f"sub-{str(s).replace('sub-', '')}" for s in subjects})
    rows = []
    for sub_dir in sorted(root.glob('sub-*')):
        if want is not None and sub_dir.name not in want:
            continue
        scores = []
        for ses_dir in sorted(sub_dir.glob('ses-*')):
            p = (ses_dir / EHR_SUBDIR
                 / f'{sub_dir.name}_{ses_dir.name}_pain-scores.csv')
            if p.exists():
                scores.append(pd.read_csv(p))
        if not scores:
            continue
        s = pd.concat(scores, ignore_index=True)['max_pain'].dropna()
        if s.empty:
            continue
        rows.append({'subject_id': sub_dir.name, 'n_reports': int(len(s)),
                     'nrs_mean': float(s.mean()), 'nrs_median': float(s.median()),
                     'nrs_sd': float(s.std(ddof=1)) if len(s) > 1 else np.nan,
                     'nrs_max': float(s.max()), 'nrs_min': float(s.min()),
                     'frac_zero': float((s == 0).mean())})
    out = pd.DataFrame(rows)
    logger.info('pain summary over %d subjects, %d total ratings',
                len(out), int(out['n_reports'].sum()) if len(out) else 0)
    return out


def epoch_lookup(labels):
    """`labels` as the subject-level join key the model frames expect.

    A single `subject_id -> dx_state` float column. It is a FLOAT rather than a
    bool because patsy would otherwise build `C(dx_state)` levels named
    'True'/'False' and the interaction term's name would stop matching the
    plain-string lookups in `mixed_model.term_stats`.
    """
    return labels[['subject_id', 'dx_state']].assign(
        dx_state=labels['dx_state'].astype(float))
