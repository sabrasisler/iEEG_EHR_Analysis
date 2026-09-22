"""
Medication exposure around a PAIN EPOCH.

The first thing in this package whose unit is a pain epoch rather than a drug
administration, so it is also the first that has to obey the cohort rule:
epochs are the neural analysis unit, and `CLAUDE.md` makes the hold-out
UNREACHABLE by default. `load_epochs` therefore takes a split and defaults to
`discovery`.

WHY THE JOIN IS EXACT. The epoch definitions carry `pain_time`, the wall-clock
timestamp of the pain assessment an epoch is anchored to, and it matches the
`date` column of the pain-score export exactly. So an epoch can be placed
against `taken_dt` directly — no run start times, and none of the incomplete
`sherlock_file_registry` timing that makes Fig 3's denominator approximate.
That is worth stating because it is the one place in this project where the
neural and EHR clocks meet without an approximation in between.

WHAT "DOSED" MEANS. An epoch is dosed if at least one analgesic was given in
the `window_hours` BEFORE the assessment the epoch is anchored to — measured
to `pain_time`, not to the epoch's 5-minute start, because the question is
about exposure the patient was under when the score was given. A dose stamped
in the same minute as the assessment counts (gap 0); this is the same
minute-resolution charting argument as `pain_link`.

NOT CAUSAL. A dose before an epoch does not make the epoch's pain score a
response to it, and a higher score in dosed epochs does not mean the drug
failed — the patient in more pain is the one who gets dosed. Confounding by
indication runs through every figure built on this module, and it runs in the
direction that makes analgesia look harmful.
"""

import glob
import logging

import pandas as pd

from ieeg_ehr import config
from ieeg_ehr.config import cohorts, med_taxonomy
from ieeg_ehr.med_analysis import load

logger = logging.getLogger(__name__)

#: Exposure window before the assessment, in hours.
WINDOW_HOURS = 2.0

DEFS_COLUMNS = ('epoch_id', 'subject_id', 'session_id', 'pain_event_id',
                'pain_score', 'pain_time')


def load_epochs(split='discovery', minutes_before=None):
    """Every pain epoch in `split`, one row each, with its wall-clock anchor.

    `split='discovery'` by default and `'heldout'` is not a legal value —
    `cohorts.subjects_for_split` raises on it — so an exploratory figure
    cannot reach the hold-out by forgetting a flag.
    """
    pattern = str(config.pain_epoch_unit_dir(minutes_before) / 'epoch_defs'
                  / '*_defs.parquet')
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f'no epoch definitions at {pattern}')

    defs = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    missing = set(DEFS_COLUMNS) - set(defs.columns)
    if missing:
        raise ValueError(f'epoch defs missing columns {sorted(missing)}')

    defs['subject'] = defs['subject_id'].str.replace('sub-', '', regex=False)
    defs['session'] = defs['session_id'].str.replace('ses-', '', regex=False)
    defs['pain_time'] = pd.to_datetime(defs['pain_time'], errors='coerce')
    defs['pain_score'] = pd.to_numeric(defs['pain_score'], errors='coerce')

    n_all = len(defs)
    allowed = set(cohorts.subjects_for_split(
        split, available=sorted(defs['subject'].unique())))
    defs = defs[defs['subject'].isin(allowed)].copy()

    logger.info('epochs: %d of %d in split %r — %d subjects, %d sessions',
                len(defs), n_all, split, defs['subject'].nunique(),
                defs.groupby(['subject', 'session']).ngroups)
    return defs.sort_values(['subject', 'session', 'pain_time']).reset_index(
        drop=True)


def load_analgesics():
    """Every analgesic administration, with its session bounds."""
    admin = load.load_administrations(paths=config.med_admin_files())
    return admin[admin['level2'].isin(
        set(med_taxonomy.ANALGESIC_SUBCLASSES))].copy()


def restrict_to_observable(epochs, admin):
    """Drop epochs whose session has no MAR export.

    Without an export "no drug" is unobserved rather than false, exactly as in
    `pain_link.response_by_assessment`. Observability comes from the export
    EXISTING, so this takes the full administration frame, not the analgesic
    subset — a session with an export and no analgesics is a real zero.
    """
    have = set(map(tuple, admin[['subject', 'session']].drop_duplicates()
                   .to_numpy()))
    keep = epochs.set_index(['subject', 'session']).index.isin(have)
    n_dropped = int((~keep).sum())
    if n_dropped:
        logger.warning('%d epoch(s) dropped: no MAR export for the session',
                       n_dropped)
    return epochs[keep].copy(), n_dropped


def exposure_before_epochs(epochs, analgesics, window_hours=WINDOW_HOURS,
                           drugs=None):
    """Per epoch: which analgesics were given in the window before its anchor.

    Returns `(per_epoch, per_epoch_drug)`. `per_epoch` gains `n_doses`,
    `n_drugs` and `dosed`; `per_epoch_drug` is the long form, one row per
    (epoch, drug) actually present, which is what a per-drug figure needs and
    what would be wrong to reconstruct by re-joining.

    This is a many-to-many interval join rather than a `merge_asof`: an epoch
    can have several doses in two hours and a dose can precede several epochs
    (the epochs overlap in time). Both are correct here — the question is
    exposure, not attribution, so nothing needs to be assigned uniquely. That
    is the opposite of `pain_link`, where double-counting WOULD be a bug, and
    the difference is worth keeping straight.
    """
    window = pd.Timedelta(hours=window_hours)
    cols = ['subject', 'session', 'drug', 'level2', 'taken_dt']
    adm = analgesics.dropna(subset=['taken_dt'])[cols]
    if drugs is not None:
        adm = adm[adm['drug'].isin(set(drugs))]

    pairs = epochs[['epoch_id', 'subject', 'session', 'pain_time']].merge(
        adm, on=['subject', 'session'], how='inner')
    gap = pairs['pain_time'] - pairs['taken_dt']
    pairs = pairs[(gap >= pd.Timedelta(0)) & (gap <= window)].copy()
    pairs['gap_minutes'] = gap[pairs.index].dt.total_seconds() / 60.0

    key = ['epoch_id', 'subject', 'session']
    agg = (pairs.groupby(key)
           .agg(n_doses=('drug', 'size'), n_drugs=('drug', 'nunique'))
           .reset_index())

    per_epoch = epochs.merge(agg, on=key, how='left')
    per_epoch['n_doses'] = per_epoch['n_doses'].fillna(0).astype(int)
    per_epoch['n_drugs'] = per_epoch['n_drugs'].fillna(0).astype(int)
    per_epoch['dosed'] = per_epoch['n_doses'] > 0

    logger.info('%d/%d epochs dosed within %g h before the anchor (%.1f%%)',
                int(per_epoch['dosed'].sum()), len(per_epoch), window_hours,
                100 * per_epoch['dosed'].mean() if len(per_epoch) else 0)
    return per_epoch, pairs


def subject_deviation(per_epoch):
    """Add each epoch's pain relative to its SESSION's mean pain.

    Session, not subject: two sessions of one subject are re-anchored
    independently by de-identification and are separate admissions clinically,
    so pooling them would centre on an average of two different baselines.
    """
    out = per_epoch.copy()
    session_mean = out.groupby(['subject', 'session'])['pain_score'].transform(
        'mean')
    out['session_mean_pain'] = session_mean
    out['pain_deviation'] = out['pain_score'] - session_mean
    return out


def paired_by_subject(per_epoch, value_col):
    """Per subject: the mean of `value_col` in dosed and in undosed epochs.

    Only subjects with at least one epoch on BOTH sides are returned — a
    subject contributing a single arm carries no within-subject contrast, and
    keeping them would silently turn a paired comparison into an unpaired one.
    """
    grouped = (per_epoch.groupby(['subject', 'dosed'])[value_col]
               .agg(['mean', 'size']).reset_index())
    wide = grouped.pivot(index='subject', columns='dosed', values='mean')
    sizes = grouped.pivot(index='subject', columns='dosed', values='size')
    wide = wide.rename(columns={True: 'dosed', False: 'undosed'})
    sizes = sizes.rename(columns={True: 'n_dosed', False: 'n_undosed'})

    for col in ('dosed', 'undosed'):
        if col not in wide.columns:
            wide[col] = float('nan')
    for col in ('n_dosed', 'n_undosed'):
        if col not in sizes.columns:
            sizes[col] = 0

    out = wide.join(sizes).reset_index()
    out['n_dosed'] = out['n_dosed'].fillna(0).astype(int)
    out['n_undosed'] = out['n_undosed'].fillna(0).astype(int)
    both = out['dosed'].notna() & out['undosed'].notna()
    n_dropped = int((~both).sum())
    if n_dropped:
        logger.info('%d subject(s) have epochs on only one side and cannot '
                    'contribute a within-subject contrast', n_dropped)
    out = out[both].copy()
    out['difference'] = out['dosed'] - out['undosed']
    return out.reset_index(drop=True), n_dropped


def dose_probability_by_score(epochs, analgesics, window_hours=WINDOW_HOURS):
    """Per (subject, pain score): P(a dose follows within the window).

    The FORWARD direction, and the one place in this module where attribution
    matters: with several assessments inside one window a dose must not be
    credited to all of them, so each dose goes to its NEAREST PRECEDING
    assessment — identical to truncating each window at the next assessment.
    Same rule and the same argument as `pain_link.response_by_assessment`,
    just anchored on epochs and at a wider window.
    """
    from ieeg_ehr.med_analysis import pain_link

    anchors = (epochs[['subject', 'session', 'pain_time', 'pain_score']]
               .drop_duplicates()
               .rename(columns={'pain_time': 'score_dt'})
               .sort_values(['subject', 'session', 'score_dt'])
               .reset_index(drop=True))

    linked, _ = pain_link.link_to_prior_score(
        analgesics, anchors, window_minutes=int(window_hours * 60),
        allow_exact=True)
    responded = set(map(tuple, linked[['subject', 'session', 'score_dt']]
                        .drop_duplicates().to_numpy()))

    anchors['responded'] = [
        (s, ses, t) in responded
        for s, ses, t in zip(anchors['subject'], anchors['session'],
                             anchors['score_dt'])]

    per = (anchors.groupby(['subject', 'pain_score'])['responded']
           .agg(n_assessments='size', n_responded='sum').reset_index())
    per['p_dose'] = per['n_responded'] / per['n_assessments']
    return per
