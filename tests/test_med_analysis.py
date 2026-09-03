"""Tests for the medication administration pipeline.

Two halves, deliberately separated:

- Pure-function tests (name cleaning, taxonomy, dedup, interval algebra, the
  combination-product guard). No Oak, milliseconds.
- `@pytest.mark.slow` tests that read the real MAR export and check the loader
  against totals the colleague's benzodiazepine analysis published independently:
  98 files, 7,340 rows, 6,919 unique administrations after collapsing 421
  multi-product duplicates, 380 benzodiazepines, 0 unmatched drug names. That is
  a strong end-to-end check — it validates name cleaning, the dedup key, and the
  taxonomy join in one shot against numbers this code did not produce.
"""
import numpy as np
import pandas as pd
import pytest

from ieeg_ehr.config import med_taxonomy
from ieeg_ehr.med_analysis import load, plot_coadmin_peth, plot_hospital_day
from ieeg_ehr.med_analysis import recording_hours

pytestmark = []


# ============================================================================
# name cleaning
# ============================================================================

@pytest.mark.parametrize('raw,expected', [
    ('ACETAMINOPHEN 325 MG PO TABS', 'ACETAMINOPHEN'),
    ('OXYCODONE 5 MG PO TABS', 'OXYCODONE'),
    # The cut is on whitespace tokens, so a hyphenated combination name survives
    # intact and only the "5-325" strength is removed.
    ('OXYCODONE-ACETAMINOPHEN 5-325 MG PO TABS', 'OXYCODONE-ACETAMINOPHEN'),
    ('HYDROCODONE-ACETAMINOPHEN 10-325 MG PO TABS', 'HYDROCODONE-ACETAMINOPHEN'),
    ('BUTALBITAL-ACETAMINOPHEN-CAFF 50-325-40 MG PO TABS',
     'BUTALBITAL-ACETAMINOPHEN-CAFF'),
    # All three fentanyl products consolidate onto one drug.
    ('FENTANYL (PF) 50 MCG/ML INJECTION', 'FENTANYL'),
    ('FENTANYL CITRATE (PF) 50 MCG/ML INJ SOLN (WRAPPER RECORD)', 'FENTANYL'),
    ('FENTANYL 100 MCG/2 ML (SHC PHARMACY-COMPOUNDED SYR)', 'FENTANYL'),
    # No numeric token at all: consolidation is what merges this with
    # "MORPHINE 2 MG/ML INJ SYRG".
    ('MORPHINE INJECTABLE SYRINGE', 'MORPHINE'),
    ('MORPHINE 2 MG/ML INJ SYRG', 'MORPHINE'),
    ('KETOROLAC 30 MG/ML (1 ML) INJ SOLN', 'KETOROLAC'),
    ('BUPRENORPHINE-NALOXONE 2-0.5 MG SL FILM', 'BUPRENORPHINE-NALOXONE'),
])
def test_extract_medication_name(raw, expected):
    assert load.extract_medication_name(raw) == expected


def test_extract_name_does_not_cut_on_embedded_digits():
    """"VITAMIN B-12" must not be cut at "B-12" — the token has no leading digit."""
    assert load.extract_medication_name(
        'CYANOCOBALAMIN (VITAMIN B-12) 1000 MCG') == 'CYANOCOBALAMIN (VITAMIN B-12)'


# ============================================================================
# taxonomy
# ============================================================================

def test_analgesic_subclasses_are_populated():
    for subclass in med_taxonomy.ANALGESIC_SUBCLASSES:
        drugs = med_taxonomy.drugs_in_subclasses([subclass])
        assert drugs, f'{subclass} has no members'


def test_non_opioid_analgesics_are_classified():
    """The source table leaves these unclassified; that is the gap we filled."""
    assert med_taxonomy.classify('ACETAMINOPHEN') == ('Analgesics', 'Acetaminophen')
    assert med_taxonomy.classify('KETOROLAC') == ('Analgesics', 'NSAIDs')
    assert med_taxonomy.classify('IBUPROFEN') == ('Analgesics', 'NSAIDs')


def test_combination_opioids_classify_as_opioids_not_acetaminophen():
    """This is what keeps the Fig 4 acetaminophen column single-ingredient."""
    for drug in ('HYDROCODONE-ACETAMINOPHEN', 'OXYCODONE-ACETAMINOPHEN',
                 'ACETAMINOPHEN-CODEINE'):
        assert med_taxonomy.classify(drug) == ('Analgesics', 'Opioids')
        assert drug in med_taxonomy.COMBINATION_DRUGS


def test_unknown_drug_raises_rather_than_dropping():
    with pytest.raises(KeyError, match='not in the medication taxonomy'):
        med_taxonomy.classify('NOT A REAL DRUG')


def test_known_but_unclassified_drug_returns_empty():
    assert med_taxonomy.classify('PANTOPRAZOLE') == ('', '')


def test_anesthetics_excluded_from_analgesic_set():
    assert 'PROPOFOL' not in med_taxonomy.ANALGESIC_DRUGS
    assert 'LIDOCAINE' not in med_taxonomy.ANALGESIC_DRUGS
    assert 'PROPOFOL' in med_taxonomy.ANESTHETIC_DRUGS


def test_coadmin_class_splits_anxiolytics():
    assert med_taxonomy.coadmin_class('Anxiolytics', 'Benzodiazepines') == \
        'Benzodiazepines'
    assert med_taxonomy.coadmin_class(
        'Anxiolytics', 'Gabapentinoids (Calcium channel alpha2delta ligands)') == \
        'Gabapentinoids'
    assert med_taxonomy.coadmin_class('Analgesics', 'Opioids') == 'Opioids'
    assert med_taxonomy.coadmin_class('', '') is None


# ============================================================================
# unit safety
# ============================================================================

def test_assert_single_unit_rejects_mixed():
    df = pd.DataFrame({'dose_unit': ['mg', 'tablet']})
    with pytest.raises(ValueError, match='mixes units'):
        load.assert_single_unit(df, 'test')


def test_assert_single_unit_accepts_uniform():
    df = pd.DataFrame({'dose_unit': ['mcg', 'mcg']})
    assert load.assert_single_unit(df, 'test') == 'mcg'


# ============================================================================
# interval algebra / hospital day
# ============================================================================

def test_merge_intervals_unions_overlaps_and_keeps_gaps():
    t = pd.Timestamp('2000-01-01 00:00')
    h = pd.Timedelta(hours=1)
    merged = recording_hours._merge_intervals([
        (t, t + 2 * h), (t + h, t + 3 * h),      # overlapping -> one
        (t + 5 * h, t + 6 * h),                  # separate -> kept
    ])
    assert merged == [(t, t + 3 * h), (t + 5 * h, t + 6 * h)]


def test_split_interval_by_day_splits_at_midnight():
    anchor = pd.Timestamp('2000-01-01')
    pieces = recording_hours._split_interval_by_day(
        pd.Timestamp('2000-01-01 22:00'), pd.Timestamp('2000-01-02 03:00'), anchor)
    assert pieces == [(0, 2.0), (1, 3.0)]


def test_hospital_day_is_anchored_per_session():
    """Day 0 tracks the session's own start date, not a global constant."""
    when = pd.Series(pd.to_datetime(['2000-01-05 10:00', '2000-01-07 10:00']))
    start = pd.Series(pd.to_datetime(['2000-01-05 08:00', '2000-01-05 08:00']))
    assert list(load.hospital_day(when, start)) == [0, 2]

    # A session shifted off the usual epoch still starts at day 0, which is the
    # whole point of anchoring per-session.
    when = pd.Series(pd.to_datetime(['1999-12-31 23:00']))
    start = pd.Series(pd.to_datetime(['1999-12-31 20:00']))
    assert list(load.hospital_day(when, start)) == [0]


def test_normalize_by_personal_max():
    series = {'a': {0: 5.0, 1: 10.0}, 'b': {0: 2.0}, 'c': {0: 0.0}}
    norm, skipped = plot_hospital_day.normalize_by_personal_max(series)
    assert norm['a'] == {0: 0.5, 1: 1.0}
    assert norm['b'] == {0: 1.0}
    assert 'c' in skipped        # personal max of 0 cannot be normalized


def test_mean_sem_by_day_reports_contributing_n():
    series = {'a': {0: 1.0, 1: 0.5}, 'b': {0: 0.5}}
    out = plot_hospital_day.mean_sem_by_day(series, [0, 1, 2])
    assert list(out['hospital_day']) == [0, 1]        # day 2 has no data at all
    assert out.loc[0, 'mean'] == pytest.approx(0.75)
    assert list(out['n_subjects']) == [2, 1]


def test_interval_table_stays_within_session():
    """A gap between two sessions is not an inter-dose interval."""
    admin = pd.DataFrame({
        'subject': ['01', '01', '01'],
        'session': ['01', '01', '02'],
        'drug': ['OXYCODONE'] * 3,
        'route': ['Oral'] * 3,
        'taken_dt': pd.to_datetime(['2000-01-01 08:00', '2000-01-01 14:00',
                                    '2000-01-01 20:00']),
    })
    from ieeg_ehr.med_analysis.plot_admin_timing import interval_table
    out = interval_table(admin)
    assert len(out) == 1
    assert out['interval_h'].iloc[0] == pytest.approx(6.0)


# ============================================================================
# PETH
# ============================================================================

def test_combination_leak_guard_fires():
    bad = pd.DataFrame({'coadmin_class': ['Acetaminophen'],
                        'is_combination': [True],
                        'drug': ['HYDROCODONE-ACETAMINOPHEN']})
    with pytest.raises(AssertionError, match='zero bin by definition'):
        plot_coadmin_peth._assert_no_combination_leak(bad, ['Acetaminophen'])


def test_peth_counts_each_anchor_once_per_bin():
    """Two co-administrations of one class in one bin count once, not twice."""
    admin = pd.DataFrame({
        'subject': ['01'] * 3,
        'session': ['01'] * 3,
        'drug': ['OXYCODONE', 'LEVETIRACETAM', 'LAMOTRIGINE'],
        'route': ['Oral'] * 3,
        'taken_dt': pd.to_datetime(['2000-01-01 08:00', '2000-01-01 08:05',
                                    '2000-01-01 08:10']),
        'coadmin_class': ['Opioids', 'Anticonvulsants', 'Anticonvulsants'],
    })
    counts, hist = plot_coadmin_peth.aggregate(
        admin, [('OXYCODONE', 'Oral', 'PO Oxycodone')], ['Anticonvulsants'])
    assert counts['PO Oxycodone'] == 1
    zero_bin = np.argmin(np.abs(plot_coadmin_peth.BIN_CENTERS))
    assert hist['PO Oxycodone']['Anticonvulsants'][zero_bin] == 1.0


def test_peth_excludes_the_anchor_from_its_own_class():
    admin = pd.DataFrame({
        'subject': ['01'], 'session': ['01'], 'drug': ['OXYCODONE'],
        'route': ['Oral'], 'taken_dt': pd.to_datetime(['2000-01-01 08:00']),
        'coadmin_class': ['Opioids'],
    })
    counts, hist = plot_coadmin_peth.aggregate(
        admin, [('OXYCODONE', 'Oral', 'PO Oxycodone')], ['Opioids'])
    assert counts['PO Oxycodone'] == 1
    assert hist['PO Oxycodone']['Opioids'].sum() == 0.0


# ============================================================================
# end-to-end against the real export
# ============================================================================

@pytest.mark.slow
def test_loader_reproduces_published_corpus_totals():
    """Independent check: these numbers come from the colleague's analysis."""
    admin = load.load_administrations()
    assert len(admin) == 6919, 'unique administrations after multi-product collapse'
    assert admin['subject'].nunique() == 96
    assert admin.groupby(['subject', 'session']).ngroups == 98
    benzos = admin[admin['level2'] == 'Benzodiazepines']
    assert len(benzos) == 380


@pytest.mark.slow
def test_analgesic_set_is_stable():
    admin = load.load_administrations(
        subclasses=med_taxonomy.ANALGESIC_SUBCLASSES)
    assert len(admin) == 1754
    assert dict(admin['level2'].value_counts()) == {
        'Opioids': 1062, 'Acetaminophen': 643, 'NSAIDs': 49}
    per_drug = admin['drug'].value_counts()
    assert per_drug['FENTANYL'] == 94, 'three fentanyl products consolidate'
    assert per_drug['MORPHINE'] == 25, 'two morphine products consolidate'
    # Every analgesic administration has a resolvable dose in a single unit per
    # (drug, route) — the guard Fig 1 and Fig 3 both rely on.
    for (drug, route), g in admin.groupby(['drug', 'route']):
        load.assert_single_unit(g, f'{drug}/{route}')


@pytest.mark.slow
def test_hospital_days_are_non_negative():
    """Per-session anchoring must not produce a negative day for any session."""
    admin = load.load_administrations()
    assert admin['hospital_day'].min() >= 0


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
