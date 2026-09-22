"""Tests for medication exposure around pain epochs.

The exposure join is deliberately many-to-many (the opposite of `pain_link`'s
unique attribution), the deviation is centred per SESSION rather than per
subject, and the paired comparison must drop one-armed subjects. Those three
are what these tests pin.
"""

import pandas as pd
import pytest

from ieeg_ehr.med_analysis import epoch_meds

T0 = pd.Timestamp('2000-01-02 12:00')


def epochs(rows):
    return pd.DataFrame([
        {'epoch_id': i, 'subject': r.get('subject', '001'),
         'session': r.get('session', '01'), 'pain_time': pd.Timestamp(r['t']),
         'pain_score': float(r['score'])}
        for i, r in enumerate(rows)])


def admin(rows):
    return pd.DataFrame([
        {'subject': r.get('subject', '001'), 'session': r.get('session', '01'),
         'drug': r.get('drug', 'OXYCODONE'), 'level2': 'Opioids',
         'taken_dt': pd.Timestamp(r['t'])}
        for r in rows])


def test_dose_inside_the_window_marks_the_epoch_dosed():
    per, pairs = epoch_meds.exposure_before_epochs(
        epochs([{'t': T0, 'score': 7}]),
        admin([{'t': '2000-01-02 10:30'}]), window_hours=2)
    assert bool(per['dosed'].iloc[0])
    assert per['n_doses'].iloc[0] == 1
    assert pairs['gap_minutes'].iloc[0] == pytest.approx(90.0)


def test_dose_just_outside_the_window_does_not():
    per, _ = epoch_meds.exposure_before_epochs(
        epochs([{'t': T0, 'score': 7}]),
        admin([{'t': '2000-01-02 09:59'}]), window_hours=2)
    assert not bool(per['dosed'].iloc[0])
    assert per['n_doses'].iloc[0] == 0


def test_dose_after_the_anchor_does_not_count():
    per, _ = epoch_meds.exposure_before_epochs(
        epochs([{'t': T0, 'score': 7}]),
        admin([{'t': '2000-01-02 12:01'}]), window_hours=2)
    assert not bool(per['dosed'].iloc[0])


def test_one_dose_may_precede_several_overlapping_epochs():
    """Exposure, not attribution: epochs overlap, so this is NOT double count."""
    per, _ = epoch_meds.exposure_before_epochs(
        epochs([{'t': '2000-01-02 11:00', 'score': 5},
                {'t': '2000-01-02 12:00', 'score': 7}]),
        admin([{'t': '2000-01-02 10:30'}]), window_hours=2)
    assert list(per['dosed']) == [True, True]


def test_several_doses_before_one_epoch_are_all_counted():
    per, _ = epoch_meds.exposure_before_epochs(
        epochs([{'t': T0, 'score': 7}]),
        admin([{'t': '2000-01-02 10:30'},
               {'t': '2000-01-02 11:30', 'drug': 'ACETAMINOPHEN'}]),
        window_hours=2)
    assert per['n_doses'].iloc[0] == 2
    assert per['n_drugs'].iloc[0] == 2


def test_a_dose_never_crosses_subjects_or_sessions():
    per, _ = epoch_meds.exposure_before_epochs(
        epochs([{'t': T0, 'score': 7, 'subject': '001'}]),
        admin([{'t': '2000-01-02 11:30', 'subject': '002'}]), window_hours=2)
    assert not bool(per['dosed'].iloc[0])

    per, _ = epoch_meds.exposure_before_epochs(
        epochs([{'t': T0, 'score': 7, 'session': '01'}]),
        admin([{'t': '2000-01-02 11:30', 'session': '02'}]), window_hours=2)
    assert not bool(per['dosed'].iloc[0])


def test_drug_filter_restricts_what_counts_as_exposure():
    per, _ = epoch_meds.exposure_before_epochs(
        epochs([{'t': T0, 'score': 7}]),
        admin([{'t': '2000-01-02 11:30', 'drug': 'MORPHINE'}]),
        window_hours=2, drugs=['OXYCODONE'])
    assert not bool(per['dosed'].iloc[0])


def test_deviation_is_centred_on_the_SESSION_not_the_subject():
    """Two sessions are separate admissions with independently shifted clocks."""
    per = epochs([{'t': T0, 'score': 2, 'session': '01'},
                  {'t': T0, 'score': 4, 'session': '01'},
                  {'t': T0, 'score': 8, 'session': '02'},
                  {'t': T0, 'score': 10, 'session': '02'}])
    out = epoch_meds.subject_deviation(per)
    assert list(out['session_mean_pain']) == [3.0, 3.0, 9.0, 9.0]
    assert list(out['pain_deviation']) == [-1.0, 1.0, -1.0, 1.0]


def test_paired_by_subject_needs_both_arms():
    per = pd.DataFrame([
        {'subject': 'A', 'dosed': True, 'pain_score': 6.0},
        {'subject': 'A', 'dosed': False, 'pain_score': 2.0},
        {'subject': 'B', 'dosed': True, 'pain_score': 5.0},
    ])
    paired, dropped = epoch_meds.paired_by_subject(per, 'pain_score')
    assert list(paired['subject']) == ['A']
    assert paired['difference'].iloc[0] == pytest.approx(4.0)
    assert dropped == 1


def test_paired_by_subject_averages_within_each_arm():
    per = pd.DataFrame([
        {'subject': 'A', 'dosed': True, 'pain_score': 6.0},
        {'subject': 'A', 'dosed': True, 'pain_score': 8.0},
        {'subject': 'A', 'dosed': False, 'pain_score': 1.0},
    ])
    paired, _ = epoch_meds.paired_by_subject(per, 'pain_score')
    assert paired['dosed'].iloc[0] == pytest.approx(7.0)
    assert paired['n_dosed'].iloc[0] == 2
    assert paired['n_undosed'].iloc[0] == 1


def test_restrict_to_observable_drops_sessions_with_no_mar():
    e = epochs([{'t': T0, 'score': 7, 'session': '01'},
                {'t': T0, 'score': 3, 'session': '02'}])
    a = admin([{'t': T0, 'session': '01'}])
    kept, dropped = epoch_meds.restrict_to_observable(e, a)
    assert dropped == 1
    assert list(kept['session']) == ['01']
