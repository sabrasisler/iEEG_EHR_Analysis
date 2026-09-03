"""Tests for linking a medication administration to the pain score before it.

The window and the same-minute rule are the two decisions this figure rests on,
so they are pinned here rather than left to the one call site.
"""

import pandas as pd
import pytest

from ieeg_ehr.med_analysis import pain_link


def admin(taken, subject='001', session='01', drug='OXYCODONE'):
    return pd.DataFrame([{
        'subject': subject, 'session': session, 'drug': drug,
        'taken_dt': pd.Timestamp(taken),
    }])


def scores(rows, subject='001', session='01'):
    return pd.DataFrame([
        {'subject': subject, 'session': session,
         'score_dt': pd.Timestamp(when), 'pain_score': float(score)}
        for when, score in rows])


def test_score_inside_window_is_linked():
    linked, stats = pain_link.link_to_prior_score(
        admin('2000-01-01 12:00'), scores([('2000-01-01 11:45', 7)]))
    assert len(linked) == 1
    assert linked['pain_score'].iloc[0] == 7
    assert linked['gap_minutes'].iloc[0] == pytest.approx(15.0)
    assert stats['n_linked'] == 1
    assert stats['n_dropped_no_recent_score'] == 0


def test_score_outside_window_drops_the_administration():
    """The window is an inclusion criterion, not a NaN to handle downstream."""
    linked, stats = pain_link.link_to_prior_score(
        admin('2000-01-01 12:00'), scores([('2000-01-01 11:29', 9)]))
    assert linked.empty
    assert stats['n_linked'] == 0
    assert stats['n_dropped_no_recent_score'] == 1


def test_window_boundary_is_inclusive():
    linked, _ = pain_link.link_to_prior_score(
        admin('2000-01-01 12:00'), scores([('2000-01-01 11:30', 5)]))
    assert len(linked) == 1
    assert linked['gap_minutes'].iloc[0] == pytest.approx(30.0)


def test_same_minute_score_counts_as_prior_by_default():
    """45% of real administrations are exactly this case; see pain_link.__doc__."""
    linked, stats = pain_link.link_to_prior_score(
        admin('2000-01-01 12:00'), scores([('2000-01-01 12:00', 8)]))
    assert len(linked) == 1
    assert linked['pain_score'].iloc[0] == 8
    assert linked['gap_minutes'].iloc[0] == 0
    assert stats['n_exact_timestamp_ties'] == 1


def test_strict_prior_excludes_the_same_minute_score():
    linked, stats = pain_link.link_to_prior_score(
        admin('2000-01-01 12:00'), scores([('2000-01-01 12:00', 8)]),
        allow_exact=False)
    assert linked.empty
    assert stats['n_linked'] == 0


def test_stats_report_the_strict_count_even_when_exact_is_allowed():
    """The audit trail has to make the choice checkable from the artifact."""
    _, stats = pain_link.link_to_prior_score(
        admin('2000-01-01 12:00'), scores([('2000-01-01 12:00', 8)]))
    assert stats['n_linked'] == 1
    assert stats['n_linked_if_strictly_prior'] == 0


def test_score_after_the_dose_is_never_linked():
    linked, _ = pain_link.link_to_prior_score(
        admin('2000-01-01 12:00'), scores([('2000-01-01 12:10', 10)]))
    assert linked.empty


def test_most_recent_eligible_score_wins():
    linked, _ = pain_link.link_to_prior_score(
        admin('2000-01-01 12:00'),
        scores([('2000-01-01 11:40', 2), ('2000-01-01 11:55', 6)]))
    assert linked['pain_score'].iloc[0] == 6


def test_a_score_from_another_subject_does_not_link():
    linked, _ = pain_link.link_to_prior_score(
        admin('2000-01-01 12:00', subject='001'),
        scores([('2000-01-01 11:50', 7)], subject='002'))
    assert linked.empty


def test_a_score_from_another_session_does_not_link():
    """`taken_dt` is only comparable within a session — dates are re-anchored."""
    linked, _ = pain_link.link_to_prior_score(
        admin('2000-01-01 12:00', session='01'),
        scores([('2000-01-01 11:50', 7)], session='02'))
    assert linked.empty


def test_administration_without_a_timestamp_is_counted_and_dropped():
    # Built in one frame rather than concatenated: an all-NaT column in a
    # concat operand trips a pandas dtype FutureWarning that has nothing to do
    # with what is being tested.
    a = pd.DataFrame([
        {'subject': '001', 'session': '01', 'drug': 'OXYCODONE',
         'taken_dt': pd.Timestamp('2000-01-01 12:00')},
        {'subject': '001', 'session': '01', 'drug': 'OXYCODONE',
         'taken_dt': pd.NaT},
    ])
    linked, stats = pain_link.link_to_prior_score(
        a, scores([('2000-01-01 11:50', 4)]))
    assert stats['n_administrations_in'] == 2
    assert stats['n_no_timestamp'] == 1
    assert stats['n_linked'] == 1


def test_counts_by_score_spans_the_full_scale():
    """Score 1 is nearly empty in the real data; it must not close the gap."""
    linked, _ = pain_link.link_to_prior_score(
        admin('2000-01-01 12:00'), scores([('2000-01-01 11:50', 7)]))
    counts = pain_link.counts_by_score(linked, ['OXYCODONE'])
    assert list(counts.index) == list(range(0, 11))
    assert counts.loc[7, 'OXYCODONE'] == 1
    assert counts.loc[0, 'OXYCODONE'] == 0
    assert counts.to_numpy().sum() == 1


def test_counts_by_score_keeps_a_requested_drug_with_no_rows():
    linked, _ = pain_link.link_to_prior_score(
        admin('2000-01-01 12:00'), scores([('2000-01-01 11:50', 7)]))
    counts = pain_link.counts_by_score(linked, ['OXYCODONE', 'MORPHINE'])
    assert list(counts.columns) == ['OXYCODONE', 'MORPHINE']
    assert counts['MORPHINE'].sum() == 0


def test_out_of_range_score_raises_rather_than_clipping(tmp_path):
    """A score off the 0-10 scale means the export changed; do not guess."""
    path = tmp_path / 'sub-001_ses-01_pain-scores.csv'
    pd.DataFrame([
        {'sub_id': 'sub-001', 'ses_id': 'ses-01',
         'date': '2000-01-01 12:00:00', 'max_pain': 11.0},
    ]).to_csv(path, index=False)
    with pytest.raises(ValueError, match='outside 0-10'):
        pain_link.load_pain_scores(paths=[path])


def test_per_drug_summary_reports_position_on_the_scale():
    a = pd.concat([admin('2000-01-01 12:00'), admin('2000-01-01 14:00')],
                  ignore_index=True)
    s = scores([('2000-01-01 11:50', 2), ('2000-01-01 13:50', 8)])
    linked, _ = pain_link.link_to_prior_score(a, s)
    summary = pain_link.per_drug_summary(linked, ['OXYCODONE'])
    assert summary['n_linked'].iloc[0] == 2
    assert summary['score_median'].iloc[0] == pytest.approx(5.0)
    assert summary['frac_at_7_plus'].iloc[0] == pytest.approx(0.5)
