"""Tests for the forward question: after an assessment, was a drug given?

The attribution rule is the load-bearing decision — two assessments inside one
window must not both claim the same dose — so it is pinned first and hardest.
"""

import numpy as np
import pandas as pd
import pytest

from ieeg_ehr.med_analysis import pain_link
from ieeg_ehr.med_analysis import plot_pain_score_response as psr

START = pd.Timestamp('2000-01-01 00:00')
END = pd.Timestamp('2000-01-08 00:00')


def scores(rows, subject='001', session='01'):
    return pd.DataFrame([
        {'subject': subject, 'session': session,
         'score_dt': pd.Timestamp(when), 'pain_score': float(score)}
        for when, score in rows])


def admin(rows, subject='001', session='01', end=END):
    return pd.DataFrame([
        {'subject': subject, 'session': session,
         'drug': drug, 'taken_dt': pd.Timestamp(when),
         'session_start': START, 'session_end': end}
        for when, drug in rows])


def bounds_for(admin_df):
    return pain_link.session_bounds(admin_df)


def run(score_rows, admin_rows, **kw):
    a = admin(admin_rows)
    s = scores(score_rows)
    return pain_link.response_by_assessment(s, a, bounds_for(a), **kw)


def test_dose_inside_window_is_attributed():
    per, stats = run([('2000-01-02 12:00', 7)],
                     [('2000-01-02 12:20', 'OXYCODONE')])
    assert len(per) == 1
    assert per['n_drugs'].iloc[0] == 1
    assert per['drugs'].iloc[0] == ('OXYCODONE',)
    assert stats['n_with_any_analgesic'] == 1


def test_dose_outside_window_is_not_attributed():
    per, stats = run([('2000-01-02 12:00', 7)],
                     [('2000-01-02 12:31', 'OXYCODONE')])
    assert per['n_drugs'].iloc[0] == 0
    assert stats['n_with_any_analgesic'] == 0


def test_dose_before_the_assessment_is_not_attributed():
    per, _ = run([('2000-01-02 12:00', 7)],
                 [('2000-01-02 11:50', 'OXYCODONE')])
    assert per['n_drugs'].iloc[0] == 0


def test_two_assessments_in_one_window_do_not_both_claim_the_dose():
    """THE case Sabra raised. The dose goes to the nearer (later) assessment."""
    per, stats = run([('2000-01-02 12:00', 3), ('2000-01-02 12:10', 8)],
                     [('2000-01-02 12:15', 'OXYCODONE')])
    per = per.sort_values('score_dt')
    assert list(per['pain_score']) == [3.0, 8.0]
    # Exactly one assessment claims it, and it is the later one.
    assert list(per['n_drugs']) == [0, 1]
    assert stats['n_with_any_analgesic'] == 1
    assert stats['n_clustered_within_window'] == 1


def test_clustered_assessments_can_be_excluded_as_a_sensitivity_check():
    per, stats = run([('2000-01-02 12:00', 3), ('2000-01-02 12:10', 8)],
                     [('2000-01-02 12:15', 'OXYCODONE')],
                     exclude_clustered=True)
    # The first assessment had a neighbour inside the window, so it goes.
    assert list(per['pain_score']) == [8.0]
    assert stats['exclude_clustered'] is True
    assert stats['n_assessments_scored'] == 1


def test_assessment_in_a_session_with_no_mar_export_is_unobservable():
    """"No drug given" there is unobserved, not false."""
    a = admin([('2000-01-02 12:20', 'OXYCODONE')], session='01')
    s = pd.concat([scores([('2000-01-02 12:00', 7)], session='01'),
                   scores([('2000-01-02 12:00', 9)], session='02')],
                  ignore_index=True)
    per, stats = pain_link.response_by_assessment(s, a, bounds_for(a))
    assert stats['n_dropped_session_has_no_mar'] == 1
    assert list(per['pain_score']) == [7.0]


def test_assessment_whose_window_runs_past_session_end_is_censored():
    a = admin([('2000-01-02 12:20', 'OXYCODONE')],
              end=pd.Timestamp('2000-01-02 12:20'))
    s = scores([('2000-01-02 12:00', 7)])
    per, stats = pain_link.response_by_assessment(s, a, bounds_for(a))
    assert stats['n_dropped_right_censored'] == 1
    assert per.empty


def test_two_distinct_drugs_are_one_assessment_with_two_drugs():
    per, stats = run([('2000-01-02 12:00', 7)],
                     [('2000-01-02 12:05', 'OXYCODONE'),
                      ('2000-01-02 12:10', 'ACETAMINOPHEN')])
    assert per['n_drugs'].iloc[0] == 2
    assert per['n_administrations'].iloc[0] == 2
    assert stats['n_with_multiple_drugs'] == 1


def test_a_dose_never_crosses_sessions():
    a = admin([('2000-01-02 12:20', 'OXYCODONE')], session='02')
    s = scores([('2000-01-02 12:00', 7)], session='02')
    extra = scores([('2000-01-02 12:00', 4)], session='01')
    a2 = pd.concat([a, admin([], session='01')], ignore_index=True)
    per, _ = pain_link.response_by_assessment(
        pd.concat([s, extra], ignore_index=True), a2, bounds_for(a2))
    got = dict(zip(per['pain_score'], per['n_drugs']))
    assert got[7.0] == 1


def test_categorize_maps_named_other_multi_and_none():
    per = pd.DataFrame([
        {'n_drugs': 1, 'drugs': ('OXYCODONE',)},
        {'n_drugs': 1, 'drugs': ('MORPHINE',)},
        {'n_drugs': 2, 'drugs': ('OXYCODONE', 'ACETAMINOPHEN')},
        {'n_drugs': 0, 'drugs': ()},
    ])
    out = psr.categorize(per, ['OXYCODONE', 'ACETAMINOPHEN'])
    assert list(out['category']) == ['OXYCODONE', psr.OTHER, psr.MULTI, psr.NONE]


def test_percentages_sum_to_100_within_each_assessed_score():
    per = pd.DataFrame([
        {'pain_score': 7, 'score_dt': START, 'n_drugs': 1,
         'drugs': ('OXYCODONE',)},
        {'pain_score': 7, 'score_dt': START, 'n_drugs': 0, 'drugs': ()},
        {'pain_score': 3, 'score_dt': START, 'n_drugs': 0, 'drugs': ()},
    ])
    cat = psr.categorize(per, ['OXYCODONE'])
    counts, pct, totals = psr.response_table(cat, ['OXYCODONE'])
    assert totals.loc[7] == 2
    assert pct.loc[7].sum() == pytest.approx(100.0)
    assert pct.loc[7, 'OXYCODONE'] == pytest.approx(50.0)
    assert pct.loc[3, psr.NONE] == pytest.approx(100.0)


def test_a_score_with_no_assessments_is_nan_not_zero():
    """A 0% bar would claim it was measured; it was not."""
    per = pd.DataFrame([{'pain_score': 7, 'score_dt': START, 'n_drugs': 0,
                         'drugs': ()}])
    cat = psr.categorize(per, ['OXYCODONE'])
    counts, pct, totals = psr.response_table(cat, ['OXYCODONE'])
    assert totals.loc[10] == 0
    assert np.isnan(pct.loc[10, psr.NONE])
    assert list(counts.index) == list(range(0, 11))
