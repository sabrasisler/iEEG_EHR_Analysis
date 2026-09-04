"""Unit tests for the per-epoch analgesic-state join.

Synthetic timestamps only -- the point is the window arithmetic, which is where
an off-by-one is silent and would quietly move epochs between strata.
"""

import numpy as np
import pandas as pd
import pytest

from ieeg_ehr.analysis import med_state


def _defs(times, subject='001', session='01'):
    return pd.DataFrame({
        'subject': subject, 'session': session,
        'epoch_id': range(len(times)),
        'pain_score': np.arange(len(times), dtype=float),
        'pain_time': pd.to_datetime(list(times)),
    })


def _admin(times, subject='001', session='01'):
    return pd.DataFrame({
        'subject': subject, 'session': session,
        'taken_dt': pd.to_datetime(list(times)),
    })


def test_dose_inside_window_is_flagged():
    defs = _defs(['2000-01-01 12:00'])
    admin = _admin(['2000-01-01 11:00'])
    out = med_state.epoch_med_state(defs, admin, hours=2.0)
    assert bool(out['med_state'].iloc[0]) is True
    assert out['n_admins'].iloc[0] == 1
    assert out['minutes_since_last'].iloc[0] == pytest.approx(60.0)


def test_dose_outside_window_is_not_flagged():
    defs = _defs(['2000-01-01 12:00'])
    admin = _admin(['2000-01-01 09:00'])          # 3 h before a 2 h window
    out = med_state.epoch_med_state(defs, admin, hours=2.0)
    assert bool(out['med_state'].iloc[0]) is False
    assert out['n_admins'].iloc[0] == 0
    assert np.isnan(out['minutes_since_last'].iloc[0])


def test_window_is_half_open_exclusive_at_the_far_edge():
    """A dose EXACTLY `hours` before the score is outside.

    The window is (t - hours, t]. Pinned because the boundary is arbitrary but
    has to be stable: a run that silently flips it would move epochs between
    strata and change every map without changing any code that looks relevant.
    """
    defs = _defs(['2000-01-01 12:00'])
    admin = _admin(['2000-01-01 10:00'])          # exactly 2 h
    out = med_state.epoch_med_state(defs, admin, hours=2.0)
    assert bool(out['med_state'].iloc[0]) is False


def test_same_minute_dose_counts_as_prior():
    """Charting is minute-resolution; a zero gap is 'just before', not 'after'.

    Same reasoning as med_analysis.pain_link, which documents that 45% of
    administrations carry a score stamped in the same minute.
    """
    defs = _defs(['2000-01-01 12:00'])
    admin = _admin(['2000-01-01 12:00'])
    out = med_state.epoch_med_state(defs, admin, hours=2.0)
    assert bool(out['med_state'].iloc[0]) is True
    assert out['minutes_since_last'].iloc[0] == pytest.approx(0.0)


def test_dose_after_the_score_is_not_flagged():
    defs = _defs(['2000-01-01 12:00'])
    admin = _admin(['2000-01-01 12:30'])
    out = med_state.epoch_med_state(defs, admin, hours=2.0)
    assert bool(out['med_state'].iloc[0]) is False


def test_minutes_since_last_uses_the_most_recent_dose():
    defs = _defs(['2000-01-01 12:00'])
    admin = _admin(['2000-01-01 10:30', '2000-01-01 11:45'])
    out = med_state.epoch_med_state(defs, admin, hours=2.0)
    assert out['n_admins'].iloc[0] == 2
    assert out['minutes_since_last'].iloc[0] == pytest.approx(15.0)


def test_session_with_no_administrations_is_false_not_missing():
    """'No analgesic was given' is an observation, not absent data."""
    defs = _defs(['2000-01-01 12:00', '2000-01-01 15:00'])
    admin = _admin([])
    out = med_state.epoch_med_state(defs, admin, hours=2.0)
    assert list(out['med_state']) == [False, False]
    assert out['n_admins'].sum() == 0


def test_doses_do_not_leak_across_subjects_or_sessions():
    defs = pd.concat([_defs(['2000-01-01 12:00'], subject='001'),
                      _defs(['2000-01-01 12:00'], subject='002')],
                     ignore_index=True)
    admin = _admin(['2000-01-01 11:30'], subject='001')
    out = med_state.epoch_med_state(defs, admin, hours=2.0)
    got = dict(zip(out['subject'], out['med_state']))
    assert got['001'] is np.True_ or bool(got['001']) is True
    assert bool(got['002']) is False


def test_window_length_changes_membership():
    defs = _defs(['2000-01-01 12:00'])
    admin = _admin(['2000-01-01 09:30'])          # 2.5 h before
    assert not bool(med_state.epoch_med_state(defs, admin, hours=2.0)
                    ['med_state'].iloc[0])
    assert bool(med_state.epoch_med_state(defs, admin, hours=4.0)
                ['med_state'].iloc[0])


def test_stratum_summary_reports_both_strata_and_the_nrs_gap():
    defs = _defs(['2000-01-01 12:00', '2000-01-01 20:00'])
    defs['pain_score'] = [8.0, 1.0]
    admin = _admin(['2000-01-01 11:00'])
    summary = med_state.stratum_summary(
        med_state.epoch_med_state(defs, admin, hours=2.0))
    assert set(summary['stratum']) == {'medicated', 'unmedicated'}
    med = summary.set_index('stratum').loc['medicated']
    unmed = summary.set_index('stratum').loc['unmedicated']
    assert med['nrs_mean'] > unmed['nrs_mean']
    assert med['n_epochs'] == 1 and unmed['n_epochs'] == 1


def test_drug_sets_are_named_and_opioids_is_a_subset():
    assert set(med_state.DRUG_SETS) == {'analgesics', 'opioids'}
    assert set(med_state.DRUG_SETS['opioids']) < set(med_state.DRUG_SETS['analgesics'])
