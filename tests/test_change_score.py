"""Unit tests for the paired change-score construction.

The gap window and the both-epochs-present rule are where a silent bug would
live: either one, done wrong, biases the difference toward zero without
producing an error.
"""

import numpy as np
import pandas as pd
import pytest

from ieeg_ehr.analysis import change_score as cs


def _defs(times, scores, subject='001', session='01'):
    return pd.DataFrame({
        'subject': subject, 'session': session,
        'epoch_id': range(len(times)),
        'pain_score': list(scores),
        'pain_time': pd.to_datetime(list(times)),
    })


def _admin(times, routes=None, subject='001', session='01'):
    times = list(times)
    return pd.DataFrame({
        'subject': subject, 'session': session,
        'taken_dt': pd.to_datetime(times),
        'route': routes if routes is not None else ['Oral'] * len(times),
    })


def test_pair_within_window_is_kept_with_correct_delta():
    defs = _defs(['2000-01-01 12:00', '2000-01-01 13:00'], [3, 7])
    p = cs.build_pairs(defs, _admin([]), 30, 240)
    assert len(p) == 1
    r = p.iloc[0]
    assert r['gap_m'] == pytest.approx(60.0)
    assert r['pain_1'] == 3 and r['pain_2'] == 7
    assert r['d_pain'] == pytest.approx(4.0)
    assert r['med_between'] == 0.0


def test_gap_below_minimum_is_excluded():
    """The overlap guard: 5-min epochs 20 min apart are fine, 10 min apart are not."""
    defs = _defs(['2000-01-01 12:00', '2000-01-01 12:10'], [3, 4])
    with pytest.raises(ValueError):
        cs.build_pairs(defs, _admin([]), 30, 240)
    assert len(cs.build_pairs(defs, _admin([]), 5, 240)) == 1


def test_gap_above_maximum_is_excluded():
    defs = _defs(['2000-01-01 12:00', '2000-01-01 20:00'], [3, 4])
    with pytest.raises(ValueError):
        cs.build_pairs(defs, _admin([]), 30, 240)


def test_window_bounds_are_inclusive():
    defs = _defs(['2000-01-01 12:00', '2000-01-01 12:30'], [3, 4])
    assert len(cs.build_pairs(defs, _admin([]), 30, 240)) == 1


def test_only_consecutive_pairs_are_formed():
    """A skipping pair would hide an assessment's dose history inside itself."""
    defs = _defs(['2000-01-01 12:00', '2000-01-01 13:00', '2000-01-01 14:00'],
                 [2, 5, 1])
    p = cs.build_pairs(defs, _admin([]), 30, 240)
    assert len(p) == 2
    assert set(zip(p.e1, p.e2)) == {(0, 1), (1, 2)}


def test_dose_between_is_flagged_and_routes_counted():
    defs = _defs(['2000-01-01 12:00', '2000-01-01 14:00'], [8, 4])
    admin = _admin(['2000-01-01 12:30', '2000-01-01 13:15'],
                   routes=['Intravenous', 'Oral'])
    r = cs.build_pairs(defs, admin, 30, 240).iloc[0]
    assert r['n_med'] == 2 and r['med_between'] == 1.0
    assert r['n_fast'] == 1 and r['n_slow'] == 1


def test_dose_outside_the_interval_is_not_counted():
    defs = _defs(['2000-01-01 12:00', '2000-01-01 14:00'], [8, 4])
    admin = _admin(['2000-01-01 11:00', '2000-01-01 15:00'])
    assert cs.build_pairs(defs, admin, 30, 240).iloc[0]['n_med'] == 0


def test_pairs_do_not_span_sessions_or_subjects():
    defs = pd.concat([
        _defs(['2000-01-01 12:00'], [3], subject='001', session='01'),
        _defs(['2000-01-01 13:00'], [7], subject='001', session='02'),
        _defs(['2000-01-01 14:00'], [5], subject='002', session='01'),
    ], ignore_index=True)
    with pytest.raises(ValueError):     # every session has a single assessment
        cs.build_pairs(defs, _admin([]), 30, 240)


def test_change_frame_differences_within_channel_and_cancels_level():
    pairs = cs.build_pairs(_defs(['2000-01-01 12:00', '2000-01-01 13:00'], [3, 7]),
                           _admin([]), 30, 240)
    # Channel A sits 10 units higher than B; both rise by exactly 0.5.
    cell = pd.DataFrame({
        'subject': 'sub-001',
        'channel_uid': ['sub-001|A', 'sub-001|B', 'sub-001|A', 'sub-001|B'],
        'epoch_id': [0, 0, 1, 1],
        'log10_power': [10.0, 0.0, 10.5, 0.5],
    })
    out = cs.build_change_frame(cell, pairs)
    assert len(out) == 2
    # The 10-unit level difference is gone; only the common change survives.
    assert np.allclose(out['d_log10_power'], 0.5)


def test_channel_missing_from_one_epoch_is_dropped_not_zeroed():
    pairs = cs.build_pairs(_defs(['2000-01-01 12:00', '2000-01-01 13:00'], [3, 7]),
                           _admin([]), 30, 240)
    cell = pd.DataFrame({
        'subject': 'sub-001',
        'channel_uid': ['sub-001|A', 'sub-001|B', 'sub-001|A'],
        'epoch_id': [0, 0, 1],
        'log10_power': [1.0, 2.0, 1.5],
    })
    out = cs.build_change_frame(cell, pairs)
    assert list(out['channel_uid']) == ['sub-001|A']


def test_pair_summary_exposes_the_baseline_imbalance():
    defs = _defs(['2000-01-01 12:00', '2000-01-01 13:00',
                  '2000-01-01 18:00', '2000-01-01 19:00'], [8, 5, 2, 3])
    admin = _admin(['2000-01-01 12:30'])
    s = cs.pair_summary(cs.build_pairs(defs, admin, 30, 240))
    assert set(s['stratum']) == {'dose between', 'no dose between'}
    dosed = s.set_index('stratum').loc['dose between']
    undosed = s.set_index('stratum').loc['no dose between']
    assert dosed['pain_1_mean'] > undosed['pain_1_mean']
