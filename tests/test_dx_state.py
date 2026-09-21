"""Tests for the diagnosis-label join. No Oak, no NWB -- synthetic frames only.

The label is the part of the MDD analysis most likely to be silently wrong: an
ICD prefix that matches one code too many, a window that includes the admission
it is supposed to exclude, or a source filter that quietly empties an arm all
produce a plausible-looking number with no error. So each of those is pinned.

Runnable either way: `pytest tests/test_dx_state.py` or
`python -m tests.test_dx_state`.
"""
import numpy as np
import pandas as pd

from ieeg_ehr.analysis import dx_state


START = pd.Timestamp('2000-01-10 12:00:00')


def make_dx(rows):
    """Build a diagnoses frame in the shape `load_diagnoses` returns."""
    df = pd.DataFrame(rows)
    df['subject_id'] = df.get('subject_id', 'sub-001')
    df['session_id'] = df.get('session_id', 'ses-01')
    df['session_start'] = df.get('session_start', START)
    df['session_end'] = START + pd.Timedelta(days=5)
    for col in ('icd9_code', 'icd10_code'):
        if col not in df:
            df[col] = None
    df['source'] = df.get('source', 'Encounter Diagnosis')
    df['date'] = pd.to_datetime(df['date'])
    df['icd9_norm'] = df['icd9_code'].map(dx_state._norm_code)
    df['icd10_norm'] = df['icd10_code'].map(dx_state._norm_code)
    return df


# ---------------------------------------------------------------- code sets

def test_mdd_matches_f32_f33_dotted_and_undotted():
    dx = make_dx([{'date': START, 'icd10_code': c}
                  for c in ('F32.1', 'F321', 'F33.2', 'F32.A')])
    assert dx_state.matches_condition(dx, 'mdd').all()


def test_mdd_matches_icd9_296_2_and_296_3():
    dx = make_dx([{'date': START, 'icd9_code': c}
                  for c in ('296.20', '296.32', '2962', '296.3')])
    assert dx_state.matches_condition(dx, 'mdd').all()


def test_mdd_excludes_bipolar_anxiety_and_dysthymia():
    """The boundaries that would silently inflate the case arm."""
    dx = make_dx([
        {'date': START, 'icd10_code': 'F31.9'},    # bipolar
        {'date': START, 'icd10_code': 'F41.1'},    # anxiety
        {'date': START, 'icd10_code': 'F34.1'},    # dysthymia
        {'date': START, 'icd9_code': '296.44'},    # ICD-9 bipolar
        {'date': START, 'icd9_code': '311'},       # depressive disorder NEC
        {'date': START, 'icd9_code': '300.4'},     # ICD-9 dysthymia
    ])
    assert not dx_state.matches_condition(dx, 'mdd').any()


def test_broad_set_adds_dysthymia_and_311_but_still_not_bipolar():
    dx = make_dx([
        {'date': START, 'icd10_code': 'F34.1'},
        {'date': START, 'icd9_code': '311'},
        {'date': START, 'icd10_code': 'F31.9'},
    ])
    hit = dx_state.matches_condition(dx, 'depression_broad')
    assert list(hit) == [True, True, False]


def test_never_matches_on_description_text():
    """'depression' also means ST-segment and respiratory depression."""
    dx = make_dx([{'date': START, 'icd10_code': 'R94.31',
                   'description': 'ST segment depression'},
                  {'date': START, 'icd10_code': 'R06.89',
                   'description': 'Respiratory depression'}])
    assert not dx_state.matches_condition(dx, 'mdd').any()


# ------------------------------------------------------------------ window

def test_code_inside_window_is_a_case():
    dx = make_dx([{'date': START - pd.Timedelta(days=30), 'icd10_code': 'F32.1'}])
    labels, _ = dx_state.subject_labels(dx, window_days=90)
    assert bool(labels['dx_state'].iloc[0]) is True


def test_code_older_than_window_is_a_control():
    dx = make_dx([{'date': START - pd.Timedelta(days=200), 'icd10_code': 'F32.1'}])
    labels, _ = dx_state.subject_labels(dx, window_days=90)
    assert bool(labels['dx_state'].iloc[0]) is False


def test_the_same_code_is_a_case_under_ever():
    """window_days=0 means EVER -- the pre-specified sensitivity analysis."""
    dx = make_dx([{'date': START - pd.Timedelta(days=200), 'icd10_code': 'F32.1'}])
    labels, _ = dx_state.subject_labels(dx, window_days=0)
    assert bool(labels['dx_state'].iloc[0]) is True


def test_code_dated_during_the_admission_never_counts():
    """The encounter that produced the iEEG must not define its own predictor."""
    dx = make_dx([{'date': START + pd.Timedelta(days=2), 'icd10_code': 'F32.1'}])
    for window in (90, 0):
        labels, _ = dx_state.subject_labels(dx, window_days=window)
        assert bool(labels['dx_state'].iloc[0]) is False, f'window={window}'


# ------------------------------------------------------------------ sources

def test_clinical_source_filter_drops_a_billing_only_case():
    dx = make_dx([{'date': START - pd.Timedelta(days=10), 'icd10_code': 'F32.1',
                   'source': 'Professional Billing Code'}])
    assert bool(dx_state.subject_labels(dx, sources='any')[0]['dx_state'].iloc[0])
    assert not bool(
        dx_state.subject_labels(dx, sources='clinical')[0]['dx_state'].iloc[0])


def test_billing_only_is_counted_in_the_label_table():
    dx = make_dx([{'date': START - pd.Timedelta(days=10), 'icd10_code': 'F32.1',
                   'source': 'Professional Billing Code'}])
    labels, _ = dx_state.subject_labels(dx, sources='any')
    assert int(labels['n_billing_codes'].iloc[0]) == 1
    assert int(labels['n_clinical_codes'].iloc[0]) == 0


# ------------------------------------------------------- multi-session rollup

def test_sessions_are_rolled_up_with_any():
    """One admission in window, one not -> the subject is a case, once."""
    dx = pd.concat([
        make_dx([{'date': START - pd.Timedelta(days=10), 'icd10_code': 'F32.1',
                  'session_id': 'ses-01'}]),
        make_dx([{'date': START - pd.Timedelta(days=900), 'icd10_code': 'F32.1',
                  'session_id': 'ses-02'}]),
    ], ignore_index=True)
    labels, detail = dx_state.subject_labels(dx, window_days=90)
    assert len(labels) == 1
    assert bool(labels['dx_state'].iloc[0]) is True
    assert sorted(detail['dx_state']) == [False, True]


def test_control_subject_has_no_codes_and_a_nan_age():
    dx = make_dx([{'date': START - pd.Timedelta(days=10), 'icd10_code': 'I10'}])
    labels, _ = dx_state.subject_labels(dx, window_days=90)
    assert bool(labels['dx_state'].iloc[0]) is False
    assert int(labels['n_codes'].iloc[0]) == 0
    assert np.isnan(labels['first_days_before'].iloc[0])


def test_stratum_summary_survives_an_all_nan_control_arm():
    """The control arm has no code age BY CONSTRUCTION; it must not warn."""
    dx = pd.concat([
        make_dx([{'date': START - pd.Timedelta(days=10), 'icd10_code': 'F32.1',
                  'subject_id': 'sub-001'}]),
        make_dx([{'date': START - pd.Timedelta(days=10), 'icd10_code': 'I10',
                  'subject_id': 'sub-002'}]),
    ], ignore_index=True)
    labels, _ = dx_state.subject_labels(dx, window_days=90)
    summary = dx_state.stratum_summary(labels)
    assert set(summary['stratum']) == {'case', 'control'}
    assert np.isnan(summary.loc[summary['stratum'] == 'control',
                                'median_days_before'].iloc[0])


# ---------------------------------------------------------------- the lookup

def test_epoch_lookup_is_float_not_bool():
    """patsy would name a bool factor's levels and break term-name matching."""
    dx = make_dx([{'date': START - pd.Timedelta(days=10), 'icd10_code': 'F32.1'}])
    labels, _ = dx_state.subject_labels(dx, window_days=90)
    look = dx_state.epoch_lookup(labels)
    assert look['dx_state'].dtype == float
    assert list(look.columns) == ['subject_id', 'dx_state']


if __name__ == '__main__':
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f'PASS {fn.__name__}')
        except Exception as exc:                        # noqa: BLE001
            failed += 1
            print(f'FAIL {fn.__name__}: {type(exc).__name__}: {exc}')
    print(f'\n{len(fns) - failed}/{len(fns)} passed')
    sys.exit(1 if failed else 0)
