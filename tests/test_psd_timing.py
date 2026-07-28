"""
Tests for the PSD windowing-design check.

The oracle is real: both params strings below were read verbatim off NWBs on Oak
(sub-247 run-IA6193AN and run-IA6193AA), so `classify_design` is tested against
what the two pipeline generations actually wrote, not against a guess.
"""

import numpy as np
import pandas as pd
import pytest

from ieeg_ehr import config
from ieeg_ehr.qc import psd_timing as pt

# --- verbatim from sub-247 run-IA6193AN (the SUPERSEDED two-level design) -------
DESC_OUTER = ('Bipolar Welch PSD, log-spaced frequency bins. Params: '
              '{"outer_window_sec": 60.0, "inner_segment_sec": 2.0, '
              '"overlap_frac": 0.5, "window_function": "hann", '
              '"scaling": "density", "psd_chunk_max_hours": null}')

# --- verbatim shape from sub-247 run-IA6193AA (the CURRENT single-level design),
#     truncated but keeping the nested `git` object that a naive regex would break on
DESC_SINGLE = ('Bipolar Welch PSD, log-spaced frequency bins. Params: '
               '{"window_sec": 2.0, "overlap_frac": 0.5, "window_function": "hann", '
               '"scaling": "density", "psd_chunk_max_hours": null, '
               '"git": {"available": true, "commit": "1c99930", "dirty": true, '
               '"modified_files": ["a.py", "b.py"]}, '
               '"log_bin_edges_hz": [1.0, 1.1167573003382756], "n_bins": 50}')


# ---------------------------------------------------------------------------
# classify_design
# ---------------------------------------------------------------------------

def test_classifies_the_two_real_designs():
    assert pt.classify_design(DESC_OUTER) == pt.DESIGN_OUTER_WINDOW
    assert pt.classify_design(DESC_SINGLE) == pt.DESIGN_SINGLE_LEVEL


def test_nested_json_does_not_break_the_parser():
    """The current blob nests `git` and a 51-element array; a `\\{.*\\}` regex or a
    first-`}`-wins parse would mangle it and fall through to 'unknown'."""
    assert pt.classify_design(DESC_SINGLE) == pt.DESIGN_SINGLE_LEVEL
    assert pt._params_from_description(DESC_SINGLE)['git']['commit'] == '1c99930'
    assert pt._params_from_description(DESC_SINGLE)['n_bins'] == 50


def test_outer_window_wins_when_both_keys_present():
    """If a writer ever emitted both, the presence of an outer window is decisive --
    it means segments were Welch-averaged before storage, which is the thing that
    freezes the log-vs-linear choice into the file."""
    desc = 'Params: {"window_sec": 2.0, "outer_window_sec": 60.0}'
    assert pt.classify_design(desc) == pt.DESIGN_OUTER_WINDOW


@pytest.mark.parametrize('desc', ['', None, 'no params here',
                                  'Params: {not valid json',
                                  'Params: {"something_else": 1}'])
def test_unparseable_descriptions_are_unknown_not_ok(desc):
    """Unknown must never be treated as fine: silence is not approval."""
    if desc == 'Params: {"something_else": 1}':
        assert pt.classify_design(desc) == pt.DESIGN_UNKNOWN
    else:
        assert pt.classify_design(desc) in (pt.DESIGN_UNKNOWN, pt.DESIGN_OUTER_WINDOW)


def test_prose_fallback_only_when_no_params_blob():
    assert pt.classify_design('60s outer_window design, no params') == pt.DESIGN_OUTER_WINDOW
    # a parsed blob outvotes prose
    assert pt.classify_design('outer window mentioned. Params: {"window_sec": 2.0}') \
        == pt.DESIGN_SINGLE_LEVEL


# ---------------------------------------------------------------------------
# hop arithmetic, derived from config
# ---------------------------------------------------------------------------

def test_expected_hop_is_derived_from_config_not_a_literal():
    assert pt.EXPECTED_HOP_SEC == config.PSD_WINDOW_SEC * (1.0 - config.PSD_OVERLAP_FRAC)
    assert pt.EXPECTED_HOP_SEC == pytest.approx(1.0)   # today's config


def test_hop_from_rate():
    assert pt.hop_from_rate(1.0) == pytest.approx(1.0)
    assert pt.hop_from_rate(1.0 / 60.0) == pytest.approx(60.0)


def test_hop_from_rate_is_nan_not_an_exception_for_bad_rate():
    """One corrupt run must not abort a 6236-file sweep."""
    for bad in (0.0, -1.0):
        assert np.isnan(pt.hop_from_rate(bad))
    assert not pt.is_expected_hop(pt.hop_from_rate(0.0))


def test_is_expected_hop():
    assert pt.is_expected_hop(1.0)
    assert not pt.is_expected_hop(60.0)
    assert not pt.is_expected_hop(float('nan'))


# ---------------------------------------------------------------------------
# describe_run
# ---------------------------------------------------------------------------

def _fake_reader(rate, description, starting_time=0.0, n_time=100):
    def _read(_path):
        return starting_time, rate, n_time, description
    return _read


def test_describe_run_flags_the_superseded_design(tmp_path, monkeypatch):
    nwb = tmp_path / 'x.nwb'
    nwb.write_bytes(b'')
    monkeypatch.setattr(pt, 'read_run_timing',
                        _fake_reader(1.0 / 60.0, DESC_OUTER, n_time=120))
    row = pt.describe_run('247', '01', 'IA6193AN', nwb_path=nwb)
    assert row['design'] == pt.DESIGN_OUTER_WINDOW
    assert row['hop_sec'] == pytest.approx(60.0)
    assert row['ok'] is False
    assert 'design=outer_window' in row['reason'] and 'hop=60' in row['reason']
    # 120 rows x 60 s = 2.00 h -- the consistency that proved these are genuine
    # coarse-resolution files rather than a mislabeled rate.
    assert row['duration_h'] == pytest.approx(2.0)


def test_describe_run_accepts_the_current_design(tmp_path, monkeypatch):
    nwb = tmp_path / 'x.nwb'
    nwb.write_bytes(b'')
    monkeypatch.setattr(pt, 'read_run_timing',
                        _fake_reader(1.0, DESC_SINGLE, n_time=7199))
    row = pt.describe_run('247', '01', 'IA6193AA', nwb_path=nwb)
    assert row['design'] == pt.DESIGN_SINGLE_LEVEL
    assert row['ok'] is True and row['reason'] == ''
    assert row['params_has_git'] is True


def test_describe_run_records_a_missing_file_rather_than_raising(tmp_path):
    row = pt.describe_run('999', '01', 'ZZZ', nwb_path=tmp_path / 'nope.nwb')
    assert row['ok'] is False and row['reason'] == 'nwb_missing'


def test_describe_run_survives_an_unreadable_nwb(tmp_path, monkeypatch):
    nwb = tmp_path / 'x.nwb'
    nwb.write_bytes(b'not an hdf5 file')

    def _boom(_p):
        raise OSError('truncated file')
    monkeypatch.setattr(pt, 'read_run_timing', _boom)
    row = pt.describe_run('247', '01', 'BAD', nwb_path=nwb)
    assert row['ok'] is False and 'unreadable' in row['reason']


def test_a_good_hop_with_the_wrong_design_is_still_not_ok():
    """Design is the primary test. A file could carry a 1 s rate and still have been
    written by the two-level code path, and it must not pass."""
    row = dict(design=pt.DESIGN_OUTER_WINDOW)
    assert row['design'] != pt.DESIGN_SINGLE_LEVEL


# ---------------------------------------------------------------------------
# Enforcement from the cached table -- no NWB reads
# ---------------------------------------------------------------------------

def _table():
    return pd.DataFrame([
        {'subject_id': 'sub-247', 'run_id': 'run-A', 'ok': True, 'reason': ''},
        {'subject_id': 'sub-247', 'run_id': 'run-B', 'ok': False,
         'reason': 'design=outer_window; hop=60s'},
        {'subject_id': 'sub-231', 'run_id': 'run-C', 'ok': True, 'reason': ''},
    ])


def test_refuse_raises_for_a_mixed_subject():
    with pytest.raises(pt.NonstandardPsdError, match='not written by the current PSD design'):
        pt.assert_subject_ok('247', policy='refuse', table=_table())


def test_refuse_passes_a_clean_subject():
    assert pt.assert_subject_ok('231', policy='refuse', table=_table()) == []


def test_drop_returns_the_offending_runs():
    assert pt.assert_subject_ok('247', policy='drop', table=_table()) == ['run-B']


def test_allow_only_warns():
    assert pt.assert_subject_ok('247', policy='allow', table=_table()) == ['run-B']


def test_an_unaudited_subject_is_refused():
    """Absence of evidence must not read as evidence of absence."""
    with pytest.raises(pt.NonstandardPsdError, match='absent from the PSD timing audit'):
        pt.assert_subject_ok('999', policy='refuse', table=_table())


def test_enforcement_opens_no_nwb(monkeypatch):
    """The whole point of caching the audit: analysis code must not re-derive it."""
    def _forbidden(_p):
        raise AssertionError('enforcement must not read NWBs')
    monkeypatch.setattr(pt, 'read_run_timing', _forbidden)
    pt.assert_subject_ok('231', policy='refuse', table=_table())


def test_subject_id_accepts_both_spellings():
    assert pt.assert_subject_ok('sub-231', table=_table()) == []
    assert pt.assert_subject_ok('231', table=_table()) == []


def test_bad_policy_is_rejected():
    with pytest.raises(ValueError, match='unknown policy'):
        pt.assert_subject_ok('231', policy='maybe', table=_table())


def test_rerun_and_ok_subject_lists_partition_the_cohort():
    table = _table()
    assert pt.rerun_subjects(table) == ['sub-247']
    assert pt.ok_subjects(table) == ['sub-231']
    assert not set(pt.rerun_subjects(table)) & set(pt.ok_subjects(table))


def test_missing_audit_table_raises_with_the_fix_in_the_message(tmp_path, monkeypatch):
    """The error must name the command that fixes it.

    Points run_timing_path() at a nonexistent file rather than relying on the real
    one being absent -- the first version of this test passed only until the audit
    was actually built, which makes a test a function of the filesystem instead of
    the code.
    """
    monkeypatch.setattr(pt, 'run_timing_path', lambda: tmp_path / 'nope.parquet')
    with pytest.raises(FileNotFoundError, match='audit_psd_timing'):
        pt.load_run_timing()
    # and 'none' is the explicit opt-out, which must NOT raise
    assert pt.load_run_timing(on_missing='none') is None
