"""Sanity tests for the analyses_run.md index helper. No Oak, no NWB.

Runnable either way: `pytest tests/test_analysis_log.py` or `python -m
tests.test_analysis_log`. Every case writes to a temp file via the `path=`
override, so the real docs/analyses_run.md is never touched.
"""
import tempfile
from pathlib import Path

from ieeg_ehr.io.analysis_log import log_analysis, mark_logged, read_entries


def test_creates_file_with_header_and_one_line():
    with tempfile.TemporaryDirectory() as d:
        idx = Path(d) / 'analyses_run.md'
        line = log_analysis('bandpower sweep, HFA x pain', '/tmp/out/run_a', path=idx)
        assert line is not None
        assert idx.exists()
        assert idx.read_text().startswith('# analyses_run.md')
        entries = read_entries(idx)
        assert len(entries) == 1
        assert entries[0]['desc'] == 'bandpower sweep, HFA x pain'
        assert entries[0]['logged'] is False


def test_dedupes_same_path_and_hash():
    """A rerun of the same script at the same commit must not double-log --
    this is what makes the helper safe to call from every task of an array."""
    with tempfile.TemporaryDirectory() as d:
        idx = Path(d) / 'analyses_run.md'
        first = log_analysis('sweep', '/tmp/out/run_a', git_hash='abc123', path=idx)
        second = log_analysis('sweep', '/tmp/out/run_a', git_hash='abc123', path=idx)
        assert first is not None
        assert second is None
        assert len(read_entries(idx)) == 1


def test_new_commit_is_a_new_line():
    """Same output dir at a different commit is genuinely a different run."""
    with tempfile.TemporaryDirectory() as d:
        idx = Path(d) / 'analyses_run.md'
        log_analysis('sweep', '/tmp/out/run_a', git_hash='abc123', path=idx)
        log_analysis('sweep', '/tmp/out/run_a', git_hash='def456', path=idx)
        assert len(read_entries(idx)) == 2


def test_description_is_flattened():
    """Pipes are the field separator and newlines would end the record."""
    with tempfile.TemporaryDirectory() as d:
        idx = Path(d) / 'analyses_run.md'
        log_analysis('has | pipe\nand a newline', '/tmp/out/run_a', path=idx)
        entry = read_entries(idx)[0]
        assert entry['desc'] == 'has / pipe and a newline'
        assert '\n' not in entry['desc']


def test_empty_description_does_not_break_the_format():
    with tempfile.TemporaryDirectory() as d:
        idx = Path(d) / 'analyses_run.md'
        log_analysis('   ', '/tmp/out/run_a', path=idx)
        assert read_entries(idx)[0]['desc'] == '(no description)'


def test_mark_logged_flips_only_the_matching_entry():
    with tempfile.TemporaryDirectory() as d:
        idx = Path(d) / 'analyses_run.md'
        log_analysis('sweep a', '/tmp/out/run_a', git_hash='abc123', path=idx)
        log_analysis('sweep b', '/tmp/out/run_b', git_hash='abc123', path=idx)

        assert mark_logged('/tmp/out/run_a', path=idx) == 1
        by_path = {e['path']: e['logged'] for e in read_entries(idx)}
        assert by_path['/tmp/out/run_a'] is True
        assert by_path['/tmp/out/run_b'] is False

        # Idempotent: already [x], nothing to flip.
        assert mark_logged('/tmp/out/run_a', path=idx) == 0


def test_mark_logged_on_missing_file_is_a_noop():
    with tempfile.TemporaryDirectory() as d:
        assert mark_logged('/tmp/out/run_a', path=Path(d) / 'nope.md') == 0


def test_deriv_paths_are_abbreviated():
    """$DERIV/ keeps lines readable; a non-Oak path stays absolute."""
    from ieeg_ehr.config.paths import DERIVATIVES_BASE
    with tempfile.TemporaryDirectory() as d:
        idx = Path(d) / 'analyses_run.md'
        log_analysis('sweep', Path(DERIVATIVES_BASE) / 'analysis' / 'pain' / 'run_a',
                     git_hash='abc123', path=idx)
        assert read_entries(idx)[0]['path'] == '$DERIV/analysis/pain/run_a'


def test_failure_is_swallowed_not_raised():
    """Indexing must never kill a six-hour array job."""
    # A path whose parent cannot be created (a file, not a dir).
    with tempfile.NamedTemporaryFile() as f:
        bad = Path(f.name) / 'child' / 'analyses_run.md'
        assert log_analysis('sweep', '/tmp/out/run_a', path=bad) is None


if __name__ == '__main__':
    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith('test_') or not callable(fn):
            continue
        try:
            fn()
            print(f'PASS {name}')
            passed += 1
        except Exception as exc:                                # noqa: BLE001
            print(f'FAIL {name}: {exc.__class__.__name__}: {exc}')
            failed += 1
    print(f'\n{passed} passed, {failed} failed')
    raise SystemExit(1 if failed else 0)
