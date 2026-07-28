"""Sanity tests for the IO conventions (P0.3). No Oak, no NWB, no Slurm.

Everything writes into a temp directory, so nothing here can touch the
derivatives tree or the repo's own tracking files.

Runnable either way: `pytest tests/test_io_conventions.py` or
`python -m tests.test_io_conventions`.
"""
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from ieeg_ehr import io
from ieeg_ehr.io.sidecar import StaleArtifactError

DF = pd.DataFrame({'epoch_id': [0, 0, 1], 'channel': ['a', 'b', 'a'],
                   'log_power': [1.5, 2.5, 3.5]})


# ---------------------------------------------------------------- sidecar paths

def test_file_sidecar_appends_suffix():
    """Append, not replace: x.parquet and x.csv must not share a sidecar."""
    assert io.sidecar_path('/d/x.parquet').name == 'x.parquet.provenance.json'
    assert io.sidecar_path('/d/x.csv').name == 'x.csv.provenance.json'


def test_directory_sidecar_is_provenance_json():
    with tempfile.TemporaryDirectory() as d:
        assert io.sidecar_path(d) == Path(d) / 'provenance.json'
    # A suffix-less path that does not exist yet is still treated as a directory.
    assert io.sidecar_path('/d/run_20260727-120000').name == 'provenance.json'


def test_reads_legacy_suffix_replaced_sidecar():
    """The pre-P0.3 form (`x.provenance.json` beside `x.csv`) still resolves --
    the legacy pain caches under outdated/ are written that way."""
    with tempfile.TemporaryDirectory() as d:
        artifact = Path(d) / 'sub-085_ses-01_epoch_channel_power.csv'
        artifact.write_text('a,b\n1,2\n')
        legacy = artifact.with_suffix('.provenance.json')
        legacy.write_text('{"kind": "legacy"}')
        assert io.read_sidecar(artifact)['kind'] == 'legacy'
        assert io.find_sidecar(artifact) == legacy


def test_missing_sidecar_reads_as_none():
    with tempfile.TemporaryDirectory() as d:
        artifact = Path(d) / 'x.parquet'
        artifact.write_text('not really parquet')
        assert io.read_sidecar(artifact) is None


# ---------------------------------------------------------------------- hashing

def test_config_hash_is_order_independent():
    assert io.config_hash({'a': 1, 'b': 2}) == io.config_hash({'b': 2, 'a': 1})
    assert io.config_hash({'a': 1}) != io.config_hash({'a': 2})


def test_file_digest_refuses_large_files(monkeypatch):
    """The guardrail that stops someone sha256-ing a multi-GB cache file inside
    an array job."""
    from ieeg_ehr.io import sidecar
    monkeypatch.setattr(sidecar, 'DIGEST_HARD_MAX_BYTES', 8)
    with tempfile.TemporaryDirectory() as d:
        big = Path(d) / 'big.bin'
        big.write_bytes(b'0' * 64)
        with pytest.raises(ValueError):
            sidecar.file_digest(big)


# ------------------------------------------------------------ write_table round trip

def test_write_table_emits_parquet_and_sidecar():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / 'cache' / 'sub-085_ses-01_epochs.parquet'
        io.write_table(DF, path, params={'mask_label': 'm1'}, subjects=['085'],
                       float_dtype='float32')

        assert path.exists()
        back = io.read_table(path)
        assert list(back.columns) == list(DF.columns)
        assert back['log_power'].dtype == 'float32'

        sc = io.read_sidecar(path)
        assert sc['kind'] == 'table'
        assert sc['schema_version'] == io.sidecar.SCHEMA_VERSION
        assert sc['subjects'] == ['085']
        assert sc['params'] == {'mask_label': 'm1'}
        assert sc['config_hash'] == io.config_hash({'mask_label': 'm1'})
        # script is auto-detected from the calling frame, not this io module
        assert sc['script'].endswith('test_io_conventions.py')


def test_write_table_column_subset_read():
    """Partial column reads are the reason the cache is Parquet at all."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / 't.parquet'
        io.write_table(DF, path, params={})
        assert list(io.read_table(path, columns=['log_power']).columns) == ['log_power']


def test_write_table_rejects_unknown_extension():
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(ValueError):
            io.write_table(DF, Path(d) / 't.tsv', params={})


def test_save_table_dispatches_on_extension():
    """Existing QC call sites pass .csv and must stay CSV."""
    with tempfile.TemporaryDirectory() as d:
        csv_path = Path(d) / 'legacy.csv'
        io.save_table(DF, csv_path)
        assert csv_path.read_text().startswith('epoch_id,channel,log_power')
        assert io.find_sidecar(csv_path) is None      # legacy writer, no sidecar

        pq_path = Path(d) / 'new.parquet'
        io.save_table(DF, pq_path)
        assert pd.read_parquet(pq_path).shape == DF.shape


# ------------------------------------------------------------------- manifests

def test_manifest_and_view_staleness_cycle():
    """The full P1.3 contract: cache unit -> manifest -> materialized view ->
    staleness check when the manifest changes underneath it."""
    with tempfile.TemporaryDirectory() as d:
        unit = Path(d) / 'epoch-5min-pre_mask-m1'
        io.write_manifest(unit, params={'epoch_minutes': 5.0, 'mask_label': 'm1',
                                        'dtype': 'float32'})
        assert io.read_manifest(unit)['params']['mask_label'] == 'm1'

        view_config = {'domain': 'log', 'normalize': 'zscore'}
        view_path = unit / 'views' / f'hfa_{io.config_hash(view_config)}' / 'table.parquet'
        io.write_table(DF, view_path, kind='view', params=view_config,
                       parents=[io.manifest_ref(unit)])

        assert io.check_view_fresh(view_path, view_config=view_config,
                                   cache_manifest=unit, on_stale='refuse')

        # A different view config is a different view, not a stale one.
        reasons = io.check_stale(view_path, config={'domain': 'linear'})
        assert any('config differs' in r for r in reasons)

        # Rebuilding the cache unit invalidates the saved view.
        io.write_manifest(unit, params={'epoch_minutes': 5.0, 'mask_label': 'm1',
                                        'dtype': 'float64'})
        reasons = io.check_stale(view_path, parents=[io.manifest_ref(unit)])
        assert any('input changed' in r for r in reasons)
        with pytest.raises(StaleArtifactError):
            io.check_view_fresh(view_path, view_config=view_config,
                                cache_manifest=unit, on_stale='refuse')


def test_parent_ref_carries_parents_own_provenance():
    """One hop of chain traceability without opening every file by hand."""
    with tempfile.TemporaryDirectory() as d:
        parent = Path(d) / 'defs.parquet'
        io.write_table(DF, parent, params={'stage': 'epoch_defs'})
        ref = io.parent_ref(parent)
        assert ref['provenance']['config_hash'] == io.config_hash({'stage': 'epoch_defs'})
        assert 'digest' in ref            # small file -> digested automatically


def test_stale_warns_by_default_and_refuses_on_request(capsys):
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / 'no_sidecar.parquet'
        io.write_table(DF, path, params={}, sidecar=False)

        assert io.assert_fresh(path, on_stale='warn') is False
        assert 'no sidecar' in capsys.readouterr().out
        with pytest.raises(StaleArtifactError):
            io.assert_fresh(path, on_stale='refuse')


# ---------------------------------------------------------------------- models

def test_save_and_load_model_round_trip():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / 'model.joblib'
        io.save_model({'coef': [1.0, 2.0]}, path, params={'spec': 'glmm-binary'},
                      subjects=['085', '071'])
        sc = io.read_sidecar(path)
        assert sc['kind'] == 'model'
        assert sc['environment']['python'].startswith('3.')
        assert sc['subjects'] == ['071', '085']       # sorted on write
        assert io.load_model(path, on_stale='ignore') == {'coef': [1.0, 2.0]}


def test_save_model_rejects_non_joblib_suffix():
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(ValueError):
            io.save_model({'a': 1}, Path(d) / 'model.pkl', params={})


# ------------------------------------------------------------------ path scheme

def test_pain_epoch_paths_are_on_oak_and_carry_the_epoch_definition():
    from ieeg_ehr.config import paths

    unit = paths.pain_epoch_unit_dir()
    # The unit is keyed on the EPOCH DEFINITION ONLY. It deliberately does not
    # carry the QC mask: the cache stores raw slices and masking is a view-time
    # join, so a mask change must not invalidate the cache (paths.py, 2026-07-27).
    assert unit.name == 'epoch-5min-pre'
    assert 'mask' not in unit.name
    assert str(unit).startswith('/oak/')

    cache = paths.pain_epoch_cache_path('085', '01')
    assert cache.name == 'sub-085_ses-01_epochs.parquet'
    assert cache.parent.name == 'cache'
    assert paths.pain_epoch_defs_path('085', '01').name == 'sub-085_ses-01_defs.parquet'
    # config spells 'manifest.json' out; io owns the constant. They must agree,
    # since io.manifest_ref() resolves the same file config points scripts at.
    assert paths.pain_epoch_manifest_path().name == io.MANIFEST_NAME
    assert io.manifest_ref(unit)['path'] == str(
        paths.pain_epoch_manifest_path())


def test_analysis_run_dir_follows_the_five_level_scheme():
    from ieeg_ehr.config import paths

    run = paths.analysis_run_dir('exploratory-sweep', 'heatmap', 'hfa-first-look',
                                 view_scheme='absolute', timestamp='20260727-120000')
    assert run.parts[-5:] == ('pain', 'exploratory-sweep', 'heatmap', 'absolute',
                              'hfa-first-look_20260727-120000')
    # view_scheme is optional -> one level shorter
    run_no_scheme = paths.analysis_run_dir('exploratory-sweep', 'heatmap', 'x',
                                           timestamp='20260727-120000')
    assert run_no_scheme.parts[-4:] == ('pain', 'exploratory-sweep', 'heatmap',
                                        'x_20260727-120000')
    assert str(run).startswith('/oak/')


def test_run_dir_always_gets_a_timestamp():
    from ieeg_ehr.config import paths
    a = paths.analysis_run_dir('q', 'heatmap', 'label')
    assert a.name.startswith('label_') and len(a.name) > len('label_')


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
