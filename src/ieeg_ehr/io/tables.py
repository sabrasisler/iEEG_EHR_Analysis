"""
Table IO.

Convention (CLAUDE.md "IO / naming"): **Parquet for tables**, joblib for fitted
models (`io.models`), JSON for manifests and sidecars (`io.sidecar`). Never
pickle tabular data — pickle is version-fragile, non-portable, and unsafe from
untrusted sources.

Two entry points, deliberately separate:

- **`write_table` / `read_table` — use these in new code.** Parquet by default,
  and the write emits a provenance sidecar in the same call. This is the point
  where "never a bare `to_parquet`" is enforced rather than merely documented.

- **`save_table` / `append_table` / `reset_table` — the existing QC writers.**
  These stay CSV. The raw-voltage QC tree is ~85 subject-sessions of per-window
  CSVs with a working metric/threshold split; converting it would invalidate
  on-disk artifacts to no analytical benefit. `save_table` now dispatches on the
  file extension, so a `.parquet` path writes Parquet and a `.csv` path writes
  CSV — existing call sites (all `.csv`) are unchanged.

New artifacts only. Do NOT bulk-convert existing CSVs; convert one when you next
touch it, and give it a sidecar when you do.
"""

from pathlib import Path

from ieeg_ehr.io.sidecar import assert_fresh, write_sidecar

PARQUET_COMPRESSION = 'snappy'


def _require_pyarrow():
    """Fail with the fix rather than a bare ImportError from deep inside pandas.

    pyarrow's modern wheels are manylinux_2_28 and Sherlock is glibc 2.17, so a
    plain `pip install pyarrow` tries a source build and dies on a missing Rust
    toolchain. `--only-binary=:all:` makes pip back off to the newest version
    that still ships a manylinux2014 wheel (20.0.0 as of 2026-07).
    """
    try:
        import pyarrow    # noqa: F401
    except ImportError as exc:
        raise ImportError(
            'pyarrow is required to read/write Parquet. Install it into the shared venv '
            'from a compute node (never the login node):\n'
            '  srun -p dev --time=00:20:00 --mem=8G bash -c \'module load python/3.12 && '
            'source $GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate && '
            'pip install --no-deps --only-binary=:all: pyarrow joblib\''
        ) from exc


def downcast_floats(df, dtype='float32'):
    """Return a copy with every float column cast to `dtype`.

    The cache standardizes on `config.CACHE_FLOAT_DTYPE` (float32, settled by the
    P0.6 audit: bit-exact round-trip through Parquet, epoch averages agreeing
    with a float64 pipeline to 8.1 sig figs). Pass that constant rather than the
    string, so there is one source of truth — this module deliberately does not
    import `config`, because `config` imports it.

    Kept as an explicit call rather than a silent default in `write_table`:
    quietly halving the precision of an effect-size or p-value table is exactly
    the kind of thing that should be visible in the diff of the script doing it.
    And note the other half of the P0.6 finding — narrow storage does NOT license
    narrow arithmetic. Anything that reduces over windows upcasts to
    `config.CACHE_ACCUMULATE_DTYPE` first.
    """
    float_cols = df.select_dtypes(include=['float64', 'float32', 'float16']).columns
    if not len(float_cols):
        return df
    return df.astype({col: dtype for col in float_cols})


# ============================================================================
# THE SANCTIONED WRITER
# ============================================================================

def write_table(df, path, *, params=None, parents=None, subjects=None,
                script=None, extra=None, kind='table', float_dtype=None,
                index=False, compression=PARQUET_COMPRESSION, sidecar=True):
    """Write a dataframe (Parquet by extension) AND its provenance sidecar.

    Args:
        path: `.parquet` (preferred) or `.csv`. The extension decides the format;
            nothing here silently rewrites your path.
        params: the config that produced this table — hashed into the sidecar's
            `config_hash`, which is what staleness comparisons use.
        parents: input artifact paths (or `parent_ref`/`manifest_ref` dicts).
        subjects: resolved cohort, for a table that spans subjects.
        float_dtype: pass `config.CACHE_FLOAT_DTYPE` for cache/feature tables.
            Default None = write whatever dtype the frame already has.
        sidecar: False ONLY for a streaming/append target whose sidecar is
            written once by the caller after the last chunk. Everything else
            keeps the sidecar in the same call as the data.

    Returns the artifact path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if float_dtype is not None:
        df = downcast_floats(df, float_dtype)

    if path.suffix == '.parquet':
        _require_pyarrow()
        df.to_parquet(path, index=index, compression=compression)
    elif path.suffix == '.csv':
        df.to_csv(path, index=index)
    else:
        raise ValueError(f'unsupported table extension {path.suffix!r} for {path} — '
                         f'use .parquet (preferred) or .csv')

    if sidecar:
        write_sidecar(path, kind=kind, script=script, params=params,
                      parents=parents, subjects=subjects, extra=extra)
    return path


def read_table(path, columns=None, *, parents=None, config=None,
               check_commit=False, on_stale='warn'):
    """Read a table, checking its sidecar for staleness first.

    `columns` is the reason the cache is Parquet: a view that only needs a few
    frequency bins reads only those columns off disk.

    Pass `parents` / `config` (what you expect this file to have been built from)
    to get a real staleness check; with neither, the check degrades to "does it
    have a sidecar at all", which is still worth surfacing. Use
    `on_stale='refuse'` for anything a reported number comes out of.
    """
    import pandas as pd

    path = Path(path)
    assert_fresh(path, parents=parents, config=config,
                 check_commit=check_commit, on_stale=on_stale)

    if path.suffix == '.parquet':
        _require_pyarrow()
        return pd.read_parquet(path, columns=columns)
    if path.suffix == '.csv':
        return pd.read_csv(path, usecols=columns)
    raise ValueError(f'unsupported table extension {path.suffix!r} for {path}')


# ============================================================================
# EXISTING QC WRITERS (CSV; extension-dispatched)
# ============================================================================

def save_table(df, path):
    """Write a dataframe, format chosen by the path's extension. NO sidecar.

    Retained for the existing QC call sites, which pass `.csv` paths and stay
    CSV on purpose (see module docstring). New code should call `write_table`,
    which is Parquet-first and cannot forget the sidecar.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == '.parquet':
        _require_pyarrow()
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def append_table(df, path):
    """
    Append a dataframe to a CSV, writing the header only if the file doesn't
    exist yet. Used to stream per-run results to disk instead of holding every
    run's rows in memory for an entire (possibly 100+ run) session.

    CSV-only by nature: Parquet has no meaningful append-a-few-rows mode (each
    write is a new file or row group), so a streaming metrics target stays CSV.
    A script that wants Parquet output should accumulate and call `write_table`
    once, or write one Parquet file per chunk into a directory.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, mode='a', header=not path.exists(), index=False)


def reset_table(path):
    """Delete a per-window table if it exists, so a re-run starts clean instead
    of appending onto a stale/partial file from a previous attempt."""
    path = Path(path)
    if path.exists():
        path.unlink()
