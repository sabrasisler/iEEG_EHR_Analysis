"""
Table IO.

Target convention (docs/kickoff_plan.md): Parquet for tables, joblib for fitted
models, JSON for manifests and sidecars. Never pickle tabular data.

Currently CSV — pyarrow is not yet in the venv. P0.3 installs it and switches
save_table to Parquet, and adds the provenance/staleness sidecar writer that
every artifact write is supposed to go through. Existing CSVs are NOT bulk
converted; convert one when it is next touched.
"""

from pathlib import Path


def save_table(df, path):
    """Write a dataframe to disk. CSV for now; swap to df.to_parquet(path) once
    pyarrow is added to the venv (P0.3)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def append_table(df, path):
    """
    Append a dataframe to a CSV, writing the header only if the file doesn't
    exist yet. Used to stream per-run results to disk instead of holding every
    run's rows in memory for an entire (possibly 100+ run) session.
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
