"""
The IO layer: one place that knows how artifacts are written, read, and traced.

The contract every new script follows (`docs/io_conventions.md` has the full
version, CLAUDE.md the short one):

1. **Parquet** for tables, **joblib** for fitted objects, **JSON** for manifests
   and sidecars. Never pickle tabular data.
2. **Every write emits a provenance sidecar in the same call.** Use
   `write_table` / `save_model` / `write_manifest` / `write_run_provenance`
   rather than `df.to_parquet` or `joblib.dump` directly.
3. **Every read of a saved artifact checks staleness.** `read_table` /
   `load_model` do it; `assert_fresh` is there for anything else (a figure's
   source table, a materialized view).
4. **Commit + push before a definitive run**, so the commit hash the sidecar
   records describes the code that actually ran. `warn_if_dirty()` at script
   start makes a dirty tree loud.
5. One line per analysis run into the repo's text index: `log_analysis(desc, run_dir)`.

Flat namespace on purpose — `from ieeg_ehr import io` then `io.write_table(...)`,
so a caller never has to remember which submodule a helper lives in.

Submodules: `provenance` (git + timestamp) · `sidecar` (the envelope, staleness)
· `tables` (Parquet/CSV) · `models` (joblib) · `analysis_log` (the run index) ·
`nwb` (raw-file readers) · `build_file_registry` (the raw-file registry). The
last two are NOT re-exported here — they pull in pynwb/h5py, which is a slow and
unnecessary import for a script that only wants to write a table.
"""

from ieeg_ehr.io.provenance import (         # noqa: F401
    git_provenance,
    run_timestamp,
    warn_if_dirty,
)
from ieeg_ehr.io.sidecar import (            # noqa: F401
    MANIFEST_NAME,
    SIDECAR_SUFFIX,
    StaleArtifactError,
    assert_fresh,
    check_stale,
    check_view_fresh,
    config_hash,
    file_digest,
    find_sidecar,
    manifest_ref,
    parent_ref,
    read_manifest,
    read_sidecar,
    sidecar_envelope,
    sidecar_path,
    write_manifest,
    write_run_provenance,
    write_sidecar,
    write_view_sidecar,
)
from ieeg_ehr.io.tables import (             # noqa: F401
    append_table,
    downcast_floats,
    read_table,
    reset_table,
    save_table,
    write_table,
)
from ieeg_ehr.io.models import (             # noqa: F401
    load_model,
    save_model,
)
from ieeg_ehr.io.analysis_log import (       # noqa: F401
    log_analysis,
)
