"""
The single config namespace: `from ieeg_ehr import config`.

Split by concern across paths.py / qc_params.py / psd_params.py /
pain_params.py / cache_params.py and re-exported here, so callers see one flat
namespace (`config.RAW_DIR`, `config.GROSS_STD_THRESH`, `config.PAIN_BIN_EDGES`,
`config.CACHE_FLOAT_DTYPE`, ...) regardless of which file a name lives in.

Two things are defined HERE rather than re-exported, deliberately:

1. The mutable output-dir globals (OUTPUT_DIR, PER_WINDOW_DIR, SUMMARY_DIR,
   PLOTS_DIR, RAIL_VALUES_CSV) and set_output_dir(). Seven QC modules read
   these as `config.X` *after* calling `config.set_output_dir(...)`. If they
   were re-exported from a submodule, `set_output_dir` would rebind the
   submodule's global while `config.X` kept pointing at the stale original —
   a silent wrong-output-directory bug. Keeping the state and its mutator in
   one namespace makes the rebind visible to every reader.

2. save_table / git_provenance / run_timestamp / warn_if_dirty, re-exported
   from ieeg_ehr.io so existing `config.save_table(...)` call sites keep
   working. New code should import them from ieeg_ehr.io directly.
"""

from pathlib import Path

from ieeg_ehr.config.paths import *          # noqa: F401,F403
from ieeg_ehr.config.paths import (          # explicit: used below
    DEFAULT_LEVEL_ROOT,
    metrics_root,
)
from ieeg_ehr.config.qc_params import *      # noqa: F401,F403
from ieeg_ehr.config.feature_qc_params import *  # noqa: F401,F403
from ieeg_ehr.config.psd_params import *     # noqa: F401,F403
from ieeg_ehr.config.pain_params import *    # noqa: F401,F403
from ieeg_ehr.config.cache_params import *   # noqa: F401,F403

# Re-exported for backwards compatibility with `config.<name>` call sites.
from ieeg_ehr.io.provenance import (         # noqa: F401
    git_provenance,
    run_timestamp,
    warn_if_dirty,
)
from ieeg_ehr.io.tables import (             # noqa: F401
    append_table,
    reset_table,
    save_table,
)

# ============================================================================
# MUTABLE OUTPUT-DIR STATE  (see note 1 in the module docstring)
# ============================================================================
# Detection points these at a level's metrics/; other scripts repoint via
# set_output_dir(), which every CLI does before doing any work.

OUTPUT_DIR = metrics_root(DEFAULT_LEVEL_ROOT)
PER_WINDOW_DIR = OUTPUT_DIR / 'per_window'
SUMMARY_DIR = OUTPUT_DIR / 'summary'
PLOTS_DIR = OUTPUT_DIR / 'plots'
RAIL_VALUES_CSV = SUMMARY_DIR / 'saturation_rails.csv'


def set_output_dir(path):
    """
    Point all output paths at an alternate root (e.g. a different level's
    metrics/, or a pipeline version whose results aren't comparable to the
    default). Call once, before anything else in a script runs.
    """
    global OUTPUT_DIR, PER_WINDOW_DIR, SUMMARY_DIR, PLOTS_DIR, RAIL_VALUES_CSV
    OUTPUT_DIR = Path(path)
    PER_WINDOW_DIR = OUTPUT_DIR / 'per_window'
    SUMMARY_DIR = OUTPUT_DIR / 'summary'
    PLOTS_DIR = OUTPUT_DIR / 'plots'
    RAIL_VALUES_CSV = SUMMARY_DIR / 'saturation_rails.csv'


def ensure_output_dirs():
    for d in (PER_WINDOW_DIR, SUMMARY_DIR, PLOTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
