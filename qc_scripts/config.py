"""
Shared configuration for the QC diagnostics pipeline (saturation, flatline,
non-neural gross artifact). All thresholds/paths live here — no magic numbers
in the detection/analysis modules.
"""

from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================

RAW_DIR = '/oak/stanford/groups/ckeller1/data/iEEG_EHR/iEEG_NWB'
FILE_REGISTRY_CSV = f'{RAW_DIR}/sherlock_file_registry.csv'
OUTPUT_DIR = Path('/oak/stanford/groups/ckeller1/data/iEEG_EHR/derivatives/sisler/analysis/qc')
PER_WINDOW_DIR = OUTPUT_DIR / 'per_window'
SUMMARY_DIR = OUTPUT_DIR / 'summary'
PLOTS_DIR = OUTPUT_DIR / 'plots'
RAIL_VALUES_CSV = SUMMARY_DIR / 'saturation_rails.csv'


def set_output_dir(path):
    """
    Point all output paths at an alternate root (e.g. a new folder for a
    pipeline version whose results aren't comparable to the default
    OUTPUT_DIR's, such as the session-level-rail / decoupled-baseline
    version). Call once, before anything else in a script runs.
    """
    global OUTPUT_DIR, PER_WINDOW_DIR, SUMMARY_DIR, PLOTS_DIR, RAIL_VALUES_CSV
    OUTPUT_DIR = Path(path)
    PER_WINDOW_DIR = OUTPUT_DIR / 'per_window'
    SUMMARY_DIR = OUTPUT_DIR / 'summary'
    PLOTS_DIR = OUTPUT_DIR / 'plots'
    RAIL_VALUES_CSV = SUMMARY_DIR / 'saturation_rails.csv'

# ============================================================================
# SUBJECT SUBSET
# ============================================================================

N_SUBJECTS = 20        # used when SUBJECT_LIST is None
SUBJECT_LIST = None    # e.g. ['217', '222'] to run an explicit set instead
RANDOM_SEED = 42

# ============================================================================
# STEP 1: AMPLIFIER SATURATION
# ============================================================================

SAT_WINDOW_SEC = 2.0

# Rail inference: a channel's saturation threshold is inferred from its own
# data, pooled across the ENTIRE SESSION (all runs), rather than assumed or
# computed per-run — amplifier gain/rail voltage varies across subjects but is
# physically shared by every channel/run within one session. Digital clipping
# shows up as the exact same extreme value repeating many times; a real
# (non-clipped) signal essentially never hits the identical floating-point
# value more than once or twice by chance.
SAT_MIN_REPEATS = 5             # occurrences (session-wide) of a channel's own abs_max needed
                                 # to call it a rail, when falling back to per-channel inference
SAT_AGREEMENT_THRESHOLD = 0.25  # if this fraction (or more) of a session's channels independently
                                 # agree on the same abs_max value, use it as the rail for EVERY
                                 # channel in the session (including ones that never saturate and
                                 # so can't infer a rail on their own)
SAT_MIN_SAMPLES = 1             # samples at/beyond the rail needed to flag a window

# Last-resort override ONLY — used if infer_rail finds no repeated extreme
# (i.e. no evidence of clipping for that channel). NOT the primary detection
# path; kept for cases where you explicitly want a hard cutoff regardless.
SAT_FALLBACK_THRESHOLD_UV = None   # None = don't flag saturation for channels with no inferred rail

# ============================================================================
# STEP 2: FLATLINED CHANNELS
# ============================================================================

FLATLINE_WINDOW_SEC = 2.0
FLATLINE_VAR_THRESH = 0.5e-12   # V^2 — same default as raw_voltage_qc.FLATLINE_VAR_THRESHOLD

# ============================================================================
# STEP 3: NON-NEURAL GROSS ARTIFACT (session-level)
# ============================================================================

GROSS_WINDOW_SEC = 60.0
GROSS_STD_THRESH = 5.0

# ============================================================================
# SUMMARY / REVIEW FLAGGING
# ============================================================================

FLAG_REVIEW_STD_THRESH = 3.0    # flag subject/channel if pct_windows_excluded is this many
                                 # cross-subject stds above the mean, per artifact type

ARTIFACT_TYPES = ['saturation', 'flatline', 'gross_artifact']


def save_table(df, path):
    """Write a dataframe to disk. CSV for now (pyarrow isn't installed);
    swap to df.to_parquet(path) once pyarrow is added to the venv."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def append_table(df, path):
    """
    Append a dataframe to a CSV, writing the header only if the file doesn't
    exist yet. Used to stream per-run results to disk instead of holding
    every run's rows in memory for an entire (possibly 100+ run) session.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, mode='a', header=not path.exists(), index=False)


def reset_table(path):
    """Delete a per-window table if it exists, so a re-run starts clean
    instead of appending onto a stale/partial file from a previous attempt."""
    path = Path(path)
    if path.exists():
        path.unlink()


def ensure_output_dirs():
    for d in (PER_WINDOW_DIR, SUMMARY_DIR, PLOTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
