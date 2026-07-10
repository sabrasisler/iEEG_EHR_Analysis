"""
Shared configuration for the QC diagnostics pipeline (saturation, flatline,
non-neural gross artifact). All thresholds/paths live here — no magic numbers
in the detection/analysis modules.
"""

import datetime
import json
import subprocess
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent   # the git repo root (parent of qc_scripts/)


def run_timestamp():
    """Local date+time a script ran, ISO-8601 with tz offset — recorded in every
    sidecar (run_info/*.json, params.json) so you can tell when outputs were made."""
    return datetime.datetime.now().astimezone().isoformat()

# ============================================================================
# PATHS
# ============================================================================

RAW_DIR = '/oak/stanford/groups/ckeller1/data/iEEG_EHR/iEEG_NWB'
FILE_REGISTRY_CSV = f'{RAW_DIR}/sherlock_file_registry.csv'

# Canonical analysis root + the qc/<level>/ layout (see the plan). A "level" is a
# data-processing stage (raw_voltage now; bipolar / features later). Within a
# level: metrics/ (expensive detection outputs, once), exclusions/<type>/<label>/
# (cheap per-artifact-type thresholded 60s tables), masks/<label>/ (combined mask
# → bipolar), _validation/ (diagnostic scratch).
ANALYSIS_DIR = Path('/oak/stanford/groups/ckeller1/data/iEEG_EHR/derivatives/sisler/analysis')
DEFAULT_LEVEL_ROOT = ANALYSIS_DIR / 'qc' / 'raw_voltage'


def metrics_root(level_root):
    return Path(level_root) / 'metrics'

def metrics_per_window_dir(level_root):
    return metrics_root(level_root) / 'per_window'

def metrics_run_info_dir(level_root):
    # per-subject JSON records of how the metrics were produced (detection params
    # + git provenance + run_timestamp). One file per subject to avoid parallel
    # array tasks racing on a shared file.
    return metrics_root(level_root) / 'run_info'

def exclusion_dir(level_root, artifact_type, label):
    return Path(level_root) / 'exclusions' / artifact_type / label

def mask_dir(level_root, label):
    return Path(level_root) / 'masks' / label

def validation_dir(level_root):
    return Path(level_root) / '_validation'


def git_provenance():
    """
    Record what code actually ran: commit hash + whether the working tree is
    dirty + the list of modified files. A bare hash is misleading when the tree
    has uncommitted changes, so callers should warn when `dirty` is True and the
    recommended workflow is to commit+push before a definitive run.

    Uses cwd=REPO_DIR rather than `git -C` — the compute nodes' system git
    (/usr/bin/git) is old and lacks -C. If git is unavailable/fails, returns
    available=False rather than silently reporting a clean tree.
    """
    def _git(*args):
        try:
            r = subprocess.run(['git', *args], cwd=str(REPO_DIR),
                               capture_output=True, text=True)
        except FileNotFoundError:
            return None
        return r.stdout.strip() if r.returncode == 0 else None

    commit = _git('rev-parse', 'HEAD')
    if commit is None:
        return {'available': False, 'commit': None, 'dirty': None, 'modified_files': []}
    porcelain = _git('status', '--porcelain') or ''
    modified = [line[3:] for line in porcelain.splitlines()] if porcelain else []
    return {'available': True, 'commit': commit, 'dirty': bool(modified),
            'modified_files': modified}


def warn_if_dirty(prov=None):
    """Print a loud warning (and return it) if the recorded code state is dirty."""
    prov = prov if prov is not None else git_provenance()
    if not prov.get('available'):
        print("  WARNING: could not read git provenance (git unavailable here) — "
              "commit hash NOT recorded. Capture it at submission time on the login node.",
              flush=True)
    elif prov['dirty']:
        print(f"  WARNING: git working tree is DIRTY ({len(prov['modified_files'])} modified "
              f"files) — recorded commit {prov['commit']} does NOT reflect what ran. "
              f"Commit + push before a definitive run for faithful provenance.", flush=True)
    return prov


# --- Back-compat globals (per_window/summary/plots under one dir) used by the
#     detection pass + summarize/plot scripts. Detection points these at a
#     level's metrics/; other scripts repoint via set_output_dir(). ---
OUTPUT_DIR = metrics_root(DEFAULT_LEVEL_ROOT)
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
# STEP 2b: SQUARE-WAVE / TWO-LEVEL ARTIFACT (see detect_square_wave.py)
# ============================================================================
# A digital/relay-style artifact where nearly all samples in a window sit at
# two discrete levels (e.g. a 0-50µV square wave): flatline misses it (high
# variance), saturation misses it (not at the rail), gross_artifact misses it
# (mean-neutral). Metric = fraction of samples pinned within EPS_FRAC of the
# window's own min/max — dimensionless, so amplitude- AND frequency-independent
# (no per-case tuning). Only two shape knobs; the range guard is derived from
# FLATLINE_VAR_THRESH, not a free parameter.
SQUARE_WINDOW_SEC = 2.0         # shared 2s granularity, consistent with flatline/saturation
SQUARE_EPS_FRAC = 0.05          # band around each level, as a fraction of the window's range
SQUARE_FRAC_THRESH = 0.9        # exclude if >= this fraction of samples sit at the two levels
# Range floor: below this peak-to-peak swing a window is effectively flat
# (flatline's job) and every sample is trivially "near" both extremes. Tied to
# the flatline threshold — the p2p range whose implied variance == FLATLINE_VAR_THRESH.
SQUARE_MIN_RANGE_V = 2 * (FLATLINE_VAR_THRESH ** 0.5)   # ≈ 1.41e-6 V (1.4µV)

# ============================================================================
# STEP 3: NON-NEURAL GROSS ARTIFACT (session-relative high-variance/amplitude
# bursts, e.g. unplug/replug — NOT DC-offset/drift; see detect_gross_artifact.py)
# ============================================================================

GROSS_WINDOW_SEC = 60.0
GROSS_STD_THRESH = 5.0   # one-sided: only anomalously HIGH variance is excluded

# ============================================================================
# SUMMARY / REVIEW FLAGGING
# ============================================================================

FLAG_REVIEW_STD_THRESH = 3.0    # flag subject/channel if pct_windows_excluded is this many
                                 # cross-subject stds above the mean, per artifact type

ARTIFACT_TYPES = ['saturation', 'flatline', 'square_wave', 'gross_artifact']


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
