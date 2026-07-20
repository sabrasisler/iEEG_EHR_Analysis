"""
Shared configuration for the QC diagnostics pipeline (saturation, flatline,
non-neural gross artifact). All thresholds/paths live here — no magic numbers
in the detection/analysis modules.
"""

import datetime
import json
import os
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
# → bipolar), validation/ (diagnostic scratch, incl. threshold_sweeps/).
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
    return Path(level_root) / 'validation'

def threshold_sweep_dir(level_root):
    return validation_dir(level_root) / 'threshold_sweeps'

def bipolar_fft_params_path(psd_out_root, subject):
    # Subject-level sidecar living IN the bipolar_fft derivatives tree itself
    # (not under qc/), so it survives independently of whether QC has been run
    # or re-run: re-referencing type + FFT/PSD params + git provenance for
    # whatever preprocessing pass most recently produced this subject's PSD
    # output. One file per subject, overwritten each time that subject's
    # preprocessing (run_pipeline_bipolar.py) is re-run.
    return Path(psd_out_root) / f'sub-{subject}' / f'sub-{subject}_bipolar_fft_params.json'


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
        # rstrip (not strip): `git status --porcelain`'s leading space is a
        # significant part of the XY status code (e.g. " M" = unstaged
        # modification) -- stripping it ate one character off the FIRST
        # modified file's path whenever that file's status began with a
        # space, e.g. "M preprocessing/x.py" -> line[3:] == "reprocessing/x.py".
        # Found via a real provenance JSON showing a truncated filename.
        return r.stdout.rstrip('\n') if r.returncode == 0 else None

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


# ============================================================================
# BIPOLAR RE-REFERENCE + PSD LEVEL (preprocessing/, not qc_scripts/ -- see
# preprocessing/CONTEXT-style notes: preprocessing computes, qc_scripts decides
# exclusions. Shared here since both sides need one provenance/path module.)
# ============================================================================

BIPOLAR_LEVEL_ROOT = ANALYSIS_DIR / 'qc' / 'bipolar'   # parallels DEFAULT_LEVEL_ROOT

# WHY 2.0s: matches raw_voltage's SAT/FLATLINE window so the bipolar variance
# detector's bins are directly alignable against raw_voltage masks (60s bins,
# which are just 30x this) without resampling.
BIPOLAR_VARIANCE_WINDOW_SEC = 2.0

# PSD params, given in seconds (resolved per-run via each run's own sfreq, not
# a fixed sample count) since sfreq varies across subjects in this dataset
# (mostly 1000/2000 Hz, occasionally 500 Hz).
#
# Single-level windowing (per lab discussion, superseding an earlier 60s
# outer-window design): each PSD_WINDOW_SEC window is its own periodogram-
# style estimate (no multi-segment Welch averaging within a coarser window),
# stepped by PSD_OVERLAP_FRAC overlap -- default 2s window / 50% overlap ->
# a PSD estimate every 1s. Matches the variance metric's 2s granularity far
# more closely than the old 60s-window scheme did, at the cost of a noisier
# per-window spectral estimate (accepted tradeoff, in exchange for much finer
# time resolution).
PSD_WINDOW_SEC = 2.0           # WHY: sets frequency resolution (sfreq/nperseg = 0.5 Hz)
                               # AND the time granularity of the PSD output.
PSD_OVERLAP_FRAC = 0.5         # WHY: 50% overlap -> 1s hop for the default 2s window.
PSD_WINDOW_FN = 'hann'
PSD_N_LOG_BINS = 50
PSD_FREQ_MIN_HZ = 1.0
PSD_FREQ_MAX_HZ = 250.0        # WHY: Nyquist-safe ceiling given rare 500 Hz-sampled subjects
                               # (Nyquist=250Hz there); can always restrict further downstream,
                               # can't recover truncated data later.
PSD_LINE_NOISE_FREQS_HZ = (60.0, 120.0, 180.0, 240.0)
PSD_LINE_NOISE_GUARD_HZ = 2.0  # +/- band around each harmonic flagged contains_line_noise

# Canonical bands for the separate downstream aggregation helper
# (preprocessing/bipolar_bands.py) -- NOT computed by the fused reref+PSD pass.
# Edges chosen to fall strictly between line-noise harmonics so no canonical
# band straddles a 60 Hz-multiple notch by construction.
CANONICAL_BANDS_HZ = {
    'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 12), 'beta': (13, 30),
    'low_gamma': (30, 58), 'high_gamma1': (65, 115), 'high_gamma2': (125, 175),
    'high_gamma3': (185, 235),
}

# WHY: same one-sided high-variance convention as GROSS_STD_THRESH, applied
# post-bipolar-derivation instead of on raw monopolar channels.
BIPOLAR_VARIANCE_STD_THRESH = 5.0


def bipolar_trace_cache_dir():
    """Scratch cache of full bipolar-referenced traces for a sample of runs
    (preprocessing/save_bipolar_sample_traces.py), so QC threshold/mask
    experiments (qc_scripts/plot_bipolar_flagged_runs.py) don't need to
    re-read raw NWB + re-reference every time. Deliberately on $SCRATCH, not
    $OAK -- this is throwaway/reproducible-on-demand data, not a derivative
    worth keeping long-term (see org policy: scratch/job I/O belongs on
    $SCRATCH, 90-day inactivity purge). Raises if $SCRATCH isn't set rather
    than silently falling back to something durable."""
    return Path(os.environ['SCRATCH']) / 'bipolar_trace_cache'

# Derivatives root for PSD NWB outputs -- deliberately separate from
# analysis/qc/ (BIDS-like derivatives/ convention, keeps large NWB outputs out
# of the CSV-oriented analysis tree). Nested under sisler/ (matching
# derivatives/sisler/analysis/'s existing convention) rather than directly
# under derivatives/, to keep this user's outputs namespaced alongside theirs.
DERIVATIVES_DIR = Path('/oak/stanford/groups/ckeller1/data/iEEG_EHR/derivatives')
BIPOLAR_PSD_DERIV_ROOT = DERIVATIVES_DIR / 'sisler' / 'preprocessed' / 'bipolar_fft'

# HDF5 chunking: default is uncapped (whole run's time axis in one chunk per
# channel). PSD rows are now spaced by the hop (PSD_WINDOW_SEC * (1 -
# PSD_OVERLAP_FRAC), ~1s by default) -- ~60x denser than the old 60s-window
# scheme, but a channel's entire run is still only single-digit MB even for
# long recordings (e.g. 2hr run: ~1.4MB/channel; 24hr: ~17MB/channel), still
# comfortably one chunk per channel. This differs from raw-voltage chunking
# (dense samples), which DOES need small time-chunks to hit a reasonable
# byte-size-per-chunk. Only set a cap for unusually long recordings.
PSD_HDF5_CHUNK_MAX_HOURS = None   # e.g. 4.0 to cap chunk size for exceptionally long runs
