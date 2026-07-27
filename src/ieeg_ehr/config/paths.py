"""
Single source of truth for every path this project reads or writes.

CODE/DATA BOUNDARY (see CLAUDE.md): this repo holds code only. Every output
path below resolves under DERIVATIVES_BASE on Oak — never a repo-relative path.
If a script has no explicit output base it must use these constants, not './'.

Layout under DERIVATIVES_BASE (architecture.md PART 4):

    preprocessed/   stored feature families, continuous + per-window (PSD, later PAC)
    qc/             data-quality facts, inherited by every view
                      raw_voltage/  bipolar/  feature_level/
    features/       event-sliced per-window caches + epoch definitions
    cohorts/        subject_id -> cohort assignments (SAFE axes only, no age)
    analysis/       terminal outputs, organised by question (5-level scheme)
    outdated/       superseded derivatives kept for reference
"""

from pathlib import Path

from ieeg_ehr._repo import REPO_DIR   # noqa: F401  (re-exported; provenance + callers use it)

# ============================================================================
# RAW INPUT
# ============================================================================

RAW_DIR = '/oak/stanford/groups/ckeller1/data/iEEG_EHR/iEEG_NWB'
FILE_REGISTRY_CSV = f'{RAW_DIR}/sherlock_file_registry.csv'

# ============================================================================
# DERIVATIVES ROOT + TOP-LEVEL TREE
# ============================================================================

DERIVATIVES_DIR = Path('/oak/stanford/groups/ckeller1/data/iEEG_EHR/derivatives')
DERIVATIVES_BASE = DERIVATIVES_DIR / 'sisler'

PREPROCESSED_ROOT = DERIVATIVES_BASE / 'preprocessed'
QC_ROOT           = DERIVATIVES_BASE / 'qc'
FEATURES_ROOT     = DERIVATIVES_BASE / 'features'
COHORTS_ROOT      = DERIVATIVES_BASE / 'cohorts'
ANALYSIS_DIR      = DERIVATIVES_BASE / 'analysis'
OUTDATED_ROOT     = DERIVATIVES_BASE / 'outdated'

# ============================================================================
# QC LEVELS
# ============================================================================
# A "level" is a processing stage the QC machinery runs against. Within a level:
#   metrics/     expensive detection outputs, computed once
#   exclusions/<artifact_type>/<label>/   cheap thresholded 60s tables
#   masks/<label>/                        OR'd union of exclusions
#   validation/                           diagnostics, incl. threshold_sweeps/
# The metric/threshold split means thresholds can be swept without recomputing
# metrics.

DEFAULT_LEVEL_ROOT   = QC_ROOT / 'raw_voltage'
BIPOLAR_LEVEL_ROOT   = QC_ROOT / 'bipolar'
FEATURE_LEVEL_ROOT   = QC_ROOT / 'feature_level'


def metrics_root(level_root):
    return Path(level_root) / 'metrics'


def metrics_per_window_dir(level_root):
    return metrics_root(level_root) / 'per_window'


def metrics_run_info_dir(level_root):
    # Per-subject JSON records of how the metrics were produced (detection
    # params + git provenance + run_timestamp). One file per subject so
    # parallel array tasks can't race on a shared file.
    return metrics_root(level_root) / 'run_info'


def exclusion_dir(level_root, artifact_type, label):
    return Path(level_root) / 'exclusions' / artifact_type / label


def mask_dir(level_root, label):
    return Path(level_root) / 'masks' / label


def validation_dir(level_root):
    return Path(level_root) / 'validation'


def threshold_sweep_dir(level_root):
    return validation_dir(level_root) / 'threshold_sweeps'


# ============================================================================
# PINNED QC MASK
# ============================================================================
# Baked into every epoch cache, so changing it means a full expensive re-run.
#
# TODO(P0.1): formally pin this. The two full-cohort (83 subject-session)
# candidates are 'gross-std3_satmargin15_sw' and
# 'gross-std3_satmargin15_sw_logz4'. The latter is the current default: it is
# the stricter of the two and the only one with summary/ and
# plots/{flagged,random}_examples/ built.
#
# NOTE: this replaces pain_analysis.config.DEFAULT_MASK_LABEL, which pointed at
# 'gross-std3_satmargin5_logz4' — a 17-subject PILOT label, not a full-cohort
# mask. That was a bug; the pilot labels were deleted in the 2026-07 refactor.
CANONICAL_MASK_LABEL = 'gross-std3_satmargin15_sw_logz4'

RAW_VOLTAGE_MASK_DIR = DEFAULT_LEVEL_ROOT / 'masks'


def raw_voltage_mask_dir(label=None):
    """The pinned (or named) raw-voltage mask directory.

    Distinct from mask_dir(level_root, label) above, which is the generic
    two-argument builder used by the QC machinery.
    """
    return RAW_VOLTAGE_MASK_DIR / (label or CANONICAL_MASK_LABEL)


def mask_csv(subject, session, label=None):
    return raw_voltage_mask_dir(label) / f'sub-{subject}_ses-{session}.csv'


# ============================================================================
# PREPROCESSED FEATURE FAMILIES
# ============================================================================
# Deliberately separate from qc/ (BIDS-like derivatives convention; keeps large
# NWB outputs out of the CSV-oriented QC tree).

BIPOLAR_PSD_DERIV_ROOT = PREPROCESSED_ROOT / 'bipolar_fft'


def bipolar_fft_params_path(psd_out_root, subject):
    # Subject-level sidecar living IN the bipolar_fft tree itself (not under
    # qc/), so it survives independently of whether QC has been run: re-ref
    # type + FFT/PSD params + git provenance for whatever preprocessing pass
    # most recently produced this subject's PSD. Overwritten on each re-run.
    return Path(psd_out_root) / f'sub-{subject}' / f'sub-{subject}_bipolar_fft_params.json'


def bipolar_psd_nwb_path(subject, session, run):
    return (BIPOLAR_PSD_DERIV_ROOT / f'sub-{subject}' / f'ses-{session}'
            / f'sub-{subject}_ses-{session}_run-{run}_bipolar_psd.nwb')


def bipolar_trace_cache_dir():
    """Scratch cache of full bipolar-referenced traces for a sample of runs, so
    QC threshold/mask experiments don't re-read raw NWB every time.

    Deliberately on $SCRATCH, not $OAK — throwaway and reproducible on demand,
    not a derivative worth keeping (90-day inactivity purge applies). Raises if
    $SCRATCH isn't set rather than silently falling back to something durable.
    """
    import os
    return Path(os.environ['SCRATCH']) / 'bipolar_trace_cache'


# ============================================================================
# PAIN EPOCH FEATURES + ANALYSIS
# ============================================================================

PAIN_ANALYSIS_ROOT = ANALYSIS_DIR / 'pain'
PAIN_FEATURES_ROOT = FEATURES_ROOT / 'pain' / 'psd_epochs'

# Legacy CSV cache from the pre-Phase-1 pipeline. Archived under outdated/ in
# the 2026-07 refactor; P1.1 replaces it with a per-window Parquet cache under
# PAIN_FEATURES_ROOT. Kept as a constant so the old plot scripts still resolve.
LEGACY_PAIN_CACHE_DIR = OUTDATED_ROOT / 'legacy_65_subjects' / 'cache'
CACHE_DIR = LEGACY_PAIN_CACHE_DIR
PLOTS_ROOT = ANALYSIS_DIR / 'scratch'


def pain_scores_csv(subject, session):
    return (Path(RAW_DIR) / f'sub-{subject}' / f'ses-{session}' / 'ehr'
            / f'sub-{subject}_ses-{session}_pain-scores.csv')


def epoch_channel_power_csv(subject, session):
    return CACHE_DIR / f'sub-{subject}_ses-{session}_epoch_channel_power.csv'


def epoch_channel_power_provenance_json(subject, session):
    return CACHE_DIR / f'sub-{subject}_ses-{session}_epoch_channel_power.provenance.json'


# ============================================================================
# COHORTS
# ============================================================================
# Age is PHI and is NOT on Sherlock: demographic matching happens offline, and
# only the anonymised subject_id -> cohort assignment plus SAFE matching axes
# cross over. Cohort membership lives here, never in a folder name.

EXPLORATORY_SUBJECTS_TXT = COHORTS_ROOT / 'subjects_qc_raw_voltage_normal.txt'

# sub-236's raw_voltage exclusion rollups are incomplete at the newer sweep
# labels (docs/qc_context.md, "sub-236 gap"), so it can't safely be combined
# into the pinned mask yet.
_EXCLUDE_FROM_EXPLORATORY = {'236'}


def exploratory_subjects():
    subjects = [
        line.strip() for line in EXPLORATORY_SUBJECTS_TXT.read_text().splitlines()
        if line.strip()
    ]
    return [s for s in subjects if s not in _EXCLUDE_FROM_EXPLORATORY]
