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

from datetime import datetime
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
PSD_TIMING_ROOT      = QC_ROOT / 'psd_timing'


def psd_timing_dir():
    """Which PSD runs were written by the CURRENT windowing design.

    Under qc/ rather than preprocessed/ because it is a data-QUALITY fact about the
    stored PSD that every downstream consumer inherits, which is what this tree is
    for — not a feature. It deliberately does NOT follow the
    metrics/exclusions/masks layout of the other levels: there is no threshold to
    sweep here. A run either was or was not written by the current algorithm, so
    there is one table and no metric/threshold split to make."""
    return PSD_TIMING_ROOT


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
# NO LONGER baked into the epoch cache. As of 2026-07-27 the cache stores raw
# per-window slices with no mask applied and no mask column; masking is a
# view-time join on (run, channel, 60s bin). So switching masks is now FREE —
# it does not invalidate the cache. This label is what views and analyses use
# by default.
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
# FEATURE-LEVEL QC  (P2.1)
# ============================================================================
# The third QC level, alongside raw_voltage/ and bipolar/, and it reuses the same
# metrics/ + exclusions/<type>/<label>/ + masks/<label>/ + validation/ layout via
# the generic builders above. Two differences from those levels, both deliberate:
#
# 1. NEW ARTIFACTS ARE PARQUET, not CSV. The raw-voltage tree stays CSV because
#    it already exists and works (docs/io_conventions.md §7); this level is new,
#    so it follows the current convention. Per-window feature metrics are also
#    far too large for CSV.
#
# 2. METRICS ARE SCOPED BY THE RAW-VOLTAGE MASK LABEL. The baseline mean/std
#    excludes windows that mask already flagged (feature_qc_params.
#    FEATURE_BASELINE_EXCLUDES_RAW_VOLTAGE), so the metric is a function of which
#    mask was subtracted first. Baking the label into the path keeps two
#    candidates' metrics from silently overwriting each other -- exactly the
#    collision build_bipolar_exclusions.py hit on 2026-07-27 (SCRATCHPAD.md) by
#    NOT namespacing on the mask that produced it.
#
# Four metric tables per subject/session, because they answer different
# questions at very different sizes:
#
#   baseline/    per (channel, bin): mean, std, counts. Tiny. The actual metric.
#   per_window/  per (run, channel, window): z order statistics. SPARSE -- only
#                rows above FEATURE_METRIC_STORE_FLOOR (see that constant; the
#                denominators live in summary/ so nothing is silently capped).
#   summary/     per (run, channel): n_windows / n_stored / n_nonfinite /
#                n_rv_excluded. Dense, tiny, and the denominator for every rate.
#   zhist/       per (run, channel): histogram of the z order statistic on a
#                fixed grid. Dense, tiny, and preserves distribution SHAPE for
#                structural threshold-setting even though per_window/ is sparse.


# Which QC level's mask was subtracted before the baseline was taken. The prefix
# is part of the scope directory name because the two are NOT interchangeable:
# a bipolar mask is (raw_voltage[anode] | raw_voltage[cathode]) | bipolar_variance,
# i.e. a strict superset of the raw-voltage-only projection, so it yields a
# different mean/std for the same channel. Without the prefix the two would
# collide on the label -- and the bipolar label already ENDS in the raw-voltage
# label it was rolled against ('std10_rv-gross-...'), which would read as a
# raw-voltage scope.
FEATURE_MASK_LEVEL_PREFIX = {'raw_voltage': 'rv', 'bipolar': 'bp'}


def feature_mask_scope(mask_label=None, level='raw_voltage'):
    """The scope string naming which upstream mask was subtracted, e.g.
    'bp-std10_rv-gross-std3_satmargin15_sw_logz4' or 'rv-gross-std3_...'.

    ONE definition, used for BOTH the metrics scope directory and the downstream
    exclusion/mask labels. That is the point: if the two were built separately they
    could drift, and an exclusion label that did not name its upstream mask would
    let two different baselines write into one directory -- the same collision
    build_bipolar_exclusions.py hit on 2026-07-27, and the reason
    bipolar_mask_label() appends the raw-voltage label it was rolled against.
    """
    if level not in FEATURE_MASK_LEVEL_PREFIX:
        raise ValueError(f'unknown feature mask level: {level} '
                         f'(expected one of {sorted(FEATURE_MASK_LEVEL_PREFIX)})')
    prefix = FEATURE_MASK_LEVEL_PREFIX[level]
    return f'{prefix}-{mask_label}' if mask_label else f'{prefix}-none'


def feature_metrics_dir(kind, mask_label=None, level='raw_voltage'):
    """One of the four feature-level metric tables, scoped by the mask subtracted.

    kind: 'baseline' | 'per_window' | 'summary' | 'zhist'.
    level: which QC level the mask came from -- see FEATURE_MASK_LEVEL_PREFIX.
    mask_label=None means the baseline was computed UNMASKED, recorded as
    '<prefix>-none' rather than being indistinguishable from a masked run.
    """
    if kind not in ('baseline', 'per_window', 'summary', 'zhist'):
        raise ValueError(f'unknown feature metric kind: {kind}')
    return (metrics_root(FEATURE_LEVEL_ROOT) / kind
            / feature_mask_scope(mask_label, level))


def feature_metrics_path(kind, subject, session, mask_label=None, level='raw_voltage'):
    return (feature_metrics_dir(kind, mask_label, level)
            / f'sub-{subject}_ses-{session}.parquet')


# Subjects whose stored PSD is being re-extracted (the superseded 60s-hop design).
# Feature-level metrics read psd_log_bins, so they are invalid for these subjects
# until the re-run lands -- rerun_psd_nonstandard.sbatch says so itself.
#
# READ FROM THE AUDIT, never hardcoded, so this cannot drift from what the timing
# audit actually found. Same principle rerun_psd_nonstandard.sbatch applies to its
# own subject list, and it is the file that script consumes.
#
# PSD_TIMING_ROOT itself is declared with the other QC level roots above, beside
# psd_timing_dir() -- the two arrived from different branches and must not be
# redeclared here, or the two spellings could drift apart silently.
PSD_RERUN_SUBJECTS_TXT = PSD_TIMING_ROOT / 'psd_rerun_subjects.txt'


def psd_rerun_subjects():
    """Subject IDs whose PSD is being re-extracted; empty set if the audit file is
    absent (i.e. no audit has run, so nothing is known to be stale)."""
    if not PSD_RERUN_SUBJECTS_TXT.exists():
        return set()
    return {line.strip() for line in PSD_RERUN_SUBJECTS_TXT.read_text().splitlines()
            if line.strip()}


def feature_metrics_deferred_path(subject, session):
    """Marker recording that a subject/session was DEFERRED rather than processed.

    A durable, greppable record so "which subjects still need adding?" is answerable
    from the artifact tree rather than from a scrollback of log lines. One file per
    subject/session, so parallel array tasks cannot race on it -- same reason
    metrics_run_info_dir is per-subject.
    """
    return (metrics_run_info_dir(FEATURE_LEVEL_ROOT)
            / f'sub-{subject}_ses-{session}_DEFERRED.json')


def feature_metrics_run_info_path(subject, session):
    """Per-subject/session record of how the metrics were produced (thresholds,
    parent mask, git provenance). One file per subject/session so parallel array
    tasks cannot race on a shared file -- same reason as
    metrics_run_info_dir(level_root) at the raw-voltage level."""
    return metrics_run_info_dir(FEATURE_LEVEL_ROOT) / f'sub-{subject}_ses-{session}.json'


def feature_exclusion_path(subject, session, artifact_type, label):
    return (exclusion_dir(FEATURE_LEVEL_ROOT, artifact_type, label)
            / f'sub-{subject}_ses-{session}.parquet')


def feature_mask_path(subject, session, label):
    return mask_dir(FEATURE_LEVEL_ROOT, label) / f'sub-{subject}_ses-{session}.parquet'


# ============================================================================
# BIPOLAR-LEVEL QC  (the pair-keyed level)
# ============================================================================
# Two naming asymmetries here are inherited, not chosen, so they are spelled out
# rather than silently absorbed by callers:
#
# 1. Exclusions are ONE FILE PER SUBJECT (`sub-019.csv`) with session_id as a
#    COLUMN, because build_bipolar_exclusions.py reads a per-subject metric CSV.
#    Every other level is one file per subject/session. Masks below go back to
#    per-subject/session, matching raw_voltage and feature_level.
# 2. Exclusions are CSV (existing artifacts, ~82 subjects on disk); the mask is
#    PARQUET. That follows io_conventions.md §7 -- do not bulk-convert what
#    exists, write everything NEW as Parquet -- and matches feature_mask_path()
#    one level over, which is already Parquet.
#
# There is only ONE bipolar artifact type (`bipolar_variance`), so unlike
# raw_voltage (four detectors) the exclusions table is already that level's
# union. The mask below is therefore not an OR across bipolar detectors; it is
# the JOIN of this level with the raw-voltage level, projected onto pairs.

BIPOLAR_ARTIFACT_TYPE = 'bipolar_variance'


def bipolar_exclusion_dir(label):
    return exclusion_dir(BIPOLAR_LEVEL_ROOT, BIPOLAR_ARTIFACT_TYPE, label)


def bipolar_exclusion_path(subject, label):
    """Per-SUBJECT (not per-session) bipolar_variance exclusions -- see note 1."""
    return bipolar_exclusion_dir(label) / f'sub-{subject}.csv'


def bipolar_mask_dir(label):
    return mask_dir(BIPOLAR_LEVEL_ROOT, label)


def bipolar_mask_path(subject, session, label):
    """The rolled-up bipolar mask: raw-voltage (projected to pairs) OR
    bipolar_variance. Parquet -- see note 2."""
    return bipolar_mask_dir(label) / f'sub-{subject}_ses-{session}.parquet'


def bipolar_mask_label(bipolar_variance_label, raw_voltage_label=None):
    """Default output label, encoding BOTH inputs.

    A bare `std10` is not a safe mask name: the same bipolar label rolled against
    a different raw-voltage mask is a DIFFERENT mask, and re-running would
    silently overwrite the earlier one. That already happened once at the
    exclusions level on 2026-07-27 -- an 82-subject `std10` run clobbered a
    17-subject `std10` run for all 17 overlapping subjects (SCRATCHPAD). Encoding
    both inputs in the directory name makes the collision impossible instead of
    merely documented.
    """
    return f'{bipolar_variance_label}_rv-{raw_voltage_label or CANONICAL_MASK_LABEL}'


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

# Throwaway exploration plots. NOTE the deviation: architecture.md PART 4 draws
# this as analysis/pain/scratch/, i.e. per-event. Left flat deliberately — runs
# already exist at this path and it is deletable output either way; whether
# scratch should be per-event at all is an open scratchpad question. A deliberate
# analysis output is NOT this: it goes through analysis_run_dir() below.
PLOTS_ROOT = ANALYSIS_DIR / 'scratch'


def pain_scores_csv(subject, session):
    return (Path(RAW_DIR) / f'sub-{subject}' / f'ses-{session}' / 'ehr'
            / f'sub-{subject}_ses-{session}_pain-scores.csv')


def med_admin_csv(subject, session):
    """The MAR (medication administration record) export for one session.

    Sibling of pain_scores_csv in the same ehr/ folder. 98 of these exist across
    96 subjects; not every subject with iEEG has one, which is why the med
    cohort is defined by med_admin_files() rather than by the file registry.
    """
    return (Path(RAW_DIR) / f'sub-{subject}' / f'ses-{session}' / 'ehr'
            / f'sub-{subject}_ses-{session}_med-admin.csv')


def med_admin_files():
    """Every MAR export on disk, sorted. THIS is the medication cohort.

    Deliberately a glob rather than a cohort file: the question these figures
    answer is "what was administered in this dataset", so the denominator is
    "has an EHR medication export", not "has usable iEEG". A subject whose
    recording failed QC still received the same drugs.
    """
    return sorted(Path(RAW_DIR).glob('sub-*/ses-*/ehr/*_med-admin.csv'))


def epoch_channel_power_csv(subject, session):
    return CACHE_DIR / f'sub-{subject}_ses-{session}_epoch_channel_power.csv'


def epoch_channel_power_provenance_json(subject, session):
    return CACHE_DIR / f'sub-{subject}_ses-{session}_epoch_channel_power.provenance.json'


# ============================================================================
# PAIN EPOCH CACHE — the Phase 1 base unit  (P1.1)
# ============================================================================
# ONE base unit = one (epoch definition, QC mask) pair, because those two are
# the only things baked INTO the cache and therefore the only two that force an
# expensive rebuild (architecture.md PART 4):
#
#   features/pain/psd_epochs/<epoch_label>_mask-<mask_label>/
#     manifest.json                                  the unit's self-description
#     cache/       sub-XXX_ses-YY_epochs.parquet      per-window log-power, masked, PRE-norm
#     epoch_defs/  sub-XXX_ses-YY_defs.parquet        run + window indices + pain label + mask ref
#     views/       <label>_<config_hash>/             OPTIONAL materialized views
#
# Subject/session live in the FILENAME, not in rows; epochs stack inside one
# file under an epoch_id column. The only fingerprinted name in the whole tree
# is a materialized view's directory (io.sidecar.config_hash) — runs and plots
# get a human label plus a timestamp.

CACHE_SUBDIR = 'cache'
EPOCH_DEFS_SUBDIR = 'epoch_defs'
VIEWS_SUBDIR = 'views'


def epoch_label(minutes_before=None):
    """The epoch definition's directory-name fragment, e.g. 'epoch-5min-pre'.

    pain_params is imported lazily so this module stays importable on its own
    and paths <-> params can never become a circular import.
    """
    if minutes_before is None:
        from ieeg_ehr.config.pain_params import EPOCH_MINUTES_BEFORE
        minutes_before = EPOCH_MINUTES_BEFORE
    return f'epoch-{minutes_before:g}min-pre'


def pain_epoch_unit_dir(minutes_before=None):
    """The base-unit directory for one epoch definition, e.g. 'epoch-5min-pre'.

    CHANGED 2026-07-27: this used to be keyed on (epoch definition, QC mask) —
    `epoch-5min-pre_mask-<label>`. The cache no longer applies or records a QC
    mask; masking moved to the view layer, so the cache depends ONLY on the
    epoch definition. Keeping the mask in the name would have asserted a
    dependency that no longer exists, and would have forced a full rebuild of a
    ~47 GB artifact every time the mask changed.

    Consequence: CLAUDE.md's "new cache ONLY for a new epoch length or a new QC
    mask" collapses to just epoch length, and P0.1 (pinning the mask) no longer
    blocks building the cache. See architecture.md PART 1.
    """
    return PAIN_FEATURES_ROOT / epoch_label(minutes_before)


def pain_epoch_manifest_path(minutes_before=None):
    """The base unit's manifest.json — window length, anchor, mask label, bin
    edges, dtype, git, date. Written once per unit (io.write_manifest), not once
    per subject, and digested by every view sidecar that depends on the unit.

    The filename is spelled out rather than imported from io.sidecar.MANIFEST_NAME
    to keep config free of an io dependency (io does not import config either, and
    that one-way boundary is worth a literal); tests assert both spellings agree."""
    return pain_epoch_unit_dir(minutes_before) / 'manifest.json'


def pain_epoch_cache_path(subject, session, minutes_before=None):
    return (pain_epoch_unit_dir(minutes_before) / CACHE_SUBDIR
            / f'sub-{subject}_ses-{session}_epochs.parquet')


def pain_epoch_defs_path(subject, session, minutes_before=None):
    return (pain_epoch_unit_dir(minutes_before) / EPOCH_DEFS_SUBDIR
            / f'sub-{subject}_ses-{session}_defs.parquet')


CHANNEL_META_SUBDIR = 'channel_meta'


def pain_epoch_channel_meta_path(subject, session, minutes_before=None):
    """Per-channel metadata the cache cannot hold: pair ORDER (which decodes the
    cache's C-order ravel) and DK labels (which the region axis needs), keyed
    (run_id, pair_index).

    Lives beside the cache rather than in views/ because it is a property of the
    cache's own encoding, not of any one view config -- every view of this unit
    reads the same table. Run TIMING is deliberately NOT here: it is constant per
    run, so it goes in the epoch_defs index instead."""
    return (pain_epoch_unit_dir(minutes_before) / CHANNEL_META_SUBDIR
            / f'sub-{subject}_ses-{session}_channels.parquet')


def pain_epoch_views_dir(view_label, config_hash, minutes_before=None):
    """Where a MATERIALIZED view lands — disposable performance cache, deletable.

    A view that a human reads or a model consumes is not this; that is an
    ANALYSIS output under analysis_run_dir(). Materialize here only when
    recompute is *measured* slow and something depends on it."""
    return (pain_epoch_unit_dir(minutes_before) / VIEWS_SUBDIR
            / f'{view_label}_{config_hash}')


# ============================================================================
# ANALYSIS RUN DIRECTORIES — the 5-level scheme
# ============================================================================
# 1 <event>/  2 <question>/  3 <output_type>/  4 <view_scheme>/(optional)
# 5 <run_name>_<timestamp>/
#
# Levels 1-2 are opened DELIBERATELY: a new event domain, or a NAMED question
# that already exists in the exploration log. Levels 3-5 are created freely per
# run. Sweep combinatorics go into ROWS of a sweeps/ results.parquet, never into
# folders. Discovery vs confirmation is a cohort reference in config, not a
# folder level — and which subjects were in a run is read from the run's
# provenance.json subjects[], never from its name.

def _run_stamp(timestamp=None):
    return timestamp or datetime.now().strftime('%Y%m%d-%H%M%S')


def analysis_run_dir(question, output_type, run_name, view_scheme=None,
                     event='pain', timestamp=None):
    """Build (do not create) the level-5 run directory.

    A timestamp is ALWAYS appended so two runs can never overwrite each other's
    provenance.json — that has bitten this project once already.
    """
    path = ANALYSIS_DIR / event / question / output_type
    if view_scheme:
        path = path / view_scheme
    stamp = _run_stamp(timestamp)
    return path / (f'{run_name}_{stamp}' if run_name else stamp)


def sweep_run_dir(run_name, event='pain', timestamp=None):
    """A tiered nomination run. All grid combinatorics live as ROWS in this
    run's results table — never as sibling folders."""
    return ANALYSIS_DIR / event / 'sweeps' / f'{run_name}_{_run_stamp(timestamp)}'


# ============================================================================
# MEDICATION ADMINISTRATION  —  level-1 event 'meds'
# ============================================================================
# Opened deliberately (PLANNING.md, "Medication administration patterns"). This
# is a second level-1 event beside 'pain': the unit of analysis is a drug
# administration from the EHR, not a pain epoch, and nothing in it touches the
# PSD cache or the view chain. Level 2 is the named question; there is no level-4
# view_scheme because there are no views here.

MED_EVENT = 'meds'
MED_ANALYSIS_ROOT = ANALYSIS_DIR / MED_EVENT
MED_DEFAULT_QUESTION = 'administration_patterns'

#: Frozen snapshot of the taxonomy source table (config/med_taxonomy.py records
#: the live path). Copied once so a run's provenance points at something stable.
MED_TAXONOMY_SNAPSHOT = MED_ANALYSIS_ROOT / 'medications_classified_snapshot.csv'


def med_run_dir(output_type, run_name, question=MED_DEFAULT_QUESTION,
                timestamp=None):
    """Build (do not create) a run directory under the 'meds' event.

    Thin wrapper over analysis_run_dir with event='meds' and no view_scheme.
    Deliberately NOT analysis.view_tables.resolve_run_dir, which hardcodes
    event='pain' and expects a view directory these scripts do not have.
    """
    return analysis_run_dir(question=question, output_type=output_type,
                            run_name=run_name, event=MED_EVENT,
                            timestamp=timestamp)


# ============================================================================
# COHORTS
# ============================================================================
# Age is PHI and is NOT on Sherlock: demographic matching happens offline, and
# only the anonymised subject_id -> cohort assignment plus SAFE matching axes
# cross over. Cohort membership lives here, never in a folder name.

EXPLORATORY_SUBJECTS_TXT = COHORTS_ROOT / 'subjects_qc_raw_voltage_normal.txt'

# EMPTY as of 2026-07-28. sub-236 was the only entry: its raw_voltage exclusion
# rollups were incomplete at the newer sweep labels (docs/qc_context.md, "sub-236
# gap"), so it could not be combined into the pinned mask. That is fixed — its one
# real gap was `square_wave/frac0.9`, its metrics were complete all along (107 of
# 107 readable runs), and it now has a pinned raw-voltage mask. Its PSD is also
# clean: 107/107 runs on the current single-level design at a 1 s hop
# (qc/psd_timing/, 2026-07-28), so it is not among the subjects awaiting a PSD
# re-run.
#
# Kept as an empty set rather than deleted: it is the sanctioned place to park a
# subject that must not enter exploratory work, and re-adding one should be a
# one-line change rather than re-plumbing exploratory_subjects().
_EXCLUDE_FROM_EXPLORATORY = set()


def exploratory_subjects():
    subjects = [
        line.strip() for line in EXPLORATORY_SUBJECTS_TXT.read_text().splitlines()
        if line.strip()
    ]
    return [s for s in subjects if s not in _EXCLUDE_FROM_EXPLORATORY]
