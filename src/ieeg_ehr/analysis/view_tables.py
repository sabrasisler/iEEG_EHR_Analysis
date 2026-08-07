"""
Loading, group-level aggregation, and output placement for P1.3 view tables --
shared by every figure built from them.

Shared rather than duplicated per plot script for one reason: subject weighting.
Every helper here treats the SUBJECT as the unit of replication -- not the
electrode and not the epoch -- and if two scripts implemented that separately,
two figures of the same data could disagree and nothing would say which was
right. `plot_pain_view_heatmaps.py` and `plot_pain_view_spectra.py` both read
through this module.

The tables themselves come from `views/build_pain_epoch_view.py`, one pair per
subject/session:

    view_subject_sub-XXX_ses-YY.parquet   (subject_id, session_id, pain_bin,
                                           region, freq_bin_index, value,
                                           n_epochs, n_channels)
    view_epochs_sub-XXX_ses-YY.parquet    same, one row per epoch (epoch_id,
                                           pain_event_id, pain_score)

Nothing here recomputes a view. Reading the tables rather than the cache is what
makes a figure and the numbers behind it provably the same values.
"""

import logging

import numpy as np
import pandas as pd

from ieeg_ehr import config, io

logger = logging.getLogger(__name__)

# Which pain bins are DRAWN, in order, per binarization scheme (view_registry
# AXIS 7). 'none' is deliberately absent from both: under any baseline
# normalization it is its own reference and sits at ~0 by construction, so it
# carries no information as a panel or a line. It is still worth checking -- see
# `log_baseline_check` -- because a 'none' far from 0 means the baseline leaked.
PANELS = {
    'subject_relative': ['low', 'high'],
    'absolute': ['low', 'medium', 'high'],
}


def load_view_tables(view_dir, kind):
    """Concatenate every subject's `view_<kind>_sub-*.parquet` in `view_dir`."""
    paths = sorted(view_dir.glob(f'view_{kind}_sub-*.parquet'))
    if not paths:
        raise FileNotFoundError(f'no view_{kind}_*.parquet in {view_dir}')
    frames = [io.read_table(p, on_stale='warn') for p in paths]
    return pd.concat(frames, ignore_index=True), paths


def view_params_from(subject_paths):
    """(view_params, ViewConfig) read from a view table's own sidecar.

    From the ARTIFACT, never from a CLI flag: units, normalization and the scheme
    code all come from here, so a figure cannot claim a normalization it was not
    built with. Returns view_config=None if the sidecar is unreadable or predates
    a field, rather than raising -- the caller can still plot, it just has to be
    told the units.
    """
    sidecar = io.read_sidecar(subject_paths[0]) or {}
    params = sidecar.get('params', {})

    from ieeg_ehr.views.view_config import ViewConfig
    fields = {f.name for f in ViewConfig.__dataclass_fields__.values()}
    try:
        # Drop the non-axis keys the sidecar also carries (`split`,
        # `roi_scheme_contents`) -- they are provenance, not view axes.
        view = ViewConfig(**{k: v for k, v in params.items() if k in fields})
    except (TypeError, ValueError) as exc:
        logger.warning('could not reconstruct ViewConfig from %s: %s',
                       io.sidecar_path(subject_paths[0]), exc)
        view = None
    return params, view


def per_subject(subject_tables):
    """One value per (subject, pain_bin, region, freq_bin_index).

    Collapses sessions first. A subject with two sessions must count ONCE, and
    the view tables carry one row per subject/session -- so aggregating the raw
    rows would silently give that subject double weight. No-op today (the arrays
    run `--session 01`), which is exactly why it is worth doing before it isn't.
    """
    return (subject_tables
            .groupby(['subject_id', 'pain_bin', 'region', 'freq_bin_index'],
                     dropna=False)['value']
            .mean().reset_index())


def subject_stats(subject_tables):
    """(pain_bin, region, freq_bin_index) -> mean, sd, sem, n_subjects ACROSS
    subjects, equal-weighted.

    EQUAL-WEIGHTED because the subject is the unit of replication: a subject with
    200 contacts must not outvote one with 30. `sd` is the sample SD (ddof=1), so
    a cell backed by a single subject gets NaN rather than a fabricated 0 -- and
    therefore no error ribbon, which is the honest rendering.
    """
    grouped = (per_subject(subject_tables)
               .groupby(['pain_bin', 'region', 'freq_bin_index'], dropna=False)['value'])
    stats = grouped.agg(mean='mean', sd=lambda s: s.std(ddof=1),
                        n_subjects='count').reset_index()
    stats['sem'] = stats['sd'] / np.sqrt(stats['n_subjects'])
    return stats


def group_table(subject_tables):
    """The heatmaps' long (pain_bin, region, freq_bin_index) -> value table.

    Defined as `subject_stats`' mean rather than as its own aggregation, so the
    line a spectrum draws and the cell a heatmap shades are the same number by
    construction and cannot drift apart.
    """
    return (subject_stats(subject_tables)
            .rename(columns={'mean': 'value'})[['pain_bin', 'region',
                                                'freq_bin_index', 'value']])


def wide_by_bin(long_table, index_cols, panels):
    """Long -> one column per plotted pain bin, in `panels` order.

    Missing panels are materialized as all-NaN columns so downstream code can
    index them unconditionally instead of branching on presence.
    """
    wide = long_table.pivot_table(index=index_cols, columns='pain_bin', values='value')
    for panel in panels:
        if panel not in wide.columns:
            wide[panel] = np.nan
    return wide[panels].reset_index()


def epoch_counts(epoch_tables, by_subject=False):
    """(region, pain_bin) -> distinct contributing epochs.

    De-duplicated to one row per (subject, epoch, region) first: a freq-bin row is
    not a distinct epoch, and counting rows would inflate n by 50x.
    """
    keys = ['subject_id', 'region', 'pain_bin'] if by_subject else ['region', 'pain_bin']
    deduped = epoch_tables.drop_duplicates(['subject_id', 'epoch_id', 'region', 'pain_bin'])
    counts = deduped.groupby(keys).size()
    if by_subject:
        counts.index = counts.index.set_names(['subject', 'region', 'pain_bin'])
    return counts


def subjects_per_region(stats, panels):
    """region -> contributing subjects, as the MINIMUM across the plotted bins.

    The minimum, not the union: both lines of a two-line panel have to be backed
    by subjects for the comparison between them to mean anything, so the smaller
    count is the one that governs the panel. The full per-(region, bin, freq bin)
    breakdown is written to the run's table for anyone who needs it.
    """
    per_bin = (stats[stats['pain_bin'].isin(panels)]
               .groupby(['region', 'pain_bin'])['n_subjects'].max()
               .unstack('pain_bin').reindex(columns=panels))
    return per_bin.min(axis=1, skipna=False).fillna(0).astype(int)


# ============================================================================
# WITHIN-SUBJECT STANDARDIZATION  (for figures that DRAW the 0-pain level)
# ============================================================================
# Lives here, not in a plot script, for this module's founding reason: two
# figures that standardize separately can disagree and nothing says which is
# right.
#
# `plot_slope_violin.py` calls these. `plot_band_violin_view.py` STILL HAS ITS
# OWN COPY -- it had uncommitted in-flight changes when this was factored out
# (2026-08-05) and editing it would have clobbered them. That duplication is a
# known and temporary wart, queued in TASKS.md; the two implementations agree
# line-for-line today, which is exactly the state that quietly stops being true.
#
# THE POINT: a figure whose violins include 'none' cannot use the view-level
# 0-pain baseline, because then the 0-pain violin is 0 by construction and the
# other two are measured against it -- the same circularity that makes the
# cluster test's `none` bin a noise floor rather than a control
# (docs/cluster_permutation.md 6). So the reference is each subject's OWN OVERALL
# level, pooled across all three pain levels, which privileges none of them.
#
# The honest cost: pooling all levels into the SD puts some of the between-level
# variance the figure is looking for into the denominator, so effects are
# slightly SHRUNK. It is the same shrinkage for every subject, so comparisons
# between levels stay valid; the absolute z is not comparable to the heatmaps'.

def exclude_thin_baseline_subjects(epoch_tables, min_none_epochs=5):
    """Drop subjects whose 0-pain baseline is built on too few epochs.

    Returns (filtered_table, excluded_subject_ids).

    WHY THIS IS NOT COSMETIC. Every pain contrast in this project is referenced to
    the subject's own 0-pain epochs, so the precision of that reference sets the
    precision of the contrast. Measured on the discovery slope tables 2026-08-07:
    subjects with < 5 zero-pain epochs carry a median SEM of 0.0828 on their 0-pain
    mean, against 0.0393 for everyone else -- and the `high - none` effect being
    measured is +0.0515. THEIR BASELINE ERROR IS LARGER THAN THE SIGNAL.

    It also explained an artifact. The long negative tail on the `none` violin of
    the 2026-08-06 figure came from four subjects (230, 210, 183, 109), and all
    four are below this floor. That tail was baseline instability, not physiology.

    SUBJECT-level, not cell-level: the count is over the subject's distinct 0-pain
    epochs across the session. A subject either has a usable baseline or does not,
    and letting them into some regions but not others would make the cohort differ
    per panel for a reason that has nothing to do with anatomy.

    Ten of 56 discovery subjects fall below the default: 039, 067, 071, 109, 124,
    183, 206, 209, 210, 230.
    """
    if not min_none_epochs:
        return epoch_tables, []

    per_subject = (epoch_tables[epoch_tables['pain_bin'] == 'none']
                   .groupby('subject_id')['epoch_id'].nunique())
    all_subjects = epoch_tables['subject_id'].unique()
    counts = per_subject.reindex(all_subjects).fillna(0).astype(int)
    excluded = sorted(counts[counts < min_none_epochs].index)

    if excluded:
        logger.info('%d/%d subject(s) excluded for < %d zero-pain epochs (their '
                    '0-pain reference is noisier than the effect): %s',
                    len(excluded), len(all_subjects), min_none_epochs,
                    {s: int(counts[s]) for s in excluded})
    return epoch_tables[~epoch_tables['subject_id'].isin(excluded)], excluded


def within_subject_z(epoch_values, value_col, min_epochs=4, keys=('subject_id', 'region')):
    """z-score each epoch against its (subject, region) mean/SD over ALL epochs.

    Pooled over pain levels ON PURPOSE -- see above. A (subject, region) with
    fewer than `min_epochs` epochs, or zero variance, yields NaN rather than a
    fabricated z: an SD from two epochs is not a scale, and dividing by it would
    manufacture enormous values out of nothing.
    """
    keys = list(keys)
    grouped = epoch_values.groupby(keys, dropna=False)[value_col]
    stats = grouped.agg(subject_mean='mean', subject_sd=lambda s: s.std(ddof=1),
                        n_epochs='size').reset_index()
    merged = epoch_values.merge(stats, on=keys, how='left')

    usable = (merged['n_epochs'] >= min_epochs) & (merged['subject_sd'] > 0)
    merged['z'] = np.where(usable,
                           (merged[value_col] - merged['subject_mean'])
                           / merged['subject_sd'].replace(0, np.nan), np.nan)
    dropped = int((~usable).sum())
    if dropped:
        logger.info('%d/%d epoch rows have no usable within-subject scale '
                    '(< %d epochs for that subject/region, or zero SD)',
                    dropped, len(merged), min_epochs)
    return merged


def subject_level(epoch_z, panels, value_col='z'):
    """One value per (subject, region, pain_bin): mean over that subject's epochs.

    ONE DOT IS ONE SUBJECT, which is what makes the resulting violin honest.
    Epochs are nested within subjects, and treating them as independent is the
    pseudo-replication that would inflate any statistic computed off the figure --
    the same reason the cluster test takes a 56-subject matrix and not a
    ~700-epoch one.
    """
    out = (epoch_z[epoch_z['pain_bin'].isin(panels)]
           .groupby(['subject_id', 'region', 'pain_bin'], dropna=False)
           .agg(value=(value_col, 'mean'), n_epochs=('epoch_id', 'nunique'))
           .reset_index()
           .dropna(subset=['value']))
    out['subject'] = out['subject_id']          # the violin helper's column name
    return out


def regions_by_min_subjects(subject_values, panels, roi_regions, min_subjects):
    """(regions to plot, per-region subject count) using the MINIMUM across levels.

    The minimum, not the union: every violin in a panel has to be backed by
    subjects for the comparison between them to mean anything, so the smallest
    count governs the panel.
    """
    counts = (subject_values.groupby(['region', 'pain_bin'])['subject'].nunique()
              .unstack('pain_bin').reindex(columns=panels))
    per_region = counts.min(axis=1, skipna=False).fillna(0).astype(int)
    regions = [r for r in roi_regions if per_region.get(r, 0) >= min_subjects]
    below = {r: int(per_region.get(r, 0)) for r in roi_regions
             if 0 < per_region.get(r, 0) < min_subjects}
    if below:
        logger.info('%d region(s) below the %d-subject floor, not plotted: %s',
                    len(below), min_subjects, below)
    return regions, per_region, below


def roi_regions_for(view_params):
    """The view's OWN ordered region list, from its recorded roi_scheme.

    NOT `config.ROI_REGIONS`, which is resolved at import from the `default`
    scheme. Filtering a view's regions against that constant silently keeps only
    the regions whose NAMES happen to appear in the default 15 -- so a 21-region
    view would render 8 rows and look entirely normal. The scheme is recorded in
    every view table's sidecar precisely so the figure can ask the artifact instead
    of a module-level default.
    """
    from ieeg_ehr.config import roi_schemes
    return roi_schemes.roi_regions(view_params.get('roi_scheme') or 'default')


def regions_with_min_subjects(stats, panels, min_subjects, regions=None):
    """Regions passing the coverage floor, in the scheme's fixed display order.

    `regions` is the scheme's display order (see `roi_regions_for`) -- a fixed
    anatomical order rather than this run's own data, so panels sit in the same
    place in every figure and two runs can be compared side by side. It defaults
    to config.ROI_REGIONS only for callers still on the default scheme.
    """
    regions = list(regions or config.ROI_REGIONS)
    counts = subjects_per_region(stats, panels)
    keep = set(counts[counts >= min_subjects].index)
    dropped = {r: int(counts[r]) for r in counts.index if r not in keep}
    if dropped:
        logger.info('%d region(s) below the %d-subject floor, not plotted: %s',
                    len(dropped), min_subjects, dropped)
    # A region with data that the scheme does not display is a real mismatch, not a
    # coverage fact -- loud, because this is exactly how the 15-vs-21 bug hid.
    unknown = sorted(keep - set(regions))
    if unknown:
        logger.warning('%d region(s) present in the data but NOT in the scheme display '
                       'order, so they will NOT be plotted: %s. Does the roi_scheme in '
                       'the view sidecar match the data?', len(unknown), unknown)
    return [r for r in regions if r in keep], counts


# ============================================================================
# WHERE THE FIGURES GO
# ============================================================================
# The 5-level analysis scheme (architecture.md PART 5), applied identically by
# every view figure so two plot types of the same view land as siblings:
#
#   analysis/ pain / <question> / <output_type> / <view_scheme> / <run>_<ts>/
#             ^event  ^level 2     ^level 3        ^level 4        ^level 5
#
# Levels 1-2 are opened DELIBERATELY -- a named question. Levels 3-5 are created
# freely per run. Level 4 is the view's own scheme_code ('blsub-rel'), NOT the
# pain-bin scheme alone: binarization is one of seven axes and not the one that
# most changes the numbers, so naming a folder after it buried the normalization.

DEFAULT_QUESTION = 'psd_physiology'


def add_output_arguments(parser, question=DEFAULT_QUESTION):
    """--question / --view-scheme / --scratch / --out-root, shared by both figure
    scripts so they cannot offer different placement vocabularies."""
    g = parser.add_argument_group('output location (architecture.md PART 5)')
    g.add_argument('--question', default=question,
                   help=f'Level-2 question folder (default: {question}). Opening a '
                        'NEW one is deliberate -- it must be a question named in the '
                        'exploration log, else use --scratch.')
    g.add_argument('--view-scheme', default=None,
                   help="Level-4 folder. Default: the view's own scheme_code, e.g. "
                        "'blsub-rel'.")
    g.add_argument('--scratch', action='store_true',
                   help=f'Throwaway run: write under {config.PLOTS_ROOT} instead of '
                        'the analysis tree. For iterating on a figure before it is '
                        'worth keeping.')
    g.add_argument('--out-root', default=None,
                   help='Explicit destination root, overriding both of the above.')
    return parser


def resolve_run_dir(args, output_type, view, run_name=None):
    """Build and create the run directory for one figure run.

    A timestamp is always appended, so two runs can never overwrite each other's
    provenance.json -- that has bitten this project once. The timestamp is taken
    ONCE here and reused across the three destinations, so the same run cannot end
    up with two different names depending on which flag was passed.
    """
    from datetime import datetime
    from pathlib import Path

    run_name = run_name if run_name is not None else args.run_name
    scheme = args.view_scheme or (view.scheme_code if view is not None else 'unknown')
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    leaf = f'{run_name}_{stamp}' if run_name else stamp

    if args.out_root:
        run_dir = Path(args.out_root) / output_type / scheme / leaf
    elif args.scratch:
        run_dir = config.PLOTS_ROOT / output_type / scheme / leaf
    else:
        run_dir = config.analysis_run_dir(question=args.question,
                                          output_type=output_type,
                                          run_name=run_name, view_scheme=scheme,
                                          timestamp=stamp)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def log_baseline_check(subject_tables):
    """Log how far the 'none' bin sits from 0 -- free correctness check.

    Under any baseline normalization the 0-pain epochs are their own reference,
    so they must come back at ~0. A 'none' mean far from 0 means the baseline
    leaked. Logged rather than plotted: as a panel it would be a band of white
    that crushes a shared colour scale, and as a line it would be a flat 0.
    """
    none_rows = subject_tables.loc[subject_tables['pain_bin'] == 'none', 'value']
    if none_rows.empty:
        return
    logger.info("baseline check -- 'none' bin mean %.2e, max |value| %.2e "
                '(should be ~0: it is its own reference)',
                float(none_rows.mean()), float(none_rows.abs().max()))
