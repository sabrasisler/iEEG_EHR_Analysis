"""
Shared loading/aggregation helpers used by both plotting scripts
(plot_pain_heatmaps.py, plot_pain_epoch_scatter.py). Kept here rather than
duplicated so region grouping, subject weighting, and provenance conventions
stay identical across plot types.
"""

import json
import logging
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pain_analysis import config
from qc_scripts import config as qc_config

logger = logging.getLogger(__name__)


def load_cache(subjects=None):
    paths = sorted(config.CACHE_DIR.glob('sub-*_ses-*_epoch_channel_power.csv'))
    if subjects is not None:
        subjects = set(subjects)
        paths = [p for p in paths if p.name.split('_')[0].replace('sub-', '') in subjects]
    if not paths:
        raise FileNotFoundError(f'No cached epoch/channel power CSVs found in {config.CACHE_DIR}')
    df = pd.concat(
        (pd.read_csv(p, dtype={'subject': str, 'session': str, 'run': str}) for p in paths),
        ignore_index=True,
    )

    # Defensive guard: a single -inf/inf/NaN mean_log_power (e.g. a dead/flat
    # channel with literally zero stored power, log10(0) = -inf, that slipped
    # past the raw_voltage QC mask) silently poisons every downstream mean --
    # and for a heatmap's shared color scale, poisons the ENTIRE figure (one
    # -inf makes vmax infinite, washing every other cell to white). Drop such
    # rows here, once, for every script that loads the cache, rather than
    # relying on each aggregation step to filter defensively on its own.
    non_finite = ~np.isfinite(df['mean_log_power'])
    n_bad = int(non_finite.sum())
    if n_bad:
        bad_channels = df.loc[non_finite, ['subject', 'channel']].drop_duplicates()
        logger.warning(
            '%d/%d cache rows have non-finite mean_log_power (dead/flat channel PSD slipping past QC?) -- '
            'dropping. Affected (subject, channel) pairs:\n%s',
            n_bad, len(df), bad_channels.to_string(index=False),
        )
        df = df[~non_finite].copy()

    return df, paths


def assign_relative_pain_bins(df, subject_col='subject', event_col='pain_event_id', score_col='pain_score'):
    """Recompute pain_bin under the subject-relative scheme (config.
    pain_bin_order('subject_relative')): 'none' for score == 0, else
    'low'/'high' split at that SAME subject's own mean pain score among
    their non-zero events. Computed from distinct (subject, pain_event_id)
    events, not the exploded channel/freq_bin rows, so a subject's mean
    isn't skewed by how many channels/bins they happen to have. Returns a
    Series aligned to df's index -- caller does `df['pain_bin'] = ...`.

    A subject with zero non-zero-pain events has an undefined baseline mean
    (NaN); comparisons against NaN are always False, but that's moot since
    there's no non-zero score left to mis-bin for that subject anyway."""
    events = df.drop_duplicates([subject_col, event_col])[[subject_col, event_col, score_col]]
    nonzero = events[events[score_col] > 0]
    subject_mean = nonzero.groupby(subject_col)[score_col].mean().rename('_subject_mean_nonzero_pain')

    merged = df[[subject_col, score_col]].merge(subject_mean, on=subject_col, how='left')
    bins = np.where(
        merged[score_col] == 0, 'none',
        np.where(merged[score_col] >= merged['_subject_mean_nonzero_pain'], 'high', 'low'),
    )
    return pd.Series(bins, index=df.index, name='pain_bin')


def add_region(df):
    df = df.copy()
    df['region'] = df['dk_anode_label'].apply(config.region_for_dk_label)
    n_dropped = int(df['region'].isna().sum())
    logger.info('Dropping %d/%d channel-epoch-freqbin rows with unmapped region '
                '(e.g. occipital, white matter, unknown)', n_dropped, len(df))
    return df[df['region'].notna()].copy()


def bin_label_table(df):
    """freq_bin_index -> (bin_low_hz, bin_high_hz), for axis labeling."""
    return (df[['freq_bin_index', 'bin_low_hz', 'bin_high_hz']]
            .drop_duplicates('freq_bin_index')
            .set_index('freq_bin_index')
            .sort_index())


def subject_region_table(df):
    """(subject, pain_bin, region, freq_bin_index) -> mean_log_power, averaged
    within subject: first across epochs per channel, then across channels
    within each region -- both steps equal-weighted, per the subject-weighting
    decision (subject is the unit of replication, not electrode or epoch
    count)."""
    channel_mean = (
        df.groupby(['subject', 'channel', 'pain_bin', 'region', 'freq_bin_index'])['mean_log_power']
        .mean().reset_index()
    )
    return (
        channel_mean.groupby(['subject', 'pain_bin', 'region', 'freq_bin_index'])['mean_log_power']
        .mean().reset_index()
    )


def subject_region_epoch_table(df):
    """(subject, pain_event_id, pain_bin, region, freq_bin_index) ->
    mean_log_power, averaged across channels within region -- epoch-level
    granularity preserved (not yet averaged across epochs), so callers can
    z-score each epoch against a per-subject/region/freq_bin baseline before
    collapsing to subject- or group-level means."""
    return (
        df.groupby(['subject', 'pain_event_id', 'pain_bin', 'region', 'freq_bin_index'])['mean_log_power']
        .mean().reset_index()
    )


def compute_subject_zscores(epoch_table, value_col='mean_log_power', group_col='freq_bin_index',
                             min_baseline_epochs=None):
    """Z-score every epoch's region/group_col power against that SAME
    subject's own 'none'-bin distribution (mean/std across that subject's
    none-bin epochs for that region/group_col), then average z-scores across
    epochs within each (subject, pain_bin, region, group_col) --
    equal-weighted, matching the log-power averaging convention elsewhere.
    group_col is 'freq_bin_index' for the per-bin z-score heatmap, or 'band'
    for the canonical-band violin plots -- same logic either granularity.

    NaN (not imputed) when a subject has fewer than min_baseline_epochs
    'none'-bin epochs for that region/group_col, or baseline std is 0 -- a
    z-score from a handful of baseline epochs is not trustworthy."""
    min_baseline_epochs = min_baseline_epochs or config.ZSCORE_MIN_BASELINE_EPOCHS

    baseline = epoch_table[epoch_table['pain_bin'] == 'none']
    baseline_stats = (
        baseline.groupby(['subject', 'region', group_col])[value_col]
        .agg(baseline_mean='mean', baseline_std='std', baseline_n='count')
        .reset_index()
    )

    merged = epoch_table.merge(baseline_stats, on=['subject', 'region', group_col], how='left')
    valid = (merged['baseline_n'] >= min_baseline_epochs) & (merged['baseline_std'] > 0)
    merged['zscore'] = np.where(
        valid, (merged[value_col] - merged['baseline_mean']) / merged['baseline_std'], np.nan,
    )

    n_invalid_epochs = int((~valid).sum())
    logger.info('%d/%d epochs have no valid z-score baseline (subject/region/%s has < %d '
                'none-bin epochs, or baseline std == 0)', n_invalid_epochs, len(merged), group_col, min_baseline_epochs)

    return (
        merged.groupby(['subject', 'pain_bin', 'region', group_col])['zscore']
        .mean().reset_index()
    )


def aggregate_epoch_table_to_bands(epoch_table, bin_labels, bands=None):
    """Aggregate per-freq-bin epoch-level log power into per-canonical-band
    epoch-level log power: average LINEAR power (10**mean_log_power) across
    bins whose geometric-mean center falls in each band, then log10 --
    matches preprocessing/bipolar_bands.py's aggregate_to_bands convention
    (avoids Jensen's-inequality bias from averaging log values directly).
    This deliberately differs from the direct-log-averaging used for
    epoch time-averaging elsewhere in this pipeline -- see
    FEATURIZATION_PLAN.md's "Follow-ups" section for that tradeoff.

    Returns (subject, pain_event_id, pain_bin, region, band, band_log_power),
    epoch-level granularity preserved (same shape/use as
    subject_region_epoch_table, just on a coarser frequency axis)."""
    bands = bands or config.CANONICAL_BANDS_HZ
    centers = np.sqrt(bin_labels['bin_low_hz'] * bin_labels['bin_high_hz'])

    frames = []
    for band_name, (fmin, fmax) in bands.items():
        bin_idx = centers[(centers >= fmin) & (centers < fmax)].index
        if len(bin_idx) == 0:
            logger.warning('No freq bins fall within %s band (%s-%s Hz), skipping', band_name, fmin, fmax)
            continue
        band_rows = epoch_table[epoch_table['freq_bin_index'].isin(bin_idx)].copy()
        band_rows['linear_power'] = 10.0 ** band_rows['mean_log_power']
        agg = (
            band_rows.groupby(['subject', 'pain_event_id', 'pain_bin', 'region'])['linear_power']
            .mean().reset_index()
        )
        with np.errstate(divide='ignore'):
            agg['band_log_power'] = np.log10(agg['linear_power'])
        agg['band'] = band_name
        frames.append(agg.drop(columns='linear_power'))

    return pd.concat(frames, ignore_index=True)


def draw_violin_with_subject_dots(ax, values_df, subject_color, value_col='value', pain_bins=None,
                                   jitter_width=0.12):
    """One violin panel: x = pain_bin (config.PAIN_BIN_ORDER by default), y =
    values_df[value_col], with per-subject colored dots overlaid (jittered,
    deterministically spread -- not random -- so figures are exactly
    reproducible from the same cache). values_df needs ['pain_bin',
    'subject', value_col] columns. Pain bins with zero data are skipped in
    the violin (matplotlib errors on empty arrays) but still get an x tick."""
    pain_bins = pain_bins or config.PAIN_BIN_ORDER

    violin_positions, violin_data = [], []
    for i, pain_bin in enumerate(pain_bins):
        vals = values_df.loc[values_df['pain_bin'] == pain_bin, value_col].dropna().to_numpy()
        if len(vals) > 0:
            violin_positions.append(i)
            violin_data.append(vals)
    if violin_data:
        parts = ax.violinplot(violin_data, positions=violin_positions, showmeans=True, showextrema=True)
        for body in parts['bodies']:
            body.set_facecolor('lightgray')
            body.set_edgecolor('gray')
            body.set_alpha(0.5)

    for i, pain_bin in enumerate(pain_bins):
        sub = values_df[values_df['pain_bin'] == pain_bin].dropna(subset=[value_col]).sort_values('subject')
        n = len(sub)
        if n == 0:
            continue
        offsets = np.linspace(-jitter_width, jitter_width, n) if n > 1 else np.zeros(1)
        for (_, row), offset in zip(sub.iterrows(), offsets):
            ax.scatter(i + offset, row[value_col], color=subject_color[row['subject']], s=18, zorder=3,
                       edgecolors='black', linewidths=0.3)

    ax.set_xticks(range(len(pain_bins)))
    ax.set_xticklabels(pain_bins, fontsize=7)


def draw_seaborn_violin_with_subject_dots(ax, values_df, subject_color, value_col='value', pain_bins=None,
                                           jitter_width=0.12):
    """Seaborn-styled version of draw_violin_with_subject_dots -- same
    per-subject colored dots (deterministic jitter, not sns.stripplot's own
    random jitter, so figures stay exactly reproducible), but the violin body
    itself is drawn with seaborn (smoother KDE, quartile lines, seaborn's
    default look) instead of matplotlib's plain ax.violinplot."""
    import seaborn as sns

    pain_bins = pain_bins or config.PAIN_BIN_ORDER
    present = [b for b in pain_bins if not values_df.loc[values_df['pain_bin'] == b, value_col].dropna().empty]

    if present:
        sns.violinplot(
            data=values_df[values_df['pain_bin'].isin(present)], x='pain_bin', y=value_col,
            order=present, ax=ax, inner='quartile', color='0.85', cut=0, linewidth=1, saturation=1,
        )

    for i, pain_bin in enumerate(pain_bins):
        sub = values_df[values_df['pain_bin'] == pain_bin].dropna(subset=[value_col]).sort_values('subject')
        n = len(sub)
        if n == 0:
            continue
        # present.index(pain_bin) gives the actual seaborn x-position (only
        # bins with data get a violin slot); pain_bins may include bins with
        # no data at all for this panel, which seaborn never draws a slot for.
        x = present.index(pain_bin) if pain_bin in present else i
        offsets = np.linspace(-jitter_width, jitter_width, n) if n > 1 else np.zeros(1)
        for (_, row), offset in zip(sub.iterrows(), offsets):
            ax.scatter(x + offset, row[value_col], color=subject_color[row['subject']], s=18, zorder=3,
                       edgecolors='black', linewidths=0.3)

    ax.set_xticks(range(len(present)))
    ax.set_xticklabels(present, fontsize=7)
    ax.set_xlabel('')


def subject_color_map(subjects, cmap_name='tab20'):
    cmap = plt.get_cmap(cmap_name)
    return {s: cmap(i % 20) for i, s in enumerate(sorted(subjects))}


def add_band_boundary_lines(ax, bin_labels=None):
    """Vertical reference lines at approximate classic EEG band boundaries
    (config.FREQ_BAND_BOUNDARIES_HZ). If bin_labels is given, each boundary
    Hz is snapped to the nearest freq_bin_index (for heatmaps with a
    categorical bin-index x-axis); otherwise lines are drawn directly at the
    Hz value (for continuous log-Hz x-axes like the scatter plot)."""
    for hz in config.FREQ_BAND_BOUNDARIES_HZ.values():
        if bin_labels is not None:
            nearest_bin = (bin_labels['bin_low_hz'] - hz).abs().idxmin()
            x = bin_labels.index.get_loc(nearest_bin) - 0.5
        else:
            x = hz
        ax.axvline(x=x, color='gray', linestyle=':', linewidth=0.8, alpha=0.6, zorder=0)


def epoch_counts(df, by_subject=False):
    """(region, pain_bin) [or (subject, region, pain_bin) if by_subject] ->
    number of distinct epochs (subject, pain_event_id pairs) contributing to
    that region/bin's average -- a channel/freq_bin_index row is not a unique
    epoch, so this de-dupes down to one row per epoch first."""
    keys = ['subject', 'region', 'pain_bin'] if by_subject else ['region', 'pain_bin']
    return (df.drop_duplicates(['subject', 'pain_event_id', 'region', 'pain_bin'])
              .groupby(keys).size())


def pivot_for_plot(table, value_col, regions, freq_bins):
    pivot = table.pivot(index='region', columns='freq_bin_index', values=value_col)
    return pivot.reindex(index=regions, columns=freq_bins)


def cluster_region_order(table, value_cols, freq_bins, regions=None, method='average'):
    """Reorder regions by hierarchical clustering (scipy, Euclidean distance
    on the concatenated [value_cols x freq_bins] pattern per region) instead
    of the fixed anatomical config.ROI_REGIONS order -- puts regions with
    similar spectral response patterns next to each other. NaNs (regions
    with no data for a bin) are treated as 0 for clustering purposes only,
    which pulls all-NaN/sparse regions toward each other -- fine for this
    exploratory reordering, not used for any displayed value."""
    from scipy.cluster.hierarchy import leaves_list, linkage

    regions = regions or config.ROI_REGIONS
    mats = [pivot_for_plot(table, col, regions, freq_bins).to_numpy() for col in value_cols]
    feat = np.nan_to_num(np.concatenate(mats, axis=1), nan=0.0)
    if feat.shape[0] < 3:
        return regions
    order = leaves_list(linkage(feat, method=method, metric='euclidean'))
    return [regions[i] for i in order]


def effect_size_region_order(table, value_cols, regions=None):
    """Reorder regions by mean absolute value across value_cols/freq_bins,
    descending -- puts the largest-effect regions at the top."""
    regions = regions or config.ROI_REGIONS
    magnitude = table.groupby('region')[value_cols].apply(lambda d: np.nanmean(np.abs(d.to_numpy())))
    return magnitude.reindex(regions).sort_values(ascending=False).index.tolist()


def epoch_count_labels(counts, regions, bin_order=None):
    """One label per region row, e.g. '245/123/106/38' for none/low/med/high
    epoch counts (0 if a bin has no epochs at all for that region). counts is
    a region/pain_bin -> n Series (from epoch_counts()). bin_order defaults
    to config.PAIN_BIN_ORDER -- pass config.pain_bin_order('subject_relative')
    (3 bins, no 'medium') when plotting under that scheme."""
    bin_order = bin_order or config.PAIN_BIN_ORDER
    labels = []
    for region in regions:
        n = [int(counts.get((region, b), 0)) for b in bin_order]
        labels.append('/'.join(str(x) for x in n))
    return labels


def plot_region_freq_heatmaps(pivots, col_titles, bin_labels, counts, title, out_path, cbar_label,
                               cmap='RdBu_r', regions=None, count_bin_order=None):
    """One imshow panel per (col_title, pivot) pair, region rows x freq_bin_index
    columns, shared symmetric-around-zero color scale, EEG-band reference
    lines, and per-region epoch-count annotations on the rightmost panel.
    Shared by plot_pain_heatmaps.py (delta log-power) and
    plot_pain_zscore_heatmaps.py (z-score) -- same layout, different values.

    `regions` must match the row order the pivots were already built with
    (via pivot_for_plot(..., regions, ...)) -- it's only used here for the
    y-axis tick labels, defaults to config.ROI_REGIONS. `count_bin_order`
    controls the pain-bin order in the epoch-count annotation text (see
    epoch_count_labels)."""
    regions = regions or config.ROI_REGIONS
    count_bin_order = count_bin_order or config.PAIN_BIN_ORDER
    freq_bins = bin_labels.index.tolist()

    abs_maxes = [np.nanmax(np.abs(p.to_numpy())) for p in pivots if not np.all(np.isnan(p.to_numpy()))]
    vmax = max(abs_maxes) if abs_maxes else 1.0

    freq_tick_labels = [f'{bin_labels.loc[b, "bin_low_hz"]:.0f}' for b in freq_bins]
    count_labels = epoch_count_labels(counts, regions, bin_order=count_bin_order) if counts is not None else None

    fig_height = max(5, 0.4 * len(regions))
    fig, axes = plt.subplots(1, len(pivots), figsize=(6.3 * len(pivots) + 1, fig_height), sharey=True)
    axes = np.atleast_1d(axes)
    im = None
    for ax, col, pivot in zip(axes, col_titles, pivots):
        im = ax.imshow(pivot.to_numpy(), aspect='auto', cmap=cmap, vmin=-vmax, vmax=vmax)
        ax.set_title(col)
        ax.set_xticks(range(len(freq_bins)))
        ax.set_xticklabels(freq_tick_labels, rotation=90, fontsize=6)
        ax.set_xlabel('Freq bin low edge (Hz)')
        add_band_boundary_lines(ax, bin_labels)
    axes[0].set_yticks(range(len(regions)))
    axes[0].set_yticklabels(regions)
    axes[0].set_ylabel('Region')

    if count_labels is not None:
        last_ax = axes[-1]
        for row, label in enumerate(count_labels):
            last_ax.text(len(freq_bins) - 0.4, row, label, va='center', ha='left',
                         fontsize=6, clip_on=False)
        header = 'n (' + '/'.join(b[:3] for b in count_bin_order) + ')'
        last_ax.text(len(freq_bins) - 0.4, -1, header, va='center', ha='left',
                     fontsize=6, fontweight='bold', clip_on=False)

    fig.suptitle(title)
    fig.colorbar(im, ax=list(axes), shrink=0.8, label=cbar_label)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info('Wrote %s', out_path)


def _source_cache_provenance(cache_paths):
    """Read each cache CSV's sidecar provenance.json (written by
    build_pain_epoch_power.py), when present, so a plot run's own provenance
    traces all the way back to the mask/epoch params used to build the
    cache -- not just the cache file paths themselves."""
    entries = []
    for csv_path in cache_paths:
        prov_path = csv_path.with_suffix('').with_suffix('.provenance.json')
        entry = {'cache_csv': str(csv_path)}
        if prov_path.exists():
            entry['cache_provenance'] = json.loads(prov_path.read_text())
        else:
            entry['cache_provenance'] = None
        entries.append(entry)
    return entries


def make_run_dir(run_name, n_subjects, category=None):
    """category groups a plot type's runs under their own subdirectory of
    config.PLOTS_ROOT, e.g. 'delta_heatmap/absolute' or
    'band_violin_grid/subject_relative' (see pain_analysis/CONTEXT.md for the
    full naming convention) -- keeps different plot types/variants from
    being siblings in one flat plots/ directory.

    run_name is a LABEL ONLY (never re-encodes plot type/scheme -- those are
    the parent category folders). A timestamp is ALWAYS appended so reruns
    never collide or silently overwrite a prior run's provenance.json (this
    bit us once: two scripts sharing an identical category+run_name
    overwrote each other's provenance)."""
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    name = f'{run_name}_{timestamp}' if run_name else f'{timestamp}_n{n_subjects}subj'
    root = config.PLOTS_ROOT / category if category else config.PLOTS_ROOT
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_run_provenance(run_dir, script_name, args, cache_paths, subjects=None, extra=None):
    """subjects: explicit list of subject IDs included in this run -- written
    into provenance.json so a run's cohort is traceable without having to
    parse it back out of cache_paths' filenames."""
    provenance = {
        'script': script_name,
        'git': qc_config.git_provenance(),
        'args': vars(args),
        'subjects': sorted(subjects) if subjects is not None else None,
        'roi_regions': config.ROI_REGIONS,
        'pain_bin_edges': config.PAIN_BIN_EDGES,
        'source_cache_files': _source_cache_provenance(cache_paths),
    }
    if extra:
        provenance.update(extra)
    (run_dir / 'provenance.json').write_text(json.dumps(provenance, indent=2))
