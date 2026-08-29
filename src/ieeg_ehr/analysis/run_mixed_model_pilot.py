"""
PHASE 1 PILOT: per-cell mixed-effects models on ~20 hand-picked region x frequency
cells, to decide how the full grid must be inferred.

This is a SIZING AND CALIBRATION run, not a result. Its one load-bearing output is
the answer to: are statsmodels' Wald p-values for `NRS_within` calibrated against a
within-subject pain-score shuffle, or ANTICONSERVATIVE? If they are
anticonservative the full grid has to use permutation p-values, which costs about
three orders of magnitude more compute and changes the Phase 2 driver's whole
shape. Everything else the pilot reports -- channel random-slope necessity,
singular fits, agreement with the two-stage map, runtime -- is secondary to that.

STAGES, because the permutation is the only expensive part
----------------------------------------------------------
    fit      one job (~minutes). Loads the view, builds the inventory, picks the
             cells, fits full + reduced + channel-slope models, writes the model
             frames and every non-calibration figure. PRINTS THE RUN DIR.
    perm     one ARRAY TASK PER CELL. Reloads only that cell's saved frame,
             refits it for warm-start parameters, runs the shuffles, writes a
             shard.
    collect  one job. Merges the shards, computes the calibration comparison,
             writes the calibration figure, METHODS.md and the final provenance.

Splitting this way keeps the expensive stage embarrassingly parallel without
making the cheap stage pay for a scheduler round-trip.

WHAT IS DELIBERATELY NOT HERE
-----------------------------
The full grid (Phase 2). No covariates of any kind -- not time of day, not time
since admission, not arousal. No aperiodic decomposition, no split-half
reliability, no clinical moderators. And no multiple-comparison correction: with
~20 purposively chosen cells, a BH q over them would be a correction over a family
that was picked by looking at the answer.

EXPLORATORY. Discovery cohort only. Nominations, not findings.
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ieeg_ehr import config, io
from ieeg_ehr.analysis import mixed_model as mm
from ieeg_ehr.analysis import pain_coef, reference_run, view_tables
from ieeg_ehr.views import cache_reader, channel_meta, view_config as vc

logger = logging.getLogger(__name__)

OUTPUT_TYPE = 'univariate_analysis'
VIEW_SCHEME = 'cont_pain_scratch'
# The run name says PILOT and TEST in the folder itself. This lands in the
# permanent analysis tree beside the reference run, so the name has to stop
# anyone reading it months from now as a finished result.
RUN_NAME = 'pilot_test_mixedlm'
VIEW_LABEL = 'chan-raw-relpain-roiv2'

DISCLAIMER = ('EXPLORATORY pilot -- discovery cohort, NOMINATIONS NOT FINDINGS. '
              'Nothing here is corrected for multiple comparisons or confirmed '
              'out of sample.')

# ---------------------------------------------------------------------------
# The pilot cells: (region, target Hz, group, why this cell)
# ---------------------------------------------------------------------------
# Chosen to span the EFFECT LANDSCAPE of the existing two-stage heatmap rather
# than to be representative -- the pilot is asking whether the machinery behaves,
# and machinery misbehaves at the extremes. Target frequencies are resolved to the
# nearest bin by GEOMETRIC centre, because the axis is log-spaced.
PILOT_CELLS = [
    ('Thalamus',   2.0,  'strong_negative_low',  'large low-frequency decrease'),
    ('PCC',        2.0,  'strong_negative_low',  'large low-frequency decrease'),
    ('Auditory',   2.0,  'strong_negative_low',  'large low-frequency decrease'),
    ('Amygdala',   2.0,  'strong_negative_low',  'large low-frequency decrease'),

    ('M1',        20.0,  'strong_positive_mid',  'sensorimotor beta increase'),
    ('S2/PO',     20.0,  'strong_positive_mid',  'sensorimotor beta increase'),
    ('dlPFC',     20.0,  'strong_positive_mid',  'significant beta cluster'),

    ('dmPFC/SMA',  2.0,  'sign_flip',            'sign flips between its two clusters'),
    ('dmPFC/SMA', 20.0,  'sign_flip',            'sign flips between its two clusters'),
    ('IFG/vlPFC',  2.0,  'sign_flip',            'sign flips between its two clusters'),
    ('IFG/vlPFC', 20.0,  'sign_flip',            'sign flips between its two clusters'),

    ('Amygdala',  20.0,  'hypothesized_hetero',  'blank in the mean map; hetero?'),
    ('Insula',    20.0,  'hypothesized_hetero',  'blank in the mean map; hetero?'),
    ('lOFC',      20.0,  'hypothesized_hetero',  'blank in the mean map; hetero?'),

    ('Occipital', 10.0,  'quasi_control',        'largest unoutlined cell in the '
                                                 'quasi-control row'),
]
N_NULL_CELLS = 4     # picked data-driven from the reference map, see `pick_null_cells`


# ============================================================================
# LOADING
# ============================================================================

def resolve_view_dir(explicit=None, mask_label=None, roi_scheme='roi_v2'):
    """The per-channel view directory, resolved the way the builder resolves it.

    Never a hard-coded config_hash: ONE code path computes it, so the builder and
    this script cannot disagree about which directory is meant.
    """
    if explicit:
        return Path(explicit)
    view = vc.ViewConfig(normalization='none', region='none', domain='log',
                         freq='log_bins_50', pain_bins='subject_relative',
                         mask_level='bipolar', mask_label=mask_label,
                         roi_scheme=roi_scheme).resolved()
    params = dict(view.provenance(), split='discovery')
    return config.pain_epoch_views_dir(VIEW_LABEL, io.config_hash(params),
                                       view.epoch_minutes)


def view_subject_paths(view_dir):
    paths = sorted(Path(view_dir).glob('view_epochs_sub-*.parquet'))
    if not paths:
        raise SystemExit(
            f'no view_epochs_*.parquet in {view_dir}.\n'
            'Build the per-channel view first:\n'
            '    sbatch sbatch/build_chan_view_array.sbatch')
    return paths


def subject_session_of(path):
    """('085', '01') from view_epochs_sub-085_ses-01.parquet."""
    stem = path.stem
    subject = stem.split('sub-')[1].split('_')[0]
    session = stem.split('ses-')[1].split('_')[0]
    return subject, session


def load_epoch_scores(paths):
    """(subject_id, epoch_id, pain_score), one row per epoch. Columns only.

    Read with `columns=` so eligibility costs three narrow columns rather than the
    whole per-channel table -- which is the reason the cache and the views are
    Parquet in the first place.
    """
    frames = []
    for p in paths:
        df = io.read_table(p, columns=['subject_id', 'epoch_id', 'pain_score'],
                           on_stale='warn')
        frames.append(df.drop_duplicates(['subject_id', 'epoch_id']))
    return pd.concat(frames, ignore_index=True)


def roi_maps(paths, subjects, roi_scheme):
    """{subject_id: {channel: ROI}} from the CACHED channel_meta tables."""
    out, missing = {}, []
    for p in paths:
        subject, session = subject_session_of(p)
        sid = f'sub-{subject}'
        if sid not in subjects:
            continue
        try:
            meta = channel_meta.build(subject, session, [])
        except FileNotFoundError:
            missing.append(sid)
            continue
        mapping = {c: r for c, r in channel_meta.region_map(meta, roi_scheme).items()
                   if r is not None}
        if not mapping:
            missing.append(sid)
            continue
        out[sid] = mapping
    if missing:
        logger.warning('%d subject(s) contribute NO ROI-labelled channel and cannot '
                       'enter any cell: %s', len(missing), sorted(missing))
    return out, missing


def load_cell_frames(paths, subjects, wanted, roi_by_subject):
    """{(region, freq_bin_index): model frame} for just the cells we need.

    One subject's file at a time, filtered to the wanted bins BEFORE the ROI join,
    so peak memory is one subject's slice rather than the ~15M-row cohort table.
    The pilot needs ~20 of 924 cells; reading the whole thing would be ~50x the
    work for the same answer.
    """
    wanted_bins = sorted({b for _, b in wanted})
    collected = {cell: [] for cell in wanted}

    for p in paths:
        subject, _ = subject_session_of(p)
        sid = f'sub-{subject}'
        if sid not in subjects or sid not in roi_by_subject:
            continue
        df = io.read_table(p, columns=['subject_id', 'epoch_id', 'pain_score',
                                       'region', 'freq_bin_index', 'value'],
                           on_stale='warn')
        df = df[df['freq_bin_index'].isin(wanted_bins)]
        if df.empty:
            continue
        # `region` holds the CHANNEL name in a --region none view. Renaming here
        # rather than living with it: two different things called `region` in one
        # function is how a join goes quietly wrong.
        df = df.rename(columns={'region': 'channel'})
        df['roi'] = df['channel'].map(roi_by_subject[sid])
        df = df.dropna(subset=['roi'])
        for (roi, b), grp in df.groupby(['roi', 'freq_bin_index'], sort=False):
            if (roi, b) in collected:
                collected[(roi, b)].append(grp)

    frames = {}
    for cell, parts in collected.items():
        if not parts:
            frames[cell] = pd.DataFrame(columns=['subject_id', 'channel', 'epoch_id',
                                                 'pain_score', 'value'])
            continue
        rows = pd.concat(parts, ignore_index=True)
        frames[cell] = mm.build_cell_frame(rows, region=cell[0], freq_bin_index=cell[1])
    return frames


# ============================================================================
# CELL SELECTION
# ============================================================================

def bin_for_hz(bin_table, target_hz):
    """Nearest bin by GEOMETRIC centre -- the axis is log-spaced, so an arithmetic
    nearest-centre would drift low."""
    centres = np.sqrt(bin_table['bin_low_hz'].to_numpy()
                      * bin_table['bin_high_hz'].to_numpy())
    idx = int(np.argmin(np.abs(np.log(centres) - np.log(target_hz))))
    return int(bin_table.index[idx]), float(centres[idx])


def pick_null_cells(ref_pain_coef, bin_table, taken, coverage, n=N_NULL_CELLS,
                    min_subjects=mm.MIN_SUBJECTS):
    """The `n` cells closest to zero in the reference map, one per region.

    DATA-DRIVEN rather than guessed. "Clearly null in the existing map" is a claim
    about the reference map, so it should be read off the reference map -- naming
    regions from memory would pick cells that merely LOOK empty on a diverging
    colour scale, where near-white covers a wide range.

    One per region so the nulls are not four bins of the same quiet row, and
    restricted to cells that actually clear the coverage floor in OUR data, since
    a null cell that cannot be fitted tests nothing.
    """
    cand = ref_pain_coef.dropna(subset=['pain_coef']).copy()
    cand = cand[cand['freq_bin_index'].isin(bin_table.index)]
    keep = [(r.region, int(r.freq_bin_index)) not in taken
            and coverage.get((r.region, int(r.freq_bin_index)), 0) >= min_subjects
            for r in cand.itertuples()]
    cand = cand[keep]
    if cand.empty:
        logger.warning('no eligible null cells in the reference map')
        return []
    cand['abs_coef'] = cand['pain_coef'].abs()
    chosen = (cand.sort_values('abs_coef')
              .drop_duplicates('region')
              .head(n))
    return [(r.region, int(r.freq_bin_index), float(r.pain_coef))
            for r in chosen.itertuples()]


def build_cell_manifest(bin_table, ref_pain_coef, coverage, min_subjects):
    """The pilot's cell list, as a table: region, bin, group, rationale."""
    rows, taken = [], set()
    for region, target_hz, group, why in PILOT_CELLS:
        b, centre = bin_for_hz(bin_table, target_hz)
        if (region, b) in taken:
            logger.info('skipping duplicate pilot cell %s bin %d', region, b)
            continue
        taken.add((region, b))
        rows.append({'region': region, 'freq_bin_index': b, 'group': group,
                     'target_hz': target_hz, 'bin_centre_hz': centre,
                     'rationale': why})

    for region, b, coef in pick_null_cells(ref_pain_coef, bin_table, taken,
                                           coverage, min_subjects=min_subjects):
        taken.add((region, b))
        centre = float(np.sqrt(bin_table.loc[b, 'bin_low_hz']
                               * bin_table.loc[b, 'bin_high_hz']))
        rows.append({'region': region, 'freq_bin_index': b, 'group': 'null',
                     'target_hz': np.nan, 'bin_centre_hz': centre,
                     'rationale': f'smallest |pain_coef| in the reference map '
                                  f'({coef:+.5f}) among covered cells in this region'})

    man = pd.DataFrame(rows)
    man.insert(0, 'cell_index', range(len(man)))
    man['bin_low_hz'] = man['freq_bin_index'].map(bin_table['bin_low_hz'])
    man['bin_high_hz'] = man['freq_bin_index'].map(bin_table['bin_high_hz'])
    return man


# ============================================================================
# INVENTORY (Stage 0)
# ============================================================================

def inventory_subjects(scores, diagnostics):
    """Per subject: report count and the shape of their NRS distribution."""
    g = scores.groupby('subject_id')['pain_score']
    inv = pd.DataFrame({
        'n_reports': g.size(), 'nrs_mean': g.mean(),
        'nrs_sd': g.std(ddof=1), 'nrs_min': g.min(), 'nrs_max': g.max(),
        'nrs_range': g.max() - g.min(), 'n_distinct': g.nunique(),
    }).reset_index()
    return inv.merge(diagnostics[['subject_id', 'included', 'excluded_because']],
                     on='subject_id', how='left')


def inventory_regions(roi_by_subject, regions):
    """Per region: how many subjects have any contact there, and how many contacts."""
    rows = []
    for region in regions:
        subs = [s for s, m in roi_by_subject.items() if region in set(m.values())]
        n_chan = sum(sum(1 for r in m.values() if r == region)
                     for m in roi_by_subject.values())
        rows.append({'region': region, 'n_subjects': len(subs), 'n_channels': n_chan})
    return pd.DataFrame(rows)


def inventory_cells(frames, manifest):
    """Per pilot cell: rows, subjects, channels, epochs actually available."""
    rows = []
    for r in manifest.itertuples():
        df = frames[(r.region, r.freq_bin_index)]
        rows.append({
            'cell_index': r.cell_index, 'region': r.region,
            'freq_bin_index': r.freq_bin_index, 'group': r.group,
            'n_rows': int(len(df)),
            'n_subjects': int(df['subject'].nunique()) if len(df) else 0,
            'n_channels': int(df['channel_uid'].nunique()) if len(df) else 0,
            'n_epochs': int(df.groupby('subject')['epoch_id'].nunique().sum())
                        if len(df) else 0,
        })
    return pd.DataFrame(rows)


def coverage_map(paths, subjects, roi_by_subject, bins):
    """{(region, bin): n subjects with coverage}. Bin-independent in practice, so
    computed once from the channel maps rather than by reading every cell."""
    per_region = {}
    for sid, mapping in roi_by_subject.items():
        if sid not in subjects:
            continue
        for region in set(mapping.values()):
            per_region[region] = per_region.get(region, 0) + 1
    return {(region, b): n for region, n in per_region.items() for b in bins}


# ============================================================================
# FITTING ONE CELL
# ============================================================================

def fit_one_cell(df, meta, *, with_channel_slope=True, tol=mm.BOUNDARY_TOL):
    """(record, blups, full results object or None) for one pilot cell."""
    region, b = meta['region'], meta['freq_bin_index']
    lo, hi = meta['bin_low_hz'], meta['bin_high_hz']

    ok, reason = mm.cell_is_fittable(df)
    if not ok:
        logger.warning('%s bin %d NOT FITTABLE: %s', region, b, reason)
        return mm.failed_record(region, b, lo, hi, reason, df=df), [], None

    t0 = time.time()
    try:
        res, warn_full = mm.fit_cell(df, mm.VC_FULL)
    except mm.CellFitError as exc:
        logger.error('%s bin %d full model FAILED: %s', region, b, exc)
        return (mm.failed_record(region, b, lo, hi, f'full: {exc}', df=df,
                                 fit_seconds=time.time() - t0), [], None)
    t_full = time.time() - t0

    try:
        res_red, warn_red = mm.fit_cell(df, mm.VC_REDUCED)
    except mm.CellFitError as exc:
        logger.error('%s bin %d reduced model FAILED: %s', region, b, exc)
        res_red, warn_red = None, [f'reduced failed: {exc}']

    rec = mm.cell_record(res, res_red, df, region=region, freq_bin_index=b,
                         bin_low_hz=lo, bin_high_hz=hi, fit_seconds=t_full,
                         warnings_full=warn_full, warnings_reduced=warn_red, tol=tol)
    rec['cell_index'] = meta['cell_index']
    rec['group'] = meta['group']
    rec['fit_seconds_reduced'] = time.time() - t0 - t_full

    # PILOT QUESTION 1: is a channel random SLOPE needed? The prior is
    # intercept-only. If the slope improves fit consistently, ROI v2 regions are
    # too coarse a unit -- contacts inside one parcel would be responding
    # differently, which is a finding about the parcellation, not about pain.
    rec['channel_slope_lrt_stat'] = np.nan
    rec['p_channel_slope'] = np.nan
    rec['channel_slope_converged'] = None
    if with_channel_slope:
        t1 = time.time()
        try:
            res_cs, _ = mm.fit_cell(df, mm.VC_CHANNEL_SLOPE)
            stat, p = mm.lrt(res_cs, res)
            rec['channel_slope_lrt_stat'] = stat
            rec['p_channel_slope'] = p
            rec['channel_slope_converged'] = bool(res_cs.converged)
            rec['var_channel_slope'] = float(
                mm.vcomp_by_name(res_cs).get('channel_slope', np.nan))
        except (mm.CellFitError, KeyError) as exc:
            logger.warning('%s bin %d channel-slope model failed: %s', region, b, exc)
            rec['channel_slope_error'] = str(exc)[:200]
        rec['fit_seconds_channel_slope'] = time.time() - t1

    blups = mm.blup_rows(res, df, region=region, freq_bin_index=b)
    logger.info('%-18s bin %2d (%6.1f-%6.1f Hz) | n=%2d subj %4d chan %6d rows | '
                'beta %+.5f z %+5.2f p %.4g | LRT %6.2f p %.4g | %s | %.1fs',
                region, b, lo, hi, rec['n_subjects'], rec['n_channels'],
                rec['n_rows'], rec['beta_nrs_within'], rec['z'], rec['p'],
                rec['lrt_stat'], rec['p_lrt_mixture'],
                rec['boundary_components'] or 'no boundary', t_full)
    return rec, blups, res


# ============================================================================
# FIGURES
# ============================================================================

def _footnote(ax_fig, text):
    ax_fig.text(0.01, 0.005, text, fontsize=6.5, va='bottom', ha='left',
                color='0.35', wrap=True)


def fig_vs_twostage(cells, ref_pain_coef, out_path):
    """PILOT QUESTION 4: does the mixed model agree with the two-stage map?"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    merged = cells.merge(ref_pain_coef, on=['region', 'freq_bin_index'], how='left')
    merged = merged.dropna(subset=['beta_nrs_within', 'pain_coef'])
    if merged.empty:
        logger.warning('no cells to compare against the two-stage map')
        return None

    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    groups = sorted(merged['group'].unique())
    cmap = plt.get_cmap('tab10')
    for i, grp in enumerate(groups):
        sub = merged[merged['group'] == grp]
        ax.errorbar(sub['pain_coef'], sub['beta_nrs_within'], yerr=sub['se'],
                    fmt='o', ms=6, lw=1, capsize=2, color=cmap(i % 10),
                    label=f'{grp} (n={len(sub)})', alpha=0.85)

    lim = float(np.nanmax(np.abs(np.concatenate(
        [merged['pain_coef'].to_numpy(), merged['beta_nrs_within'].to_numpy()])))) * 1.25
    ax.plot([-lim, lim], [-lim, lim], color='0.5', lw=1, ls='--', zorder=0,
            label='identity')
    ax.axhline(0, color='0.85', lw=0.8, zorder=0)
    ax.axvline(0, color='0.85', lw=0.8, zorder=0)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.set_xlabel('two-stage group mean pain_coef\n(within-subject OLS, '
                  'equal-weighted over subjects)')
    ax.set_ylabel('mixed-model beta_NRS_within\n(+/- 1 SE)')

    finite = merged[['pain_coef', 'beta_nrs_within']].to_numpy()
    r = float(np.corrcoef(finite[:, 0], finite[:, 1])[0, 1]) if len(finite) > 2 else np.nan
    slope = float(np.polyfit(finite[:, 0], finite[:, 1], 1)[0]) if len(finite) > 2 else np.nan
    ax.set_title(f'Mixed model vs the existing two-stage map\n'
                 f'{len(merged)} pilot cells   r = {r:.3f}   OLS slope = {slope:.3f}')
    ax.legend(fontsize=7, loc='upper left')
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    _footnote(fig,
              'Systematic departure from the identity line is EXPECTED and has two '
              'separable causes. (1) Precision weighting: the mixed model downweights '
              'subjects with fewer reports or contacts; the two-stage mean does not. '
              '(2) Channel aggregation: the two-stage map averages an ROI\'s contacts '
              'LINEAR-then-log before fitting, whereas the mixed model keeps per-channel '
              'log values and absorbs contact level in a random intercept -- effectively '
              'a mean of logs. These are different estimands, not a discrepancy.\n'
              + DISCLAIMER)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return {'r': r, 'ols_slope': slope, 'n_cells': int(len(merged))}


def fig_variance(cells, out_path):
    """PILOT QUESTION 3: variance components, boundary flags, and heterogeneity."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    d = cells.sort_values(['group', 'region', 'freq_bin_index']).reset_index(drop=True)
    labels = [f'{r.region} {r.freq_bin_low:.0f}-{r.freq_bin_high:.0f} Hz'
              for r in d.itertuples()]
    y = np.arange(len(d))

    fig, axes = plt.subplots(1, 3, figsize=(14, max(4.5, 0.34 * len(d))),
                             sharey=True,
                             gridspec_kw={'width_ratios': [1.5, 1.1, 1.1]})

    ax = axes[0]
    comps = [('var_channel', 'channel', '#4C78A8'),
             ('var_subj_int', 'subj_int', '#72B7B2'),
             ('var_subj_slope * var(NRS)', 'subj_slope (scaled)', '#E45756'),
             ('var_resid', 'residual', '#BAB0AC')]
    width = 0.2
    for k, (col, label, colour) in enumerate(comps):
        vals = (d['var_subj_slope'] * d['nrs_within_var'] if 'slope' in col
                else d[col])
        ax.barh(y + (k - 1.5) * width, vals, height=width, label=label, color=colour)
    ax.set_xscale('log')
    ax.set_xlabel('variance contributed to the linear predictor (log scale)')
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.legend(fontsize=7)
    ax.set_title('Variance components')

    ax = axes[1]
    sig = -np.log10(d['p_lrt_mixture'].clip(lower=1e-12))
    colours = ['#E45756' if s else '#4C78A8' for s in d['singular_flag']]
    ax.barh(y, sig, color=colours, height=0.6)
    ax.axvline(-np.log10(0.05), color='0.4', ls='--', lw=1)
    ax.set_xlabel('-log10 p (heterogeneity LRT, 50:50 mixture)')
    ax.set_title('Heterogeneity\nred = a component at the boundary')

    ax = axes[2]
    ax.barh(y, -np.log10(d['p'].clip(lower=1e-12)), color='#54A24B', height=0.6)
    ax.axvline(-np.log10(0.05), color='0.4', ls='--', lw=1)
    ax.set_xlabel('-log10 p (NRS_within Wald z)')
    ax.set_title('Fixed effect')

    fig.suptitle('Pilot cells: variance components, heterogeneity, fixed effect')
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    _footnote(fig,
              'subj_slope is plotted SCALED by var(NRS_within) so it is on the same '
              'footing as the intercept-like components -- a raw slope variance is in '
              'different units and looks tiny for arithmetic reasons alone. Wald p '
              'values here are NOT yet known to be calibrated; that is what the '
              'permutation stage tests.\n' + DISCLAIMER)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig_calibration(cells, out_path):
    """PILOT QUESTION 2, THE ONE THAT MATTERS: parametric vs permutation p."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    d = cells.dropna(subset=['p', 'p_perm']).copy()
    if d.empty:
        logger.warning('no cells have a permutation p; skipping the calibration figure')
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))

    ax = axes[0]
    lo = float(min(d['p'].min(), d['p_perm'].min())) * 0.5 or 1e-4
    groups = sorted(d['group'].unique())
    cmap = plt.get_cmap('tab10')
    for i, grp in enumerate(groups):
        s = d[d['group'] == grp]
        ax.scatter(s['p'], s['p_perm'], s=42, color=cmap(i % 10),
                   label=f'{grp} (n={len(s)})', alpha=0.85, edgecolor='white', lw=0.6)
    ax.plot([lo, 1], [lo, 1], color='0.5', ls='--', lw=1, label='calibrated')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(lo, 1.2)
    ax.set_ylim(lo, 1.2)
    ax.axhline(0.05, color='0.8', lw=0.8)
    ax.axvline(0.05, color='0.8', lw=0.8)
    ax.set_xlabel('parametric p (Wald z on NRS_within)')
    ax.set_ylabel('permutation p (within-subject NRS shuffle)')
    ax.set_title('Below the dashed line = parametric p is ANTICONSERVATIVE')
    ax.legend(fontsize=7, loc='lower right')

    ax = axes[1]
    ratio = np.log10(d['p_perm'] / d['p'])
    order = np.argsort(ratio.to_numpy())
    labels = [f'{r.region} {r.freq_bin_low:.0f} Hz' for r in d.itertuples()]
    ax.barh(np.arange(len(d)), ratio.to_numpy()[order],
            color=['#E45756' if v > 0 else '#4C78A8' for v in ratio.to_numpy()[order]],
            height=0.65)
    ax.set_yticks(np.arange(len(d)))
    ax.set_yticklabels([labels[i] for i in order], fontsize=7)
    ax.invert_yaxis()
    ax.axvline(0, color='0.4', lw=1)
    ax.set_xlabel('log10(permutation p / parametric p)\n'
                  'positive = parametric p too small = anticonservative')
    ax.set_title('Per-cell disagreement')

    n_anti = int((d['p_perm'] > d['p']).sum())
    fig.suptitle(f'p-value calibration on {len(d)} pilot cells: '
                 f'parametric p is smaller than permutation p in {n_anti}/{len(d)}')
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    _footnote(fig,
              'This figure DETERMINES THE INFERENCE PLAN for the full grid. If the '
              'parametric p values are systematically anticonservative, Phase 2 must '
              'use permutation p values throughout. Permutation p is two-sided by '
              'magnitude and floored at 1/(n_perm+1), so it cannot resolve below that '
              'value -- points on the lower edge are censored, not calibrated.\n'
              + DISCLAIMER)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return {'n_anticonservative': n_anti, 'n_compared': int(len(d)),
            'median_log10_ratio': float(np.median(ratio))}


# ============================================================================
# STAGES
# ============================================================================

def stage_fit(args):
    ref = reference_run.load(args.reference_run)
    ref.describe()

    view_dir = resolve_view_dir(args.view_dir, mask_label=args.mask_label or
                                ref.view_params.get('mask_label'),
                                roi_scheme=ref.view_params.get('roi_scheme', 'roi_v2'))
    logger.info('per-channel view: %s', view_dir)
    paths = view_subject_paths(view_dir)
    logger.info('%d subject view table(s) present', len(paths))

    view_params, view = view_tables.view_params_from(paths)
    if view_params.get('region') != 'none':
        raise SystemExit(
            f"{view_dir} is a REGION-AGGREGATED view (region="
            f"{view_params.get('region')!r}). The mixed model's row is one channel x "
            'one epoch, and this view has already averaged channels within an ROI. '
            'Build the per-channel view: sbatch sbatch/build_chan_view_array.sbatch')
    if view_params.get('normalization') != 'none':
        raise SystemExit(f'{view_dir} is normalized ('
                         f"{view_params.get('normalization')!r}); regressing an "
                         'already-baselined quantity on pain score is a different and '
                         'largely meaningless number. Use --normalization none.')

    # -- cohort ------------------------------------------------------------
    scores = load_epoch_scores(paths)
    eligible, diagnostics = pain_coef.eligible_subjects(
        scores,
        min_epochs=ref.criteria.get('min_epochs', pain_coef.MIN_EPOCHS),
        min_range=ref.criteria.get('min_range', pain_coef.MIN_RANGE),
        min_non_modal=ref.criteria.get('min_non_modal', pain_coef.MIN_NON_MODAL))
    subjects = set(eligible)
    ref.assert_cohort_matches(subjects, allow_drift=args.allow_cohort_drift)

    roi_scheme = ref.view_params.get('roi_scheme', 'roi_v2')
    roi_by_subject, no_roi = roi_maps(paths, subjects, roi_scheme)
    regions = view_tables.roi_regions_for({'roi_scheme': roi_scheme})

    # -- axes --------------------------------------------------------------
    epoch_minutes = ref.view_params.get('epoch_minutes')
    bin_table = cache_reader.bin_edges(epoch_minutes).set_index('freq_bin_index')
    line_noise = list(ref.line_noise_bins_removed
                      or cache_reader.line_noise_bins(epoch_minutes))
    bin_table = bin_table.drop(index=[b for b in line_noise if b in bin_table.index])
    logger.info('%d frequency bins after removing line-noise bins %s',
                len(bin_table), line_noise)

    min_subjects = int(ref.criteria.get('min_subjects', mm.MIN_SUBJECTS))
    coverage = coverage_map(paths, subjects, roi_by_subject, bin_table.index)
    manifest = build_cell_manifest(bin_table, ref.group_pain_coef(), coverage,
                                   min_subjects)
    logger.info('%d pilot cells:\n%s', len(manifest),
                manifest[['cell_index', 'region', 'freq_bin_index', 'bin_low_hz',
                          'group']].to_string(index=False))

    # -- load just those cells ---------------------------------------------
    wanted = {(r.region, r.freq_bin_index) for r in manifest.itertuples()}
    t0 = time.time()
    frames = load_cell_frames(paths, subjects, wanted, roi_by_subject)
    logger.info('loaded %d cell frame(s) in %.1fs (%d rows total)', len(frames),
                time.time() - t0, sum(len(f) for f in frames.values()))

    # -- run dir -------------------------------------------------------------
    run_dir = view_tables.resolve_run_dir(args, OUTPUT_TYPE, view, run_name=args.run_name)
    (run_dir / 'frames').mkdir(exist_ok=True)
    (run_dir / 'models').mkdir(exist_ok=True)
    logger.info('run dir: %s', run_dir)

    common = dict(params={'reference_run': str(ref.run_dir),
                          'view_dir': str(view_dir),
                          'min_subjects': min_subjects,
                          'line_noise_bins_removed': line_noise,
                          **{k: v for k, v in ref.criteria.items()}},
                  script='ieeg_ehr/analysis/run_mixed_model_pilot.py',
                  subjects=sorted(subjects))

    # -- inventory (Stage 0) --------------------------------------------------
    io.write_table(inventory_subjects(scores, diagnostics),
                   run_dir / 'inventory_subjects.parquet',
                   extra={'status': DISCLAIMER}, **common)
    io.write_table(inventory_regions(roi_by_subject, regions),
                   run_dir / 'inventory_regions.parquet',
                   extra={'status': DISCLAIMER}, **common)
    io.write_table(manifest, run_dir / 'pilot_cell_manifest.parquet',
                   extra={'status': DISCLAIMER}, **common)

    # -- fit -----------------------------------------------------------------
    records, all_blups = [], []
    for r in manifest.itertuples():
        df = frames[(r.region, r.freq_bin_index)]
        meta = {'cell_index': r.cell_index, 'region': r.region,
                'freq_bin_index': r.freq_bin_index, 'group': r.group,
                'bin_low_hz': r.bin_low_hz, 'bin_high_hz': r.bin_high_hz}
        rec, blups, res = fit_one_cell(df, meta,
                                       with_channel_slope=not args.no_channel_slope)
        rec['nrs_within_var'] = (float(np.var(df['NRS_within'], ddof=0))
                                 if len(df) else np.nan)
        records.append(rec)
        all_blups.extend(blups)
        if len(df):
            # The frame, so the permutation array does not re-read the whole view.
            io.write_table(df, run_dir / 'frames' / f'cell_{r.cell_index:03d}.parquet',
                           extra={'status': DISCLAIMER}, **common)
        if res is not None and not args.no_save_models:
            io.save_model(res, run_dir / 'models' /
                          f'cell_{r.cell_index:03d}.joblib',
                          params=common['params'], script=common['script'],
                          subjects=sorted(df['subject'].unique()))

    cells = pd.DataFrame(records).sort_values('cell_index').reset_index(drop=True)
    cells = cells.merge(manifest[['cell_index', 'target_hz', 'bin_centre_hz',
                                  'rationale']], on='cell_index', how='left')
    io.write_table(cells, run_dir / 'pilot_cells.parquet',
                   extra={'status': DISCLAIMER}, **common)
    io.write_table(inventory_cells(frames, manifest),
                   run_dir / 'inventory_cells.parquet',
                   extra={'status': DISCLAIMER}, **common)
    if all_blups:
        io.write_table(pd.DataFrame(all_blups), run_dir / 'pilot_blups.parquet',
                       extra={'status': DISCLAIMER}, **common)

    # -- figures that do not need the permutation ----------------------------
    conv = fig_vs_twostage(cells, ref.group_pain_coef(),
                           run_dir / 'fig_pilot_vs_twostage.png')
    fig_variance(cells, run_dir / 'fig_pilot_variance.png')

    report_fit(cells, conv)
    io.write_run_provenance(
        run_dir, script=common['script'], params=vars(args),
        parents=[str(ref.run_dir / 'provenance.json'), str(view_dir)],
        subjects=sorted(subjects),
        extra={'stage': 'fit', 'status': DISCLAIMER,
               'n_cells': int(len(cells)),
               'n_subjects': len(subjects),
               'subjects_without_roi': sorted(no_roi),
               'confound_caveat': CONFOUND_CAVEAT,
               **ref.provenance_summary()})
    print(run_dir)
    return run_dir


def stage_perm(args):
    run_dir = Path(args.run_dir)
    manifest = io.read_table(run_dir / 'pilot_cell_manifest.parquet', on_stale='ignore')
    row = manifest[manifest['cell_index'] == args.cell_index]
    if row.empty:
        raise SystemExit(f'no cell_index {args.cell_index} in {run_dir}')
    row = row.iloc[0]

    frame_path = run_dir / 'frames' / f'cell_{args.cell_index:03d}.parquet'
    if not frame_path.exists():
        logger.warning('cell %d has no saved frame (it was not fittable); '
                       'writing an empty shard', args.cell_index)
        out = pd.DataFrame(columns=['perm', 'beta', 'z', 'converged', 'error'])
    else:
        df = io.read_table(frame_path, on_stale='ignore')
        logger.info('cell %d: %s bin %d | %d rows, %d subjects, %d channels',
                    args.cell_index, row.region, row.freq_bin_index, len(df),
                    df['subject'].nunique(), df['channel_uid'].nunique())
        t0 = time.time()
        res, _ = mm.fit_cell(df, mm.VC_FULL)
        logger.info('observed fit %.1fs, beta %+.6f z %+.3f p %.4g',
                    time.time() - t0, res.fe_params['NRS_within'],
                    res.tvalues['NRS_within'], res.pvalues['NRS_within'])
        t0 = time.time()
        out = mm.permutation_null(df, args.n_perm, seed=args.seed,
                                  start_params=res.params_object,
                                  n_jobs=args.n_jobs)
        logger.info('%d shuffles in %.1fs (%.2fs/shuffle, %d jobs), %d failed',
                    args.n_perm, time.time() - t0,
                    (time.time() - t0) / max(args.n_perm, 1), args.n_jobs,
                    int((~out['converged']).sum()))

    out.insert(0, 'cell_index', args.cell_index)
    (run_dir / 'perm').mkdir(exist_ok=True)
    io.write_table(out, run_dir / 'perm' / f'perm_{args.cell_index:03d}.parquet',
                   script='ieeg_ehr/analysis/run_mixed_model_pilot.py',
                   params={'n_perm': args.n_perm, 'seed': args.seed,
                           'cell_index': args.cell_index},
                   extra={'status': DISCLAIMER})
    return run_dir


def stage_collect(args):
    run_dir = Path(args.run_dir)
    cells = io.read_table(run_dir / 'pilot_cells.parquet', on_stale='ignore')
    shards = sorted((run_dir / 'perm').glob('perm_*.parquet'))
    if not shards:
        raise SystemExit(f'no permutation shards in {run_dir / "perm"} -- run '
                         '--stage perm first (one array task per cell)')
    nulls = pd.concat([io.read_table(p, on_stale='ignore') for p in shards],
                      ignore_index=True)
    logger.info('%d shard(s), %d null fits, %d failed', len(shards), len(nulls),
                int((~nulls['converged'].fillna(False)).sum()))

    rows = []
    for ci, grp in nulls.groupby('cell_index'):
        obs = cells.loc[cells['cell_index'] == ci, 'beta_nrs_within']
        obs = float(obs.iloc[0]) if len(obs) else np.nan
        p_perm, n_used = mm.permutation_p(obs, grp['beta'].to_numpy())
        rows.append({'cell_index': int(ci), 'p_perm': p_perm, 'n_perm_used': n_used,
                     'n_perm_failed': int(len(grp) - n_used)})
    cells = cells.merge(pd.DataFrame(rows), on='cell_index', how='left')

    calib = fig_calibration(cells, run_dir / 'fig_pilot_calibration.png')
    io.write_table(cells, run_dir / 'pilot_cells.parquet',
                   script='ieeg_ehr/analysis/run_mixed_model_pilot.py',
                   params={'stage': 'collect'},
                   extra={'status': DISCLAIMER, 'calibration': calib})
    io.write_table(nulls, run_dir / 'pilot_permutation_null.parquet',
                   script='ieeg_ehr/analysis/run_mixed_model_pilot.py',
                   params={'stage': 'collect'}, extra={'status': DISCLAIMER})

    report_collect(cells, calib, run_dir)
    io.log_analysis(
        f'PILOT/TEST (Phase 1, not a result): mixed-effects models on '
        f'{len(cells)} region x frequency cells, p-value calibration vs a '
        f'within-subject NRS shuffle, n={cells["n_subjects"].max()}', run_dir)
    return run_dir


# ============================================================================
# REPORTING
# ============================================================================

CONFOUND_CAVEAT = (
    'The inherited QC mask is SIGNAL QUALITY ONLY (gross artifact, saturation, '
    'square wave, flatline, bipolar variance). Opioid-administration windows and '
    'post-ictal periods are NOT excluded -- no medication-state or seizure-proximity '
    'table exists in this project yet (PLANNING.md BG.6, unstarted). Both are '
    'first-order confounds for low-frequency power, so every low-frequency cell here '
    'carries that caveat.')


def report_fit(cells, conv):
    ok = cells[cells['converged'].fillna(False)]
    logger.info('\n' + '=' * 72)
    logger.info('PILOT FIT SUMMARY  (%d cells, %d fitted)', len(cells), len(ok))
    logger.info('  non-convergent / unfittable : %d  %s', len(cells) - len(ok),
                list(cells.loc[~cells['converged'].fillna(False),
                               'region'] + ' ' +
                     cells.loc[~cells['converged'].fillna(False),
                               'freq_bin_index'].astype(str)))
    logger.info('  singular (a VC at boundary) : %d', int(ok['singular_flag'].sum()))
    for comp in ('subj_slope', 'channel', 'subj_int'):
        n = int(ok['boundary_components'].fillna('').str.contains(comp).sum())
        logger.info('      %-11s at boundary : %d', comp, n)
    logger.info('  runtime  : median %.1fs  max %.1fs  total %.0fs',
                ok['fit_seconds'].median(), ok['fit_seconds'].max(),
                cells['fit_seconds'].sum())
    logger.info('  fixed effect p<0.05 (Wald, NOT yet calibrated) : %d',
                int((ok['p'] < 0.05).sum()))
    logger.info('  heterogeneity LRT p<0.05                       : %d',
                int((ok['p_lrt_mixture'] < 0.05).sum()))
    interesting = ok[(ok['p_lrt_mixture'] < 0.05) & (ok['p'] >= 0.05)]
    logger.info('  LRT significant but fixed effect NOT (the interesting ones): %d',
                len(interesting))
    for r in interesting.itertuples():
        logger.info('      %-18s %6.1f-%6.1f Hz  beta %+.5f (p %.3f)  LRT p %.4g',
                    r.region, r.freq_bin_low, r.freq_bin_high, r.beta_nrs_within,
                    r.p, r.p_lrt_mixture)
    if 'p_channel_slope' in ok:
        n_cs = int((ok['p_channel_slope'] < 0.05).sum())
        logger.info('  channel random SLOPE improves fit (p<0.05) : %d/%d %s',
                    n_cs, int(ok['p_channel_slope'].notna().sum()),
                    '<- ROI v2 may be too coarse' if n_cs > len(ok) / 2 else
                    '(prior of intercept-only holds)')
    if conv:
        logger.info('  vs two-stage map: r=%.3f  OLS slope=%.3f  over %d cells',
                    conv['r'], conv['ols_slope'], conv['n_cells'])
    logger.info('  PHASE 2 SIZING: %d regions x 44 bins = 924 cells at the median '
                'full+reduced cost = %.0f min single-core',
                21, 924 * (ok['fit_seconds'].median()
                           + ok.get('fit_seconds_reduced',
                                    pd.Series([0.0])).median()) / 60)
    logger.info(CONFOUND_CAVEAT)
    logger.info('=' * 72)


def report_collect(cells, calib, run_dir):
    d = cells.dropna(subset=['p', 'p_perm'])
    logger.info('\n' + '=' * 72)
    logger.info('P-VALUE CALIBRATION  (%d cells with a permutation p)', len(d))
    if calib:
        logger.info('  parametric p SMALLER than permutation p (anticonservative): '
                    '%d/%d', calib['n_anticonservative'], calib['n_compared'])
        logger.info('  median log10(p_perm / p_parametric) : %+.3f',
                    calib['median_log10_ratio'])
    for r in d.sort_values('p').itertuples():
        logger.info('  %-18s %6.1f Hz | beta %+.5f | p_param %.4g | p_perm %.4g '
                    '| ratio %6.2f', r.region, r.freq_bin_low, r.beta_nrs_within,
                    r.p, r.p_perm, r.p_perm / r.p if r.p else np.nan)
    logger.info('  -> INFERENCE PLAN for Phase 2 follows from the line above. '
                'A consistent ratio > 1 means the Wald p is anticonservative and '
                'the full grid must use permutation p values.')
    logger.info('  run: %s', run_dir)
    logger.info('=' * 72)


# ============================================================================
# CLI
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', choices=['fit', 'perm', 'collect'], default='fit')
    ap.add_argument('--run-dir', default=None,
                    help='Existing run directory. Required for --stage perm/collect.')
    ap.add_argument('--cell-index', type=int, default=None,
                    help='Which pilot cell this array task handles (--stage perm).')
    ap.add_argument('--view-dir', default=None,
                    help='Per-channel view directory. Default: resolved the same way '
                         'the builder resolves it, so the hash is never spelled twice.')
    ap.add_argument('--reference-run', default=str(reference_run.CONTPAIN_HEATMAP),
                    help='Run whose cohort, exclusions and thresholds are inherited.')
    ap.add_argument('--mask-label', default=None,
                    help="Override the reference run's mask label when resolving the "
                         'view directory. Almost never right.')
    ap.add_argument('--n-perm', type=int, default=1000)
    ap.add_argument('--n-jobs', type=int,
                    default=int(__import__('os').environ.get('SLURM_CPUS_PER_TASK', 1)))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--allow-cohort-drift', action='store_true',
                    help='Continue when the re-derived cohort differs from the '
                         "reference's. Every comparison then spans two cohorts.")
    ap.add_argument('--no-channel-slope', action='store_true',
                    help='Skip the channel random-slope comparison (pilot question 1).')
    ap.add_argument('--no-save-models', action='store_true')
    ap.add_argument('--run-name', default=RUN_NAME)
    view_tables.add_output_arguments(ap)
    ap.set_defaults(view_scheme=VIEW_SCHEME)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    io.warn_if_dirty()

    if args.stage == 'fit':
        stage_fit(args)
    elif args.stage == 'perm':
        if not args.run_dir or args.cell_index is None:
            raise SystemExit('--stage perm needs --run-dir and --cell-index')
        stage_perm(args)
    else:
        if not args.run_dir:
            raise SystemExit('--stage collect needs --run-dir')
        stage_collect(args)


if __name__ == '__main__':
    main()
