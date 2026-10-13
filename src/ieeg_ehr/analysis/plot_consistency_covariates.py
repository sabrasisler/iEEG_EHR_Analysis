"""Does anything about the patient track how much their pain map resembles the group's?

    python -m ieeg_ehr.analysis.plot_consistency_covariates --run-dir <run>

`fig_consistency_rfull_hist` shows one number per subject: `r_full`, the Pearson
r between the subject's slope map and the leave-one-out group map. This script
redraws that histogram once per subject-level covariate, each subject a dot
coloured by the covariate, and beside it the covariate against `r_full`
directly. A final overview puts every covariate panel on one page.

Every covariate comes from the SAME epochs and channels that entered the fit,
not from the whole record:

    mdd            MDD ICD code within 90 d before admission (`dx_state`
                   defaults). A billing artefact, not a diagnosis -- see the
                   `dx_state` docstring.
    nrs_mean       mean NRS over model epochs (`inventory_subjects.parquet`).
    nrs_sd         SD of the same. The range a slope is fitted over sets its
                   noise, so this is the first suspect for a low r.
    pct_opioid     % of model epochs with an opioid dose in the 2 h before the
                   score (`med_state.epoch_med_state`, the fit's own join).
    pct_analgesic  the same for any analgesic.
    n_channels     bipolar channels in the model frames.
    n_regions      ROIs with at least one slope (`consistency_subjects.csv`).
    n_epochs       model epochs (subject-session keyed, from the view the fit read).

A subject absent from the MAR export reads as NaN for the two dosing columns,
not 0 %: "no export" is unobserved, not undosed (`epoch_meds.restrict_to_observable`
makes the same call).

Statistics are one point per subject, so no per-subject structure is pooled
away: Spearman rho for the continuous covariates, the median difference and
rank-biserial r for MDD, each with its raw p and a BH q across the covariates.

Output: `<run>/consistency/rfull_covariates_<ts>/`.

EXPLORATORY. Discovery cohort only. Nominations, not findings.
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ieeg_ehr import io
from ieeg_ehr.analysis import dx_state, fullres_cells, med_state
from ieeg_ehr.analysis.plot_bandpower_consistency import (ACCENT_A, ACCENT_B,
                                                          OUT_SUBDIR,
                                                          RFULL_POSTER_FS)
from ieeg_ehr.decoding.aggregate import benjamini_hochberg

logger = logging.getLogger(__name__)

SCRIPT = 'ieeg_ehr/analysis/plot_consistency_covariates.py'

CONTINUOUS = {
    'nrs_mean': 'mean pain score (NRS)',
    'nrs_sd': 'pain score SD (NRS)',
    'pct_opioid': '% of epochs dosed by opioids\n(2 h before the score)',
    'pct_analgesic': '% of epochs dosed by any analgesic\n(2 h before the score)',
    'n_channels': 'bipolar channels in model',
    'n_regions': 'ROIs covered',
    'n_epochs': 'epochs in model',
}
MDD_LABEL = 'MDD code (90 d before admission)'
COVARIATES = ('mdd', *CONTINUOUS)

#: magma reversed so a darker dot is a HIGHER value, trimmed at both ends: the
#: top of magma is near-white and vanishes on the page, the bottom is flat black.
MAGMA_RANGE = (0.90, 0.05)
MISSING_COLOR = '#bdbdbd'
BIN_WIDTH = 0.05


def build_table(run_dir):
    """One row per subject with `r_full`: the covariates over the model's own data."""
    sim = io.read_table(run_dir / OUT_SUBDIR / 'consistency_subjects.csv',
                        on_stale='warn')
    tab = sim.loc[sim['r_full'].notna(), ['subject', 'r_full', 'n_regions']]
    subjects = sorted(tab['subject'])
    bare = [s.replace('sub-', '') for s in subjects]

    inv = io.read_table(run_dir / 'inventory_subjects.parquet', on_stale='warn')
    tab = tab.merge(inv.rename(columns={'subject_id': 'subject'})
                    [['subject', 'nrs_mean', 'nrs_sd']], on='subject', how='left')

    frames = pd.concat(
        [io.read_table(p, columns=['subject', 'channel_uid', 'epoch_id'],
                       on_stale='ignore')
         for p in sorted((run_dir / 'frames').glob('cell_*.parquet'))],
        ignore_index=True)
    frames = frames[frames['subject'].isin(subjects)]
    tab = tab.merge(frames.groupby('subject')['channel_uid'].nunique()
                    .rename('n_channels').reset_index(), on='subject', how='left')

    # The model's epochs. The frames drop the session, and epoch_id is unique
    # only within a subject-session, so the session comes back from the view
    # the fit read: one file per subject-session.
    params = json.loads((run_dir / 'provenance.json').read_text())['params']
    view = []
    for p in fullres_cells.subject_paths(params['view_dir']):
        subject, session = fullres_cells.subject_session_of(p)
        view.append(io.read_table(p, columns=['epoch_id'], on_stale='warn')
                    .drop_duplicates().assign(subject=subject, session=session))
    used = (frames[['subject', 'epoch_id']].drop_duplicates()
            .assign(subject=lambda d: d['subject'].str.replace('sub-', '',
                                                               regex=False))
            .merge(pd.concat(view, ignore_index=True),
                   on=['subject', 'epoch_id'], how='inner'))
    defs = med_state.load_epoch_defs(
        epoch_minutes=params['view_params']['epoch_minutes'], subjects=bare)
    defs = defs.merge(used, on=['subject', 'session', 'epoch_id'], how='inner')
    if len(defs) != len(used):
        raise ValueError(f'{len(used) - len(defs)} model epoch(s) have no '
                         'epoch definition')
    tab['n_epochs'] = tab['subject'].str.replace('sub-', '', regex=False).map(
        defs.groupby('subject').size())

    admin = med_state.load_admin_table()
    unobserved = sorted(set(bare) - set(admin['subject']))
    if unobserved:
        logger.warning('%d subject(s) absent from the MAR export, dosing NaN: %s',
                       len(unobserved), unobserved)
    for col, drug_set in (('pct_opioid', 'opioids'),
                          ('pct_analgesic', 'analgesics')):
        state = med_state.epoch_med_state(
            defs, admin[admin['level2'].isin(med_state.DRUG_SETS[drug_set])],
            hours=med_state.DEFAULT_WINDOW_HOURS)
        pct = 100 * state.groupby('subject')['med_state'].mean()
        pct[pct.index.isin(unobserved)] = np.nan
        tab[col] = tab['subject'].str.replace('sub-', '', regex=False).map(pct)

    labels, _ = dx_state.subject_labels(dx_state.load_diagnoses(subjects=bare),
                                        condition='mdd')
    tab['mdd'] = tab['subject'].map(labels.set_index('subject_id')['dx_state']
                                    .astype(float))
    if tab['mdd'].isna().any():
        logger.warning('no diagnoses table, MDD unknown: %s',
                       tab.loc[tab['mdd'].isna(), 'subject'].tolist())
    return tab[['subject', 'r_full', *COVARIATES]].reset_index(drop=True)


def covariate_stats(tab):
    rows = []
    for col in COVARIATES:
        d = tab[['r_full', col]].dropna()
        if col == 'mdd':
            a = d.loc[d['mdd'] == 1, 'r_full']
            b = d.loc[d['mdd'] == 0, 'r_full']
            u, p = stats.mannwhitneyu(a, b, alternative='two-sided')
            rows.append({'covariate': col, 'test': 'Mann-Whitney U', 'n': len(d),
                         'n_mdd': len(a), 'n_no_mdd': len(b),
                         'effect': 2 * u / (len(a) * len(b)) - 1,
                         'effect_name': 'rank-biserial r (MDD > no MDD)',
                         'median_diff': a.median() - b.median(), 'p': p})
        else:
            rho, p = stats.spearmanr(d[col], d['r_full'])
            rows.append({'covariate': col, 'test': 'Spearman', 'n': len(d),
                         'effect': rho, 'effect_name': 'Spearman rho', 'p': p})
    out = pd.DataFrame(rows)
    out['q_bh'] = benjamini_hochberg(out['p'])
    return out


def _magma():
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    return ListedColormap(plt.cm.magma(np.linspace(*MAGMA_RANGE, 256)))


def point_colors(tab, col):
    """Per-subject RGBA, plus the (cmap, norm) a colorbar needs (None for MDD)."""
    from matplotlib.colors import Normalize, to_rgba
    v = tab[col].to_numpy(dtype=float)
    if col == 'mdd':
        lut = {1.0: ACCENT_B, 0.0: ACCENT_A}
        return np.array([to_rgba(lut.get(x, MISSING_COLOR)) for x in v]), None
    cmap, norm = _magma(), Normalize(np.nanmin(v), np.nanmax(v))
    rgba = cmap(norm(np.nan_to_num(v)))
    rgba[~np.isfinite(v)] = to_rgba(MISSING_COLOR)
    return rgba, (cmap, norm)


def _style(ax, fs):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=fs['tick'], colors='black')


def draw_dot_hist(ax, tab, col, colors, fs):
    """`fig_consistency_rfull_hist` as stacked dots, one per subject."""
    r = tab['r_full'].to_numpy(dtype=float)
    lo = np.floor(r.min() / BIN_WIDTH) * BIN_WIDTH
    b = np.floor((r - lo) / BIN_WIDTH + 1e-9).astype(int)
    v = tab[col].to_numpy(dtype=float)
    y = np.zeros(len(r))
    for k in np.unique(b):
        idx = np.flatnonzero(b == k)
        # Within a bin, lower values sit at the bottom; missing values go last.
        idx = idx[np.argsort(np.where(np.isfinite(v[idx]), v[idx], np.inf),
                             kind='stable')]
        y[idx] = np.arange(1, len(idx) + 1)
    x = lo + (b + 0.5) * BIN_WIDTH
    ax.scatter(x, y, s=70, c=colors, edgecolors='#4d4d4d', linewidths=0.5,
               zorder=3)
    med = float(np.median(r))
    ax.axvline(0, color='black', lw=0.8, ls='--', zorder=2)
    ax.axvline(med, color='black', lw=1.2, zorder=2)
    ax.set_ylim(0.3, y.max() + 0.9)
    ax.text(med, ax.get_ylim()[1], f'  median r = {med:.2f}', ha='left',
            va='top', fontsize=fs['note'])
    ax.yaxis.get_major_locator().set_params(integer=True)
    ax.set_xlabel(fs['xlabel'], fontsize=fs['label'])
    ax.set_ylabel(f'subjects (n = {len(r)})', fontsize=fs['label'])
    _style(ax, fs)


def _stat_text(row):
    if row['covariate'] == 'mdd':
        head = (f"median Δr = {row['median_diff']:+.2f}, "
                f"rank-biserial = {row['effect']:+.2f}")
    else:
        head = f"Spearman ρ = {row['effect']:+.2f}"
    return f"{head}\np = {row['p']:.3f}, BH q = {row['q_bh']:.3f}, n = {row['n']}"


def draw_relation(ax, tab, col, colors, row, fs):
    """The covariate against `r_full`: a scatter, or a strip plot for MDD."""
    r = tab['r_full'].to_numpy(dtype=float)
    v = tab[col].to_numpy(dtype=float)
    ok = np.isfinite(v)
    if col == 'mdd':
        rng = np.random.default_rng(0)
        x = v + rng.uniform(-0.12, 0.12, len(v))
        for g in (0.0, 1.0):
            ax.hlines(np.median(r[v == g]), g - 0.25, g + 0.25, color='black',
                      lw=1.6, zorder=4)
        ax.set_xticks([0, 1],
                      [f'no MDD\n(n = {int((v == 0).sum())})',
                       f'MDD\n(n = {int((v == 1).sum())})'])
        ax.set_xlim(-0.6, 1.6)
        ax.set_xlabel(MDD_LABEL, fontsize=fs['label'])
    else:
        x = v
        ax.set_xlabel(CONTINUOUS[col], fontsize=fs['label'])
    ax.scatter(x[ok], r[ok], s=55, c=colors[ok], edgecolors='#4d4d4d',
               linewidths=0.5, zorder=3)
    ax.axhline(0, color='black', lw=0.8, ls='--', zorder=2)
    ax.set_ylabel('Pearson r vs. leave-one-out group map',
                  fontsize=fs['label'])
    ax.text(0.02, 0.98, _stat_text(row), transform=ax.transAxes, ha='left',
            va='top', fontsize=fs['note'])
    ax.set_ylim(min(-0.4, r.min() - 0.05), r.max() + 0.35)
    _style(ax, fs)


def _colorbar(fig, ax, mappable_spec, col, fs):
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    cmap, norm = mappable_spec
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.02,
                      fraction=0.05)
    cb.set_label(CONTINUOUS[col].replace('\n', ' '), fontsize=fs['label'])
    cb.ax.tick_params(labelsize=fs['tick'])
    plt.setp(cb.ax.get_yticklabels(), color='black')


def fig_covariate(tab, col, row, out, fs=RFULL_POSTER_FS):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    colors, spec = point_colors(tab, col)
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 5.0), layout='constrained',
                               gridspec_kw={'width_ratios': [1.15, 1]})
    draw_dot_hist(a, tab, col, colors, fs)
    draw_relation(b, tab, col, colors, row, fs)
    if spec is None:
        handles = [Line2D([], [], ls='', marker='o', ms=8, mfc=c, mec='#4d4d4d',
                          label=lab)
                   for c, lab in ((ACCENT_B, 'MDD'), (ACCENT_A, 'no MDD'))]
        a.legend(handles=handles, frameon=False, fontsize=fs['note'],
                 loc='upper left')
    else:
        _colorbar(fig, a, spec, col, fs)
    n_missing = int(tab[col].isna().sum())
    if n_missing:
        a.text(0.02, 0.98, f'grey: {n_missing} subject(s) with no value',
               transform=a.transAxes, ha='left', va='top',
               fontsize=fs['note'], color='#595959')
    title = MDD_LABEL if col == 'mdd' else CONTINUOUS[col].replace('\n', ' ')
    fig.suptitle(f'Similarity to the group map, by {title}',
                 fontsize=fs['title'], fontweight=fs['title_weight'])
    fig.savefig(out, dpi=300, facecolor='white')
    plt.close(fig)
    logger.info('wrote %s', out)


def fig_overview(tab, st, out, fs=RFULL_POSTER_FS):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(20, 9.5), layout='constrained')
    for ax, col in zip(axes.flat, COVARIATES):
        colors, _ = point_colors(tab, col)
        draw_relation(ax, tab, col, colors,
                      st.set_index('covariate').loc[col].to_dict() | {'covariate': col},
                      fs)
    fig.suptitle('Does any subject-level variable track similarity to the group '
                 'map?  (one dot per subject; darker = higher)',
                 fontsize=fs['title'], fontweight=fs['title_weight'])
    fig.savefig(out, dpi=300, facecolor='white')
    plt.close(fig)
    logger.info('wrote %s', out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-dir', required=True,
                    help='A band-power mixed-model run with a consistency/ '
                         'subfolder from plot_bandpower_consistency.py.')
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    io.warn_if_dirty()
    import matplotlib
    matplotlib.use('Agg')

    run_dir = Path(args.run_dir)
    tab = build_table(run_dir)
    st = covariate_stats(tab)
    logger.info('\n%s', st.to_string(index=False))

    out_dir = (run_dir / OUT_SUBDIR
               / f"rfull_covariates_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    out_dir.mkdir(parents=True)
    rows = st.set_index('covariate')
    for col in COVARIATES:
        fig_covariate(tab, col, rows.loc[col].to_dict() | {'covariate': col},
                      out_dir / f'fig_rfull_by_{col}.png')
    fig_overview(tab, st, out_dir / 'fig_rfull_covariates_overview.png')

    parents = [str(run_dir / OUT_SUBDIR / 'consistency_subjects.csv'),
               str(run_dir / 'inventory_subjects.parquet'),
               str(run_dir / 'frames'), str(med_state.ADMIN_TABLE)]
    params = {'dx': 'mdd', 'dx_window_days': dx_state.DEFAULT_WINDOW_DAYS,
              'dx_sources': 'any',
              'med_window_hours': med_state.DEFAULT_WINDOW_HOURS,
              'drug_sets': {k: list(med_state.DRUG_SETS[k])
                            for k in ('opioids', 'analgesics')},
              'source_run': str(run_dir)}
    subjects = sorted(tab['subject'])
    io.write_table(tab, out_dir / 'rfull_covariates.csv', params=params,
                   parents=parents, subjects=subjects, script=SCRIPT)
    io.write_table(st, out_dir / 'rfull_covariate_stats.csv', params=params,
                   parents=parents, subjects=subjects, script=SCRIPT)
    io.write_run_provenance(out_dir, script=SCRIPT, params=params,
                            parents=parents, subjects=subjects,
                            extra={'status': 'EXPLORATORY -- nominations, not '
                                             'findings.'})
    io.log_analysis('consistency r_full by subject covariates (MDD, pain, '
                    'dosing, coverage) -- EXPLORATORY', out_dir)


if __name__ == '__main__':
    main()
