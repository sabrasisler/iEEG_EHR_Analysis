"""Is the medication interaction a finding, or two unstable estimates differenced?

`fig_interaction_map` shows the baseline slope, the change, and their sum, which
is enough to see THAT the estimates differ and not enough to decide whether the
difference should be believed. Two figures here separate the readings.

`fig_simple_slopes` -- four panels, region x frequency:

    1  unmedicated slope    beta_NRS_within            (the model's med_state=0 slope)
    2  MEDICATED slope      beta_NRS_within + med_ix   (the simple slope; derived)
    3  interaction          med_ix_beta                (the difference, panel 2 - 1)
    4  offset               med_main_beta              (power at average pain)

Panels 1-3 SHARE a colour scale, because they are the same quantity and the
whole question is how panel 2 compares to panel 1. Panel 4 is a level shift in
log10 power rather than a slope, is an order of magnitude larger, and gets its
own bar -- putting it on the shared scale would saturate it and flatten the
three panels that matter.

THE READING THIS FIGURE IS FOR. If medication genuinely abolishes pain encoding,
panel 2 is flat near zero everywhere and clean, panel 3 is close to the negative
of panel 1, and RMS(medicated) is far below RMS(unmedicated). If instead panel 2
is structured and noisy in a pattern complementary to panel 1 -- comparable RMS,
comparable smoothness, opposite sign -- then both stratum slopes are noisy and
their difference is being over-read. The summary statistics for exactly that call
are computed onto the figure and into `simple_slopes_summary.csv`.

`fig_nrs_range_by_condition` -- the other half of the diagnostic, and the one
that can kill the comparison outright. A slope is only as trustworthy as the
range of the predictor it was fitted over. Patients are medicated BECAUSE they
are in pain, so the unmedicated epochs may occupy a compressed band of NRS, and a
slope fitted over a compressed range is both attenuated and unstable. This plots
the per-region, per-condition distribution of NRS_within, in BOTH centrings that
the models actually use: pooled over strata (what the interaction model sees) and
re-centred within stratum (what the stratified medpos/medneg fits see). If the
unmedicated spread is visibly narrower, be skeptical of the difference before
interpreting any cell of it.

    python -m ieeg_ehr.analysis.plot_med_simple_slopes --run-dir <med strata run>
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ieeg_ehr import io
from ieeg_ehr.analysis import view_tables
from ieeg_ehr.analysis.plot_med_design_checks import load_design
from ieeg_ehr.analysis.plot_mixed_model_grid import order_regions, pivot
from ieeg_ehr.analysis.run_mixed_model_pilot import (roi_maps, resolve_view_dir,
                                                     view_subject_paths)
from ieeg_ehr.features import common

logger = logging.getLogger(__name__)

DISCLAIMER = ('EXPLORATORY -- discovery cohort, NOMINATIONS NOT FINDINGS. Parametric '
              'Wald p (pilot permutation put the null z SD at ~1.03). Not confirmed '
              'out of sample.')

MED_COLOR = '#c1442f'
UNMED_COLOR = '#4a7ba7'

#: Panel 2 is arithmetic on panels 1 and 3, so it has no BH family of its own and
#: never gets an outline. The other three each have one.
REJECT_COLUMN = {'unmedicated': 'p_bh_reject', 'medicated': None,
                 'interaction': 'med_ix_bh_reject', 'offset': 'med_main_bh_reject'}


# ============================================================================
# PANEL 2: THE SIMPLE SLOPE
# ============================================================================

def simple_slopes(cells):
    """`cells` plus a `beta_medicated` column: the slope implied at med_state=1.

    beta_NRS_within is the slope when med_state=0 and med_ix_beta is how much it
    moves when med_state=1, so their SUM is the medicated slope. It is not fitted
    and has no standard error here -- the variance of a sum needs the covariance
    of the two coefficients, which the cell records do not store. `medpos` is the
    independently-fitted version with a real SE if one is needed.
    """
    need = {'beta_nrs_within', 'med_ix_beta', 'med_main_beta'}
    missing = need - set(cells.columns)
    if missing:
        raise SystemExit(f'grid_cells.parquet lacks {sorted(missing)} -- this is '
                         'not an interaction run')
    out = cells.copy()
    out['beta_medicated'] = out['beta_nrs_within'] + out['med_ix_beta']
    return out


def _smoothness(mat):
    """Correlation between neighbouring frequency bins within a region.

    The statistic that separates "structured" from "noisy", which RMS cannot do:
    a panel of independent noise has a neighbour correlation near 0 however large
    its RMS, while a panel carrying real spectral structure has a high one
    because adjacent log-spaced bins share FFT frequencies and physiology. Taken
    ACROSS frequency within a row, never down a column, since regions are not
    ordered and a column neighbour means nothing.
    """
    a = mat.to_numpy(dtype=float)
    left, right = a[:, :-1].ravel(), a[:, 1:].ravel()
    ok = np.isfinite(left) & np.isfinite(right)
    if ok.sum() < 3:
        return np.nan
    return float(np.corrcoef(left[ok], right[ok])[0, 1])


def _rms(mat):
    a = mat.to_numpy(dtype=float)
    a = a[np.isfinite(a)]
    return float(np.sqrt(np.mean(a ** 2))) if a.size else np.nan


def stability_summary(panels):
    """The numbers that decide "abolished encoding" vs "two unstable estimates".

    Returned as rows rather than printed, so the call is auditable later against
    the figure that was read at the time.
    """
    unmed, med = panels['unmedicated'], panels['medicated']
    ix = panels['interaction']
    u, m, i = (x.to_numpy(dtype=float).ravel() for x in (unmed, med, ix))
    ok_um = np.isfinite(u) & np.isfinite(m)
    ok_ui = np.isfinite(u) & np.isfinite(i)

    rows = [{'statistic': f'rms_{name}', 'value': _rms(mat),
             'reads_as': 'typical |coefficient| in the panel'}
            for name, mat in panels.items()]
    rows += [{'statistic': f'freq_neighbour_corr_{name}', 'value': _smoothness(mat),
              'reads_as': 'high = spectrally structured, ~0 = noise'}
             for name, mat in panels.items()]
    rows.append({'statistic': 'rms_ratio_medicated_over_unmedicated',
                 'value': _rms(med) / _rms(unmed) if _rms(unmed) else np.nan,
                 'reads_as': '<<1 supports abolished encoding, ~1 supports two '
                             'comparably noisy estimates'})
    rows.append({'statistic': 'corr_unmedicated_medicated',
                 'value': (float(np.corrcoef(u[ok_um], m[ok_um])[0, 1])
                           if ok_um.sum() > 2 else np.nan),
                 'reads_as': 'strongly negative = the two panels mirror each other'})
    rows.append({'statistic': 'corr_unmedicated_interaction',
                 'value': (float(np.corrcoef(u[ok_ui], i[ok_ui])[0, 1])
                           if ok_ui.sum() > 2 else np.nan),
                 'reads_as': 'NEGATIVELY BIASED BY CONSTRUCTION -- see figure note; '
                             'not evidence on its own'})
    rows.append({'statistic': 'frac_cells_medicated_smaller',
                 'value': (float(np.mean(np.abs(m[ok_um]) < np.abs(u[ok_um])))
                           if ok_um.any() else np.nan),
                 'reads_as': 'fraction of cells where |medicated| < |unmedicated|'})
    return pd.DataFrame(rows)


def fig_simple_slopes(cells, regions, bins, bin_labels, out_path,
                      outline_significance=False):
    """Four panels: unmedicated, medicated, interaction, offset."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    panels = {'unmedicated': pivot(cells, 'beta_nrs_within', regions, bins),
              'medicated': pivot(cells, 'beta_medicated', regions, bins),
              'interaction': pivot(cells, 'med_ix_beta', regions, bins),
              'offset': pivot(cells, 'med_main_beta', regions, bins)}
    stats = stability_summary(panels).set_index('statistic')['value']

    # ONE scale over the three slope panels, set by all three together. Scaling
    # each to its own range is the specific mistake this figure exists to avoid:
    # it makes a slope that reversed and a slope that halved look identical.
    slope_names = ('unmedicated', 'medicated', 'interaction')
    vmax = float(np.nanmax(np.abs(np.concatenate(
        [panels[n].to_numpy(dtype=float).ravel() for n in slope_names]))))
    off_max = float(np.nanmax(np.abs(panels['offset'].to_numpy(dtype=float))))

    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad('0.85')

    titles = {
        'unmedicated': ('pain slope when UNMEDICATED\n(beta_NRS_within; fitted)'),
        'medicated': (f'pain slope when MEDICATED\n(beta_NRS_within + interaction; '
                      f'derived) RMS {stats["rms_medicated"]:.4f}'),
        'interaction': ('INTERACTION\n(med_ix_beta; panel 2 minus panel 1)'),
        'offset': ('MEDICATION OFFSET\n(med_main_beta; power at average pain)'),
    }
    titles['unmedicated'] += f'  RMS {stats["rms_unmedicated"]:.4f}'

    order = ['unmedicated', 'medicated', 'interaction', 'offset']
    n_heat = len(order)
    fig, axes = plt.subplots(
        1, n_heat + 2, figsize=(5.8 * n_heat + 3.6, 0.42 * len(regions) + 3.6),
        gridspec_kw={'width_ratios': [1] * n_heat + [0.34, 0.34]})

    for i, name in enumerate(order):
        ax = axes[i]
        lim = off_max if name == 'offset' else vmax
        im = ax.imshow(panels[name].to_numpy(dtype=float), aspect='auto', cmap=cmap,
                       vmin=-lim, vmax=lim, interpolation='nearest')
        col = REJECT_COLUMN[name]
        if outline_significance and col and col in cells.columns:
            sig = pivot(cells, col, regions, bins).fillna(False).astype(bool)
            common.draw_mask_outline(ax, sig.to_numpy())
        ax.set_title(titles[name], fontsize=10)
        ax.set_xticks(range(len(bins)))
        ax.set_xticklabels([f'{bin_labels.loc[b, "bin_low_hz"]:.0f}' for b in bins],
                           fontsize=6, rotation=90)
        ax.set_yticks(range(len(regions)))
        ax.set_yticklabels(regions if i == 0 else [], fontsize=8)
        ax.set_xlabel('frequency bin, low edge (Hz)', fontsize=8)
        common.add_band_boundary_lines(ax, bin_labels.loc[bins])
        cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cb.ax.tick_params(labelsize=7)

    for ax, col, label, colour in ((axes[n_heat], 'n_subjects', 'subjects', '0.55'),
                                   (axes[n_heat + 1], 'n_channels', 'electrodes',
                                    '#4a7ba7')):
        per_region = pivot(cells, col, regions, bins).max(axis=1)
        ax.barh(range(len(regions)), per_region.to_numpy(), color=colour)
        ax.set_yticks(range(len(regions)))
        ax.set_yticklabels([])
        ax.set_ylim(len(regions) - 0.5, -0.5)
        ax.set_title(f'n {label}\nper region', fontsize=10)
        ax.set_xlabel(f'n {label}', fontsize=8)
        ax.tick_params(labelsize=7)
        vals = per_region.to_numpy()
        span = np.nanmax(vals) if np.isfinite(np.nanmax(vals)) else 1.0
        for j, v in enumerate(vals):
            if np.isfinite(v):
                ax.text(v + 0.02 * span, j, f'{int(v)}', va='center', fontsize=6.2,
                        color='0.3')

    sig_note = ('Outlines are BH at q=0.05, each panel against its own family; '
                'panel 2 is derived and has none. '
                if outline_significance else
                'NO SIGNIFICANCE IS SHOWN: every cell is drawn on its estimate '
                'alone. ')

    fig.suptitle('Medicated vs unmedicated pain slopes, their difference, and the '
                 'medication offset', fontsize=13)
    fig.tight_layout(rect=(0, 0.10, 1, 0.945))
    fig.text(0.01, 0.005,
             'PANELS 1-3 SHARE ONE COLOUR SCALE and are the same quantity; panel 4 '
             'is a level shift in log10 power, not a slope, and has its own bar '
             '(it is ~{:.0f}x larger, so sharing the scale would flatten the '
             'others). Panel 2 is arithmetic, not a separate fit: it has no '
             'standard error here, because the variance of the sum needs the '
             'covariance of the two coefficients, which the cell records do not '
             'store -- the independently fitted version with a real SE is the '
             '`medpos` map. {}\n'
             'IS THE DIFFERENCE REAL? RMS(medicated)/RMS(unmedicated) = {:.2f}. If '
             'medication abolished encoding this would be far below 1 and panel 2 '
             'would be flat and clean; near 1 means both slopes are comparably '
             'large and their difference is the small quantity. Frequency-neighbour '
             'correlation -- high means spectrally structured, ~0 means noise -- is '
             '{:.2f} unmedicated, {:.2f} medicated, {:.2f} interaction. '
             'corr(unmedicated, medicated) = {:.2f} across cells; {:.0%} of cells '
             'have |medicated| < |unmedicated|. NOTE corr(unmedicated, interaction) '
             '= {:.2f} is NOT evidence: the interaction IS medicated minus '
             'unmedicated, so noise in panel 1 propagates into panel 3 with a '
             'negative sign and biases that correlation downward by construction. '
             'Read the RMS ratio and the smoothness, not that number.\n'.format(
                 (off_max / vmax) if vmax else float('nan'), sig_note,
                 stats['rms_ratio_medicated_over_unmedicated'],
                 stats['freq_neighbour_corr_unmedicated'],
                 stats['freq_neighbour_corr_medicated'],
                 stats['freq_neighbour_corr_interaction'],
                 stats['corr_unmedicated_medicated'],
                 stats['frac_cells_medicated_smaller'],
                 stats['corr_unmedicated_interaction'])
             + 'CONFOUNDING BY INDICATION: patients are medicated BECAUSE they are '
               'in pain, so med_state and NRS_within are correlated within subject '
               'and this is not an orthogonal design. Read '
               'fig_nrs_range_by_condition before any cell of this one.\n'
             + DISCLAIMER,
             fontsize=7, va='bottom', ha='left', color='0.35', wrap=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return panels, stability_summary(panels)


# ============================================================================
# THE PREDICTOR RANGE
# ============================================================================

def region_epochs(run_dir, roi_scheme, view_dir=None):
    """Epoch-level design rows, repeated once per region the subject covers.

    Region is a property of an ELECTRODE, not of an epoch -- one pain score is
    shared by every channel that subject has -- so "the NRS range in Insula" means
    the range over the subjects who have an insula contact. That is the range each
    insula cell was actually fitted over, and it differs between regions only
    because the contributing subjects differ.
    """
    d = load_design(run_dir)
    # The run's OWN view directory, not a re-derivation: resolve_view_dir hashes
    # mask_label into the path, and re-deriving it here without the run's
    # mask_label would silently resolve to a different directory.
    mask_label = None
    if view_dir is None:
        import json
        try:
            params = json.loads(
                (Path(run_dir) / 'provenance.json').read_text()).get('params', {})
        except (OSError, ValueError):
            params = {}
        view_dir = params.get('view_dir')
        mask_label = (params.get('view_params') or {}).get('mask_label')
    paths = view_subject_paths(resolve_view_dir(view_dir, mask_label=mask_label,
                                                roi_scheme=roi_scheme))
    roi_by_subject, _ = roi_maps(paths, {f'sub-{s}' for s in d['subject'].unique()},
                                 roi_scheme)
    frames = []
    for sid, mapping in roi_by_subject.items():
        sub = d[d['subject'] == sid.replace('sub-', '')]
        if sub.empty:
            continue
        for region in sorted(set(mapping.values())):
            frames.append(sub.assign(region=region))
    if not frames:
        raise SystemExit('no subject contributed an ROI-labelled channel')
    out = pd.concat(frames, ignore_index=True)

    # The two centrings the models actually use, computed per region because the
    # subject set -- and therefore each subject's mean -- is per region.
    g = out.groupby(['region', 'subject'])['pain_score']
    out['nrs_within_pooled'] = out['pain_score'] - g.transform('mean')
    gs = out.groupby(['region', 'subject', 'med_state'])['pain_score']
    out['nrs_within_stratum'] = out['pain_score'] - gs.transform('mean')
    return out


POOLED_ROW = 'ALL REGIONS POOLED'


def pooled_epochs(d):
    """`d` with each epoch ONCE, for the pooled summary row.

    `region_epochs` repeats every epoch once per region that patient covers, so
    summing the frame would count a patient with 12 regions twelve times and
    report a region-weighted NRS mean as if it were the cohort's. Dropping the
    duplicates is exact rather than approximate here: a patient contributes ALL
    of their epochs to every region they cover, so the per-subject mean -- and
    therefore both centred columns -- is identical across their region copies.
    """
    return d.drop_duplicates(['subject', 'session', 'epoch_id'])


def nrs_spread_table(d, regions):
    """Per region x condition: n, and the spread of NRS under both centrings."""
    rows = []
    for region in regions + [POOLED_ROW]:
        sub = pooled_epochs(d) if region == POOLED_ROW else d[d['region'] == region]
        if sub.empty:
            continue
        for med in (False, True):
            s = sub[sub['med_state'] == med]
            rows.append({
                'region': region,
                'condition': 'medicated' if med else 'unmedicated',
                'n_epochs': int(len(s)),
                'n_subjects': int(s['subject'].nunique()),
                'nrs_mean': float(s['pain_score'].mean()) if len(s) else np.nan,
                'nrs_sd': float(s['pain_score'].std(ddof=0)) if len(s) else np.nan,
                'within_pooled_sd': (float(s['nrs_within_pooled'].std(ddof=0))
                                     if len(s) else np.nan),
                'within_stratum_sd': (float(s['nrs_within_stratum'].std(ddof=0))
                                      if len(s) else np.nan),
                'within_pooled_iqr': (float(s['nrs_within_pooled'].quantile(0.75)
                                            - s['nrs_within_pooled'].quantile(0.25))
                                      if len(s) else np.nan)})
    t = pd.DataFrame(rows)
    wide = t.pivot(index='region', columns='condition', values='within_stratum_sd')
    ratio = (wide['unmedicated'] / wide['medicated']).rename('sd_ratio_unmed_over_med')
    return t.merge(ratio.reset_index(), on='region', how='left')


def fig_nrs_range(d, regions, table, out_path):
    """Per region, per condition: is the unmedicated pain range compressed?"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    rows = regions + [POOLED_ROW]
    fig, axes = plt.subplots(1, 3, figsize=(19, 0.44 * len(rows) + 4.2),
                             gridspec_kw={'width_ratios': [1.5, 1, 0.8]},
                             sharey=True)

    # Panel 1: the distributions themselves. A violin rather than a box because
    # the question is whether the unmedicated epochs pile up in a narrow band,
    # and a box plot showing the same IQR can hide a bimodal spread that is not
    # a compressed one.
    ax = axes[0]
    for offset, med, colour in ((-0.19, False, UNMED_COLOR), (0.19, True, MED_COLOR)):
        data, pos = [], []
        for i, region in enumerate(rows):
            sub = (pooled_epochs(d) if region == POOLED_ROW
                   else d[d['region'] == region])
            v = sub.loc[sub['med_state'] == med, 'nrs_within_pooled'].dropna()
            # A single-point violin has no width and matplotlib raises on the
            # degenerate covariance rather than drawing nothing.
            if len(v) > 1 and v.std(ddof=0) > 0:
                data.append(v.to_numpy())
                pos.append(i + offset)
        if not data:
            continue
        parts = ax.violinplot(data, positions=pos, vert=False, widths=0.34,
                              showextrema=False, showmedians=True)
        for body in parts['bodies']:
            body.set_facecolor(colour)
            body.set_alpha(0.55)
            body.set_linewidth(0)
        parts['cmedians'].set_color('0.15')
        parts['cmedians'].set_linewidth(1.0)
    ax.axvline(0, color='0.5', lw=0.9, ls='--')
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=8)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlabel('NRS_within (pain score minus that patient\'s mean, pooled over '
                  'strata)', fontsize=8)
    ax.set_title('Distribution of the predictor, per region and condition',
                 fontsize=11)
    ax.legend(handles=[Line2D([], [], color=UNMED_COLOR, lw=6, alpha=0.6,
                              label='not medicated'),
                       Line2D([], [], color=MED_COLOR, lw=6, alpha=0.6,
                              label='medicated')], fontsize=8, loc='lower right')

    # Panel 2: the spread as a number, under BOTH centrings, because the two
    # model families see different ones and can be compressed differently.
    ax = axes[1]
    idx = {r: i for i, r in enumerate(rows)}
    for med, colour in ((False, UNMED_COLOR), (True, MED_COLOR)):
        t = table[table['condition'] == ('medicated' if med else 'unmedicated')]
        t = t[t['region'].isin(idx)]
        y = [idx[r] for r in t['region']]
        ax.scatter(t['within_pooled_sd'], y, s=34, color=colour, zorder=3,
                   label=f'{"medicated" if med else "not medicated"} (pooled centring)')
        ax.scatter(t['within_stratum_sd'], y, s=34, facecolors='none',
                   edgecolors=colour, linewidths=1.3, zorder=3,
                   label=f'{"medicated" if med else "not medicated"} (within-stratum)')
    ax.set_xlabel('SD of NRS_within', fontsize=8)
    ax.set_title('How wide is the range the slope was fitted over?\n'
                 'filled = pooled centring (interaction model), '
                 'open = re-centred within stratum (medpos/medneg)', fontsize=9.5)
    ax.grid(axis='x', color='0.9', lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=6.5, loc='lower right')

    ratio = table.drop_duplicates('region').set_index('region')[
        'sd_ratio_unmed_over_med']
    for region, i in idx.items():
        r = ratio.get(region, np.nan)
        if np.isfinite(r):
            ax.text(1.01, i, f'x{r:.2f}', transform=ax.get_yaxis_transform(),
                    va='center', fontsize=6.4,
                    color=('#b03a2e' if r < 0.8 else '0.35'))

    # Panel 3: n. A narrow spread on 30 epochs and on 300 are different problems.
    ax = axes[2]
    for offset, med, colour in ((-0.19, False, UNMED_COLOR), (0.19, True, MED_COLOR)):
        t = table[table['condition'] == ('medicated' if med else 'unmedicated')]
        t = t[t['region'].isin(idx)]
        ax.barh([idx[r] + offset for r in t['region']], t['n_epochs'], height=0.34,
                color=colour)
    ax.set_xlabel('n epochs', fontsize=8)
    ax.set_title('How many epochs?', fontsize=11)

    for a in axes:
        a.tick_params(labelsize=7)

    pooled = table[table['region'] == POOLED_ROW].set_index('condition')
    note = ''
    if {'medicated', 'unmedicated'} <= set(pooled.index):
        note = ('Pooled over all regions the unmedicated epochs have '
                f'NRS mean {pooled.loc["unmedicated", "nrs_mean"]:.2f} and '
                f'within-stratum SD {pooled.loc["unmedicated", "within_stratum_sd"]:.2f}, '
                f'the medicated {pooled.loc["medicated", "nrs_mean"]:.2f} and '
                f'{pooled.loc["medicated", "within_stratum_sd"]:.2f}. ')

    fig.suptitle('Is the unmedicated pain range compressed? NRS_within by region '
                 'and medication state', fontsize=13)
    fig.tight_layout(rect=(0, 0.11, 1, 0.94))
    fig.text(0.01, 0.005,
             'A SLOPE IS ONLY AS TRUSTWORTHY AS THE RANGE IT WAS FITTED OVER. If '
             'one condition\'s NRS_within is visibly narrower than the other\'s, '
             'its slope is estimated over less predictor variation, which both '
             'attenuates it and inflates its standard error -- so a difference '
             'between the two conditions can be produced by unequal range alone, '
             'with no difference in physiology. Ratios in the middle panel are '
             'unmedicated SD / medicated SD under the within-stratum centring; '
             f'below 0.8 is marked red. {note}\n'
             'REGION IS A PROPERTY OF THE ELECTRODE, not the epoch: one pain score '
             'is shared by all of that patient\'s channels, so a region\'s '
             'distribution is over the patients who have a contact there, and '
             'regions differ only through their contributing patients. Rows are '
             'therefore heavily overlapping, not independent. Counts are '
             'epoch-weighted; the models are row-weighted over channel-epochs and '
             'drop QC-masked rows, so a cell\'s effective range can be slightly '
             'narrower than shown. A patient with one epoch in a stratum '
             'contributes exactly 0 to the within-stratum spread, which is real -- '
             'that patient carries no within-stratum information -- but it does '
             'pull the open markers down.\n'
             + DISCLAIMER,
             fontsize=7, va='bottom', ha='left', color='0.35', wrap=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-dir', required=True,
                    help='The med-strata run directory (the one holding '
                         'provenance.json and epoch_med_state.parquet).')
    ap.add_argument('--stratum', default='interaction',
                    help='Subdirectory holding the interaction fit (default '
                         'interaction). Must carry med_ix_* and med_main_* columns.')
    ap.add_argument('--roi-scheme', default='roi_v2')
    ap.add_argument('--view-dir', default=None,
                    help='Per-channel view directory. Defaults to the one the '
                         'builder would resolve for --roi-scheme.')
    ap.add_argument('--outline-significance', action='store_true',
                    help='Draw BH outlines on the three fitted panels. OFF by '
                         'default: this figure is about whether the estimates are '
                         'stable enough to compare, which a threshold does not say.')
    ap.add_argument('--skip-nrs-range', action='store_true',
                    help='Only the slope figure. The NRS figure needs the channel '
                         'metadata for every subject, which is the slow part.')
    ap.add_argument('--suffix', default='')
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    io.warn_if_dirty()

    run_dir = Path(args.run_dir)
    cell_dir = run_dir / args.stratum
    cells = simple_slopes(io.read_table(cell_dir / 'grid_cells.parquet',
                                        on_stale='warn'))

    regions = order_regions(cells,
                            view_tables.roi_regions_for({'roi_scheme': args.roi_scheme}))
    bins = sorted(cells['freq_bin_index'].unique())
    bin_labels = (cells.drop_duplicates('freq_bin_index')
                  .set_index('freq_bin_index')[['freq_bin_low', 'freq_bin_high']]
                  .rename(columns={'freq_bin_low': 'bin_low_hz',
                                   'freq_bin_high': 'bin_high_hz'})
                  .sort_index())
    logger.info('%d cells | %d regions | %d bins', len(cells), len(regions), len(bins))

    slope_path = cell_dir / f'fig_simple_slopes{args.suffix}.png'
    _, stats = fig_simple_slopes(cells, regions, bins, bin_labels, slope_path,
                                 outline_significance=args.outline_significance)
    logger.info('wrote %s', slope_path)
    for row in stats.itertuples():
        logger.info('  %-42s %8.4f', row.statistic, row.value)

    io.write_table(cells[['region', 'freq_bin_index', 'freq_bin_low', 'freq_bin_high',
                          'beta_nrs_within', 'beta_medicated', 'med_ix_beta',
                          'med_main_beta', 'n_subjects', 'n_channels']],
                   cell_dir / f'simple_slopes{args.suffix}.csv',
                   params={'stratum': args.stratum,
                           'beta_medicated': 'beta_nrs_within + med_ix_beta'},
                   parents=[str(cell_dir / 'grid_cells.parquet')],
                   script='ieeg_ehr/analysis/plot_med_simple_slopes.py')
    io.write_table(stats, cell_dir / f'simple_slopes_summary{args.suffix}.csv',
                   params={'stratum': args.stratum},
                   parents=[str(cell_dir / 'grid_cells.parquet')],
                   script='ieeg_ehr/analysis/plot_med_simple_slopes.py')

    if not args.skip_nrs_range:
        d = region_epochs(run_dir, args.roi_scheme, view_dir=args.view_dir)
        present = [r for r in regions if r in set(d['region'])]
        table = nrs_spread_table(d, present)
        nrs_path = run_dir / 'design' / f'fig_nrs_range_by_condition{args.suffix}.png'
        nrs_path.parent.mkdir(parents=True, exist_ok=True)
        fig_nrs_range(d, present, table, nrs_path)
        logger.info('wrote %s', nrs_path)
        io.write_table(table,
                       run_dir / 'design' / f'nrs_range_by_condition{args.suffix}.csv',
                       params={'roi_scheme': args.roi_scheme,
                               'centring': 'pooled over strata and within stratum'},
                       parents=[str(run_dir / 'epoch_med_state.parquet')],
                       script='ieeg_ehr/analysis/plot_med_simple_slopes.py')

    io.log_analysis('medicated vs unmedicated simple slopes with the medication '
                    'offset, plus the per-region NRS_within range by condition -- '
                    'stability diagnostics for the interaction map (EXPLORATORY)',
                    run_dir)


if __name__ == '__main__':
    main()
