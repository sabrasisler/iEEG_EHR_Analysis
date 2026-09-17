"""Sanity checks, effect plots and diagnostics for a band-power mixed-model run.

    python -m ieeg_ehr.analysis.plot_bandpower_checks --run-dir <run> [--figure F]

Six groups, in the order a reader should look at them. Every figure is written
into the run directory it describes, beside `band_cells.parquet`.

    decomposition   Does the within/between split have anything to work with?
                    Per-subject NRS range sorted by subject mean, the
                    distribution of per-subject NRS SD, and mean vs range --
                    with the ELIGIBILITY-EXCLUDED subjects drawn too, because
                    they are the honest version of "the mass near zero".
    within_between  The model's two coefficients, side by side: spaghetti of
                    per-subject lines with the fixed effect over them, and the
                    between-subject scatter with the NRS_submean line. This is
                    where Simpson's paradox becomes visible if it is happening.
    caterpillar     Per-subject slopes sorted, UNPOOLED with their own SEs and
                    the model's BLUPs beside them. The sign-flip null is a
                    statement about this distribution being symmetric about
                    zero, so this is that null's visual form -- and the gap
                    between the two markers is the shrinkage.
    effects         Volcano of beta against -log10 p, coloured by band, with a
                    second x-axis in PERCENT POWER CHANGE per pain point so
                    "significant but trivial" is judgeable rather than asserted.
    residuals       Conditional residuals vs fitted, their QQ plot, and their
                    autocorrelation ACROSS ASSESSMENTS IN REAL TIME within
                    subject. The last one decides whether a free label shuffle
                    is defensible or whether the permutation needs blocking.
    coverage        Where the contacts are. A glass brain and per-region counts,
                    because an effect map is unreadable without knowing that one
                    region rests on 28 electrodes and another on 819.
    permutation     Null histogram for one cell and Wald-vs-permutation p for
                    all of them. Requires `--stage perm` to have run; skipped
                    with a message if it has not.

TWO PLACES WHERE THIS DEPARTS FROM WHAT WAS ASKED FOR, both deliberate:

1. A P-P PLOT OF OBSERVED p-VALUES AGAINST UNIFORM IS NOT A CALIBRATION CHECK
   when real effects are present -- under any true signal the observed p-values
   SHOULD depart from the diagonal, so a departure cannot distinguish "the test
   is miscalibrated" from "the effect is real". It is drawn (it says useful
   things about how much of the family is null) but labelled as a p-value
   DISTRIBUTION. Calibration is the Wald-vs-permutation panel, per cell, which
   compares two p-values for the SAME hypothesis on the same data.
2. THERE IS NO PER-ELECTRODE EFFECT MAP, because this model does not estimate
   one. The fixed effect is per (region, band); contacts enter through a random
   INTERCEPT, which absorbs each electrode's amplitude and says nothing about its
   pain slope. A per-contact slope would need `VC_CHANNEL_SLOPE`, which the
   Phase-1 pilot rejected (improved fit in 1 of 19 cells). `--figure coverage`
   therefore paints each electrode with its REGION's beta and says so on the
   figure; it is an anatomical rendering of a regional number, not a per-contact
   estimate.

EXPLORATORY. Discovery cohort only. Nominations, not findings.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ieeg_ehr import config, io
from ieeg_ehr.analysis import mixed_model as mm
from ieeg_ehr.analysis import view_tables
from ieeg_ehr.analysis.plot_mixed_model_subject_lines import (draw_panel, epoch_level,
                                                              subject_slopes)

logger = logging.getLogger(__name__)

SCRIPT = 'ieeg_ehr/analysis/plot_bandpower_checks.py'

FIGURES = ('decomposition', 'within_between', 'caterpillar', 'effects',
           'residuals', 'coverage', 'permutation')

DISCLAIMER = ('EXPLORATORY -- discovery cohort, NOMINATIONS NOT FINDINGS. '
              'Not confirmed out of sample.')

#: Enough lags to see structure without running off the end of a short subject.
ACF_MAX_LAG = 6


# ============================================================================
# SHARED
# ============================================================================

def load_run(run_dir):
    """Every table the figures need, with the optional ones as None."""
    run_dir = Path(run_dir)
    out = {'run_dir': run_dir,
           'cells': io.read_table(run_dir / 'band_cells.parquet', on_stale='warn')}
    for key, name in (('slopes', 'subject_slopes.parquet'),
                      ('blups', 'blups.parquet'),
                      ('inventory', 'inventory_subjects.parquet'),
                      ('nulls', 'permutation_null.parquet')):
        path = run_dir / name
        out[key] = io.read_table(path, on_stale='ignore') if path.exists() else None
        if out[key] is None:
            logger.info('%s absent; figures needing it will say so', name)
    return out


def run_params(run_dir):
    import json
    try:
        return json.loads((Path(run_dir) / 'provenance.json').read_text()).get(
            'params', {})
    except (OSError, ValueError):
        return {}


def pick_cell(cells, region=None, band=None):
    """The (region, band) to draw per-cell figures for.

    Defaults to the most significant CONVERGED cell, which is the one a reader
    will ask about first; `--region`/`--band` override it. Chosen by p rather
        than by |beta| so a large estimate on two subjects cannot win.
    """
    d = cells[cells['converged'].fillna(False).astype(bool)]
    if region:
        d = d[d['region'] == region]
    if band:
        d = d[d['band'] == band]
    if d.empty:
        raise SystemExit(f'no converged cell matches region={region!r} band={band!r}')
    row = d.loc[d['p'].idxmin()] if not d['p'].isna().all() else d.iloc[0]
    return row


def load_frame(run_dir, cell_index):
    path = Path(run_dir) / 'frames' / f'cell_{int(cell_index):03d}.parquet'
    if not path.exists():
        raise SystemExit(f'no saved frame at {path}; the cell was not fittable')
    return io.read_table(path, on_stale='ignore')


def conditional_parts(res, df):
    """(fitted WITH random effects, conditional residuals).

    `res.fittedvalues` is the FIXED-effects prediction alone, so `res.resid` still
    contains every subject and channel effect. Plotting that against fitted would
    show enormous structure and indict the log transform for something the model
    deliberately absorbs. These are the conditional quantities -- y minus Xb minus
    Zu -- which are the ones a residual diagnostic is about.
    """
    marginal = np.asarray(res.fittedvalues, dtype=float)
    re = np.zeros(len(df), dtype=float)

    subj = df['subject'].to_numpy()
    uid = df['channel_uid'].to_numpy()
    within = df['NRS_within'].to_numpy(dtype=float)

    chan_effect = {}
    for subject, eff in res.random_effects.items():
        m = subj == subject
        if not m.any():
            continue
        re[m] += float(eff.get('subj_int[Intercept]', 0.0))
        re[m] += float(eff.get('subj_slope[NRS_within]', 0.0)) * within[m]
        for key, val in eff.items():
            if key.startswith('channel[C(channel_uid)['):
                chan_effect[key.split('[C(channel_uid)[')[1].rstrip(']')] = float(val)
    if chan_effect:
        re += np.array([chan_effect.get(u, 0.0) for u in uid])

    fitted = marginal + re
    return fitted, df['log10_power'].to_numpy(dtype=float) - fitted


def epoch_times(subjects, epoch_minutes=None):
    """{(subject_id, epoch_id): pain_time} from the unit's epoch_defs.

    Real assessment times, not row order: the ACF is only interpretable if lag 1
    means "the next report", and the median gap in hours is worth printing beside
    it. A subject with two sessions can in principle repeat an epoch_id, so the
    first session's value wins and the count of collisions is logged.
    """
    out, collisions = {}, 0
    for sid in sorted(set(subjects)):
        subject = sid.replace('sub-', '')
        pattern = f'sub-{subject}_ses-*_defs.parquet'
        found = sorted(config.fullres_epoch_defs_path(subject, '01', epoch_minutes)
                       .parent.glob(pattern))
        for path in found:
            defs = io.read_table(path, columns=['epoch_id', 'pain_time'],
                                 on_stale='ignore')
            for e, t in zip(defs['epoch_id'], defs['pain_time']):
                key = (sid, int(e))
                if key in out:
                    collisions += 1
                    continue
                out[key] = t
    if collisions:
        logger.warning('%d (subject, epoch_id) collisions across sessions; the '
                       'first session won. Affects only multi-session subjects.',
                       collisions)
    return out


def _footnote(fig, text, y=0.005, size=6.8):
    fig.text(0.01, y, text, fontsize=size, va='bottom', ha='left', color='0.35',
             wrap=True)


# ============================================================================
# 1. THE DECOMPOSITION
# ============================================================================

def fig_decomposition(data, args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    inv = data['inventory']
    if inv is None:
        logger.warning('no inventory_subjects.parquet; skipping decomposition')
        return None
    inv = inv.copy()
    inv['nrs_range'] = inv['nrs_max'] - inv['nrs_min']
    inv['included'] = inv['included'].fillna(False).astype(bool)
    inc, exc = inv[inv['included']], inv[~inv['included']]

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 0.20 * len(inv) + 3.4),
                             gridspec_kw={'width_ratios': [1.5, 1, 1]})

    # -- panel 1: one strip per subject, sorted by their mean NRS -------------
    ax = axes[0]
    d = inv.sort_values('nrs_mean').reset_index(drop=True)
    for i, r in d.iterrows():
        colour = '#4a7ba7' if r['included'] else '#c9a227'
        ax.plot([r['nrs_min'], r['nrs_max']], [i, i], color=colour, lw=2.0,
                alpha=0.85, solid_capstyle='round', zorder=2)
        ax.plot(r['nrs_mean'], i, 'o', ms=3.4, color='black', zorder=3)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([f"{r.subject_id}  (n={int(r.n_reports)})" for r in
                        d.itertuples()], fontsize=5.5)
    ax.set_xlabel('NRS (0-10)', fontsize=9)
    ax.set_xlim(-0.4, 10.4)
    ax.set_ylim(-1, len(d))
    ax.set_title('Per-subject NRS RANGE, sorted by subject mean\n'
                 f'blue = in the cohort ({len(inc)}), '
                 f'gold = eligibility-excluded ({len(exc)})', fontsize=10)
    ax.grid(axis='x', color='0.9', lw=0.6)
    ax.set_axisbelow(True)

    # -- panel 2: the SD distribution, which is the compact version -----------
    ax = axes[1]
    bins = np.linspace(0, max(3.5, float(np.nanmax(inv['nrs_sd'])) * 1.05), 26)
    ax.hist([inc['nrs_sd'].dropna(), exc['nrs_sd'].dropna()], bins=bins,
            stacked=True, color=['#4a7ba7', '#c9a227'],
            label=[f'in cohort (n={len(inc)})', f'excluded (n={len(exc)})'])
    ax.set_xlabel('within-subject NRS standard deviation', fontsize=9)
    ax.set_ylabel('subjects', fontsize=9)
    ax.set_title('How much within-subject variance exists\n'
                 'mass near zero contributes nothing to beta(within)', fontsize=10)
    ax.legend(fontsize=7.5)

    # -- panel 3: is "reports high" the same subject as "reports variably"? ---
    ax = axes[2]
    for sub, colour, label in ((inc, '#4a7ba7', 'in cohort'),
                               (exc, '#c9a227', 'excluded')):
        ax.scatter(sub['nrs_mean'], sub['nrs_range'], s=26, color=colour,
                   alpha=0.8, edgecolor='white', lw=0.5, label=label)
    if len(inc) > 2:
        r = float(np.corrcoef(inc['nrs_mean'], inc['nrs_range'])[0, 1])
        ax.set_title('Subject mean NRS vs NRS range\n'
                     f'in-cohort r = {r:+.3f}', fontsize=10)
    ax.set_xlabel('subject mean NRS', fontsize=9)
    ax.set_ylabel('subject NRS range (max - min)', fontsize=9)
    ax.legend(fontsize=7.5)

    for ax in axes:
        ax.tick_params(labelsize=7.5)
    fig.suptitle('Sanity check on the within/between decomposition', fontsize=13)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    _footnote(fig,
              'THE EXCLUDED SUBJECTS ARE DRAWN because eligibility has already '
              'removed most of the low-variance mass: a subject needs >10 epochs, '
              'an NRS range >=4 and >=5 non-modal scores to enter at all, so the '
              'near-zero-SD subjects a reader expects to find were filtered '
              'upstream rather than being absent from the data. Panel 3 answers '
              'whether the variable reporters are also the high reporters -- if '
              'they were the same people, beta(within) and beta(submean) would be '
              'estimated on nearly the same contrast and the split would not '
              'separate them.\n' + DISCLAIMER)
    out = data['run_dir'] / 'fig_check_decomposition.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out


# ============================================================================
# 2. WITHIN vs BETWEEN
# ============================================================================

def fig_within_between(data, args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cell = pick_cell(data['cells'], args.region, args.band)
    df = load_frame(data['run_dir'], cell['cell_index'])
    res, _ = mm.fit_cell(df, mm.VC_FULL)

    per_epoch = epoch_level(df)
    slopes = subject_slopes(per_epoch)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.8))

    # -- panel 1: the WITHIN effect, reusing the pilot's panel verbatim -------
    info = draw_panel(axes[0], per_epoch, slopes,
                      {'beta_nrs_within': cell['beta_nrs_within'], 'se': cell['se'],
                       'p': cell['p'], 'region': cell['region'],
                       'freq_bin_low': cell['freq_bin_low'],
                       'freq_bin_high': cell['freq_bin_high'],
                       'freq_bin_index': cell['freq_bin_index'],
                       'group': f"{cell['band']}  WITHIN-subject",
                       'n_subjects': cell['n_subjects'],
                       'n_channels': cell['n_channels']})
    axes[0].set_xlabel('NRS_within  (pain minus that subject\'s own mean)', fontsize=9)
    axes[0].set_ylabel('log10 band power, centred within subject', fontsize=9)

    # -- panel 2: the BETWEEN effect, one point per subject -------------------
    ax = axes[1]
    per_subject = (per_epoch.groupby('subject')
                   .agg(mean_power=('log10_power', 'mean'),
                        mean_nrs=('NRS', 'mean'),
                        n_epochs=('epoch_id', 'nunique')).reset_index())
    beta_b = float(res.fe_params['NRS_submean'])
    se_b = float(res.bse['NRS_submean'])
    p_b = float(res.pvalues['NRS_submean'])

    ax.scatter(per_subject['mean_nrs'], per_subject['mean_power'],
               s=12 + 2.2 * per_subject['n_epochs'], color='#4a7ba7', alpha=0.75,
               edgecolor='white', lw=0.6, zorder=3)
    xs = np.linspace(per_subject['mean_nrs'].min(), per_subject['mean_nrs'].max(), 50)
    # Anchored at the data's centroid: the coefficient is a slope, and the model's
    # intercept is not comparable to these subject means (it also carries the
    # channel and subject random effects).
    x0 = float(per_subject['mean_nrs'].mean())
    y0 = float(per_subject['mean_power'].mean())
    ax.plot(xs, y0 + beta_b * (xs - x0), color='black', lw=2.4, zorder=4)
    ax.fill_between(xs, y0 + (beta_b - 1.96 * se_b) * (xs - x0),
                    y0 + (beta_b + 1.96 * se_b) * (xs - x0),
                    color='0.25', alpha=0.20, lw=0, zorder=2)
    ax.set_xlabel('subject MEAN NRS', fontsize=9)
    ax.set_ylabel('subject mean log10 band power', fontsize=9)
    ax.set_title(f"{cell['region']}  {cell['band']}  BETWEEN-subject", fontsize=8.5)
    ax.text(0.03, 0.97,
            f"beta_submean {beta_b:+.4f}\nWald p {p_b:.2g}\n"
            f"{len(per_subject)} subjects\ndot size = n epochs",
            transform=ax.transAxes, fontsize=6.4, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.28', fc='white', ec='0.8', alpha=0.85))
    ax.tick_params(labelsize=7)

    same = np.sign(cell['beta_nrs_within']) == np.sign(beta_b)
    fig.suptitle(f"Two coefficients, two questions -- {cell['region']} "
                 f"{cell['band']}\nwithin {cell['beta_nrs_within']:+.4f}   vs   "
                 f"between {beta_b:+.4f}   "
                 f"({'same' if same else 'OPPOSITE'} sign)", fontsize=12)
    fig.tight_layout(rect=(0, 0.10, 1, 0.92))
    _footnote(fig,
              'LEFT is the effect of interest: within a patient, does power move '
              'with their pain. One thin line per subject is their own unpooled '
              'OLS fit; the thick line with its band is the model fixed effect, '
              'drawn through the origin because both axes are subject-centred. '
              'RIGHT is the nuisance term: do patients who hurt more on average '
              'have different average power. OPPOSITE SIGNS BETWEEN THE PANELS IS '
              "SIMPSON'S PARADOX and is the reason NRS is split at all -- a single "
              'NRS term would report a blend of the two and be interpretable as '
              'neither. The between panel is descriptive: it plots subject means, '
              'while the coefficient is estimated jointly with everything else, so '
              'the line is anchored at the centroid rather than at the model '
              'intercept.\n' + DISCLAIMER)
    out = data['run_dir'] / 'fig_check_within_between.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info('within/between for %s %s: %s', cell['region'], cell['band'], info)
    return out


# ============================================================================
# 3. HETEROGENEITY
# ============================================================================

def fig_caterpillar(data, args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cell = pick_cell(data['cells'], args.region, args.band)
    if data['slopes'] is None:
        logger.warning('no subject_slopes.parquet; skipping caterpillar')
        return None
    s = data['slopes']
    s = s[(s['region'] == cell['region']) & (s['band'] == cell['band'])].copy()
    s = s.dropna(subset=['slope']).sort_values('slope').reset_index(drop=True)
    if s.empty:
        logger.warning('no per-subject slopes for the chosen cell')
        return None

    beta = float(cell['beta_nrs_within'])
    se = float(cell['se'])
    y = np.arange(len(s))

    blup = None
    if data['blups'] is not None and len(data['blups']):
        b = data['blups']
        b = b[(b['region'] == cell['region']) & (b['band'] == cell['band'])]
        if len(b):
            blup = (s[['subject']].merge(b[['subject', 'subject_slope']],
                                         on='subject', how='left')['subject_slope']
                    .to_numpy(dtype=float))

    fig, ax = plt.subplots(figsize=(8.2, 0.26 * len(s) + 3.2))
    ax.axvspan(beta - 1.96 * se, beta + 1.96 * se, color='0.25', alpha=0.16, lw=0,
               zorder=0)
    ax.axvline(beta, color='black', lw=2.0, zorder=1,
               label=f'fixed effect {beta:+.4f}')
    ax.axvline(0, color='0.5', lw=1.0, ls='--', zorder=1)

    agree = np.sign(s['slope'].to_numpy()) == np.sign(beta)
    ax.errorbar(s['slope'][agree], y[agree], xerr=1.96 * s['se'][agree], fmt='o',
                ms=4.2, lw=1.0, capsize=2, color='#2b6ca3', zorder=3,
                label='unpooled OLS, agrees with the group')
    ax.errorbar(s['slope'][~agree], y[~agree], xerr=1.96 * s['se'][~agree], fmt='o',
                ms=4.2, lw=1.0, capsize=2, color='#b03a2e', zorder=3,
                label='unpooled OLS, opposes the group')
    if blup is not None:
        ax.plot(blup, y, 'D', ms=3.4, color='#54A24B', alpha=0.9, zorder=4,
                label='model BLUP (shrunk)')

    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.subject} (n={int(r.n_epochs)})" for r in s.itertuples()],
                       fontsize=6)
    ax.set_ylim(-1, len(s))
    ax.set_xlabel('slope: d log10 band power per pain point', fontsize=9)
    frac = float(agree.mean())
    ax.set_title(f"Per-subject slopes -- {cell['region']} {cell['band']}\n"
                 f'{int(agree.sum())}/{len(s)} share the group sign ({frac:.0%}); '
                 f"heterogeneity LRT p = {float(cell['p_lrt_mixture']):.2g}",
                 fontsize=10.5)
    ax.legend(fontsize=7, loc='lower right')
    ax.tick_params(labelsize=7)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    _footnote(fig,
              'THIS IS THE VISUAL FORM OF THE SIGN-FLIP NULL: that null asks '
              'whether this distribution is symmetric about zero, so a column of '
              'points sitting mostly to one side is what a small permutation p '
              'means. Error bars are each subject\'s OWN evidence (their epochs '
              'alone, no pooling), which is why they are wide -- they are not the '
              'model\'s uncertainty about that subject. The green diamonds are the '
              'BLUPs; the distance from the circle to the diamond IS the '
              'shrinkage, and BLUP spread systematically understates real '
              'between-subject variation, which is why sign consistency is '
              'computed from the unpooled fits and never from these.\n'
              + DISCLAIMER)
    out = data['run_dir'] / 'fig_check_caterpillar.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out


# ============================================================================
# 4. EFFECT SIZE vs SIGNIFICANCE
# ============================================================================

def fig_effects(data, args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cells = data['cells']
    bands = list(dict.fromkeys(cells['band']))
    cmap = plt.get_cmap('tab10')
    colours = {b: cmap(i % 10) for i, b in enumerate(bands)}

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.0),
                             gridspec_kw={'width_ratios': [1.35, 1]})

    # -- panel 1: volcano ----------------------------------------------------
    ax = axes[0]
    d = cells.dropna(subset=['beta_nrs_within', 'p']).copy()
    d['nlp'] = -np.log10(d['p'].clip(lower=1e-300))
    rej = d['p_bh_reject'].fillna(False).astype(bool)
    for band in bands:
        m = d['band'] == band
        ax.scatter(d.loc[m & ~rej, 'beta_nrs_within'], d.loc[m & ~rej, 'nlp'],
                   s=26, facecolor='none', edgecolor=colours[band], lw=1.1,
                   alpha=0.85)
        ax.scatter(d.loc[m & rej, 'beta_nrs_within'], d.loc[m & rej, 'nlp'],
                   s=52, color=colours[band], alpha=0.9, edgecolor='black', lw=0.5,
                   label=band)
    if rej.any():
        # The BH threshold is a p CUTOFF that depends on rank, so it is not one
        # horizontal line -- the largest p that was rejected is, and that is the
        # honest thing to draw.
        thresh = float(d.loc[rej, 'p'].max())
        ax.axhline(-np.log10(thresh), color='0.4', ls='--', lw=1.0)
        ax.text(0.99, -np.log10(thresh), f'  BH q={args.fdr_q:g} boundary (p={thresh:.3g})',
                transform=ax.get_yaxis_transform(), fontsize=6.8, va='bottom',
                ha='right', color='0.35')
    ax.axvline(0, color='0.8', lw=0.8)
    ax.set_xlabel('beta: d log10 band power per pain point', fontsize=9)
    ax.set_ylabel('-log10 Wald p', fontsize=9)
    ax.set_title(f'Effect size vs significance, {len(d)} cells\n'
                 f'filled = BH-significant ({int(rej.sum())})', fontsize=10.5)
    ax.legend(fontsize=7.5, title='band', title_fontsize=8)
    ax.tick_params(labelsize=7.5)

    # A SECOND AXIS IN PERCENT, so "is this effect trivial" is a question the
    # figure answers rather than one it invites. 10**beta - 1 per pain point.
    sec = ax.secondary_xaxis('top', functions=(lambda b: (10 ** b - 1) * 100,
                                               lambda p: np.log10(1 + p / 100)))
    sec.set_xlabel('percent change in band power per pain point', fontsize=8.5)
    sec.tick_params(labelsize=7)

    # -- panel 2: the p-value distribution -----------------------------------
    ax = axes[1]
    p = np.sort(cells['p'].dropna().to_numpy())
    n = len(p)
    ax.plot(np.arange(1, n + 1) / (n + 1), p, 'o', ms=4, color='#4a7ba7',
            label='Wald p')
    if 'p_perm' in cells.columns and cells['p_perm'].notna().any():
        pp = np.sort(cells['p_perm'].dropna().to_numpy())
        ax.plot(np.arange(1, len(pp) + 1) / (len(pp) + 1), pp, 's', ms=4,
                color='#b03a2e', label='permutation p')
    ax.plot([0, 1], [0, 1], color='0.5', ls='--', lw=1.0, label='uniform')
    ax.set_xlabel('expected quantile if every cell were null', fontsize=9)
    ax.set_ylabel('observed p', fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('p-value DISTRIBUTION (not a calibration check)', fontsize=10.5)
    ax.legend(fontsize=7.5, loc='lower right')
    ax.tick_params(labelsize=7.5)

    fig.suptitle('Band-power effects: magnitude, significance, and the p-value '
                 'distribution', fontsize=12.5)
    fig.tight_layout(rect=(0, 0.11, 1, 0.92))
    _footnote(fig,
              'THE VOLCANO EXISTS TO CATCH "significant but trivial": read the top '
              'axis, where a beta of 0.01 is a 2.3% power change per pain point '
              'and about 26% across the full 0-10 scale. RIGHT PANEL IS NOT A '
              'CALIBRATION PLOT and is labelled accordingly -- with real effects '
              'present the observed p-values SHOULD bend away from the diagonal, '
              'so a departure cannot separate "miscalibrated test" from "true '
              'signal". It is useful for the opposite reading: the part of the '
              'curve that hugs the diagonal is roughly the null portion of the '
              'family. Calibration needs two p-values for the SAME hypothesis, '
              'which is the Wald-vs-permutation panel in fig_check_permutation.\n'
              + DISCLAIMER)
    out = data['run_dir'] / 'fig_check_effects.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out


# ============================================================================
# 5. RESIDUAL DIAGNOSTICS
# ============================================================================

def residual_acf(df, resid, times, max_lag=ACF_MAX_LAG):
    """(per-lag mean autocorrelation, n subjects per lag, median hours per lag).

    ONE VALUE PER (subject, epoch) -- residuals averaged over that epoch's
    channels -- then ordered by REAL ASSESSMENT TIME within subject. Lag 1 is
    therefore "the next pain report", not "the next row", and the median gap in
    hours is returned so the lag axis can be read as time rather than as index.
    """
    d = df.copy()
    d['resid'] = resid
    per_epoch = (d.groupby(['subject', 'epoch_id'], as_index=False)['resid'].mean())
    per_epoch['t'] = [times.get((s, int(e))) for s, e in
                      zip(per_epoch['subject'], per_epoch['epoch_id'])]
    per_epoch = per_epoch.dropna(subset=['t'])

    acf = {lag: [] for lag in range(1, max_lag + 1)}
    gaps = {lag: [] for lag in range(1, max_lag + 1)}
    for _, g in per_epoch.groupby('subject'):
        g = g.sort_values('t')
        r = g['resid'].to_numpy(dtype=float)
        t = pd.to_datetime(g['t']).to_numpy()
        r = r - r.mean()
        denom = float((r ** 2).sum())
        if denom <= 0 or len(r) < 4:
            continue
        for lag in range(1, max_lag + 1):
            if len(r) <= lag + 2:
                continue
            acf[lag].append(float((r[:-lag] * r[lag:]).sum() / denom))
            gaps[lag].append(float(np.median(
                (t[lag:] - t[:-lag]).astype('timedelta64[m]').astype(float)) / 60.0))
    rows = []
    for lag in range(1, max_lag + 1):
        vals = np.array(acf[lag], dtype=float)
        rows.append({'lag': lag, 'n_subjects': len(vals),
                     'acf_mean': float(vals.mean()) if len(vals) else np.nan,
                     'acf_sem': float(vals.std(ddof=1) / np.sqrt(len(vals)))
                                if len(vals) > 1 else np.nan,
                     'median_gap_h': float(np.median(gaps[lag])) if gaps[lag] else np.nan})
    return pd.DataFrame(rows)


def fig_residuals(data, args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy import stats

    cell = pick_cell(data['cells'], args.region, args.band)
    df = load_frame(data['run_dir'], cell['cell_index'])
    res, _ = mm.fit_cell(df, mm.VC_FULL)
    fitted, resid = conditional_parts(res, df)

    params = run_params(data['run_dir'])
    times = epoch_times(df['subject'].unique(), params.get('epoch_minutes'))
    acf = residual_acf(df, resid, times)

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.0))

    # -- residuals vs fitted --------------------------------------------------
    ax = axes[0]
    ax.scatter(fitted, resid, s=2, alpha=0.08, color='0.35', linewidths=0)
    q = pd.qcut(fitted, 20, duplicates='drop')
    binned = pd.DataFrame({'f': fitted, 'r': resid, 'q': q}).groupby('q',
                                                                    observed=True)
    ax.plot(binned['f'].mean(), binned['r'].mean(), 'o-', color='#b03a2e', lw=1.4,
            ms=4, zorder=3, label='binned mean')
    ax.axhline(0, color='0.5', lw=1.0, ls='--')
    ax.set_xlabel('fitted (fixed + random effects)', fontsize=9)
    ax.set_ylabel('conditional residual', fontsize=9)
    ax.set_title('Residuals vs fitted\nwas log10 the right transform?', fontsize=10)
    ax.legend(fontsize=7.5)

    # -- QQ -------------------------------------------------------------------
    ax = axes[1]
    s = (resid - resid.mean()) / resid.std(ddof=1)
    n = len(s)
    step = max(1, n // 20000)                 # 38k points is a smear, not a plot
    theo = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
    ax.plot(theo[::step], np.sort(s)[::step], '.', ms=2.4, color='#4a7ba7')
    lim = float(np.nanmax(np.abs(theo))) * 1.05
    ax.plot([-lim, lim], [-lim, lim], color='0.5', ls='--', lw=1.0)
    ax.set_xlim(-lim, lim)
    ax.set_xlabel('theoretical normal quantile', fontsize=9)
    ax.set_ylabel('standardised residual', fontsize=9)
    ax.set_title(f'Residual QQ\nskew {stats.skew(s):+.2f}, '
                 f'excess kurtosis {stats.kurtosis(s):+.2f}', fontsize=10)

    # -- ACF across assessments ----------------------------------------------
    ax = axes[2]
    ok = acf.dropna(subset=['acf_mean'])
    ax.bar(ok['lag'], ok['acf_mean'], yerr=ok['acf_sem'], width=0.62,
           color='#4a7ba7', capsize=3)
    ax.axhline(0, color='0.5', lw=1.0)
    if len(df['subject'].unique()) > 1:
        # +/-2/sqrt(n) per subject is the usual white-noise band; here the mean
        # over subjects is what is plotted, so the band shrinks by sqrt(n_subj).
        n_sub = int(ok['n_subjects'].max()) if len(ok) else 1
        band = 2.0 / np.sqrt(max(n_sub, 1))
        ax.axhspan(-band, band, color='0.85', alpha=0.6, lw=0, zorder=0)
    ax.set_xticks(ok['lag'])
    ax.set_xticklabels([f"{int(r.lag)}\n{r.median_gap_h:.1f} h" for r in
                        ok.itertuples()], fontsize=7.5)
    ax.set_xlabel('lag in ASSESSMENTS (median gap below)', fontsize=9)
    ax.set_ylabel('mean within-subject residual autocorrelation', fontsize=9)
    ax.set_title('Residual ACF over real time\ndecides whether a free shuffle is '
                 'defensible', fontsize=10)

    for ax in axes:
        ax.tick_params(labelsize=7.5)
    fig.suptitle(f"Residual diagnostics -- {cell['region']} {cell['band']} "
                 f"({int(cell['n_rows'])} rows, {int(cell['n_subjects'])} subjects)",
                 fontsize=12.5)
    fig.tight_layout(rect=(0, 0.12, 1, 0.92))
    _footnote(fig,
              'THESE ARE CONDITIONAL RESIDUALS -- y minus the fixed effects MINUS '
              'the fitted random effects. statsmodels\' own `resid` is marginal '
              '(it still contains every subject and channel effect), and plotting '
              'that would show huge structure and wrongly indict the log '
              'transform for variance the model absorbs on purpose. THE ACF IS THE '
              'CONSEQUENTIAL PANEL: the permutation null relabels epochs freely '
              'within a subject, which assumes no temporal dependence beyond what '
              'the model already carries. Autocorrelation at lag 1-2 above the '
              'grey band means that assumption is wrong and the shuffle should be '
              'BLOCKED (or a temporal term added), because a free shuffle would '
              'then give a null that is too narrow and p-values that are too '
              'small. The lag axis is in assessments with the median real gap '
              'printed beneath, since the reports are irregularly spaced.\n'
              + DISCLAIMER)
    out = data['run_dir'] / 'fig_check_residuals.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    io.write_table(acf, data['run_dir'] / 'residual_acf.csv',
                   params={'region': cell['region'], 'band': cell['band'],
                           'cell_index': int(cell['cell_index']),
                           'residual': 'conditional (y - Xb - Zu)',
                           'ordering': 'real assessment time within subject'},
                   script=SCRIPT)
    return out


# ============================================================================
# 6. COVERAGE
# ============================================================================

def fig_coverage(data, args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from ieeg_ehr.analysis import plot_electrode_locations as pel

    params = run_params(data['run_dir'])
    roi_scheme = params.get('roi_scheme', 'roi_v2_ofc')
    cells = data['cells']
    regions = [r for r in view_tables.roi_regions_for({'roi_scheme': roi_scheme})
               if r in set(cells['region'])]

    # provenance.json `subjects[]` is the ONLY sanctioned answer to "who was in
    # this run" (CLAUDE.md) -- not the folder name, and not whichever subjects
    # happen to appear in a downstream table.
    import json
    try:
        prov = json.loads((data['run_dir'] / 'provenance.json').read_text())
        subjects = sorted(s.replace('sub-', '') for s in (prov.get('subjects') or []))
    except (OSError, ValueError):
        subjects = []
    if not subjects:
        logger.warning('provenance.json carries no subjects[]; cannot build the '
                       'coverage map without knowing whose contacts to load')
        return None

    contacts = pel.load_contacts(subjects, '01', roi_scheme,
                                 params.get('epoch_minutes'))
    contacts = contacts[contacts['region'].isin(regions)]

    band = args.band or pick_cell(cells, args.region, args.band)['band']
    beta = (cells[cells['band'] == band].set_index('region')['beta_nrs_within']
            .reindex(regions))
    cap = float(np.nanmax(np.abs(beta.to_numpy(dtype=float))))
    cmap = plt.get_cmap('RdBu_r')
    colours = {r: matplotlib.colors.to_hex(cmap(0.5 + 0.5 * (v / cap)))
               if np.isfinite(v) else '#cccccc' for r, v in beta.items()}

    out = data['run_dir'] / f'fig_check_coverage_{band}.png'
    pel.plot_glass_brain(
        contacts, regions, colours,
        f'Contact coverage, painted with each REGION\'s {band} beta '
        f'(NOT a per-contact estimate)', out)

    # A second, plain figure for the counts: the effect map is unreadable without
    # them, because a region with 28 electrodes must not look as authoritative as
    # one with 300.
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 0.32 * len(regions) + 2.6),
                             sharey=True)
    per_region = (contacts.groupby('region')
                  .agg(n_contacts=('channel', 'nunique'),
                       n_subjects=('subject_id', 'nunique')).reindex(regions))
    y = np.arange(len(regions))
    for ax, col, label, colour in ((axes[0], 'n_contacts', 'bipolar pairs', '#4a7ba7'),
                                   (axes[1], 'n_subjects', 'subjects', '0.55')):
        vals = per_region[col].to_numpy(dtype=float)
        ax.barh(y, vals, color=colour, height=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels(regions if col == 'n_contacts' else [], fontsize=8)
        ax.set_ylim(len(regions) - 0.5, -0.5)
        ax.set_xlabel(f'n {label}', fontsize=9)
        ax.set_title(f'{label} per region', fontsize=10)
        ax.tick_params(labelsize=7.5)
        span = np.nanmax(vals) if np.isfinite(np.nanmax(vals)) else 1.0
        for i, v in enumerate(vals):
            if np.isfinite(v):
                ax.text(v + 0.02 * span, i, f'{int(v)}', va='center', fontsize=6.4,
                        color='0.3')
    fig.suptitle('Coverage behind every band estimate', fontsize=12)
    fig.tight_layout(rect=(0, 0.10, 1, 0.94))
    _footnote(fig,
              'A REGION\'S ESTIMATE IS ONLY AS GOOD AS ITS COVERAGE, and these '
              'counts vary by more than an order of magnitude, so the same beta '
              'does not carry the same weight across rows. The mixed model is '
              'precision-weighted, so a subject with many contacts in a region '
              'pulls that region harder than a subject with two -- which is a '
              'feature for estimation and a hazard for interpretation. Contacts '
              'are BIPOLAR PAIRS at their midpoint, de-duplicated across runs.\n'
              + DISCLAIMER)
    out2 = data['run_dir'] / 'fig_check_coverage_counts.png'
    fig.savefig(out2, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info('wrote %s and %s', out.name, out2.name)
    return out2


# ============================================================================
# 7. PERMUTATION
# ============================================================================

def fig_permutation(data, args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cells = data['cells']
    if data['nulls'] is None or 'p_perm' not in cells.columns:
        logger.warning(
            'no permutation results in this run. Run:\n'
            '    sbatch --export=ALL,RUN_DIR="%s" sbatch/bandpower_perm_array.sbatch\n'
            '    python -m ieeg_ehr.analysis.run_bandpower_mixed --stage collect '
            '--run-dir "%s"\nthen re-run this figure.',
            data['run_dir'], data['run_dir'])
        return None

    cell = pick_cell(cells, args.region, args.band)
    nulls = data['nulls']
    g = nulls[nulls['cell_index'] == int(cell['cell_index'])]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))

    ax = axes[0]
    vals = g['beta'].dropna().to_numpy(dtype=float)
    ax.hist(vals, bins=45, color='0.72', edgecolor='white', lw=0.4)
    obs = float(cell['beta_nrs_within'])
    ax.axvline(obs, color='#b03a2e', lw=2.4,
               label=f'observed {obs:+.4f}')
    ax.axvline(0, color='0.5', lw=1.0, ls='--')
    n_extreme = int((np.abs(vals) >= abs(obs)).sum())
    ax.set_xlabel('beta under a within-subject NRS shuffle', fontsize=9)
    ax.set_ylabel('permutations', fontsize=9)
    ax.set_title(f"{cell['region']} {cell['band']}: null vs observed\n"
                 f'{n_extreme} of {len(vals)} shuffles at least as extreme '
                 f"(p_perm = {float(cell['p_perm']):.4g})", fontsize=10)
    ax.legend(fontsize=7.5)

    ax = axes[1]
    d = cells.dropna(subset=['p', 'p_perm'])
    lo = max(min(d['p'].min(), d['p_perm'].min()) * 0.5, 1e-6)
    bands = list(dict.fromkeys(d['band']))
    cmap = plt.get_cmap('tab10')
    for i, band in enumerate(bands):
        s = d[d['band'] == band]
        ax.scatter(s['p'], s['p_perm'], s=34, color=cmap(i % 10), alpha=0.85,
                   edgecolor='white', lw=0.5, label=band)
    ax.plot([lo, 1], [lo, 1], color='0.5', ls='--', lw=1.0)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(lo, 1.3)
    ax.set_ylim(lo, 1.3)
    ax.set_xlabel('parametric Wald p', fontsize=9)
    ax.set_ylabel('permutation p', fontsize=9)
    n_anti = int((d['p_perm'] > d['p']).sum())
    ax.set_title('THE calibration panel: two p-values, one hypothesis\n'
                 f'Wald smaller than permutation in {n_anti}/{len(d)} cells',
                 fontsize=10)
    ax.legend(fontsize=7, title='band', title_fontsize=7.5)

    for ax in axes:
        ax.tick_params(labelsize=7.5)
    fig.suptitle('Permutation null and the calibration of the Wald p', fontsize=12.5)
    fig.tight_layout(rect=(0, 0.10, 1, 0.92))
    _footnote(fig,
              'LEFT is one cell, deliberately: a null histogram per cell would be '
              '120 panels nobody reads. RIGHT is the comparison that actually '
              'tests calibration, because both axes are p-values for the SAME '
              'hypothesis on the SAME data -- points BELOW the diagonal mean the '
              'Wald p is smaller than the permutation p, i.e. parametric '
              'inference is ANTICONSERVATIVE there. The permutation p cannot go '
              'below 1/(n_perm+1), so points on the lower edge are censored '
              'rather than calibrated. Both p-values still assume epochs are '
              'exchangeable within a subject -- see the residual ACF panel, which '
              'is what tests that.\n' + DISCLAIMER)
    out = data['run_dir'] / 'fig_check_permutation.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out


# ============================================================================

BUILDERS = {'decomposition': fig_decomposition,
            'within_between': fig_within_between,
            'caterpillar': fig_caterpillar,
            'effects': fig_effects,
            'residuals': fig_residuals,
            'coverage': fig_coverage,
            'permutation': fig_permutation}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--figure', nargs='*', choices=list(FIGURES) + ['all'],
                    default=['all'])
    ap.add_argument('--region', default=None,
                    help='Region for the per-cell figures. Default: the most '
                         'significant converged cell.')
    ap.add_argument('--band', default=None)
    ap.add_argument('--fdr-q', type=float, default=0.05)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    io.warn_if_dirty()

    data = load_run(args.run_dir)
    wanted = FIGURES if 'all' in args.figure else tuple(args.figure)
    written = []
    for name in wanted:
        try:
            out = BUILDERS[name](data, args)
        except Exception as exc:                        # noqa: BLE001
            # One figure failing must not cost the other six. The traceback is
            # logged rather than swallowed.
            logger.exception('figure %r failed: %s', name, exc)
            continue
        if out is not None:
            written.append(out)
            logger.info('wrote %s', out)
    if written:
        io.log_analysis(f'band-power model checks: {len(written)} figure(s) '
                        '(EXPLORATORY)', Path(args.run_dir))
    logger.info('%d/%d figure(s) written', len(written), len(wanted))


if __name__ == '__main__':
    main()
