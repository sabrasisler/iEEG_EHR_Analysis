#!/usr/bin/env python3
"""Is a domain's pain slope something most patients show, or a group average?

    python -m ieeg_ehr.analysis.plot_domain_consistency --run-dir <domain run>

`fig_domain_summary` gives one beta per (domain, band) with a Wald interval.
That is a statement about the GROUP, and a reader cannot tell from it whether
the Sensory delta effect is 40 patients doing the same thing or 12 doing it
hard. These figures answer that, on the same 5 x 6 grid, and they are the
domain-level counterpart of `plot_bandpower_consistency` -- deliberately the
same statistics and the same colour scales, so the domain map and the
region map can be read side by side.

WHY THIS IS A SEPARATE SCRIPT AND NOT A FLAG ON THE FIT
-------------------------------------------------------
It needs per-(subject, domain, band) slopes, which the fit does not save. It
recomputes them from the same view, which costs seconds -- the domain run's own
log records "loaded and aggregated in 5s" -- against 20 minutes to refit. Being
separate also means it runs against ANY existing domain run, including the
parcel-level ones, so two specifications can be compared without re-fitting
either.

WHAT A PER-SUBJECT SLOPE IS HERE
--------------------------------
One ordinary least-squares line per (subject, domain, band), through that
subject's own epochs, with the domain's channels AVERAGED PER EPOCH first
(`epoch_level`). Unpooled, deliberately: the model's BLUPs already exist and are
the wrong tool, because partial pooling drags every subject toward the group and
BLUP-based agreement comes out at 0.8-1.0 almost everywhere and separates
nothing (measured, `compute_subject_slopes_grid` docstring).

`NRS_within` is taken from the FULL frame, before the split by domain, so every
domain's slope is on the same centred predictor the model used.

THE FOUR FIGURES, AND WHY THE OBVIOUS FIFTH ONE IS NOT HERE
------------------------------------------------------------
    sign            ONE panel on the grid: the percentage of subjects sharing
                    the group's sign, annotated in every cell, with COLOUR
                    reserved for the cells that beat their own sign-flip null
                    (BH q=0.05). The null decides what is coloured; it is not
                    explained on the figure. `excess_sign`, `null_frac_mean`,
                    `p_signflip` and `p_signflip_bh` are in the CSV.
    heterogeneity   tau and I-squared: how much the subjects genuinely DIFFER,
                    with sampling noise removed.
    strip           Every subject's slope as a point, per cell. No averaging and
                    no summary statistic at all.
    matrix          Subject x domain, one panel per band. The same data as
                    `strip` arranged so one patient can be followed across
                    domains.

THE OBVIOUS FIFTH IS THE PLAIN STANDARD DEVIATION OF THE SUBJECT SLOPES, and it
is computed and written to the CSV but is NOT given a figure, because on this
data it mostly measures the wrong thing. Per-subject precision varies by close
to an order of magnitude here -- a subject with 11 epochs and one with 50 are
both one point in that SD -- so the raw spread is dominated by how noisily each
subject was measured rather than by how much they truly differ. `tau` is that
same spread with the sampling component subtracted (DerSimonian-Laird), which is
the quantity "how variable is this effect across patients" actually asks for,
and `I2` is the share of the total spread that survives the subtraction.

READ tau AND I2 AS UPPER BOUNDS. Both are driven by Q, which compares the
observed spread against the per-subject SEs, and those SEs come from an OLS fit
that assumes independent epochs. Pain reports are not independent in time, so
the SEs are optimistic, Q is inflated, and tau and I2 come out high. The
COMPARISON between cells is still informative -- every cell is biased the same
way -- but the absolute value is not a calibrated heterogeneity estimate.

EXPLORATORY. Discovery cohort. Nominations, not findings.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ieeg_ehr import io
from ieeg_ehr.analysis import cluster_permutation as cp
from ieeg_ehr.analysis import fullres_cells, mixed_model as mm, reference_run
from ieeg_ehr.analysis.plot_bandpower_consistency import (N_PERM, _signflip_p,
                                                            cell_weights,
                                                            signflip_null)
from ieeg_ehr.analysis.plot_mixed_model_subject_lines import (epoch_level,
                                                              subject_slopes)
from ieeg_ehr.analysis.run_bandpower_mixed import BAND_SETS, band_table, aggregate
from ieeg_ehr.analysis.run_domain_model import (DOMAIN_COLOURS, domain_caveat,
                                                parcel_domain_maps,
                                                roi_domain_maps, unit_word)
from ieeg_ehr.analysis.run_fullres_grid import resolve_cohort
from ieeg_ehr.config import roi_schemes

logger = logging.getLogger(__name__)

SCRIPT = 'ieeg_ehr/analysis/plot_domain_consistency.py'
OUT_SUBDIR = 'consistency'

DISCLAIMER = ('EXPLORATORY -- discovery cohort, NOMINATIONS NOT FINDINGS. '
              'Not confirmed out of sample.')

#: A cell with fewer than this many subjects carrying a fittable slope gets no
#: statistics. Three is the floor `signflip_null` itself imposes; the sign-flip
#: reference is meaningless below it and tau is undefined.
MIN_SUBJECTS = 3


# ============================================================================
# THE PER-SUBJECT SLOPES
# ============================================================================

def run_params(run_dir):
    """The run's own parameters, from its provenance -- never from argparse.

    The whole point of pointing this script at a run directory is that the run
    already recorded what it did. Re-specifying the ROI scheme or the insula
    threshold on the command line is how a consistency map ends up describing a
    different model from the one it is drawn beside.
    """
    prov = json.loads((Path(run_dir) / 'provenance.json').read_text())
    params = prov.get('params', {})
    return params, prov.get('subjects', [])


def per_subject_slopes(run_dir, args):
    """(long slope frame, coverage, domains) recomputed from the run's own view.

    One row per (subject, domain, band) with a fittable slope. Subjects whose
    epochs give no slope in a cell -- fewer than three, or a single distinct
    pain score -- come back with NaN rather than being dropped, so the
    denominator of every percentage is visible.
    """
    params, subjects_recorded = run_params(run_dir)
    unit = params.get('unit', 'parcel')
    scheme = params['roi_scheme']
    logger.info('run used unit=%r scheme=%r band_set=%r', unit, scheme,
                params.get('band_set'))

    ref_run = reference_run.load(args.reference_run)
    epoch_minutes = params.get('epoch_minutes') or ref_run.view_params.get(
        'epoch_minutes')
    view_dir = fullres_cells.resolve_view_dir(
        args.view_dir,
        mask_label=args.mask_label or ref_run.view_params.get('mask_label'),
        max_excluded_frac=ref_run.view_params.get('max_excluded_frac'),
        epoch_minutes=epoch_minutes)

    cohort_scheme = (roi_schemes.domain_scheme(scheme)['base']
                     if unit == 'roi' else scheme)
    paths, _, _, subjects, roi_by_subject, _ = resolve_cohort(
        ref_run, view_dir, cohort='reference', roi_scheme=cohort_scheme,
        insula_threshold=params.get('insula_threshold'))

    if unit == 'roi':
        parcel_of, parcel_to_domain, coverage, _ = roi_domain_maps(
            roi_by_subject, subjects, scheme)
        display = roi_schemes.domain_scheme(scheme)['display']
    else:
        parcel_of, parcel_to_domain, coverage = parcel_domain_maps(
            paths, subjects, scheme)
        display = roi_schemes.roi_regions(scheme)
    domains = [d for d in display if d in set(coverage['domain'])]

    # The cohort has to be the run's, not whatever this resolution produces --
    # if they differ, the map would be drawn over different patients from the
    # model it describes, which is the one thing it must not do.
    recomputed = set(coverage['subject_id'])
    if subjects_recorded and recomputed != set(subjects_recorded):
        only_here = sorted(recomputed - set(subjects_recorded))
        only_run = sorted(set(subjects_recorded) - recomputed)
        raise SystemExit(
            'cohort mismatch against the run being described: '
            f'{len(only_here)} extra {only_here[:5]}, '
            f'{len(only_run)} missing {only_run[:5]}. Refusing rather than '
            'drawing a consistency map over different patients.')

    kept, bands, _ = band_table(params['band_set'], args.notch_half_width_hz,
                                epoch_minutes)
    index, values = fullres_cells.load_all_parcels(
        paths, subjects, parcel_of, list(kept.index),
        epoch_minutes=epoch_minutes)
    band_values, names = aggregate(values, kept, bands)
    del values

    rows = []
    for bi, band in enumerate(names):
        frame = pd.DataFrame({
            'subject_id': index['subject_id'].to_numpy(),
            'session': index['session'].to_numpy(),
            'channel': index['channel'].to_numpy(),
            'epoch_id': index['epoch_id'].to_numpy(),
            'pain_score': index['pain_score'].to_numpy(),
            'value': band_values[:, bi],
            'parcel': index['parcel'].to_numpy()})
        frame['domain'] = frame['parcel'].map(parcel_to_domain)
        # Built over ALL domains at once, exactly as the fit did, so
        # NRS_within is the same centred predictor the model used. Splitting
        # first and centring per domain would give each domain its own x.
        df = mm.build_cell_frame(frame, extra_columns=('parcel', 'domain'))
        for dom in domains:
            sub = df[df['domain'] == dom]
            if sub.empty:
                continue
            # epoch_id is unique only within a subject-SESSION, and two
            # subjects here have two sessions, so it is made unique before the
            # per-epoch average pools two different epochs of one patient.
            sub = sub.assign(epoch_id=sub['epoch_id'].astype(str))
            s = subject_slopes(epoch_level(sub))
            s['domain'] = dom
            s['band'] = band
            s['n_channels'] = sub.groupby('subject')['channel_uid'].nunique() \
                                 .reindex(s['subject']).to_numpy()
            rows.append(s)
        logger.info('%-11s | per-subject slopes for %d domain(s)', band,
                    len(domains))

    slopes = pd.concat(rows, ignore_index=True)
    return slopes, coverage, domains, list(names), params


# ============================================================================
# THE CELL STATISTICS
# ============================================================================

def dersimonian_laird(slope, se):
    """(tau, I2, Q, k) -- between-subject SD with the sampling part removed.

    THE REASON THIS IS HERE RATHER THAN `slope.std()`. The plain standard
    deviation of the per-subject slopes answers "how spread out are these
    estimates", which is not the question. Every estimate carries its own
    measurement error, and here those errors differ by close to an order of
    magnitude between a subject with 11 epochs and one with 50. So the plain SD
    is largely a statement about how noisily each patient was measured.

    Q compares the observed spread against what the per-subject SEs alone
    predict. Whatever is left over after subtracting the expected sampling
    contribution (k-1) is attributed to real between-subject variation:

        tau^2 = max(0, (Q - (k-1)) / C),    C = sum(w) - sum(w^2)/sum(w)
        I2    = max(0, (Q - (k-1)) / Q)

    tau is in the units of the slope (d log10 power per pain point), which is
    what makes it directly comparable to the group beta drawn beside it: tau
    larger than |beta| means the between-patient spread exceeds the average
    effect. I2 is the unitless share.

    BOTH ARE UPPER BOUNDS on this data -- see the module docstring. The OLS SEs
    assume independent epochs, pain reports are autocorrelated, so the SEs are
    optimistic and Q is inflated.
    """
    ok = np.isfinite(slope) & np.isfinite(se) & (se > 0)
    y, s = np.asarray(slope)[ok], np.asarray(se)[ok]
    k = len(y)
    if k < MIN_SUBJECTS:
        return np.nan, np.nan, np.nan, k
    w = 1.0 / s ** 2
    ybar = float((w * y).sum() / w.sum())
    Q = float((w * (y - ybar) ** 2).sum())
    C = float(w.sum() - (w ** 2).sum() / w.sum())
    tau2 = max(0.0, (Q - (k - 1)) / C) if C > 0 else np.nan
    i2 = max(0.0, (Q - (k - 1)) / Q) if Q > 0 else 0.0
    return float(np.sqrt(tau2)), float(i2), Q, k


def cell_stats(slopes, group, domains, bands, n_perm=N_PERM, seed=0, fdr_q=0.05):
    """One row per (domain, band): sign agreement, its null, and heterogeneity."""
    rng = np.random.default_rng(seed)
    beta = group.set_index(['domain', 'band'])['beta']
    rej = (group.assign(r=group.get('p_bh_reject', pd.Series(False, group.index))
                        .fillna(False).astype(bool))
           .set_index(['domain', 'band'])['r'])

    rows = []
    for dom in domains:
        for band in bands:
            sub = slopes[(slopes['domain'] == dom) & (slopes['band'] == band)]
            s = sub['slope'].to_numpy(dtype=float)
            e = sub['se'].to_numpy(dtype=float)
            fit = np.isfinite(s)
            g = float(beta.get((dom, band), np.nan))
            k = int(fit.sum())
            rec = {'domain': dom, 'band': band, 'beta_group': g,
                   'p_bh_reject': bool(rej.get((dom, band), False)),
                   'n_subjects_total': int(len(sub)), 'n_with_slope': k}
            if k >= MIN_SUBJECTS and np.isfinite(g) and g != 0:
                match = int((np.sign(s[fit]) == np.sign(g)).sum())
                frac = match / k
                # `cell_weights`, not a bare 1/se^2: it floors the SE at the
                # cell's 5th percentile so one implausibly precise subject
                # cannot take the whole cell. Shared with the region-level map
                # so the two use identical weighting.
                w = cell_weights(e, s)
                nm, n95, _ = signflip_null(s, w, n_perm=n_perm, rng=rng)
                # How often a random sign assignment reaches THIS much
                # agreement. (hits+1)/(draws+1), so a permutation p is never 0.
                p_sf = _signflip_p(s, w, frac, n_perm, rng)
                tau, i2, Q, _ = dersimonian_laird(s, e)
                rec.update({
                    'n_sign_match': match, 'frac_sign': frac,
                    'null_frac_mean': nm, 'null_frac_p95': n95,
                    'excess_sign': frac - nm if np.isfinite(nm) else np.nan,
                    'beats_null_p95': bool(np.isfinite(n95) and frac > n95),
                    'p_signflip': p_sf,
                    'tau': tau, 'i2': i2, 'Q': Q,
                    # Reported because it is what "std" usually means, and so
                    # the gap to tau is visible. NOT plotted -- see the module
                    # docstring.
                    'sd_raw': float(np.nanstd(s[fit], ddof=1)) if k > 1 else np.nan,
                    'median_slope': float(np.nanmedian(s[fit])),
                    'iqr_slope': float(np.nanpercentile(s[fit], 75)
                                       - np.nanpercentile(s[fit], 25)),
                    'median_se': float(np.nanmedian(e[fit])),
                    'tau_over_abs_beta': (tau / abs(g)
                                          if np.isfinite(tau) and g else np.nan),
                })
            rows.append(rec)

    out = pd.DataFrame(rows)
    # BH over the whole grid, one family: "which cells are more consistent than
    # their own null" is one question asked 30 times. `consistent` is what the
    # figure colours in, and it is the ONLY place significance is decided --
    # deliberately separate from the group fit's `p_bh_reject`, which asks
    # whether the slope differs from zero, not whether patients agree about it.
    out['p_signflip_bh'] = np.nan
    m = out['p_signflip'].notna() if 'p_signflip' in out else pd.Series(False,
                                                                        out.index)
    if m.any():
        _, adj = cp.bh_fdr(out.loc[m, 'p_signflip'].to_numpy(), q=fdr_q)
        out.loc[m, 'p_signflip_bh'] = adj
    out['consistent'] = out['p_signflip_bh'].le(fdr_q).fillna(False)
    logger.info('sign-flip null: %d of %d cells beat it at BH q=%g',
                int(out['consistent'].sum()), len(out), fdr_q)
    return out


# ============================================================================
# FIGURES
# ============================================================================

def _footnote(fig, text):
    fig.text(0.01, 0.005, text, fontsize=6.2, va='bottom', ha='left',
             color='0.35', wrap=True)


def band_tick(band, bands_def):
    lo, hi = bands_def[band]
    return f'{band}\n{lo:g}-{hi:g} Hz'


def fig_sign(cells, domains, bands, bands_def, out, caveat):
    """ONE panel: what percentage of subjects slope the group's way.

    COLOUR MARKS SIGNIFICANCE, THE NUMBER CARRIES THE VALUE. Every cell is
    annotated with its percentage; only the cells that beat their own sign-flip
    null (BH q=0.05 over the grid) are filled. That split is what lets this be
    one uncrowded panel instead of two: the earlier version drew the raw
    fraction beside the excess-over-null because the raw fraction cannot be
    read without knowing where chance sits, and here it does not have to be --
    an uncoloured cell IS the "not above chance" statement.

    The null is still doing all the work, it is just not narrated on the
    figure. What it is and why 50% is the wrong reference is in this module's
    docstring and in `domain_consistency_cells.csv`, which carries
    `null_frac_mean`, `excess_sign`, `p_signflip` and `p_signflip_bh` per cell.
    """
    import matplotlib.pyplot as plt

    def grid(col):
        return (cells.pivot(index='domain', columns='band', values=col)
                .reindex(index=domains, columns=bands))

    frac = grid('frac_sign').to_numpy(dtype=float)
    sig = grid('consistent').fillna(False).astype(bool).to_numpy()
    nn = grid('n_with_slope').to_numpy(dtype=float)

    # Masked so only the significant cells take a colour; the rest render as
    # the colormap's 'bad' value, one flat grey.
    shown = np.ma.masked_where(~sig | ~np.isfinite(frac), frac)
    cm = plt.get_cmap('viridis').copy()
    cm.set_bad('0.92')

    # SCALED TO THE COLOURED CELLS, not to 0-100% or 50-100%. Only the
    # significant cells take a colour and they occupy a narrow band (72-85% in
    # the 2026-09-21 run), so a fixed wide axis spends most of viridis on
    # values that never appear and renders every cell the same teal. Rounded
    # out to 5% steps so the colourbar reads in round numbers, with a minimum
    # 15-point span so a set of near-identical cells is not stretched into a
    # spurious gradient.
    vals = frac[sig & np.isfinite(frac)]
    if vals.size:
        vmin = np.floor(vals.min() * 20) / 20
        vmax = np.ceil(vals.max() * 20) / 20
        if vmax - vmin < 0.15:
            pad = (0.15 - (vmax - vmin)) / 2
            vmin, vmax = max(0.0, vmin - pad), min(1.0, vmax + pad)
    else:
        vmin, vmax = 0.5, 1.0

    fig, ax = plt.subplots(figsize=(1.35 * len(bands) + 3.4,
                                    0.72 * len(domains) + 2.4))
    im = ax.imshow(shown, aspect='auto', cmap=cm, vmin=vmin, vmax=vmax,
                   interpolation='nearest')
    ax.set_xticks(np.arange(-0.5, len(bands), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(domains), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.4)
    ax.tick_params(which='minor', length=0)

    for i in range(len(domains)):
        for j in range(len(bands)):
            f = frac[i, j]
            if not np.isfinite(f) or not nn[i, j]:
                continue
            if sig[i, j]:
                # White on the dark end of viridis, near-black on the light.
                shade = (min(max(f, vmin), vmax) - vmin) / (vmax - vmin)
                colour, weight = ('white' if shade < 0.55 else '0.10'), 'bold'
            else:
                colour, weight = '0.45', 'normal'
            ax.text(j, i, f'{f:.0%}', ha='center', va='center', fontsize=12,
                    color=colour, fontweight=weight)

    ax.set_xticks(range(len(bands)))
    ax.set_xticklabels([band_tick(b, bands_def) for b in bands], fontsize=9)
    ax.set_yticks(range(len(domains)))
    ax.set_yticklabels(domains, fontsize=11)
    for t, d in zip(ax.get_yticklabels(), domains):
        t.set_color(DOMAIN_COLOURS.get(d, '0.2'))
    ax.set_title("Subjects sharing the group's pain-slope sign", fontsize=13,
                 pad=12)
    # PercentFormatter rather than set_yticklabels: the latter fixes labels to
    # whatever ticks exist at call time and warns, then silently mislabels if
    # the locator moves.
    from matplotlib.ticker import PercentFormatter
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025)
    cb.set_label('% of subjects', fontsize=9)
    cb.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _footnote(fig,
              'Coloured = more agreement than a sign-flip null (BH q=0.05); grey = not. '
              'Per-subject slopes are unpooled OLS. n = '
              + ', '.join(f'{d} {int(np.nanmax(nn[i]))}'
                          for i, d in enumerate(domains)) + '. ' + DISCLAIMER)
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info('wrote %s', out.name)


def fig_heterogeneity(cells, domains, bands, bands_def, out, caveat):
    """tau (between-subject SD, sampling noise removed) and I2, on the grid.

    tau is drawn RELATIVE TO THE GROUP EFFECT as well as in its own units,
    because the number that matters is not "how big is tau" but "how big is tau
    against the effect it is the spread of". A tau of 0.02 is enormous beside a
    beta of 0.004 and unremarkable beside a beta of 0.05.
    """
    import matplotlib.pyplot as plt
    from ieeg_ehr.features import common

    def grid(col):
        return (cells.pivot(index='domain', columns='band', values=col)
                .reindex(index=domains, columns=bands))

    tau, i2 = grid('tau'), grid('i2')
    ratio, beta = grid('tau_over_abs_beta'), grid('beta_group')
    sd = grid('sd_raw')
    sig = grid('p_bh_reject').fillna(0).astype(bool).to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 0.55 * len(domains) + 3.8))

    # -- tau, in slope units ------------------------------------------------
    arr = tau.to_numpy(dtype=float)
    cm = plt.get_cmap('magma_r').copy()
    cm.set_bad('0.85')
    im = axes[0].imshow(arr, aspect='auto', cmap=cm, vmin=0,
                        vmax=float(np.nanmax(arr)) or 1e-3,
                        interpolation='nearest')
    common.draw_mask_outline(axes[0], sig)
    axes[0].set_title('tau — between-SUBJECT SD of the pain slope\n'
                      '(sampling error removed; same units as beta)',
                      fontsize=10.5)
    fig.colorbar(im, ax=axes[0], fraction=0.045, pad=0.03,
                 label='tau (d log10 power / pain point)')
    hi = float(np.nanmax(arr)) or 1.0
    for i in range(len(domains)):
        for j in range(len(bands)):
            t, b, r = tau.iat[i, j], beta.iat[i, j], ratio.iat[i, j]
            if not np.isfinite(t):
                continue
            axes[0].text(j, i, f'τ {t:.4f}\nβ {b:+.4f}\nτ/|β| {r:.1f}',
                         ha='center', va='center', fontsize=7,
                         color='white' if t > 0.6 * hi else '0.12')

    # -- I2 -----------------------------------------------------------------
    arr2 = i2.to_numpy(dtype=float)
    cm2 = plt.get_cmap('magma_r').copy()
    cm2.set_bad('0.85')
    im2 = axes[1].imshow(arr2, aspect='auto', cmap=cm2, vmin=0, vmax=1,
                         interpolation='nearest')
    common.draw_mask_outline(axes[1], sig)
    axes[1].set_title('I² — share of the spread that is REAL difference\n'
                      'between patients rather than measurement noise',
                      fontsize=10.5)
    fig.colorbar(im2, ax=axes[1], fraction=0.045, pad=0.03, label='I²')
    for i in range(len(domains)):
        for j in range(len(bands)):
            v = arr2[i, j]
            if not np.isfinite(v):
                continue
            axes[1].text(j, i, f'{v:.0%}\nraw SD {sd.iat[i, j]:.4f}',
                         ha='center', va='center', fontsize=7.5,
                         color='white' if v > 0.6 else '0.12')

    for ax in axes:
        ax.set_xticks(range(len(bands)))
        ax.set_xticklabels([band_tick(b, bands_def) for b in bands], fontsize=8)
        ax.set_yticks(range(len(domains)))
        ax.set_yticklabels(domains, fontsize=10)
        for t, d in zip(ax.get_yticklabels(), domains):
            t.set_color(DOMAIN_COLOURS.get(d, '0.2'))

    fig.suptitle('How much do patients actually DIFFER in each domain and band?',
                 fontsize=13)
    fig.tight_layout(rect=(0, 0.20, 1, 0.94))
    _footnote(fig,
              'THE RAW STANDARD DEVIATION OF THE SUBJECT SLOPES IS PRINTED BUT NOT COLOURED, '
              'on purpose. Per-subject precision varies by close to an order of magnitude here '
              '(a patient with 11 epochs and one with 50 are each one point in that SD), so the '
              'raw spread is mostly a statement about how noisily each patient was measured. '
              'tau is that spread with the expected sampling contribution subtracted '
              '(DerSimonian-Laird: tau² = max(0, (Q-(k-1))/C)), and I² is the share that '
              'survives the subtraction. tau is in the units of beta, so tau/|beta| > 1 means '
              'the spread BETWEEN patients is larger than the average effect itself -- which is '
              'compatible with a real group effect, and is the thing a group-only figure hides. '
              'READ BOTH AS UPPER BOUNDS: Q is computed against OLS SEs that assume independent '
              'epochs, pain reports are autocorrelated in time, so the SEs are optimistic and Q '
              'is inflated. Comparisons BETWEEN cells are still informative because every cell '
              'is biased the same way; the absolute values are not calibrated. Black outlines '
              'are the BH-significant cells of the group fit. ' + caveat + '\n' + DISCLAIMER)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info('wrote %s', out.name)


def fig_strip(slopes, cells, domains, bands, bands_def, out, caveat):
    """Every subject's slope as a point. One panel per band, domains on x.

    NO SUMMARY STATISTIC AT ALL, which is the point: agreement percentages and
    tau are both single numbers standing in for a distribution, and this is the
    distribution. Point SIZE is inverse to the subject's own SE, so the eye
    weights a well-measured patient more than a noisy one -- the correction that
    sign counting, which gives every subject one vote, cannot make.
    """
    import matplotlib.pyplot as plt

    ncol = 3
    nrow = int(np.ceil(len(bands) / ncol))
    fig, axes = plt.subplots(nrow, ncol, squeeze=False,
                             figsize=(4.7 * ncol, 3.5 * nrow))
    rng = np.random.default_rng(0)
    g = cells.set_index(['domain', 'band'])

    lim = float(np.nanpercentile(np.abs(slopes['slope'].to_numpy(dtype=float)),
                                 99)) or 0.05

    for k, band in enumerate(bands):
        ax = axes[k // ncol][k % ncol]
        for x, dom in enumerate(domains):
            sub = slopes[(slopes['domain'] == dom) & (slopes['band'] == band)]
            s = sub['slope'].to_numpy(dtype=float)
            e = sub['se'].to_numpy(dtype=float)
            ok = np.isfinite(s)
            if not ok.any():
                continue
            size = np.full(ok.sum(), 14.0)
            ee = e[ok]
            good = np.isfinite(ee) & (ee > 0)
            if good.any():
                inv = 1.0 / ee[good]
                size[good] = 6 + 34 * (inv / np.nanmax(inv))
            jitter = rng.uniform(-0.17, 0.17, size=ok.sum())
            ax.scatter(x + jitter, np.clip(s[ok], -lim, lim), s=size,
                       alpha=0.55, edgecolor='none',
                       color=DOMAIN_COLOURS.get(dom, '0.4'))
            row = g.loc[(dom, band)] if (dom, band) in g.index else None
            if row is not None and np.isfinite(row['beta_group']):
                sig = bool(row['p_bh_reject'])
                ax.plot([x - 0.32, x + 0.32], [row['beta_group']] * 2,
                        color='k', lw=2.4 if sig else 1.2,
                        ls='-' if sig else '--', zorder=5)
        ax.axhline(0, color='0.5', lw=0.9, ls=':')
        ax.set_xticks(range(len(domains)))
        ax.set_xticklabels(domains, fontsize=8, rotation=20, ha='right')
        for t, d in zip(ax.get_xticklabels(), domains):
            t.set_color(DOMAIN_COLOURS.get(d, '0.2'))
        ax.set_ylim(-lim * 1.08, lim * 1.08)
        ax.set_title(band_tick(band, bands_def).replace('\n', ' '), fontsize=9.5)
        if k % ncol == 0:
            ax.set_ylabel('per-subject slope\n(d log10 power / pain point)',
                          fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=7.5)
    for k in range(len(bands), nrow * ncol):
        axes[k // ncol][k % ncol].axis('off')

    fig.suptitle('Every subject, every domain: the distribution behind each cell',
                 fontsize=13)
    fig.tight_layout(rect=(0, 0.11, 1, 0.95))
    _footnote(fig,
              'One dot = one patient\'s unpooled OLS slope in that domain and band. DOT SIZE IS '
              'INVERSE TO THAT PATIENT\'S OWN SE, so a well-measured subject draws the eye more '
              'than a noisy one -- the weighting that the sign-agreement figure, where every '
              'subject casts one equal vote, cannot apply. The black bar is the GROUP estimate '
              'from the mixed model (solid + thick = BH-significant, dashed = not). Slopes are '
              'clipped to the 99th percentile of |slope| so a single outlier cannot flatten '
              'every panel; the clip affects the drawing only, never the statistics. Horizontal '
              'jitter is cosmetic. ' + caveat + '\n' + DISCLAIMER)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info('wrote %s', out.name)


def fig_matrix(slopes, cells, domains, bands, bands_def, out, caveat):
    """Subject x domain, one panel per band. The same data as `strip`, reordered.

    `strip` shows the spread; this shows WHO. A patient is a row in every panel,
    so a subject who drives several cells at once is a bright horizontal band --
    which no per-cell statistic can show, because each of those cells looks
    ordinary on its own.
    """
    import matplotlib.pyplot as plt

    subjects = sorted(slopes['subject'].unique())
    # Ordered by each subject's mean slope in the band with the strongest group
    # omnibus effect available, so the rows are not in arbitrary ID order.
    key_band = (cells.dropna(subset=['beta_group'])
                .assign(a=lambda d: d['beta_group'].abs())
                .groupby('band')['a'].mean().idxmax())
    order_key = (slopes[slopes['band'] == key_band]
                 .groupby('subject')['slope'].mean()
                 .reindex(subjects).fillna(0.0).sort_values())
    subjects = list(order_key.index)
    pos = {s: i for i, s in enumerate(subjects)}

    cube = np.full((len(subjects), len(domains), len(bands)), np.nan)
    for r in slopes.itertuples():
        if r.domain in domains and r.band in bands and np.isfinite(r.slope):
            cube[pos[r.subject], domains.index(r.domain),
                 bands.index(r.band)] = r.slope

    cap = float(np.nanpercentile(np.abs(cube), 98)) or 0.05
    cm = plt.get_cmap('RdBu_r').copy()
    cm.set_bad('0.88')

    ncol = 3
    nrow = int(np.ceil(len(bands) / ncol))
    fig, axes = plt.subplots(nrow, ncol, squeeze=False,
                             figsize=(3.6 * ncol, 0.17 * len(subjects) * nrow + 2.8))
    g = cells.set_index(['domain', 'band'])
    for k, band in enumerate(bands):
        ax = axes[k // ncol][k % ncol]
        im = ax.imshow(cube[:, :, k], aspect='auto', cmap=cm, vmin=-cap,
                       vmax=cap, interpolation='nearest')
        # The group result as a glyph row on top, not a second colour scale:
        # subject slopes run an order of magnitude larger than the fixed
        # effect, so a shared scale would flatten the panel to white.
        for x, dom in enumerate(domains):
            row = g.loc[(dom, band)] if (dom, band) in g.index else None
            if row is None or not np.isfinite(row['beta_group']):
                continue
            ax.plot(x, -1.0, marker='^' if row['beta_group'] > 0 else 'v',
                    ms=6, color='k' if row['p_bh_reject'] else '0.6',
                    clip_on=False)
        ax.set_ylim(len(subjects) - 0.5, -1.6)
        ax.set_xticks(range(len(domains)))
        ax.set_xticklabels(domains, fontsize=7, rotation=35, ha='right')
        for t, d in zip(ax.get_xticklabels(), domains):
            t.set_color(DOMAIN_COLOURS.get(d, '0.2'))
        ax.set_title(band_tick(band, bands_def).replace('\n', ' '), fontsize=9)
        if k % ncol == 0:
            ax.set_yticks(range(len(subjects)))
            ax.set_yticklabels([s.replace('sub-', '') for s in subjects],
                               fontsize=5)
        else:
            ax.set_yticks([])
    for k in range(len(bands), nrow * ncol):
        axes[k // ncol][k % ncol].axis('off')
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02,
                 label='per-subject slope (clipped at the 98th pct)')

    fig.suptitle('Subject x domain, per band — no averaging anywhere',
                 fontsize=13)
    _footnote(fig,
              f'Rows are patients, ordered by their mean slope in {key_band} (the band with the '
              'largest average group effect), so a patient sits in the same row in every panel '
              'and a subject who drives several cells at once reads as a horizontal band -- '
              'which no per-cell statistic can show, because each of those cells looks ordinary '
              'alone. Grey = that patient has no fittable slope in that cell (fewer than three '
              'epochs, or a single distinct pain score); grey is NOT zero. Triangles along the '
              'top are the group fixed effect (up/down = sign, black = BH-significant, grey = '
              'not) and are drawn as glyphs rather than on the colour scale because subject '
              'slopes run an order of magnitude larger than the fixed effect. ' + caveat
              + '\n' + DISCLAIMER)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info('wrote %s', out.name)


# ============================================================================

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-dir', required=True,
                    help='A domain-model run written by run_domain_model.py. '
                         'Its ROI scheme, unit, band set and insula threshold '
                         'are taken from its provenance, never from flags here.')
    ap.add_argument('--reference-run', default=str(reference_run.CONTPAIN_HEATMAP))
    ap.add_argument('--view-dir', default=None)
    ap.add_argument('--mask-label', default=None)
    ap.add_argument('--notch-half-width-hz', type=float, default=None)
    ap.add_argument('--n-perm', type=int, default=N_PERM)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--fdr-q', type=float, default=0.05,
                    help='BH level for the sign-flip test, which is what '
                         'decides which cells the sign figure colours in. A '
                         'separate family from the group fit\'s own BH: '
                         '"do patients agree" is not "is the slope nonzero".')
    ap.add_argument('--out-subdir', default=OUT_SUBDIR)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    io.warn_if_dirty()

    import matplotlib
    matplotlib.use('Agg')

    run_dir = Path(args.run_dir)
    slopes, coverage, domains, bands, params = per_subject_slopes(run_dir, args)

    group = io.read_table(run_dir / 'domain_slopes.parquet', on_stale='warn')
    group = group[group['term'] == 'pain'] if 'term' in group else group
    if 'beta' not in group.columns and 'beta_pain' in group.columns:
        group = group.rename(columns={'beta_pain': 'beta'})

    cells = cell_stats(slopes, group, domains, bands, n_perm=args.n_perm,
                       seed=args.seed, fdr_q=args.fdr_q)

    logger.info('\n%s', cells[['domain', 'band', 'beta_group', 'n_with_slope',
                               'frac_sign', 'null_frac_mean', 'excess_sign',
                               'p_signflip', 'p_signflip_bh', 'consistent',
                               'tau', 'i2', 'sd_raw']].to_string(index=False))

    out_dir = run_dir / args.out_subdir if args.out_subdir else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    caveat = domain_caveat(params['roi_scheme'])
    bands_def = BAND_SETS[params['band_set']]
    fig_sign(cells, domains, bands, bands_def,
             out_dir / 'fig_domain_consistency_sign.png', caveat)
    fig_heterogeneity(cells, domains, bands, bands_def,
                      out_dir / 'fig_domain_consistency_heterogeneity.png', caveat)
    fig_strip(slopes, cells, domains, bands, bands_def,
              out_dir / 'fig_domain_consistency_strip.png', caveat)
    fig_matrix(slopes, cells, domains, bands, bands_def,
               out_dir / 'fig_domain_consistency_matrix.png', caveat)

    shared = {'n_perm': args.n_perm, 'seed': args.seed, 'fdr_q': args.fdr_q,
              'unit': params.get('unit'), 'roi_scheme': params['roi_scheme'],
              'band_set': params['band_set'],
              'min_subjects': MIN_SUBJECTS,
              'slope': 'unpooled OLS per (subject, domain, band) on the '
                       "domain's per-epoch channel mean; NRS_within centred "
                       'over the FULL frame, as the model did'}
    parents = [str(run_dir / 'domain_slopes.parquet')]
    io.write_table(cells, out_dir / 'domain_consistency_cells.csv',
                   params={**shared, 'unit_of_row': 'domain x band cell'},
                   parents=parents, script=SCRIPT,
                   extra={'status': DISCLAIMER, 'domain_caveat': caveat,
                          'heterogeneity':
                              'tau and I2 are DerSimonian-Laird and are UPPER '
                              'BOUNDS here: Q is computed against OLS SEs that '
                              'assume independent epochs, and pain reports are '
                              'autocorrelated, so the SEs are optimistic. '
                              'sd_raw is the plain SD of the subject slopes and '
                              'is reported for comparison only -- it is '
                              'dominated by per-subject measurement precision, '
                              'which varies by nearly an order of magnitude.'})
    io.write_table(slopes, out_dir / 'domain_subject_slopes.csv',
                   params={**shared, 'unit_of_row': 'subject x domain x band'},
                   parents=parents, script=SCRIPT,
                   subjects=sorted(slopes['subject'].unique()),
                   extra={'status': DISCLAIMER,
                          'reading': 'slope is UNPOOLED -- one OLS line through '
                                     "that patient's own epochs, with no "
                                     'shrinkage toward the group. p is the '
                                     'ordinary t-test on n-2 df and answers a '
                                     'much narrower question than the group p; '
                                     'it is not "this patient responds to pain".'})

    io.log_analysis(
        'domain-model across-subject consistency: sign agreement vs a sign-flip '
        'null, and DerSimonian-Laird tau/I2, per domain x band (EXPLORATORY)',
        run_dir)
    logger.info('consistency -> %s', out_dir)


if __name__ == '__main__':
    main()
