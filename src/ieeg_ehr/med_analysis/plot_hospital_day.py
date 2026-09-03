"""
Fig 3 — administration count and dose across the hospitalization.

A 2 x N grid; columns are drugs, rows are two measures that answer different
questions and are deliberately normalized differently:

- **Top row: administration rate**, administrations per 24 RECORDED hours. The
  denominator is recorded iEEG hours for that day, not calendar hours, which
  corrects for partial coverage at the start and end of a stay. A day with no
  recorded time is left as a GAP, never plotted as zero — an undefined rate and
  a rate of zero are different claims.

- **Bottom row: normalized dose**, mean +/- SEM of each patient's daily dose as a
  fraction of their own maximum daily dose, zero-dose days excluded. Normalizing
  to the personal maximum removes between-patient potency and body-size
  differences, and is unit-free, which is what makes it legitimate here at all
  given that combination products are dosed in tablets and fentanyl in mcg.
  Zero-dose days are excluded because this row is about dose MAGNITUDE; the top
  row already covers frequency, and leaving zeros in would make the two rows
  measure the same thing twice.

A companion figure plots each patient's daily dose trajectory as one line per
patient, which is the honest counterpart to the group mean: most patients receive
a variable rather than a constant daily dose, and a mean +/- SEM band hides that.

HOSPITAL DAY 0 is midnight of the session's own `session_start` date — the
calendar day the iEEG session began. That is not the same claim as "day 0 is
admission", and the caption says so. Days 0-6 are shown for comparability with
the source benzodiazepine figures, with contributing subjects annotated per day
so thinning sample size is visible rather than implied.

A monitored day with no dose contributes 0.0; an unmonitored day contributes
nothing at all. Collapsing those two would turn missing data into evidence of no
dosing.

Run on Slurm, never the login node:
    python -m ieeg_ehr.med_analysis.plot_hospital_day
"""

import logging

import numpy as np
import pandas as pd

from ieeg_ehr import config, io
from ieeg_ehr.med_analysis import load, output, recording_hours, style
from ieeg_ehr.med_analysis.style import plt

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_TYPE = 'hospital_day'
SCRIPT = 'ieeg_ehr/med_analysis/plot_hospital_day.py'

DAY_LO, DAY_HI = 0, 6


def admin_rate_by_day(admin_df, drug, subject_hours, days):
    """Administrations per 24 recorded hours, over that drug's subjects only.

    Restricting the denominator to subjects who ever received the drug is what
    makes the rate interpretable: including everyone would measure how commonly
    the drug is prescribed at all, which is Fig 1's question.
    """
    sub = admin_df[admin_df['drug'] == drug]
    subjects = set(sub['subject'])

    n_by_day = sub.groupby('hospital_day').size()
    hours = subject_hours[subject_hours['subject'].isin(subjects)]
    hours_by_day = hours.groupby('hospital_day')['hours'].sum()
    subj_by_day = hours.groupby('hospital_day')['subject'].nunique()

    rows = []
    for day in days:
        h = float(hours_by_day.get(day, 0.0))
        if h <= 0:
            continue                      # undefined, not zero — leave the gap
        rows.append({'drug': drug, 'hospital_day': day,
                     'n_admin': int(n_by_day.get(day, 0)),
                     'recorded_hours': h,
                     'n_subjects_present': int(subj_by_day.get(day, 0)),
                     'admin_per_24h': int(n_by_day.get(day, 0)) / h * 24.0})
    return pd.DataFrame(rows)


def daily_dose_series(admin_df, drug, route, day_hours, exclude_zero_days=True):
    """subject -> {day: mean dose that day}, over monitored days only.

    Dose is the raw `sig` in that formulation's native unit; the caller
    normalizes per subject before pooling, so the unit cancels.
    """
    sub = admin_df[(admin_df['drug'] == drug) & (admin_df['route'] == route)]
    load.assert_single_unit(sub, f'{drug} / {route}')
    subjects = set(sub['subject'])

    dosed = (sub.dropna(subset=['dose'])
             .groupby(['subject', 'hospital_day'])['dose'].mean())

    monitored = day_hours[(day_hours['subject'].isin(subjects))
                          & (day_hours['hours'] > 0)]

    series = {}
    n_unmonitored_dose_days = 0
    for subject, g in monitored.groupby('subject'):
        per_day = {}
        for day in g['hospital_day']:
            val = dosed.get((subject, day))
            if val is not None and not pd.isna(val):
                per_day[day] = float(val)
            elif not exclude_zero_days:
                per_day[day] = 0.0
        series[subject] = per_day

    # Doses recorded on a day with no coverage at all are a data-completeness
    # edge case, surfaced rather than folded in.
    monitored_pairs = set(zip(monitored['subject'], monitored['hospital_day']))
    for (subject, day) in dosed.index:
        if (subject, day) not in monitored_pairs:
            n_unmonitored_dose_days += 1

    return series, n_unmonitored_dose_days


def normalize_by_personal_max(series):
    """Each subject's daily dose as a fraction of their own maximum."""
    out, skipped = {}, []
    for subject, per_day in series.items():
        if not per_day:
            continue
        personal_max = max(per_day.values())
        if personal_max <= 0:
            skipped.append(subject)
            continue
        out[subject] = {d: v / personal_max for d, v in per_day.items()}
    return out, skipped


def mean_sem_by_day(series, days):
    rows = []
    for day in days:
        vals = [per_day[day] for per_day in series.values() if day in per_day]
        if not vals:
            continue
        sem = (np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
        rows.append({'hospital_day': day, 'mean': float(np.mean(vals)),
                     'sem': float(sem), 'n_subjects': len(vals)})
    return pd.DataFrame(rows)


def plot_grid(rate_frames, dose_frames, drugs, out_path, days):
    ncols = len(drugs)
    fig, axes = plt.subplots(2, ncols, squeeze=False,
                             figsize=(4.6 * ncols, 7.6), sharex='col')

    for col, drug in enumerate(drugs):
        # --- top: administration rate -----------------------------------
        ax = axes[0][col]
        rate = rate_frames[drug]
        ax.bar(rate['hospital_day'], rate['admin_per_24h'], width=0.7,
               color=style.BAR_COLOR, zorder=3)
        for r in rate.itertuples():
            ax.annotate(f'{r.n_subjects_present}',
                        (r.hospital_day, r.admin_per_24h),
                        textcoords='offset points', xytext=(0, 3), ha='center',
                        fontsize=style.FOOTNOTE_SIZE, color=style.TEXT_MUTED)
        style.style_axes(ax)
        style.label_axes(ax, None,
                         'Administrations per\n24 recorded hours' if col == 0 else None,
                         f'{drug.title()}')

        # --- bottom: normalized dose ------------------------------------
        ax = axes[1][col]
        dose = dose_frames.get(drug)
        if dose is None or dose.empty:
            ax.text(0.5, 0.5, 'no dose data', transform=ax.transAxes,
                    ha='center', va='center', color=style.TEXT_MUTED,
                    fontsize=style.LEGEND_SIZE)
            style.style_axes(ax)
        else:
            ax.plot(dose['hospital_day'], dose['mean'], color=style.NORM_COLOR,
                    linewidth=2, zorder=3)
            ax.fill_between(dose['hospital_day'], dose['mean'] - dose['sem'],
                            dose['mean'] + dose['sem'], color=style.NORM_COLOR,
                            alpha=0.22, zorder=2)
            for r in dose.itertuples():
                ax.annotate(f'{r.n_subjects}', (r.hospital_day, r.mean),
                            textcoords='offset points', xytext=(0, 5),
                            ha='center', fontsize=style.FOOTNOTE_SIZE,
                            color=style.TEXT_MUTED)
            ax.set_ylim(0, 1.05)
            style.style_axes(ax)
        style.label_axes(
            ax, 'Hospital day',
            'Fraction of personal\nmax daily dose (mean ± SEM)' if col == 0 else None)
        ax.set_xticks(list(days))

    fig.suptitle('Analgesic administration and dose across hospital days',
                 fontsize=style.TITLE_SIZE + 2, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    return style.save(
        fig, out_path,
        footnote=(f'day 0 = calendar day of iEEG session start. Small numbers = '
                  'contributing subjects. Top row denominator is recorded iEEG '
                  'hours; days with no recorded time are omitted, not zeroed. '
                  'Bottom row excludes zero-dose days.'))


def plot_subject_trajectories(dose_series, drugs, out_path, days):
    """One line per patient — the counterpart to the group mean."""
    ncols = len(drugs)
    fig, axes = plt.subplots(1, ncols, squeeze=False,
                             figsize=(4.6 * ncols, 3.8), sharey=True)

    for col, drug in enumerate(drugs):
        ax = axes[0][col]
        series = dose_series.get(drug, {})
        for per_day in series.values():
            pts = sorted((d, v) for d, v in per_day.items() if d in days)
            if len(pts) < 1:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=style.NORM_COLOR, alpha=0.35, linewidth=1,
                    marker='o', markersize=2.5, zorder=3)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(list(days))
        style.style_axes(ax)
        style.label_axes(ax, 'Hospital day',
                         'Fraction of personal\nmax daily dose' if col == 0 else None,
                         f'{drug.title()} ({len(series)} subjects)')

    fig.suptitle('Per-patient daily dose trajectories',
                 fontsize=style.TITLE_SIZE + 2, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    return style.save(fig, out_path,
                      footnote='one line per patient; zero-dose days excluded')


def main():
    parser = output.build_parser(__doc__)
    parser.add_argument('--max-drugs', type=int, default=4,
                        help='columns in the grid')
    parser.add_argument('--day-lo', type=int, default=DAY_LO)
    parser.add_argument('--day-hi', type=int, default=DAY_HI)
    args = parser.parse_args()

    io.warn_if_dirty()

    paths = config.med_admin_files()
    admin = load.load_administrations(paths=paths, subclasses=args.subclasses)

    coverage = recording_hours.session_coverage(admin)
    cov_stats = recording_hours.coverage_report(coverage)
    day_hours = recording_hours.subject_hours_by_day(coverage, admin)

    days = list(range(args.day_lo, args.day_hi + 1))
    counts = admin['drug'].value_counts()
    drugs = [d for d in counts.index if counts[d] >= args.min_admin][:args.max_drugs]
    if not drugs:
        raise SystemExit(f'no drug reaches --min-admin={args.min_admin}')

    rate_frames, dose_frames, dose_series = {}, {}, {}
    n_unmonitored = 0
    for drug in drugs:
        rate_frames[drug] = admin_rate_by_day(admin, drug, day_hours, days)

        # Dose is per (drug, route); use that drug's modal route so the unit is
        # single-valued. Which route it was is recorded in the run provenance.
        modal_route = admin.loc[admin['drug'] == drug, 'route'].mode().iloc[0]
        series, n_unmon = daily_dose_series(admin, drug, modal_route, day_hours)
        n_unmonitored += n_unmon
        normalized, _skipped = normalize_by_personal_max(series)
        dose_series[drug] = normalized
        dose_frames[drug] = mean_sem_by_day(normalized, days)

    run_dir = output.resolve_run_dir(args, OUTPUT_TYPE)
    parents = output.source_parents(paths)
    subjects = sorted(admin['subject'].unique())

    plot_grid(rate_frames, dose_frames, drugs,
              run_dir / 'fig3_hospital_day_grid.png', days)
    plot_subject_trajectories(dose_series, drugs,
                              run_dir / 'fig3b_subject_trajectories.png', days)

    rate_all = pd.concat(rate_frames.values(), ignore_index=True)
    dose_all = pd.concat(
        [f.assign(drug=d) for d, f in dose_frames.items() if not f.empty],
        ignore_index=True) if any(not f.empty for f in dose_frames.values()) \
        else pd.DataFrame(columns=['hospital_day', 'mean', 'sem', 'n_subjects', 'drug'])

    output.write_table(rate_all, run_dir, 'admin_rate_by_day', SCRIPT,
                       params=vars(args), parents=parents, subjects=subjects)
    output.write_table(dose_all, run_dir, 'normalized_dose_by_day', SCRIPT,
                       params=vars(args), parents=parents, subjects=subjects)
    output.write_table(coverage.drop(columns=['intervals']), run_dir,
                       'session_recorded_hours', SCRIPT, params=vars(args),
                       parents=parents, subjects=subjects)
    output.write_table(day_hours, run_dir, 'subject_hours_by_day', SCRIPT,
                       params=vars(args), parents=parents, subjects=subjects)

    output.write_run(
        run_dir, SCRIPT, args, admin, paths,
        extra={
            'drugs_plotted': drugs,
            'day_range': [args.day_lo, args.day_hi],
            'hospital_day_anchor': ('midnight of the session\'s own session_start '
                                    'date; day 0 = calendar day the iEEG session '
                                    'began, NOT necessarily admission'),
            'recorded_hours': cov_stats,
            'recorded_hours_caveat': (
                'The file registry only timestamps runs that have a preprocessed '
                'file, so sessions without one fall back to MAR session span. '
                'Rates are accurate to a few percent, not exact. A raw-NWB span '
                'extraction would fix this; see TASKS.md.'),
            'n_dose_days_without_coverage': int(n_unmonitored),
        },
        description=(f'analgesic administration rate + normalized dose by hospital '
                     f'day, {len(drugs)} drugs, n={len(subjects)}'))


if __name__ == '__main__':
    main()
