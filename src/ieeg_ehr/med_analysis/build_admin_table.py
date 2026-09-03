"""
Materialize the tidy administration table.

One row per administration, with drug, subclass, route, dose, unit, timestamp,
subject, and hospital day. Every figure in this package is a thin rendering layer
over this table; this script exists so it can be looked at directly.

It is NOT a standing artifact. Building it reads 98 small CSVs and takes seconds,
which puts it on the "cheap transform of stored data" side of the decision rule
in CLAUDE.md — recompute at load, do not save by default. Each figure run already
writes the slice it plotted into its own run directory. Run this when you want to
eyeball the whole thing or hand it to someone.

    python -m ieeg_ehr.med_analysis.build_admin_table            # analgesics
    python -m ieeg_ehr.med_analysis.build_admin_table --all-drugs
"""

import logging

from ieeg_ehr import config, io
from ieeg_ehr.config import med_taxonomy
from ieeg_ehr.med_analysis import load, output, recording_hours

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_TYPE = 'admin_table'
SCRIPT = 'ieeg_ehr/med_analysis/build_admin_table.py'


def main():
    parser = output.build_parser(__doc__)
    parser.add_argument('--all-drugs', action='store_true',
                        help='every classified drug, not just the analgesic set')
    parser.add_argument('--with-hours', action='store_true',
                        help='also write per-session and per-day recorded hours')
    args = parser.parse_args()

    io.warn_if_dirty()

    paths = config.med_admin_files()
    subclasses = None if args.all_drugs else args.subclasses
    admin = load.load_administrations(paths=paths, subclasses=subclasses)

    admin = admin.assign(
        coadmin_class=[med_taxonomy.coadmin_class(l1, l2)
                       for l1, l2 in zip(admin['level1'], admin['level2'])],
        formulation=[load.formulation_label(d, r)
                     for d, r in zip(admin['drug'], admin['route'])],
    )

    run_dir = output.resolve_run_dir(args, OUTPUT_TYPE)
    parents = output.source_parents(paths)
    subjects = sorted(admin['subject'].unique())

    output.write_table(admin, run_dir, 'administrations', SCRIPT,
                       params=vars(args), parents=parents, subjects=subjects)

    extra = {'all_drugs': bool(args.all_drugs)}
    if args.with_hours:
        coverage = recording_hours.session_coverage(admin)
        extra['recorded_hours'] = recording_hours.coverage_report(coverage)
        output.write_table(coverage.drop(columns=['intervals']), run_dir,
                           'session_recorded_hours', SCRIPT, params=vars(args),
                           parents=parents, subjects=subjects)
        output.write_table(
            recording_hours.subject_hours_by_day(coverage, admin), run_dir,
            'subject_hours_by_day', SCRIPT, params=vars(args), parents=parents,
            subjects=subjects)

    logger.info('%d administrations, %d drugs, %d subjects',
                len(admin), admin['drug'].nunique(), len(subjects))

    output.write_run(
        run_dir, SCRIPT, args, admin, paths, extra=extra,
        description=(f'tidy medication administration table, {len(admin)} rows, '
                     f'{admin["drug"].nunique()} drugs, n={len(subjects)}'))


if __name__ == '__main__':
    main()
