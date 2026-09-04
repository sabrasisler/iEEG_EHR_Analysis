"""
Where a med_analysis run writes, and what it records about itself.

The equivalent of `analysis/view_tables.py`'s output half, kept separate because
that one hardcodes `event='pain'` and expects a view directory these scripts do
not have.

Layout (CLAUDE.md, the 5-level scheme):

    analysis/meds/<question>/<output_type>/<run_name>_<timestamp>/

Level 4 (view_scheme) is skipped: there are no views here, so inventing a scheme
folder would be a level that never varies.

TABLES ARE CSV. Under `analysis/` these are small, terminal, and read by eye;
Parquet buys nothing and costs a pyarrow round-trip to open one. `io.write_table`
dispatches on the extension and emits the provenance sidecar either way, so
nothing about the contract changes. The cache, views, and features stay Parquet —
they are large, column-sliced, and dtype-critical.
"""

import argparse
import logging
from datetime import datetime

from ieeg_ehr import config, io
from ieeg_ehr.config import med_taxonomy

logger = logging.getLogger(__name__)

TABLE_SUFFIX = '.csv'


def add_output_arguments(parser):
    """The placement vocabulary every med_analysis script shares."""
    parser.add_argument('--question', default=config.MED_DEFAULT_QUESTION,
                        help='level-2 question folder under analysis/meds/')
    parser.add_argument('--run-name', default=None,
                        help='human label for the run dir; a timestamp is always '
                             'appended')
    parser.add_argument('--scratch', action='store_true',
                        help='write to analysis/scratch/ instead of the meds tree')
    parser.add_argument('--out-root', default=None,
                        help='explicit output root, overriding everything else')
    return parser


def add_cohort_arguments(parser):
    """Drug-set selection, shared by every figure script."""
    parser.add_argument('--subclasses', nargs='+',
                        default=list(med_taxonomy.ANALGESIC_SUBCLASSES),
                        help='Level 2 classes to include (default: the analgesic '
                             'set). Anesthetics are excluded by default because '
                             'the MAR export does not capture procedural meds.')
    parser.add_argument('--min-admin', type=int, default=20,
                        help='drop drugs with fewer administrations than this from '
                             'the per-drug panels')
    parser.add_argument('--drugs', nargs='+', default=None,
                        help='name the drugs to show, as they appear in the MAR '
                             '`drug` column. Overrides --min-admin and the '
                             'per-figure panel cap, so the same set can be held '
                             'fixed across figures; a name absent from the data '
                             'is an error, not a silent drop. Default: the '
                             'most-administered drugs clearing --min-admin.')
    return parser


def resolve_run_dir(args, output_type, timestamp=None):
    """Build and create the run directory. Timestamp is taken once, here."""
    from pathlib import Path

    stamp = timestamp or datetime.now().strftime('%Y%m%d-%H%M%S')
    run_name = args.run_name or output_type
    leaf = f'{run_name}_{stamp}'

    if args.out_root:
        run_dir = Path(args.out_root) / output_type / leaf
    elif args.scratch:
        run_dir = config.PLOTS_ROOT / 'meds' / output_type / leaf
    else:
        run_dir = config.med_run_dir(output_type=output_type, run_name=run_name,
                                     question=args.question, timestamp=stamp)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def source_parents(paths):
    """Provenance parents for a med run: the MAR files plus the file registry.

    `digest=False` for the same reason every other script in this repo does it —
    these are fingerprinted by (path, size, mtime), and digesting 98 CSVs plus a
    multi-megabyte registry on every run buys nothing.
    """
    parents = [io.parent_ref(p, digest=False) for p in paths]
    parents.append(io.parent_ref(config.FILE_REGISTRY_CSV, digest=False))
    parents.append(io.parent_ref(med_taxonomy.TAXONOMY_SOURCE_CSV, digest=False))
    return parents


def write_run(run_dir, script, args, admin_df, paths, extra=None,
              description=None):
    """Provenance + the analysis-log line. Call once, after the figures exist.

    `subjects` comes from the data actually plotted, never from the cohort that
    was requested — the two differ the moment a drug filter removes someone.
    """
    subjects = sorted(admin_df['subject'].unique())
    io.write_run_provenance(
        run_dir,
        script=script,
        params=vars(args),
        parents=source_parents(paths),
        subjects=subjects,
        extra={
            'n_administrations': int(len(admin_df)),
            'n_subjects': len(subjects),
            'n_sessions': int(admin_df.groupby(['subject', 'session']).ngroups),
            'taxonomy_note': (
                'drug classes from config/med_taxonomy.py, seeded from '
                f'{med_taxonomy.TAXONOMY_SOURCE_CSV}'),
            'status': ('EXPLORATORY descriptive characterization, not a finding '
                       '(CLAUDE.md; pending P2.6 FREEZE)'),
            **(extra or {}),
        },
    )
    if description:
        io.log_analysis(description, run_dir)
    logger.info('figures + provenance -> %s', run_dir)
    return run_dir


def write_table(df, run_dir, name, script, params=None, parents=None,
                subjects=None, extra=None):
    """Write one of a run's tables as CSV, with its sidecar. See module docstring."""
    return io.write_table(df, run_dir / f'{name}{TABLE_SUFFIX}', kind='table',
                          script=script, params=params, parents=parents,
                          subjects=subjects, extra=extra)


def build_parser(description):
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_output_arguments(parser)
    add_cohort_arguments(parser)
    return parser
