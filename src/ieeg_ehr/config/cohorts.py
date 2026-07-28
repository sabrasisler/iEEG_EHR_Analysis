"""
Which subjects an analysis is allowed to touch (P0.2).

THE RULE THIS ENFORCES (CLAUDE.md, Cohorts): the hold-out is UNREACHABLE by
default in exploratory runs, and current discovery subjects are locked as
discovery PERMANENTLY. A subject whose data has influenced any choice can never
serve as a hold-out again -- that is not a policy preference, it is what makes a
hold-out mean anything.

WHY DISCOVERY IS THE 65 AND NOT THE 60
--------------------------------------
Two defensible definitions existed and they differ by five subjects:

  60  subjects with actual legacy analysis output under
      outdated/legacy_65_subjects/cache/ -- i.e. genuinely looked at
  65  cohorts/legacy/subjects_65.txt -- the DOCUMENTED draw: 15 forced from the
      original exploratory list + 50 randomly sampled, seed 20260723, from an
      82-subject mask pool (cohorts/legacy/selection_provenance.json)

The 65 is locked (decided 2026-07-28). `122 138 212 235 259` were drawn into the
cohort but never produced output -- none has an epoch cache -- so on a strict
"has been seen" test they could have stayed hold-out-eligible. They are discovery
anyway, because the cohort was DEFINED by that documented random draw and
selectively keeping back the members that happened to fail processing would make
the discovery set a survivorship-filtered subset of its own sampling frame. The
five are simply unprocessed discovery subjects.

WHY THE REST ARE `unassigned`, NOT `heldout`
-------------------------------------------
The matched hold-out is built OFFLINE, on the PHI side, matching on {pain-range,
sEEG/ECoG, age, sex} -- age is PHI and is not on Sherlock (PLANNING P4). So no
code here may assert hold-out membership. Everything not in discovery is
`unassigned`, and `--split heldout` RAISES rather than quietly returning the
leftovers, which would silently redefine "matched hold-out" as "whatever was left".

Splits gate ANALYSIS, not preprocessing. QC, masks and PSD extraction legitimately
run over every subject on disk; it is views, sweeps, models and figures that must
respect the split.
"""

import argparse
import json
import logging

import pandas as pd

from ieeg_ehr.config.paths import COHORTS_ROOT, FILE_REGISTRY_CSV

logger = logging.getLogger(__name__)

# The locked assignment. DATED in the filename because it is a historical fact --
# a later cohort file must be a NEW file, never an edit of this one, so that any
# artifact citing this name always refers to the same subject set.
DISCOVERY_COHORT_JSON = COHORTS_ROOT / 'discovery-core-2026-07-28.json'

# Source of truth this was derived from (kept for the rebuild path).
LEGACY_65_TXT = COHORTS_ROOT / 'legacy' / 'subjects_65.txt'
LEGACY_SELECTION_JSON = COHORTS_ROOT / 'legacy' / 'selection_provenance.json'

SPLITS = ('discovery', 'unassigned', 'all')

# Selected into the 65 but never analysed -- recorded so the distinction survives
# even though all 65 are discovery. They need an epoch cache before they can enter
# a view; sub-259 needs its PSD re-extracted first (qc/psd_timing/).
SELECTED_NOT_ANALYSED = ('122', '138', '212', '235', '259')


class HoldoutUnavailableError(RuntimeError):
    """`--split heldout` was requested before the matched hold-out exists.

    Its own type so a caller cannot mistake it for "no subjects matched" and
    proceed with an empty list.
    """


def _norm(subject):
    return str(subject).replace('sub-', '').strip()


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def load_discovery_cohort(path=None):
    path = path or DISCOVERY_COHORT_JSON
    if not path.exists():
        raise FileNotFoundError(
            f'no discovery cohort file at {path}. Build it once with '
            '`python -m ieeg_ehr.config.cohorts --build`. Refusing to fall back to '
            '"every subject on disk": that would silently put hold-out-eligible '
            'subjects into an exploratory run, which cannot be undone.'
        )
    with open(path) as fh:
        return json.load(fh)


def discovery_subjects(path=None):
    """The locked discovery set, sorted."""
    return sorted(_norm(s) for s in load_discovery_cohort(path)['discovery'])


def registry_subjects():
    """Every subject in the raw-file registry -- the universe."""
    reg = pd.read_csv(FILE_REGISTRY_CSV, usecols=['sub_id'])
    return sorted({_norm(s) for s in reg['sub_id'].unique()})


def unassigned_subjects(path=None):
    """Registry minus discovery.

    COMPUTED, not stored: ~150 more subjects are still arriving, and a stored list
    would be wrong the moment one lands. Storing only the discovery side means the
    file states the irreversible fact and infers the rest.
    """
    discovery = set(discovery_subjects(path))
    return [s for s in registry_subjects() if s not in discovery]


def subjects_for_split(split='discovery', available=None, path=None):
    """Resolve a split to a subject list.

    `available` optionally intersects with what is actually on disk (e.g. subjects
    with an epoch cache), so a run reports the subjects it can really use rather
    than failing partway through. The intersection is LOGGED, because a shrinking
    denominator is a coverage confound in this dataset and must stay visible.
    """
    if split == 'heldout':
        raise HoldoutUnavailableError(
            'The matched hold-out does not exist yet (PLANNING P4). It is built '
            'OFFLINE on the PHI side, matching on age/sex which are not on '
            'Sherlock, so it cannot be derived here. Non-discovery subjects are '
            '"unassigned", not hold-out -- use --split unassigned if that is what '
            'you mean, and understand it is NOT a matched comparison set.'
        )
    if split not in SPLITS:
        raise ValueError(f'unknown split {split!r}; expected one of {SPLITS} '
                         "(or 'heldout', which raises until P4)")

    if split == 'discovery':
        subjects = discovery_subjects(path)
    elif split == 'unassigned':
        subjects = unassigned_subjects(path)
    else:
        subjects = registry_subjects()

    if available is not None:
        have = {_norm(s) for s in available}
        usable = [s for s in subjects if s in have]
        missing = [s for s in subjects if s not in have]
        if missing:
            logger.warning('split=%s: %d/%d subjects have the required artifact; '
                           '%d unavailable: %s', split, len(usable), len(subjects),
                           len(missing), missing)
        return usable
    return subjects


def subjects_with_epoch_cache(minutes_before=None):
    """Subjects that actually have a pain epoch cache -- the practical `available`
    set for a view run. Read from the tree, so it answers "what can run today"."""
    from ieeg_ehr.config.paths import CACHE_SUBDIR, pain_epoch_unit_dir
    cache_dir = pain_epoch_unit_dir(minutes_before) / CACHE_SUBDIR
    return sorted({p.name.split('_')[0].replace('sub-', '')
                   for p in cache_dir.glob('sub-*_ses-*_epochs.parquet')})


def split_of(subject, path=None):
    """'discovery' or 'unassigned' for one subject."""
    return 'discovery' if _norm(subject) in set(discovery_subjects(path)) else 'unassigned'


def assert_split_allowed(subjects, split='discovery', path=None):
    """Refuse subjects that fall outside the requested split.

    This is the guard for an explicit `--subjects` list: without it, `--split
    discovery` would be advisory and naming a hold-out-eligible subject by hand
    would still work.
    """
    allowed = set(subjects_for_split(split, path=path))
    intruders = sorted({_norm(s) for s in subjects} - allowed)
    if intruders:
        raise ValueError(
            f'subjects {intruders} are not in split={split!r}. They are '
            f'{"/".join(sorted({split_of(s, path) for s in intruders}))}. Looking at '
            'a non-discovery subject during exploration cannot be undone -- pass '
            '--split all deliberately if that is genuinely what you want.'
        )
    return sorted({_norm(s) for s in subjects})


def cohort_provenance(path=None):
    """The cohort's own record, for an analysis run's provenance.json."""
    cohort = load_discovery_cohort(path)
    return {'cohort_file': str(path or DISCOVERY_COHORT_JSON),
            'n_discovery': len(cohort['discovery']),
            'created': cohort.get('created'),
            'source': cohort.get('source')}


# ---------------------------------------------------------------------------
# One-time build
# ---------------------------------------------------------------------------

def build_discovery_cohort_file(path=None, overwrite=False):
    """Write the locked discovery cohort from the legacy draw.

    Refuses to overwrite by default: this file is a historical fact, and silently
    rewriting it would change the meaning of every artifact that cites it.
    """
    from ieeg_ehr import io

    path = path or DISCOVERY_COHORT_JSON
    if path.exists() and not overwrite:
        raise FileExistsError(
            f'{path} already exists. This file is a LOCKED historical record; a '
            'different cohort must be a NEW dated file, not an edit of this one. '
            'Pass overwrite=True only to fix a demonstrable error in it.'
        )

    subjects = sorted({_norm(line) for line in LEGACY_65_TXT.read_text().splitlines()
                       if line.strip()})
    selection = {}
    if LEGACY_SELECTION_JSON.exists():
        with open(LEGACY_SELECTION_JSON) as fh:
            raw = json.load(fh)
        selection = {k: raw[k] for k in
                     ('description', 'mask_label_pool', 'mask_pool_size',
                      'forced_subjects_from_initial_cohort', 'n_forced',
                      'n_randomly_sampled', 'random_seed') if k in raw}

    payload = {
        'schema_version': 1,
        'kind': 'cohort_assignment',
        'name': path.stem,
        'created': io.run_timestamp(),
        'git': io.git_provenance(),
        'discovery': subjects,
        'n_discovery': len(subjects),
        'source': {
            'subjects_txt': str(LEGACY_65_TXT),
            'selection_provenance': str(LEGACY_SELECTION_JSON),
            'selection': selection,
        },
        'selected_not_analysed': list(SELECTED_NOT_ANALYSED),
        'notes': [
            'DISCOVERY IS PERMANENT. These subjects have influenced analytic '
            'choices (or were drawn into the cohort that did), so none may ever '
            'serve as a hold-out.',
            'Locked as the documented 65-subject draw rather than the 60 with '
            'legacy output: withholding the members that happened to fail '
            'processing would make discovery a survivorship-filtered subset of '
            'its own sampling frame.',
            'selected_not_analysed have no epoch cache yet; sub-259 also needs its '
            'PSD re-extracted (qc/psd_timing/) before it can enter a view.',
            'Non-discovery subjects are UNASSIGNED, not hold-out. The matched '
            'hold-out is built offline on the PHI side (PLANNING P4).',
            'Four discovery subjects (093, 154, 159, 240) have no '
            'Desikan_Killiany_anode labels, so region-level analyses have n=61, '
            'not 65. Track that in the coverage denominator.',
        ],
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as fh:
        json.dump(payload, fh, indent=2)
    io.write_sidecar(path, kind='cohort_assignment',
                     script='ieeg_ehr/config/cohorts.py',
                     params={'source_txt': str(LEGACY_65_TXT),
                             'n_discovery': len(subjects)},
                     parents=[str(LEGACY_65_TXT), str(LEGACY_SELECTION_JSON)],
                     subjects=[f'sub-{s}' for s in subjects])
    logger.info('wrote %s (%d discovery subjects)', path, len(subjects))
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--build', action='store_true', help='Write the cohort file once')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--show', action='store_true', help='Print the current assignment')
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    if args.build:
        build_discovery_cohort_file(overwrite=args.overwrite)
    if args.show or not args.build:
        cohort = load_discovery_cohort()
        disc = discovery_subjects()
        unas = unassigned_subjects()
        print(f"cohort file : {DISCOVERY_COHORT_JSON}")
        print(f"created     : {cohort.get('created')}")
        print(f"discovery   : {len(disc)}  {' '.join(disc)}")
        print(f"unassigned  : {len(unas)}  {' '.join(unas[:20])}"
              f"{' ...' if len(unas) > 20 else ''}")
        print(f"selected but never analysed: {' '.join(cohort.get('selected_not_analysed', []))}")


if __name__ == '__main__':
    main()
