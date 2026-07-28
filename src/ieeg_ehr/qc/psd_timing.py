"""
Which PSD runs were written by the CURRENT windowing design, and which by the
superseded one — decided once, in the QC layer, and looked up cheaply thereafter.

WHAT A "HOP" IS
---------------
The PSD is computed on sliding windows of the bipolar voltage. Window length sets
frequency resolution; overlap sets how far consecutive windows step. The **hop**
is that step — `window_sec * (1 - overlap_frac)` — i.e. the TIME RESOLUTION of the
stored spectrogram. It reaches the NWB as `DecompositionSeries.rate` (Hz), with
`rate = 1 / hop`: rate 1.0 is one spectrum per second, rate 1/60 one per minute.

WHAT WENT WRONG, AND WHAT IT IS NOT
-----------------------------------
It is NOT a data-quality artifact and the files are NOT mislabeled. Some runs are
correct outputs of a DIFFERENT, SUPERSEDED ALGORITHM, readable straight off the
NWB description:

  old  {"outer_window_sec": 60.0, "inner_segment_sec": 2.0, "overlap_frac": 0.5}
       two-level: a 60 s outer window of ~59 overlapping 2 s inner segments,
       Welch-AVERAGED into one spectrum per minute.
  new  {"window_sec": 2.0, "overlap_frac": 0.5, ...}
       single-level: each 2 s window is its own periodogram, stepped 1 s.

`config/psd_params.py` documents that transition ("superseding an earlier 60s
outer-window design"). The mechanism is an INCOMPLETE REPROCESSING PASS:
sub-247 has both designs on disk, because `run_pipeline_bipolar.py` has no
skip-if-exists and no `--runs` flag, so a partial re-run leaves a mixed tree with
no complaint.

Why the old runs are excluded rather than down-weighted (DECISIONS.md 2026-07-28
+ its correction): the old files store `log(linear-mean of ~59 segments)` while
the new store `log(single 2 s segment)`. An epoch mean is then approximately
`log(arithmetic mean)` versus a geometric mean of per-second values — the AXIS 4
log-vs-linear choice, FROZEN INTO STORAGE where no view can undo it. It is not
simply that the old values are noisier; each is an average of ~59 segments, so
per-value they are less noisy.

WHY DESIGN, NOT HOP, IS THE PRIMARY TEST
----------------------------------------
`classify_design` parses the stored params and names the algorithm.
`hop_sec` is a SYMPTOM of it. Testing the design is more durable: if the project
ever changed `PSD_WINDOW_SEC`/`PSD_OVERLAP_FRAC` so that 60 s became the expected
hop, a hop-only test would start passing stale two-level files. Both are recorded;
`ok` requires both.

WHERE THE CHECK LIVES
---------------------
Derivation happens ONCE, here and in `audit_psd_timing.py`, and lands in
`qc/psd_timing/`. Enforcement is `assert_subject_ok()`, a one-row lookup in that
table — O(1) per subject, zero NWB opens, so analysis code never pays to
re-derive it. Nothing checks per epoch or per view.
"""

import json
import logging
import re

import numpy as np
import pandas as pd

from ieeg_ehr import config, io

logger = logging.getLogger(__name__)

# The hop the CURRENT design produces, derived from config rather than written as a
# literal so the two cannot drift apart.
EXPECTED_HOP_SEC = config.PSD_WINDOW_SEC * (1.0 - config.PSD_OVERLAP_FRAC)
EXPECTED_STARTING_TIME = 0.0

DESIGN_SINGLE_LEVEL = 'single_level'      # current: window_sec + overlap_frac
DESIGN_OUTER_WINDOW = 'outer_window'      # superseded: outer_window_sec + inner_segment_sec
DESIGN_UNKNOWN = 'unknown'                # no parseable params in the description

RUN_TIMING_COLUMNS = ['subject_id', 'session_id', 'run_id', 'starting_time', 'rate',
                      'hop_sec', 'n_time', 'duration_h', 'design', 'params_has_git',
                      'ok', 'reason']


class NonstandardPsdError(RuntimeError):
    """A subject's PSD was not written by the current windowing design.

    Its own type so callers can distinguish "this subject is excluded by decision"
    from "something broke", and so `--nonstandard-hop allow` has one thing to
    catch.
    """


# ---------------------------------------------------------------------------
# Reading one run
# ---------------------------------------------------------------------------

def read_run_timing(nwb_path):
    """(starting_time, rate, n_time, description) for one run -- METADATA ONLY.

    Never touches `decomp.data`: that is what makes a 6236-file sweep affordable
    (~0.3 s/run measured). Moved here from
    features/backfill_epoch_defs_timing._run_timing so the audit and the backfill
    cannot disagree about how timing is read.
    """
    from pynwb import NWBHDF5IO
    with NWBHDF5IO(str(nwb_path), 'r') as handle:
        decomp = handle.read().processing['ecephys']['psd_log_bins']
        return (float(decomp.starting_time), float(decomp.rate),
                int(decomp.data.shape[0]), decomp.description or '')


def _params_from_description(description):
    """The params JSON embedded in the DecompositionSeries description, or {}.

    The description is prose followed by `Params: {...}`. Parsed with a brace-match
    from the first `{` rather than a regex over the whole string, because the blob
    contains nested objects (`git`, `log_bin_edges_hz`) that a naive `\\{.*\\}`
    would mangle.
    """
    if not description:
        return {}
    start = description.find('{')
    if start < 0:
        return {}
    depth = 0
    for i, ch in enumerate(description[start:], start):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(description[start:i + 1])
                except json.JSONDecodeError:
                    return {}
    return {}


def classify_design(description):
    """DESIGN_SINGLE_LEVEL / DESIGN_OUTER_WINDOW / DESIGN_UNKNOWN.

    `outer_window_sec` is checked FIRST: the old blob carries both
    `outer_window_sec` and `inner_segment_sec`, and a future writer might emit both
    keys, in which case the presence of an outer window is the decisive fact.
    """
    params = _params_from_description(description)
    if 'outer_window_sec' in params or 'inner_segment_sec' in params:
        return DESIGN_OUTER_WINDOW
    if 'window_sec' in params:
        return DESIGN_SINGLE_LEVEL
    # Fall back to prose only if there are no params at all -- a description that
    # merely mentions the words should not outvote a parsed blob. Guarded on
    # `description` being a non-empty string: pynwb yields None for an unset
    # description, and re.search(None) raises, which would abort a whole sweep task
    # over one such run.
    if params or not description:
        return DESIGN_UNKNOWN
    if re.search(r'outer[_ ]window', description, re.I):
        return DESIGN_OUTER_WINDOW
    return DESIGN_UNKNOWN


def hop_from_rate(rate):
    """Seconds between consecutive spectra. NaN for a non-positive rate rather
    than a ZeroDivisionError, so one corrupt run cannot abort a cohort sweep."""
    rate = float(rate)
    return 1.0 / rate if rate > 0 else float('nan')


def is_expected_hop(hop_sec):
    return bool(np.isfinite(hop_sec) and np.isclose(hop_sec, EXPECTED_HOP_SEC))


def describe_run(subject, session, run, nwb_path=None):
    """One audit row for one run. Never raises on a bad file -- records why."""
    path = nwb_path or config.bipolar_psd_nwb_path(subject, session, run)
    row = {'subject_id': f'sub-{subject}', 'session_id': f'ses-{session}',
           'run_id': f'run-{run}', 'starting_time': np.nan, 'rate': np.nan,
           'hop_sec': np.nan, 'n_time': -1, 'duration_h': np.nan,
           'design': DESIGN_UNKNOWN, 'params_has_git': False, 'ok': False,
           'reason': ''}
    if not path.exists():
        row['reason'] = 'nwb_missing'
        return row
    try:
        starting_time, rate, n_time, description = read_run_timing(path)
    except Exception as exc:                       # noqa: BLE001 -- see below
        # Deliberately broad: a sweep over 6236 files must not die on one corrupt
        # or truncated NWB. The failure is RECORDED as not-ok with its reason, so
        # it shows up in the re-run list instead of vanishing.
        row['reason'] = f'unreadable: {type(exc).__name__}'
        logger.error('sub-%s ses-%s run-%s: %s (%s)', subject, session, run,
                     type(exc).__name__, exc)
        return row

    hop = hop_from_rate(rate)
    params = _params_from_description(description)
    row.update({'starting_time': starting_time, 'rate': rate, 'hop_sec': hop,
                'n_time': n_time, 'duration_h': (n_time * hop / 3600.0
                                                 if np.isfinite(hop) else np.nan),
                'design': classify_design(description),
                'params_has_git': 'git' in params})

    reasons = []
    if row['design'] != DESIGN_SINGLE_LEVEL:
        reasons.append(f"design={row['design']}")
    if not is_expected_hop(hop):
        reasons.append(f'hop={hop:g}s (expected {EXPECTED_HOP_SEC:g}s)')
    if not np.isclose(starting_time, EXPECTED_STARTING_TIME):
        reasons.append(f'starting_time={starting_time:g}')
    row['reason'] = '; '.join(reasons)
    row['ok'] = not reasons
    return row


# ---------------------------------------------------------------------------
# The audit table + enforcement
# ---------------------------------------------------------------------------

def run_timing_path():
    return config.psd_timing_dir() / 'run_timing.parquet'


def rerun_subjects_path():
    return config.psd_timing_dir() / 'psd_rerun_subjects.txt'


def rerun_runs_path():
    return config.psd_timing_dir() / 'psd_rerun_runs.csv'


def load_run_timing(on_missing='raise'):
    """The audit table, or None when `on_missing='none'`."""
    path = run_timing_path()
    if not path.exists():
        if on_missing == 'none':
            return None
        raise FileNotFoundError(
            f'no PSD timing audit at {path}. Run '
            '`sbatch sbatch/audit_psd_timing.sbatch` (then --reduce) first. '
            'Refusing to guess: without it, a subject written by the superseded '
            '60s outer-window design is indistinguishable from a good one.'
        )
    return io.read_table(path, on_stale='warn')


def subject_status(subject, table=None):
    """(ok, offending_rows) for one subject, from the cached table."""
    table = load_run_timing() if table is None else table
    sid = f'sub-{str(subject).replace("sub-", "")}'
    rows = table[table['subject_id'] == sid]
    if rows.empty:
        return None, rows
    bad = rows[~rows['ok'].astype(bool)]
    return bool(bad.empty), bad


def assert_subject_ok(subject, policy='refuse', table=None):
    """Enforce the exclusion decision. ONE table lookup; zero NWB opens.

    policy:
      'refuse' (default) -- raise NonstandardPsdError if ANY of the subject's runs
                            was not written by the current design. Chosen because a
                            subject with mixed designs cannot be analysed coherently
                            (see the module docstring on Jensen).
      'drop'             -- return the offending run_ids for the caller to exclude.
      'allow'            -- warn only. For deliberate methodological comparison.

    An UNAUDITED subject is treated as a failure under 'refuse'. That is the whole
    point: silence must not read as approval.
    """
    if policy not in ('refuse', 'drop', 'allow'):
        raise ValueError(f'unknown policy {policy!r}')
    table = load_run_timing() if table is None else table
    ok, bad = subject_status(subject, table)

    if ok is None:
        msg = (f'sub-{subject} is absent from the PSD timing audit at '
               f'{run_timing_path()}; its windowing design is unverified.')
        if policy == 'refuse':
            raise NonstandardPsdError(msg + ' Re-run the audit, or pass '
                                            'nonstandard-hop=allow to proceed anyway.')
        logger.warning(msg)
        return []
    if ok:
        return []

    detail = ', '.join(f"{r.run_id} ({r.reason})" for r in bad.head(5).itertuples())
    msg = (f'sub-{subject}: {len(bad)} run(s) not written by the current PSD design '
           f'-- {detail}{" ..." if len(bad) > 5 else ""}. These were EXCLUDED from '
           'analysis by decision (DECISIONS.md 2026-07-28); their PSD needs re-running.')
    if policy == 'refuse':
        raise NonstandardPsdError(msg)
    logger.warning(msg)
    return bad['run_id'].tolist()


def ok_subjects(table=None):
    """Subjects with every run on the current design -- the analysable set."""
    table = load_run_timing() if table is None else table
    grouped = table.groupby('subject_id')['ok'].all()
    return sorted(grouped[grouped].index)


def rerun_subjects(table=None):
    """Subjects with at least one offending run -- the PSD re-run list.

    SUBJECT level on purpose: `run_pipeline_bipolar.py` takes `--subjects` only and
    has no `--runs` flag, so a run-level list is not directly actionable.
    """
    table = load_run_timing() if table is None else table
    grouped = table.groupby('subject_id')['ok'].all()
    return sorted(grouped[~grouped].index)
