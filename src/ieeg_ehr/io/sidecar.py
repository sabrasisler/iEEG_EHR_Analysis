"""
Sidecars: the provenance + staleness record that travels beside every artifact.

The rule this module exists to make cheap (CLAUDE.md, architecture.md PART 2):
**no artifact is written without a sidecar, and no saved artifact is read back
without checking whether it went stale.** A bare `to_parquet`/`to_csv`/`savefig`
is forbidden precisely because the resulting file cannot answer "what code, what
inputs, what config made you?" six weeks later.

Three shapes of the same envelope, so one reader understands all of them:

| Sidecar | Written by | Describes |
|---|---|---|
| `<artifact>.provenance.json` | `write_sidecar` (via `io.tables`/`io.models`) | one file |
| `<dir>/manifest.json` | `write_manifest` | a cache / base-unit directory |
| `<run_dir>/provenance.json` | `write_run_provenance` | one analysis run |

Envelope keys: `schema_version, kind, artifact, created, script, git, params,
config_hash, parents[], subjects[]` plus whatever `extra` merges in at the top
level.

Two deliberate choices worth knowing before you use this:

1. **Parents are FINGERPRINTED, not content-hashed.** A per-window cache file is
   hundreds of MB to GB; sha256-ing it on every write (and again on every
   staleness check) would cost more than recomputing the view the check is
   guarding. So a parent reference is `(path, bytes, mtime)` plus a real digest
   only for small files. The one thing that IS always digested is a
   `manifest.json` — which is why view staleness is defined against the cache's
   manifest hash rather than against the cache data itself.

2. **Staleness warns by default, refuses on request.** `on_stale='warn'` keeps an
   exploratory session moving; `on_stale='refuse'` is for anything a number comes
   out of. Recomputing is always the safe fallback, because a recomputed view
   cannot be stale — which is why views default to not saving at all.
"""

import datetime
import hashlib
import inspect
import json
from pathlib import Path

from ieeg_ehr._repo import REPO_DIR
from ieeg_ehr.io.provenance import git_provenance, run_timestamp

SCHEMA_VERSION = 1

SIDECAR_SUFFIX = '.provenance.json'
DIR_SIDECAR_NAME = 'provenance.json'
MANIFEST_NAME = 'manifest.json'

# Hashing ceilings. A parent is digested automatically only if it is small
# (manifests, configs, cohort files, epoch-definition tables); above that the
# fingerprint is (bytes, mtime), which is what actually catches "someone rebuilt
# the cache under me". The hard ceiling applies even when a caller asks for a
# digest explicitly, so nobody accidentally sha256s a 500 GB tree one file at a
# time inside an array job.
DIGEST_AUTO_MAX_BYTES = 8 * 1024 * 1024
DIGEST_HARD_MAX_BYTES = 64 * 1024 * 1024


class StaleArtifactError(RuntimeError):
    """A saved artifact's sidecar disagrees with the inputs/config asked for now."""


# ============================================================================
# WHERE A SIDECAR LIVES
# ============================================================================

def sidecar_path(target):
    """The sidecar path for `target`.

    - a directory (or a suffix-less path) -> `<dir>/provenance.json`
    - a file                              -> `<file>.provenance.json`

    The file form APPENDS rather than replaces the suffix. Replacing collapses
    `x.parquet` and `x.csv` onto one sidecar name — exactly the collision you
    hit while converting a table from one format to the other, which this repo
    does deliberately and incrementally ("convert one when you next touch it").
    Appending also matches the `<figure>.png.notes.md` convention already used
    for figure notes on Oak.
    """
    target = Path(target)
    if target.is_dir() or target.suffix == '':
        return target / DIR_SIDECAR_NAME
    return target.with_name(target.name + SIDECAR_SUFFIX)


def _legacy_sidecar_path(target):
    """The pre-`io.sidecar` form: suffix REPLACED, e.g.
    `sub-085_ses-01_epoch_channel_power.csv` ->
    `sub-085_ses-01_epoch_channel_power.provenance.json`.

    Still on disk beside the legacy CSV caches under `outdated/`, so readers
    fall back to it. Nothing writes this form any more.
    """
    target = Path(target)
    if target.suffix == '' or target.is_dir():
        return None
    return target.with_suffix(SIDECAR_SUFFIX)


def find_sidecar(target):
    """The existing sidecar path for `target` (current form first, then legacy),
    or None if the artifact has no sidecar at all."""
    for path in (sidecar_path(target), _legacy_sidecar_path(target)):
        if path is not None and path.exists():
            return path
    return None


def read_sidecar(target):
    """Parse `target`'s sidecar, or None if absent/unparseable.

    A truncated sidecar (a job killed mid-write) reads as None rather than
    raising: the caller's next move is the same either way — treat the artifact
    as unverifiable and recompute.
    """
    path = find_sidecar(target)
    if path is None:
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        print(f'  WARNING: sidecar {path} is unreadable ({exc.__class__.__name__}) — '
              f'treating {Path(target).name} as unverifiable.', flush=True)
        return None


# ============================================================================
# HASHING
# ============================================================================

def config_hash(obj, length=12):
    """Stable short hash of a JSON-able config.

    Canonical JSON (sorted keys, no whitespace) so the same config always hashes
    the same regardless of dict insertion order. This is the ONLY thing in the
    project that gets fingerprinted into a directory name, and only for
    materialized views (CLAUDE.md: runs and plots get a human label + timestamp,
    never a hash).
    """
    blob = json.dumps(obj, sort_keys=True, default=str, separators=(',', ':'))
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()[:length]


def file_digest(path, length=16):
    """sha256 (truncated) of a SMALL file — a manifest, config, or cohort file.

    Raises ValueError above DIGEST_HARD_MAX_BYTES. That is a guardrail, not a
    limitation: staleness of a big cache is tracked via its manifest's digest
    plus (bytes, mtime) on the data files, because hashing the data would cost
    more than the recompute the check exists to avoid.
    """
    path = Path(path)
    size = path.stat().st_size
    if size > DIGEST_HARD_MAX_BYTES:
        raise ValueError(
            f'refusing to digest {path} ({size / 1e6:.0f} MB > '
            f'{DIGEST_HARD_MAX_BYTES / 1e6:.0f} MB). Digest its manifest.json instead — '
            f'see io.sidecar module docstring.')
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:length]


def _iso_mtime(stat_result):
    return (datetime.datetime.fromtimestamp(stat_result.st_mtime)
            .astimezone().isoformat(timespec='seconds'))


def parent_ref(path, digest=None, kind=None):
    """A reference to an input artifact, cheap enough to record on every write.

    `digest`: None = automatic (small files only), True = force (still bounded by
    DIGEST_HARD_MAX_BYTES), False = never.

    Also carries a summary of the parent's OWN sidecar when it has one, so a
    chain of artifacts is traceable one hop further without opening every file
    in the chain by hand.
    """
    path = Path(path)
    ref = {'path': str(path), 'exists': path.exists()}
    if kind is not None:
        ref['kind'] = kind
    if not ref['exists']:
        return ref

    st = path.stat()
    if path.is_file():
        ref['bytes'] = st.st_size
    ref['mtime'] = _iso_mtime(st)

    want_digest = (path.is_file() and st.st_size <= DIGEST_AUTO_MAX_BYTES
                   if digest is None else bool(digest) and path.is_file())
    if want_digest:
        try:
            ref['digest'] = file_digest(path)
        except (ValueError, OSError):
            pass    # fingerprint alone; not worth failing a write over

    prov = read_sidecar(path)
    if prov is not None:
        ref['provenance'] = {k: prov.get(k) for k in
                             ('kind', 'created', 'script', 'config_hash')}
        ref['provenance']['commit'] = (prov.get('git') or {}).get('commit')
    return ref


def manifest_ref(target):
    """A parent reference to a cache/base-unit directory's `manifest.json`.

    Accepts either the directory or the manifest path. Always digested (a
    manifest is small by construction) — this digest IS the "cache manifest
    hash" the view-staleness rule is written against.
    """
    path = Path(target)
    if path.is_dir() or path.suffix == '':
        path = path / MANIFEST_NAME
    return parent_ref(path, digest=True, kind='manifest')


def _normalize_parents(parents):
    """Accept paths, parent_ref dicts, or a mix; always return a list of dicts."""
    if parents is None:
        return []
    if isinstance(parents, (str, Path, dict)):
        parents = [parents]
    out = []
    for item in parents:
        out.append(item if isinstance(item, dict) else parent_ref(item))
    return out


# ============================================================================
# WRITING
# ============================================================================

def _caller_script():
    """Repo-relative path of the first frame outside this package, for the
    `script` field — so the lazy call (`write_sidecar(path, params=...)`) still
    records who wrote the artifact. An explicit `script=` always wins.
    """
    io_dir = Path(__file__).parent
    for frame in inspect.stack()[1:]:
        filename = Path(frame.filename)
        if filename.parent == io_dir:
            continue
        try:
            return str(filename.relative_to(REPO_DIR))
        except ValueError:
            return str(filename)
    return None


def sidecar_envelope(artifact, *, kind='artifact', script=None, params=None,
                     parents=None, subjects=None, extra=None):
    """Build the sidecar dict without writing it (used by the writers below, and
    directly by writers that embed provenance INTO their artifact — e.g. the
    NWB DecompositionSeries description field)."""
    params = {} if params is None else params
    envelope = {
        'schema_version': SCHEMA_VERSION,
        'kind': kind,
        'artifact': Path(artifact).name,
        'created': run_timestamp(),
        'script': script if script is not None else _caller_script(),
        'git': git_provenance(),
        'params': params,
        'config_hash': config_hash(params),
        'parents': _normalize_parents(parents),
        'subjects': sorted(subjects) if subjects is not None else None,
    }
    if extra:
        envelope.update(extra)
    return envelope


def write_sidecar(target, *, kind='artifact', script=None, params=None,
                  parents=None, subjects=None, extra=None):
    """Write the sidecar for an artifact that has just been written.

    Call it in the same breath as the artifact write — `io.tables.write_table`
    and `io.models.save_model` already do, which is why new code should go
    through those rather than calling this by hand.

    Args:
        target: the artifact (file) or run/cache directory the sidecar describes.
        kind: 'table' | 'manifest' | 'view' | 'run' | 'model' | 'figure' | 'artifact'.
        params: the config that produced it. Hashed into `config_hash`, which is
            what staleness compares against — so put everything that changes the
            output in here, and nothing that doesn't (no timestamps, no paths
            that vary per array task).
        parents: input artifact paths (or `parent_ref`/`manifest_ref` dicts).
        subjects: the resolved cohort for a run. Read from here, never from a
            folder name.
        extra: merged at the TOP level of the envelope — for run-specific blocks
            like `counts` or `inputs`.

    Returns the sidecar path.
    """
    path = sidecar_path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = sidecar_envelope(target, kind=kind, script=script, params=params,
                               parents=parents, subjects=subjects, extra=extra)
    path.write_text(json.dumps(envelope, indent=2, default=str))
    return path


def write_manifest(target_dir, *, params, parents=None, subjects=None,
                   script=None, extra=None):
    """Write `<target_dir>/manifest.json` for a cache / base-unit directory.

    The manifest is the base unit's self-description — window length, anchor,
    mask label, bin edges, dtype, git, date — and the anchor for every
    downstream staleness check, because it is small enough to digest cheaply
    (see `manifest_ref`). One per `features/pain/psd_epochs/<epoch>_<mask>/`,
    not one per subject.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / MANIFEST_NAME
    envelope = sidecar_envelope(path, kind='manifest', script=script, params=params,
                               parents=parents, subjects=subjects, extra=extra)
    path.write_text(json.dumps(envelope, indent=2, default=str))
    return path


def read_manifest(target):
    """Parse a base unit's `manifest.json` (accepts the directory or the file),
    or None if it isn't there."""
    path = Path(target)
    if path.is_dir() or path.suffix == '':
        path = path / MANIFEST_NAME
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def write_view_sidecar(save_path, *, view_config, cache_manifest, script=None, extra=None):
    """The staleness sidecar for a MATERIALIZED view (architecture.md PART 2).

    Records the view config (hashed), the view code's git commit, the date, and
    a digest of the cache manifest the view was computed from — the four things
    `check_view_fresh` compares on load.

    Materialize a view only when recompute is *measured* slow AND something
    depends on it; the default is to recompute, because a recomputed view cannot
    go stale.
    """
    return write_sidecar(save_path, kind='view', script=script, params=view_config,
                         parents=[manifest_ref(cache_manifest)], extra=extra)


def write_run_provenance(run_dir, *, script=None, params=None, parents=None,
                         subjects=None, extra=None):
    """Write `<run_dir>/provenance.json` for one analysis run.

    `subjects` is the resolved cohort — the only sanctioned answer to "which
    subjects were in this run?" (never the folder name). `params` is normally
    `vars(args)`.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    return write_sidecar(run_dir, kind='run', script=script, params=params,
                         parents=parents, subjects=subjects, extra=extra)


# ============================================================================
# READING / STALENESS
# ============================================================================

def _parent_reasons(recorded, expected):
    """Compare one expected parent reference against what the sidecar recorded."""
    by_path = {ref.get('path'): ref for ref in recorded}
    reasons = []
    for want in expected:
        path = want.get('path')
        have = by_path.get(path)
        if have is None:
            reasons.append(f'input not recorded in the sidecar: {path}')
            continue
        if want.get('exists') and not have.get('exists'):
            reasons.append(f'input did not exist when this was written: {path}')
            continue
        # Digest wins when both sides have one; otherwise (bytes, mtime).
        if 'digest' in want and 'digest' in have:
            if want['digest'] != have['digest']:
                reasons.append(f'input changed (content digest differs): {path}')
            continue
        for field, label in (('bytes', 'size'), ('mtime', 'mtime')):
            if field in want and field in have and want[field] != have[field]:
                reasons.append(f'input changed ({label} differs): {path}')
    return reasons


def check_stale(target, *, parents=None, config=None, check_commit=False,
                allow_dirty=True):
    """Return a list of reasons `target` is stale — empty list means fresh.

    Compares only the sidecar against what the caller wants NOW; it never reads
    the artifact, so it is cheap enough to call on every load.

    Args:
        parents: the inputs this artifact SHOULD have been built from.
        config: the config it SHOULD have been built with (compared by hash).
        check_commit: also require the current git commit to match the recorded
            one. True for materialized views (the view code defines the numbers);
            usually False for a data table, which is not wrong just because the
            repo moved on.
        allow_dirty: when False, a sidecar recorded from a dirty tree is itself a
            staleness reason — the commit hash does not describe what ran.
    """
    sc = read_sidecar(target)
    if sc is None:
        return [f'no sidecar beside {Path(target).name} — provenance unverifiable']

    reasons = []
    if config is not None and sc.get('config_hash') != config_hash(config):
        reasons.append('config differs from the one recorded in the sidecar')
    reasons += _parent_reasons(sc.get('parents') or [], _normalize_parents(parents))

    git = sc.get('git') or {}
    if check_commit:
        current = git_provenance()
        if current.get('available') and git.get('commit') != current.get('commit'):
            reasons.append(f"written at commit {str(git.get('commit'))[:12]}, "
                           f"now at {str(current.get('commit'))[:12]}")
    if not allow_dirty and git.get('dirty'):
        reasons.append('written from a dirty working tree — the recorded commit '
                       'does not describe the code that ran')
    return reasons


def assert_fresh(target, *, parents=None, config=None, check_commit=False,
                 allow_dirty=True, on_stale='warn'):
    """Check staleness and act on it. Returns True if fresh, False if stale and
    tolerated; raises StaleArtifactError when `on_stale='refuse'`.

    `on_stale`: 'warn' (default — keeps exploration moving), 'refuse' (for
    anything a reported number comes out of), 'ignore'.
    """
    reasons = check_stale(target, parents=parents, config=config,
                          check_commit=check_commit, allow_dirty=allow_dirty)
    if not reasons:
        return True
    detail = '\n'.join(f'    - {r}' for r in reasons)
    message = f'STALE: {target}\n{detail}\n    Recompute instead of trusting this file.'
    if on_stale == 'refuse':
        raise StaleArtifactError(message)
    if on_stale != 'ignore':
        print(f'  WARNING: {message}', flush=True)
    return False


def check_view_fresh(save_path, *, view_config, cache_manifest, on_stale='warn'):
    """The load-side half of `write_view_sidecar`: same four comparisons, and
    `check_commit=True` because a materialized view's numbers are defined by the
    view code that produced them."""
    return assert_fresh(save_path, parents=[manifest_ref(cache_manifest)],
                        config=view_config, check_commit=True, on_stale=on_stale)
