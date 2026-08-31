"""
Provenance: what code actually produced an artifact.

Every stored artifact writes git commit (+dirty flag), timestamp, parent
artifact reference, and subjects[] for runs. Commit AND push before any
definitive/array run so the recorded hash matches the code that ran.
"""

import datetime
import subprocess
import sys

from ieeg_ehr._repo import REPO_DIR


def run_timestamp():
    """Local date+time a script ran, ISO-8601 with tz offset — recorded in every
    sidecar (run_info/*.json, params.json) so you can tell when outputs were made."""
    return datetime.datetime.now().astimezone().isoformat()


def git_provenance():
    """
    Record what code actually ran: commit hash + whether the working tree is
    dirty + the list of modified files. A bare hash is misleading when the tree
    has uncommitted changes, so callers should warn when `dirty` is True and the
    recommended workflow is to commit+push before a definitive run.

    Uses cwd=REPO_DIR rather than `git -C` — the compute nodes' system git
    (/usr/bin/git) is old and lacks -C. If git is unavailable/fails, returns
    available=False rather than silently reporting a clean tree.
    """
    def _git(*args):
        try:
            r = subprocess.run(['git', *args], cwd=str(REPO_DIR),
                               capture_output=True, text=True)
        except FileNotFoundError:
            return None
        # rstrip (not strip): `git status --porcelain`'s leading space is a
        # significant part of the XY status code (e.g. " M" = unstaged
        # modification) -- stripping it ate one character off the FIRST
        # modified file's path whenever that file's status began with a
        # space, e.g. "M preprocessing/x.py" -> line[3:] == "reprocessing/x.py".
        # Found via a real provenance JSON showing a truncated filename.
        return r.stdout.rstrip('\n') if r.returncode == 0 else None

    commit = _git('rev-parse', 'HEAD')
    if commit is None:
        return {'available': False, 'commit': None, 'dirty': None, 'modified_files': []}
    porcelain = _git('status', '--porcelain') or ''
    modified = [line[3:] for line in porcelain.splitlines()] if porcelain else []
    return {'available': True, 'commit': commit, 'dirty': bool(modified),
            'modified_files': modified}


def warn_if_dirty(prov=None):
    """Print a loud warning (and return it) if the recorded code state is dirty.

    On STDERR, not stdout. Several scripts publish a machine-readable path as their
    only stdout output so a wrapper can do RUN_DIR=$(python -m ...); a warning on
    stdout silently corrupts that capture, and the failure surfaces far downstream
    as an unusable path rather than as a warning anyone reads.
    """
    prov = prov if prov is not None else git_provenance()
    if not prov.get('available'):
        print("  WARNING: could not read git provenance (git unavailable here) — "
              "commit hash NOT recorded. Capture it at submission time on the login node.",
              file=sys.stderr, flush=True)
    elif prov['dirty']:
        print(f"  WARNING: git working tree is DIRTY ({len(prov['modified_files'])} modified "
              f"files) — recorded commit {prov['commit']} does NOT reflect what ran. "
              f"Commit + push before a definitive run for faithful provenance.",
              file=sys.stderr, flush=True)
    return prov
