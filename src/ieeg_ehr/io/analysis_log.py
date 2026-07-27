"""
The terse machine index of everything that has been run.

`log_analysis()` appends ONE line per analysis run to `docs/analyses_run.md`.
That file is the raw "everything I've run" index — greppable, no interpretation.
The narrative written on top of it lives in `docs/labnotebook/YYYY-MM-DD.md`
(via `/lognote`), and `/lognote` reads this index to offer the list of runs that
don't have a notebook entry yet.

This extends the existing sidecar-writing pattern: the same place a script
already emits provenance/staleness sidecars gains one more responsibility, so
the cost is one call in code you were touching anyway.

CODE/DATA BOUNDARY (see CLAUDE.md): this is the one sanctioned write INTO the
repo, and it is deliberate — `docs/analyses_run.md` is a small tracked text
index, not data. It records *paths* to Oak outputs, never their contents. Keep
descriptions deidentified (anonymized subject IDs only); this file is committed
to a GitHub remote. PHI-side filtering lives in `phi_manifest.py`, outside this
repo, and nothing reachable from here should need it.

Logging is AUXILIARY: every failure mode is swallowed with a warning. A six-hour
array job must never die because an index append failed.
"""

import fcntl
import re
from pathlib import Path

from ieeg_ehr._repo import REPO_DIR
from ieeg_ehr.io.provenance import git_provenance, run_timestamp

ANALYSES_RUN_MD = REPO_DIR / 'docs' / 'analyses_run.md'

# One line per run. The trailing box is the "has a notebook entry yet?" flag,
# flipped to [x] by /lognote. It is a CACHE, not the truth -- the truth is
# whether any docs/labnotebook/*.md mentions the output path, which is what
# /lognote actually greps, so hand-written notebook entries still count.
_LINE_RE = re.compile(
    r'^- (?P<when>\S+ \S+) \| (?P<desc>.*?) \| (?P<path>.*?) \| (?P<hash>.*?) \| \[(?P<logged>.)\]\s*$'
)

_HEADER = """# analyses_run.md — the terse index of every run

Machine-appended by `log_analysis()` (`src/ieeg_ehr/io/analysis_log.py`). One
line per analysis run. **Append-only. Do not hand-edit** except to flip a
`[ ]` to `[x]`, which is what `/lognote` does.

No interpretation lives here — this is the bare "what have I run" index. The
narrative goes in `docs/labnotebook/YYYY-MM-DD.md`.

Format:

```
- YYYY-MM-DD HH:MM | <one-sentence description> | <output_path> | <git_hash> | [logged?]
```

`$DERIV/` abbreviates the Oak derivatives base. `+dirty` on a hash means the
working tree had uncommitted changes, so that commit does NOT describe what ran.
`[x]` means a `docs/labnotebook/` entry references this run.

---

"""


def _deriv_base():
    """The Oak derivatives base, or None if config isn't importable.

    Imported lazily: `config.paths` doesn't import `io`, so there is no cycle
    today, but this keeps the index helper usable from a context where the
    config module can't load.
    """
    try:
        from ieeg_ehr.config.paths import DERIVATIVES_BASE
        return Path(DERIVATIVES_BASE)
    except Exception:
        return None


def _shorten(output_path):
    """Abbreviate an Oak derivatives path to `$DERIV/...` so lines stay readable.

    Everything else is recorded absolute — an ambiguous relative path in a
    permanent index is worse than a long one.
    """
    path = Path(output_path)
    path = path.resolve() if path.is_absolute() else path
    base = _deriv_base()
    if base is not None:
        try:
            return f'$DERIV/{path.relative_to(base)}'
        except ValueError:
            pass
    return str(path)


def _clean(text):
    """Flatten a description to one pipe-free line.

    `|` is the field separator and a newline would end the record, so both are
    replaced rather than escaped — the index is meant to be read by eye and by
    `grep`, and neither benefits from an escaping scheme.
    """
    return ' '.join(str(text).replace('|', '/').split()) or '(no description)'


def _short_hash(git_hash):
    if git_hash is not None:
        return _clean(git_hash)
    prov = git_provenance()
    if not prov.get('available') or not prov.get('commit'):
        return 'no-git'
    short = prov['commit'][:12]
    return f'{short}+dirty' if prov.get('dirty') else short


def read_entries(path=None):
    """Parse `docs/analyses_run.md` into dicts. Used by /lognote."""
    path = Path(path) if path is not None else ANALYSES_RUN_MD
    if not path.exists():
        return []
    entries = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        m = _LINE_RE.match(line)
        if m:
            entry = m.groupdict()
            entry['lineno'] = lineno
            entry['logged'] = entry['logged'].lower() == 'x'
            entries.append(entry)
    return entries


def log_analysis(description, output_path, *, git_hash=None, path=None):
    """
    Append one line to `docs/analyses_run.md` recording that an analysis ran.

    Call once per script, next to the provenance sidecar write::

        log_analysis('bipolar re-reference, discovery cohort', out_dir)

    Args:
        description: one human-readable sentence, deidentified.
        output_path: where the figures/tables landed (dir or file). Pass the
            RUN directory, not a per-subject file — see idempotency.
        git_hash: defaults to the current commit (12 chars, `+dirty` suffixed
            when the tree is dirty, `no-git` when git is unavailable).
        path: override the index file. For tests.

    Idempotency: dedupes on (output_path, git_hash). Re-running the same script
    at the same commit does not double-log; re-running at a NEW commit is a new
    line, because it is genuinely a different run. This is also what makes the
    helper safe to call from every task of a Slurm array — all tasks share the
    run directory and commit, so the array collapses to the single line it
    should be, and whichever task arrives first writes it.

    Returns the line written, or None if it deduped or failed.
    """
    target = Path(path) if path is not None else ANALYSES_RUN_MD
    try:
        when = run_timestamp()[:16].replace('T', ' ')   # YYYY-MM-DD HH:MM
        desc = _clean(description)
        out = _clean(_shorten(output_path))
        ghash = _short_hash(git_hash)
        line = f'- {when} | {desc} | {out} | {ghash} | [ ]'

        target.parent.mkdir(parents=True, exist_ok=True)

        # Serialise concurrent array tasks on a lock file beside the index.
        # flock over NFS is best-effort, not a guarantee; the dedupe check below
        # is what actually keeps the file clean, and a rare duplicate line is a
        # cosmetic problem in an index, not a correctness one.
        lock_path = target.with_name(target.name + '.lock')
        with open(lock_path, 'a') as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            except OSError:
                pass

            if not target.exists():
                target.write_text(_HEADER)

            for entry in read_entries(target):
                if entry['path'] == out and entry['hash'] == ghash:
                    return None

            with open(target, 'a') as fh:
                fh.write(line + '\n')

        return line
    except Exception as exc:                                    # noqa: BLE001
        # Never let indexing kill an analysis run.
        print(f'  WARNING: log_analysis failed ({exc.__class__.__name__}: {exc}) — '
              f'run NOT indexed in docs/analyses_run.md. Add it by hand or via '
              f'/lognote "log something not in the list".', flush=True)
        return None


def mark_logged(output_path, git_hash=None, path=None):
    """Flip an entry's `[ ]` to `[x]` once a notebook entry references it.

    Called by /lognote. Matches on output path (and commit hash if given), so
    the caller doesn't have to reconstruct the exact line. Returns how many
    entries were flipped.
    """
    target = Path(path) if path is not None else ANALYSES_RUN_MD
    if not target.exists():
        return 0
    out = _clean(_shorten(output_path))
    lines = target.read_text().splitlines()
    flipped = 0
    for i, line in enumerate(lines):
        m = _LINE_RE.match(line)
        if not m or m.group('path') != out or m.group('logged').lower() == 'x':
            continue
        if git_hash is not None and m.group('hash') != _clean(git_hash):
            continue
        lines[i] = line[:line.rindex('[')] + '[x]'
        flipped += 1
    if flipped:
        target.write_text('\n'.join(lines) + '\n')
    return flipped
