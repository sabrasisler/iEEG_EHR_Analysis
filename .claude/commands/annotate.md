---
description: Annotate one specific figure with a note and/or status tags, as a .notes.md sidecar beside it on Oak
argument-hint: <figure-path> ["note"] [#tag ...] [--log]
allowed-tools: Read, Write, Edit, Glob, Grep, Bash(date *), Bash(ls *)
---

# /annotate — write a sidecar note for one figure

Annotate a single figure. The note lives in `<figure>.notes.md`, **beside the
figure on Oak** — never in the repo (figures are data; see the CODE/DATA BOUNDARY
in `CLAUDE.md`).

Arguments: `$ARGUMENTS`

## Parse the arguments loosely

- **first token** = the figure path (required). Accept `$DERIV/...` shorthand and
  expand it to the Oak derivatives base from `src/ieeg_ehr/config/paths.py`
  (`DERIVATIVES_BASE`). Accept a bare filename if it is unambiguous — glob for it
  under the analysis tree and ask only if there are several matches.
- **quoted string** = the note. Optional.
- **`#word` tokens** = status tags. Zero or more. Conventional set:
  `#review` `#artifact` `#good` `#shared` `#star`. Extensible — they are just
  strings, so accept any `#word` without complaint.
- **`--log`** = also reference this annotation from today's notebook entry.
  Opt-in, default off.

**A bare `/annotate fig.png #star` is valid** — note and tags are independently
optional. So is a note with no tags. If BOTH are missing, ask once for a note,
briefly, and accept an empty answer (in which case do nothing and say so).

If the figure path doesn't exist, say so and stop — do not create a sidecar for a
path that isn't there. That is how the sidecar's existence stays meaningful.

## Auto-capture the provenance

Do not make Sabra retype what the system knows. Look beside and above the figure
for the producing run's provenance (`provenance.json`, `run_info/*.json`,
`params.json`) and pull out the **producing analysis** and the **commit hash**.
If there is no sidecar to read, record `provenance: not found` and move on — do
not ask.

## Write the sidecar

Path is `<figure>.notes.md` — i.e. `heatmap_S85.png.notes.md`, keeping the image
extension. That makes the mapping to its figure unambiguous and 1:1, keeps it
trivial to glob, and survives many figures in one directory.

**Create it only on first annotation** — never pre-create empty sidecars. Its
mere existence is a signal that someone flagged this figure, which the future
viewer will read; empty ones would destroy that signal.

First-annotation format:

```markdown
# Notes — <figure filename>

Figure: `<full path>`
Produced by: `<analysis>` @ `<commit>`
<!-- Sidecar notes for one figure. Append-only. Tags are a space-separated set. -->

## <YYYY-MM-DD HH:MM> — #tag #tag
<note text>
```

On subsequent annotations, **append** a new dated section. Never edit or reflow
an earlier one. Tags are per-annotation, so the file accumulates a history rather
than a single mutable state.

Keep notes deidentified — anonymized subject IDs only.

## If `--log` was passed

Append a one-line **reference** (not a copy of the note) to today's notebook entry
in `docs/labnotebook/<today>.md`, creating the file with a `# <YYYY-MM-DD>` header
if needed:

> flagged `heatmap_S85.png` #artifact #review during the bandpower sweep — see
> sidecar

The point of `--log` being opt-in is that most annotations are minor. Auto-logging
every one would re-flood the notebook, which is the failure mode this whole
scheme exists to avoid. Do not suggest `--log` when it wasn't asked for.

## Report

One line: the sidecar path, whether it was created or appended, and whether a
notebook reference was added. If the tags suggest follow-up work, you may
mention `/addtask` once — but do not run it uninvited.
