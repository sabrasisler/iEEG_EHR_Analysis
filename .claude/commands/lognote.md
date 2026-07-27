---
description: Write a lab-notebook entry for something that happened (an analysis run, or a processing event)
argument-hint: [optional free-form note to log immediately]
allowed-tools: Read, Write, Edit, Glob, Grep, Bash(date *), Bash(git log *), Bash(git rev-parse *)
---

# /lognote — append a lab-notebook entry

You are helping Sabra log **something that happened** to
`docs/labnotebook/<today>.md`. Read `docs/WORKFLOW.md` §1 if you need the routing
rules.

**The single most important thing about this command: the lazy path must be fully
functional.** A bare entry — date plus which analysis, nothing else — is a
success, not a failure. It takes ten seconds and it is worth far more than the
entry Sabra doesn't write because you asked too much. Never re-prompt for a field
that was skipped. Never editorialize about a sparse entry.

## Step 1 — figure out what's unlogged

Read `docs/analyses_run.md` and parse its entries (format documented in that
file's header). An entry is **unlogged** when BOTH:

- its trailing flag is `[ ]`, **and**
- its output path does not appear anywhere in `docs/labnotebook/*.md`.

The grep is the truth and the flag is a cache — a notebook entry written by hand
still counts as logged. Check both.

If `$ARGUMENTS` is non-empty, treat it as a free-form note to log right now: skip
straight to Step 4 with that text, no questions asked.

## Step 2 — present the list and ask everything AT ONCE

Show the unlogged runs as a numbered list, newest first, human-readable
(date, description, short path). Cap at 10 and say how many more there are.
Always include:

```
0. Log something not in the list (a processing event you never scripted)
```

That option is a **first-class path, not a fallback** — most processing events
("re-referenced 60 subjects, 4 failed on memory") were never a registered
analysis and will never appear in the list.

Then ask for everything in **ONE message**, so the whole ritual is a single
round-trip. Make the skippability explicit and load-bearing:

> Reply with just a number to log a bare entry. Anything else is optional —
> answer only what you feel like, skip the rest.
>
> 1. Which one? (number, or 0 for free-form)
> 2. What did you run, and why?
> 3. What stood out? (surprises, failures, notable figures)
> 4. Anything broken or needing a rerun?
> 5. Does this suggest anything to **DO**, or anything you're still **WONDERING**
>    about?

Do not use a multiple-choice tool for questions 2–5; they are free text. Wait for
one reply and take whatever comes back. If Sabra answers only some, that is the
expected case.

## Step 3 — auto-fill everything you can

Never ask for something the system already knows. From the selected
`analyses_run.md` line, take the **output path** and **commit hash**. Get today's
date with `date +%F` and the time with `date +%H:%M`. If the run has a
provenance sidecar (`provenance.json` / `run_info/*.json`) under its output path
and you need more detail, read it rather than asking.

## Step 4 — append the entry

If `docs/labnotebook/<today>.md` doesn't exist, create it with `# <YYYY-MM-DD>`
as the first line. **Append** — never rewrite or reflow existing entries in the
file; it is an append-only record.

Follow the shape of `docs/labnotebook/_TEMPLATE.md`, but **omit any heading whose
content was skipped.** An entry with only a title, output path, and commit is a
valid entry. Rules for the prose:

- **Links, never copies.** Reference the figure *directory* and name notable
  figures by path. Do not embed figures. Reference decisions by date and
  phases/tasks by ID (P2.2, BG.1); do not restate them.
- **Nomination language, not findings.** Write "elevated in 12/17", not "we
  found that". Observations accumulate here until they earn a `DECISIONS.md`
  entry at P2.6. If Sabra's wording states a conclusion, keep her wording — just
  don't upgrade it yourself.
- **Deidentified only.** Anonymized subject IDs; no real clinical dates. This
  file goes to a GitHub remote.
- Write in Sabra's voice from her answers. Do not invent observations she didn't
  make, and do not pad a sparse entry to look fuller.

## Step 5 — mark it logged

Flip that line's `[ ]` to `[x]` in `docs/analyses_run.md` with the Edit tool.
Change nothing else in the file. (`mark_logged()` in
`src/ieeg_ehr/io/analysis_log.py` does the same thing programmatically, but do
not run Python for this — Python must not run on the Sherlock login node.)

## Step 6 — route the spawned work

From the answer to question 5, split at the doing/thinking seam and do both if
both apply:

- **checkable action** → append to `TASKS.md` (as `/addtask` would), with a
  `(→ docs/labnotebook/<today>.md)` origin reference.
- **question / hunch, no done-state** → append to `SCRATCHPAD.md` (as
  `/addscratch` would), same origin reference.

Also add a short **Spawned:** section to the notebook entry pointing at what you
added. If question 5 was skipped, skip this step silently.

## Finally

Report in two or three lines: which file you appended to, what you routed where,
and nothing more. No summary of the entry back at her — she just wrote it.
