# YYYY-MM-DD

<!--
Template for a lab-notebook day file. Copy to docs/labnotebook/<today>.md, or
just run /lognote and let it do this for you.

WHAT LIVES HERE: things that HAPPENED. Ran X, saw Y, Z broke. Append-only,
permanent. This is the center of gravity of the whole system — most daily
logging lands here and the cockpit files (TASKS/SCRATCHPAD/DECISIONS) are
satellites.

TWO KINDS OF ENTRY, both belong:
  - PROCESSING EVENTS  — what you did to the data and what broke.
      "bipolar re-referenced 60 subjects; 4 failed on memory; rerun with more"
  - ANALYTICAL OBSERVATIONS — what you saw.
      "theta+gamma elevated in high-pain epochs in 12/17; S85 looks artifactual"

RULES:
  - LINKS, NEVER COPIES. Reference the figure DIRECTORY and notable figures by
    path; do not embed them. Reference decisions by date; do not restate them.
    Reference phases/tasks by ID (P2.2, BG.1).
  - NOMINATION LANGUAGE, NOT FINDINGS. "elevated in 12/17" is a nomination.
    Observations accumulate here until they earn a DECISIONS.md entry (P2.6).
  - DEIDENTIFIED ONLY. Anonymized subject IDs, offset-anchored dates. This file
    is committed to a GitHub remote.
  - One file per day; entries APPENDED. Never edit a past entry to "fix" it —
    append a correction, so the record of what you believed at the time survives.
  - A bare entry is a good entry. Date + which analysis, nothing else, is worth
    far more than the entry you didn't write.

Delete this comment block in a real day file.
-->

## HH:MM — <short title of what happened>

**Ran:** <what and why — one or two sentences. Skippable.>

**Output:** `$DERIV/analysis/<...>/<run>_<timestamp>/`
**Commit:** `<12-char hash>` <!-- +dirty if the tree wasn't clean -->
**Index:** logged in `docs/analyses_run.md`

**Saw:** <what stood out — surprises, failures, notable figures. Skippable.>

- notable: `<figure>.png` — <one line on why it's notable>

**Broken / needs rerun:** <or omit this heading entirely.>

**Spawned:**
- task → `TASKS.md`: <action> <!-- checkable -->
- scratch → `SCRATCHPAD.md`: <question/hunch> <!-- not checkable -->

**Refs:** PLANNING P<n.n> · DECISIONS <YYYY-MM-DD> · notebook <YYYY-MM-DD>
