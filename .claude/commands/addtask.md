---
description: Append a checkable action item to TASKS.md
argument-hint: "<action>" [(→ origin)]
allowed-tools: Read, Edit, Bash(date *)
---

# /addtask — append a checkable action to TASKS.md

Action: `$ARGUMENTS`

A thin appender. Callable standalone or from inside `/lognote`. Do the smallest
correct thing and get out of the way.

## Check it belongs here

**The operational test: can you check it off?**

- **Yes** → it's a task. Proceed.
- **No** — it's a question, hunch, or maybe → it belongs in `SCRATCHPAD.md`.
  Say so in one line and offer `/addscratch` instead. Do not append it here.

If it's genuinely both — a *noticing* and a *doing* fused together — split it at
the seam and say you did: the doing goes here, the wondering goes to
`SCRATCHPAD.md`.

If `$ARGUMENTS` is empty, ask once for the action, in one line.

## Append it

Add to `TASKS.md` under **`## This week`** unless the action is clearly not
near-term, in which case use `## Next`. Format:

```markdown
- [ ] **<short handle if there's a natural one>** <action, imperative mood>
```

If the caller supplied an origin (an analysis path, a notebook date, a figure), or
you were invoked from `/lognote` and know it, append a back-reference so the task
carries where it came from:

```markdown
      (→ docs/labnotebook/2026-07-27.md)
```

Preserve the file's existing wording and structure. Do not reflow, re-sort, or
"tidy" other lines — this file is read by eye every morning and churn makes real
changes hard to spot.

## Guard the size of the file

`TASKS.md` **must stay small — near-term only.** If it has grown past roughly 15
open items, say so once, briefly, and name which items look like far-future work
that should be pushed down into a `PLANNING.md` phase instead. Suggest; do not
move anything on your own.

## Report

One line: the task you added and which section it went in.
