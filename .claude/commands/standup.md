---
description: Session-start continuity — where you left off, what's open, candidate next steps (read-only)
allowed-tools: Read, Glob, Grep, Bash(date *), Bash(git log *), Bash(squeue --me)
---

# /standup — session-start continuity briefing

Reconstruct where Sabra left off and what's worth doing next.

**This command is READ-ONLY. It proposes; it does not write.** Do not append to
any file, do not check anything off, do not create a notebook entry. If something
obviously needs recording, name it and let her run `/addtask`, `/addscratch`, or
`/lognote`.

## Read, in this order

1. **`TASKS.md`** — what's open and near-term. This is the operative list.
2. **`SCRATCHPAD.md`** — what's unresolved, half-formed, or was dumped at the end
   of the last session. The "next steps" section here is deliberately a message
   from yesterday to today; treat it as the highest-signal input.
3. **`PLANNING.md`** — which phase is live, what it depends on. Context for
   *why* the open tasks matter, not a to-do list.
4. **The most recent one or two `docs/labnotebook/*.md`** — what actually
   happened last. Often explains a task that looks stalled.
5. **`DECISIONS.md`** — skim for context only. Never re-open a settled call in a
   standup; if a decision looks wrong, that is its own conversation.

Optionally check `docs/analyses_run.md` for runs whose flag is still `[ ]` and
whose output path appears in no notebook entry — those are analyses that ran but
were never written up. Also worth a glance: `squeue --me`, since a job left
running or pending overnight changes what's actionable this morning.

## Report, in this shape

Keep it tight — this is a briefing, not a report. Aim for under 20 lines total.

**Where you left off** — one short paragraph from the latest notebook entries and
the scratchpad's next-steps dump. Concrete: what was in flight, what broke.

**Open tasks** — the near-term items, grouped as *ready* vs *blocked* (a task is
blocked when it depends on an unfinished lock — e.g. P1.1 waits on P0.1/P0.2/P0.3/P0.6).
Say what each is blocked *on*. That distinction is the main value of this command.

**Unresolved in scratch** — the open questions, one line each. Flag any that look
like they have quietly become answerable, or that have graduated into a task
without being moved.

**Jobs** — anything running or pending, if you checked.

**Candidate next steps** — two to four, ordered, with a one-line reason each.
Prefer things that unblock the most downstream work: the sequencing rule is that
anything baked into the cache (QC mask, epoch definition) or preventing
contamination (cohort split) is locked *before* building at scale. Cheap locks
first. Say plainly if the highest-value next step is a decision rather than a
task, since decisions can't be delegated to a job queue.

End there. No motivational framing, no restating the project back at her.
