# WORKFLOW.md — how the lab notebook and tracking files work

**Read this when you don't know where something goes.** Status: v0 — built
2026-07-27, meant to be reshaped from real usage. Everything here is plain text
and thin commands; nothing is expensive to change.

---

## 1. The governing principle

Every record has exactly one home, determined by **what the thing *is*** — which
turns out to be the same question as **how long it lives**. When unsure, ask
"what is this, really?"

| The thing is… | Home | Lifespan |
|---|---|---|
| A standing rule Claude should always follow | `CLAUDE.md` | permanent |
| A phase / milestone / the project's shape | `PLANNING.md` | slow, deliberate |
| A thing I will **DO** (has a done-state, checkable) | `TASKS.md` | until done |
| A thing I am **THINKING** (question/hunch, no done-state) | `SCRATCHPAD.md` | until it resolves or graduates |
| A settled call, with reasons | `DECISIONS.md` | permanent (append-only) |
| A thing that **happened** (ran X, saw Y, Z broke) | `docs/labnotebook/YYYY-MM-DD.md` | permanent (append-only) |
| A note about **one specific figure** | `<figure>.png.notes.md` sidecar (Oak) | permanent, created on demand |
| A terse index line: "ran this" | `docs/analyses_run.md` | permanent (append-only) |

### Tasks vs scratch — the operational test

**Can you check it off?**

- **Yes** → task. *"investigate S85 artifact"*
- **No, it's a question / hunch / maybe** → scratch. *"is S85 a mask failure or a
  bad channel?"*

A single realization often spawns **both**, split at the seam between the
*noticing* (scratch) and the *doing* (task). Scratch is the waiting room: an item
there either resolves, dissolves, or **graduates** — into a task when it becomes
DOable, or into a decision when it becomes a settled call. Scratch is allowed to
be half-formed. That is its job.

### Two zones

- **Cockpit** (repo root) — forward-looking control: `CLAUDE.md`, `PLANNING.md`,
  `TASKS.md`, `SCRATCHPAD.md`, `DECISIONS.md`. Claude Code reads these to know how
  to help.
- **Flight log** (`docs/labnotebook/`, `docs/analyses_run.md`, Oak sidecars) —
  backward-looking record of what happened. **The notebook is the center of
  gravity**; the cockpit files are satellites.

### Two rules that keep it from rotting

**Links are references, never copies.** Cross-reference by date, path, commit
hash, and task/phase ID. Never paste a decision into the notebook or restate a
phase in `TASKS.md` — point at it. Two copies of a fact means one of them is
wrong and you can't tell which.

**Deidentified references only.** Anonymized subject IDs; dates and intervals in
the deidentified (offset-anchored) timeline, never a real clinical date. This is
the same discipline enforced on data, extended to prose, and it is what makes
these files safe to commit to a GitHub remote.

---

## 2. The flow

```
PLANNING ──spawns──► TASKS ──execution generates──► SCRATCHPAD
    ▲                                                    │
    │                                              resolves into
    └──── (later: /updateplan) ◄── reflects ──┐          ▼
                                              │      DECISIONS  (hinge)
                                              │          │
   analysis runs ──log_analysis()──► analyses_run.md     │
                                              │      referenced by
                                         /lognote ──►    ▼
                              docs/labnotebook/YYYY-MM-DD.md  (what happened)
                                              │
                                        references (path, notable figs)
                                              ▼
                        Oak: figure.png  +  figure.png.notes.md  (/annotate)
```

`DECISIONS.md` is the hinge: everything upstream is provisional, everything
downstream of a decision treats it as settled.

---

## 3. Worked example — the S85 heatmap

You run a bandpower sweep. It self-registers via `log_analysis()` → one line in
`docs/analyses_run.md`. You see theta+gamma elevated in high-pain epochs, and S85
looks artifactual. **One moment of work, four records** — because it genuinely
*is* four different kinds of thing at once:

**1. The run → notebook.** `/lognote` shows the sweep in the unlogged list. Pick
it, note *"theta+gamma elevated high-pain, 12/17; S85 artifactual"*, skip the
rest. One narrative entry pointing at the sweep's figure directory.

**2. The figure flaw → sidecar.**

```
/annotate $DERIV/analysis/pain/.../heatmap_S85.png "flat/saturated in gamma, not physiological" #artifact #review --log
```

Creates `heatmap_S85.png.notes.md` beside the figure on Oak; `--log` also drops a
*reference* into today's notebook entry.

**3. The work it spawns → task.**

```
/addtask "investigate S85 gamma artifact — check QC mask coverage / saturation detector"
```

Checkable → task.

**4. The open question → scratch.**

```
/addscratch "is S85 a mask failure? if the detector missed it, is threshold Z too lax → affects P2.1 feature-QC?"
```

Not checkable, a hunch → scratch.

**5. No decision yet.** *"theta+gamma elevated"* is a **nomination**, not a
finding. It accumulates in the notebook across sweep rounds, robustness checks,
and the S85 resolution until it earns a `DECISIONS.md` entry — weeks out, at P2.6
FREEZE. Promoting it early is the failure mode this whole scheme exists to
prevent.

---

## 4. Commands

All live in `.claude/commands/`, so they are available when Claude Code is
launched **from the repo root**. Design posture for all of them: **the lazy path
must be fully functional.** A tool you can finish in ten seconds gets used; one
that demands five prose answers gets avoided, and then the whole system rots.

| Command | What it does |
|---|---|
| `/lognote` | The center-of-gravity ritual. Surfaces unlogged runs, prompts through an entry — **every prompt skippable** — and catches spawned tasks/scratch. |
| `/annotate <figure> ["note"] [#tag ...] [--log]` | Writes a `.notes.md` sidecar beside a figure on Oak. `--log` also references it from today's notebook entry. |
| `/addtask "<action>"` | Appends a checkable `[ ]` item to `TASKS.md`. |
| `/addscratch "<thought>"` | Appends a `[ ]` question/hunch to `SCRATCHPAD.md`. |
| `/standup` | Session-start continuity. Read-only: reports where you left off and what's open. Proposes, never writes. |

**Session-end habit:** dump next-steps into `SCRATCHPAD.md` so tomorrow's
`/standup` picks them up.

### `log_analysis()` — the one line in your analysis scripts

```python
from ieeg_ehr.io.analysis_log import log_analysis

log_analysis('bandpower sweep, HFA x pain, discovery cohort', out_dir)
```

Put it next to the provenance sidecar write. It captures the timestamp and commit
hash itself. Pass the **run directory**, not a per-subject file — the dedupe key
is `(output_path, git_hash)`, which is what makes a 60-task Slurm array collapse
to the single index line it should be.

Add it to scripts **as you next touch them**. Do not do a sweep-and-add pass;
that's a big diff with no immediate payoff.

### What `/lognote`'s "unlogged" list can and cannot see

It is precise, not a filesystem heuristic — analyses self-register. But that
cuts both ways: **a processing event you never scripted will never appear.** The
bipolar-re-referencing example ("60 subjects, 4 failed on memory") is exactly
that case. Hence `/lognote`'s "log something not in the list" escape hatch, which
is a first-class path, not a fallback.

---

## 5. Deferred (deliberately not in v0)

`/logdecision`, `/updateplan`, the HTML figure viewer, Slack sharing.
`DECISIONS.md` and `PLANNING.md` are hand-edited for now — automate only if the
manual version becomes annoying. **Reshape from real usage before adding any of
these.**
