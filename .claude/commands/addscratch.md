---
description: Append an open question or hunch to SCRATCHPAD.md
argument-hint: "<thought>" [(→ origin)]
allowed-tools: Read, Edit, Bash(date *)
---

# /addscratch — append a thought to SCRATCHPAD.md

Thought: `$ARGUMENTS`

A thin appender. Callable standalone or from inside `/lognote`.

## Check it belongs here

**The operational test: can you check it off?**

- **No** — it's a question, hunch, or "thing I'm chewing on" → scratch. Proceed.
- **Yes**, it has a clear done-state → it's a task. Say so in one line and offer
  `/addtask` instead.

**Half-formed is fine here — that is the entire point of this file.** Do not ask
Sabra to sharpen a vague thought before recording it; a vague thought recorded
beats a sharp one lost. Write it down close to how she said it.

If `$ARGUMENTS` is empty, ask once for the thought, in one line.

## Append it

Add to `SCRATCHPAD.md` under **`## Open`** as:

```markdown
- [ ] **<the question, as a question>** <any elaboration she gave>
```

Phrasing it as a question is usually the right shape, since the item exists
precisely because it isn't settled. But don't force it — if her wording is
already the clearest form, keep her wording.

If there's an origin (analysis path, notebook date, figure, config file), append a
back-reference: `(→ config/paths.py TODO(P0.1))`.

Preserve existing wording and structure. Do not reflow or re-sort other items.

## Keep it small

This file is kept deliberately small so it fits in context. If it has grown past
roughly 10 open items, mention once which ones look **stale or already resolved**
— candidates for the four fates: resolve (`[x]`, then delete after committing),
dissolve (delete), graduate to a task, or graduate to a decision. Suggest only;
delete nothing yourself.

## Report

One line: the thought you added. Nothing more — this command should feel free.
