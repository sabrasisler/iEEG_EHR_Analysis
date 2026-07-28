# analyses_run.md — the terse index of every run

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

- 2026-07-27 15:39 | P0.6 dtype audit: float32 cache round-trip + epoch-average precision, 1 subject(s) | $DERIV/qc/feature_level/validation/dtype_audit/smoke_2026-07-27T153908 | 34da3ee180a1+dirty | [ ]
- 2026-07-27 15:52 | P0.6 dtype audit: float32 cache round-trip + epoch-average precision, 3 subject(s) | $DERIV/qc/feature_level/validation/dtype_audit/p0.6_2026-07-27T155201 | 34da3ee180a1+dirty | [ ]
- 2026-07-27 16:00 | P0.6 dtype audit: float32 cache round-trip + epoch-average precision, 3 subject(s) | $DERIV/qc/feature_level/validation/dtype_audit/p0.6_2026-07-27T160009 | 34da3ee180a1+dirty | [ ]
- 2026-07-28 12:46 | P1.3 view heatmaps (zscore_vs_baseline, mask std10_rv-gross-std3_satmargin15_sw_logz4), n=4 | $DERIV/analysis/scratch/view_heatmap/subject_relative/p13_std10_zscore_vs_baseline_20260728-124613 | 8cd9e87958d7+dirty | [ ]
- 2026-07-28 12:46 | P1.3 view heatmaps (baseline_subtract, mask std10_rv-gross-std3_satmargin15_sw_logz4), n=4 | $DERIV/analysis/scratch/view_heatmap/subject_relative/p13_std10_baseline_subtract_20260728-124617 | 8cd9e87958d7+dirty | [ ]
- 2026-07-28 12:48 | P1.3 view heatmaps (zscore_vs_baseline, mask std10_rv-gross-std3_satmargin15_sw_logz4), n=4 | $DERIV/analysis/scratch/view_heatmap/subject_relative/p13_std10_zscore_nolinenoise_20260728-124839 | 8cd9e87958d7+dirty | [ ]
