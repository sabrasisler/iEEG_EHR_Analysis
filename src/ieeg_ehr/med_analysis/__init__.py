"""
Medication administration patterns from the EHR MAR export.

A self-contained domain: EHR CSVs in, figures + tables out. Nothing here touches
the PSD cache, the QC masks, or the view chain — the unit of analysis is a drug
administration, not a pain epoch. That is why it is a subpackage beside
`analysis/` rather than a module inside it, and why its outputs go under a
separate level-1 event (`analysis/meds/`).

Layout:

- `load`             MAR CSVs -> one tidy table, one row per administration
- `recording_hours`  gap-aware recorded iEEG hours, per subject per hospital day
- `style`            the shared palette and axis styling for these figures
- `build_admin_table`  materialize the tidy table on demand
- `plot_admin_burden`  Fig 1: administrations vs subjects, + the drug table
- `plot_admin_timing`  Fig 2: time-of-day and inter-dose intervals
- `plot_hospital_day`  Fig 3: administration rate and dose across hospital days
- `plot_coadmin_peth`  Fig 4: co-administration peri-event time histograms

Every figure is a thin rendering layer over `load.load_administrations()`. If a
number in a figure cannot be traced to a column of that table, it does not
belong in the figure.

Adapted from a colleague's benzodiazepine analysis
(`/home/groups/ckeller1/sisler/iEEG-EHR_Code/med_admin/`), with the drug set
swapped to analgesics. Deviations from that source are marked WHY-DIFFERENT in
the module that makes them.
"""
