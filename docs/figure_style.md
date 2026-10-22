# Figure style (poster panels 2 and 3)

DRAFT, 2026-10-22. One look for every line, point and bar figure on the poster's
panels 2 and 3. The heatmaps and glass brains keep their own style
(`plot_pain_change_heatmap.py`, `plot_pain_change_glass.py`).

## Scope

| Panel | Figure | Function |
|---|---|---|
| 2 | One figure, three axes: predicted vs actual Δpain (`dose_panel`), per-subject OLS slopes (`slope_panel`), predicted Δpain over the % of time between reports per Δpain bin (`_trajectory_lines`, `_fraction_axis`) | `plot_pain_change_poster` (`poster-panel2`) |
| 3a | Actual vs predicted Δpain per ΔC bin | `plot_pain_change_prediction.fig_dc_actual` (`dc-groups-actual`) |
| 3b | Prediction over the interval, pain fell vs no pain change, one line per ΔC bin | `plot_pain_change_prediction.fig_dc_pain` (`dc-pain-fell`) |
| 3d | Δpain coefficient vs ΔC coefficient per component | `plot_pain_change_prediction.fig_components` (`dc-components`) |

The drawing helpers of `plot_pain_change_prediction.py` (`_style`, `_zero`,
`_title`, `_legend`) carry every rule here. `pain_change_tracking.py` and
`plot_pain_change_poster.py` import them, so one edit restyles every figure of
the three scripts. Each poster axes comes from a `*_panel(ax, ...)` function
that its standalone figure also calls, so the poster and the run's own figures
cannot drift apart.

## Rules

### Title

- One title per figure, bold, centered over the plotting area.
- The title says what the figure shows in plain words, for example
  "Predicted pain change tracks reported change". Run names, weights names,
  weighting tokens and model formulas stay out of the title. The sidecar and
  `provenance.json` carry them.
- A poster figure with several axes gives each axes its own bold centered
  title and has no `suptitle`. The poster's section header names the panel.

### Axes

- No gridlines, on any axis.
- Left and bottom spines only, in MUTED.
- Tick marks MUTED, tick labels INK.
- Axis labels in sentence case and plain words, no jargon ("OLS"). Pain axes
  carry no unit: "Predicted Δpain at report 2", "True Δpain at report 2".
- Time within a pair runs on "% of time between reports" (0% at report 1,
  100% at report 2, `_fraction_axis`), never hours since report 1.

### Reference lines

- Zero: a light grey solid horizontal line at y = 0 whenever the y range
  includes 0 and 0 means "no change" (Δpain, ΔC, a coefficient). Drawn behind
  the data. Counts (histograms) need none, since their axis starts at 0.
- A vertical line at x = 0 follows the same rule (3d, where both axes are
  coefficients).
- Identity (y = x): MUTED, dashed, named in the legend.
- A summary number (median slope) goes in as text in a corner of the axes,
  not as a reference line.

### Data marks

- Lines: mean across subjects, lw 1.6, ± SEM as a fill at alpha 0.18.
- Points with intervals: `errorbar`, capsize 0.
- Colors come from the existing palettes, so a color means the same thing in
  every panel:
  - Δpain bins: `BIN_COLOURS` (blue to grey to red)
  - ΔC bins: `DC_COLOURS` (brown to grey to teal)
  - Region families: `FAMILY_COLOURS`
  - A single series: `NEUTRAL`

### Legend

- Frameless, INK text, inside the axes in the emptiest corner. Raise the y
  limit to make room rather than move the legend outside.
- No counts in an inlaid legend. They go in the `docs/analyses_run.md` line.

### Footer

- None on the poster figure. Counts go in the `docs/analyses_run.md` line,
  caveats in the figure's `.png.notes.md`.

### Output

- PNG, 300 dpi, white background, `bbox_inches='tight'`.

## Constants

| Name | Value | Use |
|---|---|---|
| INK | `#2b2b29` | text, tick labels |
| MUTED | `#6b6a66` | spines, ticks, identity lines |
| ZERO | `#d3d2cd` | the zero line |
| TITLE_SIZE | 12, bold | figure title |
| LABEL_SIZE | 10 | axis labels, subtitles |
| TICK_SIZE | 9 | tick labels |
| LEGEND_SIZE | 8 | legend text |

## Open

- Font sizes: the table above assumes the current figure sizes (about 6 x 4.5
  in). On the poster each figure prints at a few inches wide, so the type may
  need to grow. Fix this once the first figure prints at its poster size.
