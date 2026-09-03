"""
One place for the look of the medication figures.

The source analysis re-declares the same six colour constants and the same
`style_axes` in ten files, at poster font sizes (titles 39, axis labels 30). The
palette is kept — these figures are meant to sit beside the benzodiazepine ones
— but the duplication is not, and the sizes are scaled back to something that
reads in a multi-panel grid rather than on a printed poster.

`matplotlib.use('Agg')` happens here, before pyplot is imported anywhere else in
the package. Every module in med_analysis imports this one first for that reason.
"""

import math

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt      # noqa: E402

# Palette, carried over from the source analysis so the two sets of figures are
# visually of a piece.
BAR_COLOR = '#2a78d6'
NORM_COLOR = '#eb6834'
HOURS_COLOR = '#1baf7a'
GRID_COLOR = '#e1e0d9'
AXIS_COLOR = '#c3c2b7'
TEXT_PRIMARY = '#0b0b0b'
TEXT_MUTED = '#898781'
ZERO_LINE_COLOR = '#898781'

#: Categorical palette. Eight slots; the source analysis notes it never needed
#: more than seven. Analgesics use it for routes and subclasses, both of which
#: stay well inside the ceiling — but `categorical_colors` raises rather than
#: silently wrapping, because a repeated colour in a legend is a wrong figure.
PALETTE = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100',
           '#e87ba4', '#008300', '#4a3aa7', '#e34948')

MARKERS = ('o', 's', '^', 'D', 'v', 'P', 'X', '*')

TITLE_SIZE = 13
LABEL_SIZE = 11
TICK_SIZE = 9
LEGEND_SIZE = 9
FOOTNOTE_SIZE = 6

DPI = 200

#: Every exploratory figure in this repo carries this. It is the difference
#: between a nomination and a finding, and it belongs on the image rather than
#: only in the notebook, because the image is what ends up in a slide deck.
FOOTNOTE = ('EXPLORATORY, discovery cohort -- NOMINATIONS, NOT FINDINGS.')


def categorical_colors(keys):
    """key -> colour, stable in the order given."""
    keys = list(keys)
    if len(keys) > len(PALETTE):
        raise ValueError(
            f'{len(keys)} categories but only {len(PALETTE)} palette slots '
            f'({keys}). Add colours to style.PALETTE or group the categories — '
            f'wrapping would put two categories on one colour.')
    return {key: PALETTE[i] for i, key in enumerate(keys)}


def categorical_markers(keys):
    """key -> marker, stable in the order given."""
    keys = list(keys)
    if len(keys) > len(MARKERS):
        raise ValueError(f'{len(keys)} categories but only {len(MARKERS)} markers')
    return {key: MARKERS[i] for i, key in enumerate(keys)}


def style_axes(ax, grid_axis='y'):
    """The house axis style: light horizontal grid behind, no top/right spines."""
    if grid_axis in ('y', 'both'):
        ax.yaxis.grid(True, color=GRID_COLOR, linewidth=1, zorder=0)
    if grid_axis in ('x', 'both'):
        ax.xaxis.grid(True, color=GRID_COLOR, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_color(AXIS_COLOR)
    ax.spines['bottom'].set_color(AXIS_COLOR)
    ax.tick_params(colors=TEXT_MUTED, labelsize=TICK_SIZE)


def label_axes(ax, xlabel=None, ylabel=None, title=None, title_loc='left'):
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=LABEL_SIZE, color=TEXT_PRIMARY)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=LABEL_SIZE, color=TEXT_PRIMARY)
    if title is not None:
        ax.set_title(title, fontsize=TITLE_SIZE, color=TEXT_PRIMARY, pad=8,
                     loc=title_loc)


def _overlap_area(a, b):
    """Area shared by two display-space boxes; 0 if they are disjoint."""
    dx = min(a.x1, b.x1) - max(a.x0, b.x0)
    dy = min(a.y1, b.y1) - max(a.y0, b.y0)
    return dx * dy if dx > 0 and dy > 0 else 0.0


def label_points(ax, xs, ys, labels, *, marker_size=170, fontsize=None,
                 color=None, obstacles=(), pad_pt=2.5, leader_from_pt=15.0):
    """Annotate EVERY point, choosing offsets that do not collide.

    A fixed offset per point is what turns a scatter into a pile: on these
    panels the drugs that matter cluster in one corner, so any single offset
    stacks their labels on each other and a reader cannot tell which name
    belongs to which marker. This places each label instead — greedily, in the
    order given, so pass the points whose placement matters most first. Each
    label takes the CLOSEST candidate offset whose text box clears the boxes
    already placed, every marker, and anything in `obstacles` (the legend,
    typically); if nothing is clean it takes the least-overlapping candidate
    rather than giving up. A leader line is drawn only when a label had to
    travel far enough that which marker it belongs to would otherwise be a
    guess.

    Positions are measured in DISPLAY space, so the axes must already be at
    their final size: call this AFTER the limits are set and after
    `fig.tight_layout()`, or every box is measured against a stale layout.
    Labels may land outside the axes — `save` writes with `bbox_inches='tight'`,
    which grows the canvas to include them rather than clipping.
    """
    from matplotlib.transforms import Bbox

    fontsize = TICK_SIZE if fontsize is None else fontsize
    color = TEXT_PRIMARY if color is None else color

    fig = ax.figure
    fig.canvas.draw()                  # need a renderer, and final transforms
    renderer = fig.canvas.get_renderer()
    px = fig.dpi / 72.0                # points -> pixels
    pad = pad_pt * px

    xy = list(zip(xs, ys))
    pts = ax.transData.transform(xy)
    # `s` is an AREA in pt^2, so the radius is sqrt(s)/2.
    marker_r = math.sqrt(marker_size) / 2.0 * px

    # Every marker blocks every label, including its own.
    blocked = [Bbox.from_bounds(x - marker_r - pad, y - marker_r - pad,
                                2 * (marker_r + pad), 2 * (marker_r + pad))
               for x, y in pts]
    blocked += list(obstacles)
    placed = []

    # A label that leaves the axes is not clipped (`save` uses
    # `bbox_inches='tight'`) but it does stretch the canvas and push the panel
    # off-centre, so spilling out is a soft cost rather than a hard veto: a
    # label still goes outside if inside is genuinely full.
    axes_box = ax.get_window_extent(renderer)

    # Nearest ring first, and due-east first within a ring, so a sparse panel
    # still comes out looking conventionally labelled.
    rings = (11.0, 17.0, 25.0, 35.0, 48.0)
    angles = (0, 35, -35, 70, -70, 110, -110, 145, -145, 180)
    candidates = [(r, math.radians(a)) for r in rings for a in angles]

    for (x_data, y_data), text in zip(xy, labels):
        probe = ax.annotate(text, (x_data, y_data), textcoords='offset points',
                            xytext=(0, 0), fontsize=fontsize, color=color,
                            zorder=6)
        best = None
        for r, theta in candidates:
            dx, dy = r * math.cos(theta), r * math.sin(theta)
            probe.xyann = (dx, dy)
            probe.set_ha('left' if dx > 1 else 'right' if dx < -1 else 'center')
            probe.set_va('bottom' if dy > 1 else 'top' if dy < -1 else 'center')
            e = probe.get_window_extent(renderer)
            box = Bbox.from_extents(e.x0 - pad, e.y0 - pad,
                                    e.x1 + pad, e.y1 + pad)
            cost = sum(_overlap_area(box, other) for other in blocked)
            cost += sum(_overlap_area(box, other) for other in placed)
            box_area = (box.x1 - box.x0) * (box.y1 - box.y0)
            cost += box_area - _overlap_area(box, axes_box)   # the part outside
            if best is None or cost + r < best[0]:
                best = (cost + r, dx, dy, r, box)
            if cost == 0:
                break                  # closest clean spot wins outright
        probe.remove()

        _, dx, dy, r, box = best
        arrow = None
        if r >= leader_from_pt:
            arrow = dict(arrowstyle='-', color=AXIS_COLOR, linewidth=0.7,
                         shrinkA=1.5, shrinkB=marker_r / px + 1.0)
        ax.annotate(text, (x_data, y_data), textcoords='offset points',
                    xytext=(dx, dy),
                    ha='left' if dx > 1 else 'right' if dx < -1 else 'center',
                    va='bottom' if dy > 1 else 'top' if dy < -1 else 'center',
                    fontsize=fontsize, color=color, zorder=6, arrowprops=arrow)
        placed.append(box)

    return placed


def add_footnote(fig, extra=None):
    """The exploratory-status footnote, plus any run-specific caveat."""
    text = f'{extra}  |  {FOOTNOTE}' if extra else FOOTNOTE
    fig.text(0.005, 0.002, text, fontsize=FOOTNOTE_SIZE, color='0.25',
             ha='left', va='bottom')


def save(fig, out_path, footnote=None):
    """Write a figure and close it. The one place dpi/bbox are decided."""
    if footnote is not None:
        add_footnote(fig, footnote)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    return out_path
