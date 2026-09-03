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
