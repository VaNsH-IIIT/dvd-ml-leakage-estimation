"""Shared matplotlib style for project figures.

Colours are the first four slots of the dataviz skill's validated categorical
palette, assigned to the four terminal currents in a fixed order everywhere.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CURRENT_COLORS = {
    "idrain": "#2a78d6",   # blue
    "igate": "#eb6834",    # orange
    "isource": "#1baf7a",  # aqua
    "ibulk": "#eda100",    # yellow
}
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"


def apply():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 2,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })
