"""Figure style shared by every chart (palette: validated categorical order, recessive grid)."""
import matplotlib as mpl
import matplotlib.pyplot as plt

INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
C = {"ensemble": "#2a78d6", "lear": "#eb6834", "lgbm": "#1baf7a", "naive_d1": "#eda100", "clock": "#e87ba4",
     "rl": "#6250d6", "perfect": "#52514e", "naive_d7": "#008300"}
LABEL = {"perfect": "Perfect foresight", "ensemble": "Ensemble forecast + LP", "lear": "LEAR forecast + LP",
         "lgbm": "LightGBM forecast + LP", "naive_d1": "Yesterday's prices + LP", "naive_d7": "Last week's prices + LP",
         "clock": "Clock schedule (trailing-year profile)", "rl": "Q-learning agent"}


def style():
    mpl.rcParams.update({
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 9.5, "axes.titlesize": 10.5,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight", "lines.linewidth": 2,
    })


def eu(x, d=0):
    """European number format: 12.345,6"""
    s = f"{x:,.{d}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")
