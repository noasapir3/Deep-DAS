"""Figure style: fixed categorical palette (validated, CVD-safe order), thin marks, recessive axes."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# categorical slots in fixed order (blue, orange, aqua, yellow, magenta, green, violet, red)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
DL_COLOR, ZS_COLOR, THIRD = SERIES[0], SERIES[1], SERIES[2]
NEUTRAL = "#9a9892"
NEUTRAL_LIGHT = "#d9d8d3"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e1"
SURFACE = "#fcfcfb"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]   # light -> dark
ORDINAL3 = ["#86b6ef", "#2a78d6", "#0d366b"]                                              # mild, moderate, severe


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": NEUTRAL_LIGHT, "axes.linewidth": 0.8, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "grid.linestyle": "-", "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": TEXT_SECONDARY, "ytick.color": TEXT_SECONDARY, "axes.labelcolor": TEXT, "text.color": TEXT,
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "semibold", "legend.frameon": False, "legend.fontsize": 8,
        "lines.linewidth": 1.6, "lines.markersize": 5, "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
    })


def type_label(atype: str) -> str:
    return atype.replace("anomaly_", "").replace("_", " ")
