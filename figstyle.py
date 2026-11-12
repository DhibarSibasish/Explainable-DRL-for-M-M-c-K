import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
GRAY = "#6b6a63"
INK = "#2b2a27"
MARK = ["o", "s", "^", "D", "v", "P"]
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.edgecolor": "#9a998f",
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "text.color": INK,
    "axes.grid": True, "grid.color": "#e4e3dc", "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6,
    "lines.markersize": 5, "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "serif",
})
