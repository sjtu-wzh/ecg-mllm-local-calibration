"""Shared style, colours and data loaders for the paper figures (paper/fig*.py). Loaded with runpy by each figure script."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper/figures"); os.makedirs(OUT, exist_ok=True)
J = lambda p: json.load(open(os.path.join(ROOT, p)))

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "svg.fonttype": "none", "pdf.fonttype": 42,
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.major.size": 3, "ytick.major.size": 3, "legend.frameon": False, "lines.linewidth": 1.4, "savefig.dpi": 300,
})
MM = 1 / 25.4  # mm -> inch; single column 89 mm, double column 183 mm
C = {  # one colour per read-out, identical in every figure
    "B0": "#4D4D4D", "ZS": "#A8A8A8", "THR": "#42949E", "S1": "#0F4D92", "PROBE": "#B64342", "CLIP": "#B8B8B8",
    "COMBO": "#9A4D8E", "MedGemma": "#7884B4", "Qwen": "#CFCECE", "grid": "#E6E6E6", "up": "#2E9E44", "down": "#E53935",
}
DATASETS = [("cpsc", "CPSC 2018"), ("csn", "CSN"), ("g12", "G12EC"), ("sph", "SPH (pre-registered)")]
SIZES = [25, 50, 100, 200, 300, 400, 600]


def baselines(d):
    return J(f"outputs/clarity/baselines_{d}.json")["results"]


def panel_label(ax, s, x=-0.18, y=1.06):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left")


OUT_SUPP = os.path.join(ROOT, "paper/figures_supp"); os.makedirs(OUT_SUPP, exist_ok=True)


def save(fig, name, supp=False):
    d = OUT_SUPP if supp else OUT
    for ext in ("svg", "pdf", "png"):
        fig.savefig(os.path.join(d, f"{name}.{ext}"), bbox_inches="tight", facecolor="white")
    print("saved", os.path.join(d, name + ".{svg,pdf,png}"))
