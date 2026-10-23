"""Side-by-side learning curves: CPSC (out-of-domain for PULSE) vs CODE-15% (in-domain).
-> outputs/clarity/learning_curves_cpsc_vs_code15.png"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIZES = [25, 50, 100, 200, 300, 400, 600]
SURFACE, TEXT, TEXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = [  # fixed categorical order; marker shape is the secondary encoding
    ("THR", "Threshold only", "#2a78d6", "o"),
    ("S1", "S1: label-probability calibration", "#eb6834", "s"),
    ("S1B0", "S1 + generated answer (exploratory)", "#1baf7a", "^"),
]
cpsc = json.load(open(os.path.join(ROOT, "outputs/clarity/learning_curve.json")))["summary"]
code = json.load(open(os.path.join(ROOT, "outputs/clarity/code15_curve.json")))["results"]
PANELS = [
    ("CPSC 2018 — out-of-domain for PULSE (test n = 1461)", cpsc,
     [(57.6, "B0: PULSE-7B generation", "top"), (58.6, "Zero-shot, PTB-XL thresholds (best of 9 variants)", "bottom"),
      (65.4, "PULSE retrained, npj 2026 (CPSC zero-shot)", "bottom")], (44, 72)),
    ("CODE-15% — in-domain for PULSE (test n = 800)", code["curve"],
     [(round(code["B0"]["F1"], 1), "B0: PULSE-7B generation", "bottom"),
      (round(code["B1@0.5 (zero-shot)"]["F1"], 1), "Zero-shot probability (P ≥ 0.5)", "top")], (60, 90)),
]
REF_STYLES = [(0, (6, 2, 1, 2)), (0, (1, 1.6)), (0, (4, 3))]  # dash-dot / dotted / dashed: reference lines are told apart in the legend
END_DY = {"cpsc": {"THR": 0, "S1": -0.9, "S1B0": 0.9}, "code": {"THR": 0, "S1": -0.9, "S1B0": 0.9}}

plt.rcParams.update({"font.family": "Arial", "font.size": 9.5, "axes.edgecolor": GRID, "axes.labelcolor": TEXT2,
                     "xtick.color": TEXT2, "ytick.color": TEXT2})
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=200)
fig.patch.set_facecolor(SURFACE)
for ax, (title, S, refs, ylim), key in zip(axes, PANELS, ("cpsc", "code")):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    style = lambda lab: REF_STYLES[0] if lab.startswith("PULSE retrained") else REF_STYLES[1] if lab.startswith("Zero-shot") else REF_STYLES[2]
    for y, label, va in sorted(refs, key=lambda r: -r[0]):
        ls = style(label)
        ax.axhline(y, color=TEXT2, linewidth=1.1, linestyle=ls, alpha=0.85, zorder=4, label=f"{label}  {y}")
    for k, name, color, marker in SERIES:
        m = [S[f"{k}@{n}"]["F1_mean"] for n in SIZES]
        ax.fill_between(SIZES, [S[f"{k}@{n}"]["F1_p2.5"] for n in SIZES], [S[f"{k}@{n}"]["F1_p97.5"] for n in SIZES],
                        color=color, alpha=0.10, linewidth=0, zorder=2)
        ax.plot(SIZES, m, color=color, linewidth=2, marker=marker, markersize=5.5, label=name, zorder=3,
                markeredgecolor=SURFACE, markeredgewidth=1.1)
        ax.text(640, m[-1] + END_DY[key][k], k.replace("S1B0", "S1+B0"), color=TEXT, fontsize=8.5, va="center")
    ax.set_xscale("log"); ax.set_xticks(SIZES); ax.set_xticklabels([str(n) for n in SIZES]); ax.minorticks_off()
    ax.set_xlim(22, 800); ax.set_ylim(*ylim)
    ax.set_xlabel("Labeled local ECGs used for calibration (n)")
    ax.set_title(title, loc="left", color=TEXT, fontsize=10.5, pad=8)
axes[0].set_ylabel("Macro-F1 on held-out test (official metric)")
for ax in axes: ax.legend(loc="lower right", frameon=False, fontsize=7.6, labelcolor=TEXT, handlelength=3)
fig.text(0.01, 0.01, "Frozen PULSE-7B, no retraining. Lines: mean of 20 random label draws; bands: 2.5–97.5th percentile. "
         "Each panel has its own y-range. S1+B0 was added after seeing the CODE-15% result (exploratory). 65.4 is an external number (npj 2026 paper).",
         color=TEXT2, fontsize=7.5)
fig.tight_layout(rect=(0, 0.04, 1, 1))
out = os.path.join(ROOT, "outputs/clarity/learning_curves_cpsc_vs_code15.png")
fig.savefig(out, facecolor=SURFACE)
print(out)
