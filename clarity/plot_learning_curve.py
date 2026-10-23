"""Plot the CPSC learning curve (outputs/clarity/learning_curve.json) -> outputs/clarity/learning_curve.png"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S = json.load(open(os.path.join(ROOT, "outputs/clarity/learning_curve.json")))["summary"]
SIZES = [25, 50, 100, 200, 300, 400, 600]
SURFACE, TEXT, TEXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = [  # fixed categorical order: slot 1 blue, 2 orange, 3 aqua; marker shape as secondary encoding
    ("THR", "Threshold only", "#2a78d6", "o"),
    ("S1", "S1: label-probability calibration", "#eb6834", "s"),
    ("S3", "S3: S1 + evidence questions (exploratory)", "#1baf7a", "^"),
]
REFS = [(65.4, "PULSE retrained, npj 2026 (CPSC zero-shot)  65.4", "bottom"), (58.6, "Zero-shot, PTB-XL thresholds (best of 9 variants)  58.6", "bottom"),
        (57.6, "B0: PULSE-7B generation  57.6", "top")]  # label above / below its line so the close pair does not collide
END_DY = {"THR": 0, "S1": -0.7, "S3": 0.7}

plt.rcParams.update({"font.family": "Arial", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": TEXT2,
                     "xtick.color": TEXT2, "ytick.color": TEXT2})
fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=200)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
ax.grid(axis="y", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)

REF_STYLES = [(0, (6, 2, 1, 2)), (0, (1, 1.6)), (0, (4, 3))]  # dash-dot / dotted / dashed: reference lines are told apart in the legend
for (y, label, va), ls in zip(REFS, REF_STYLES):
    ax.axhline(y, color=TEXT2, linewidth=1.1, linestyle=ls, alpha=0.85, zorder=4, label=label)

for key, name, color, marker in SERIES:
    m = [S[f"{key}@{n}"]["F1_mean"] for n in SIZES]
    lo = [S[f"{key}@{n}"]["F1_p2.5"] for n in SIZES]
    hi = [S[f"{key}@{n}"]["F1_p97.5"] for n in SIZES]
    ax.fill_between(SIZES, lo, hi, color=color, alpha=0.12, linewidth=0, zorder=2)
    ax.plot(SIZES, m, color=color, linewidth=2, marker=marker, markersize=6, label=name, zorder=3,
            markeredgecolor=SURFACE, markeredgewidth=1.2)
    ax.text(640, m[-1] + END_DY[key], name.split(":")[0], color=TEXT, fontsize=9, va="center", ha="left")

ax.set_xscale("log")
ax.set_xticks(SIZES); ax.set_xticklabels([str(n) for n in SIZES]); ax.minorticks_off()
ax.set_xlim(22, 760); ax.set_ylim(44, 72)
ax.set_xlabel("Labeled CPSC ECGs used for calibration (n)")
ax.set_ylabel("Macro-F1 on CPSC test (n = 1461)")
ax.set_title("How many local labels does calibration need?", loc="left", color=TEXT, fontsize=12, pad=10)
ax.legend(loc="lower right", frameon=False, fontsize=8, labelcolor=TEXT, handlelength=3)
fig.text(0.01, 0.005, "Frozen PULSE-7B, no retraining. Lines: mean of 20 random label draws; bands: 2.5–97.5th percentile. 65.4 is an external number (npj 2026 paper).",
         color=TEXT2, fontsize=7.5)
fig.tight_layout(rect=(0, 0.03, 1, 1))
out = os.path.join(ROOT, "outputs/clarity/learning_curve.png")
fig.savefig(out, facecolor=SURFACE)
print(out)
