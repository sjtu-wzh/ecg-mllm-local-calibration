"""Three-panel learning curves on the out-of-domain datasets (CPSC, CSN, G12EC) for frozen PULSE-7B.
-> outputs/clarity/learning_curves_3ood.png"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
J = lambda p: json.load(open(os.path.join(ROOT, p)))
SIZES = [25, 50, 100, 200, 300, 400, 600]
SURFACE, TEXT, TEXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = [("THR", "Threshold only", "#2a78d6", "o"), ("S1", "S1: label-probability calibration", "#eb6834", "s")]

cpsc = J("outputs/clarity/learning_curve.json")["summary"]
cpsc_curve = {k: {"F1_mean": v["F1_mean"], "p2.5": v["F1_p2.5"], "p97.5": v["F1_p97.5"]} for k, v in cpsc.items()}
cpsc_zs = J("outputs/clarity/results_transfer.json")["cpsc_test"]["B1@0.5"]["F1"]
panels = [("CPSC 2018", "9 labels, test n = 1461", cpsc_curve,
           [(57.6, f"B0: PULSE-7B generation  57.6  =  zero-shot probability (P ≥ 0.5)  {cpsc_zs:.1f}", 90, "bottom")])]
ZS_POS = {"csn": (60, "top"), "g12": (130, "top")}  # label positions chosen to avoid the curves
for d, title in (("csn", "CSN (Chapman–Shaoxing–Ningbo)"), ("g12", "G12EC (Georgia)")):
    r = J(f"outputs/clarity/ml_curve_{d}.json")["results"]
    panels.append((title, f"{r['n_labels']} labels, test n = {r['n_test']}", r["curve"],
                   [(r["B0"][0], f"B0: PULSE-7B generation  {r['B0'][0]:.1f}", 23.5, "bottom"),
                    (r["ZS"][0], f"Zero-shot probability (P ≥ 0.5)  {r['ZS'][0]:.1f}", *ZS_POS[d]),
                    (r["BLIND"][0], f"Image-blind ceiling  {r['BLIND'][0]:.1f}", 23.5, "bottom")]))

plt.rcParams.update({"font.family": "Arial", "font.size": 9.5, "axes.edgecolor": GRID, "axes.labelcolor": TEXT2,
                     "xtick.color": TEXT2, "ytick.color": TEXT2})
fig, axes = plt.subplots(1, 3, figsize=(15, 4.9), dpi=200)
fig.patch.set_facecolor(SURFACE)
for ax, (title, sub, C, refs) in zip(axes, panels):
    ax.set_facecolor(SURFACE); ax.grid(axis="y", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    vals = [v for k in ("THR", "S1") for n in SIZES for v in (C[f"{k}@{n}"]["p2.5"], C[f"{k}@{n}"]["p97.5"])] + [r[0] for r in refs]
    lo, hi = min(vals) - 3, max(vals) + 3
    for y, label, x, va in refs:
        ax.axhline(y, color=TEXT2, linewidth=1, linestyle=(0, (4, 3)), alpha=0.7, zorder=4)
        ax.text(x, y + (0.3 if va == "bottom" else -0.3), label, color=TEXT2, fontsize=7.3, va=va, ha="left", zorder=2.5,
                bbox=dict(boxstyle="square,pad=0.1", facecolor=SURFACE, edgecolor="none", alpha=0.9))
    for k, name, color, marker in SERIES:
        m = [C[f"{k}@{n}"]["F1_mean"] for n in SIZES]
        ax.fill_between(SIZES, [C[f"{k}@{n}"]["p2.5"] for n in SIZES], [C[f"{k}@{n}"]["p97.5"] for n in SIZES],
                        color=color, alpha=0.11, linewidth=0, zorder=2)
        ax.plot(SIZES, m, color=color, linewidth=2, marker=marker, markersize=5.5, label=name, zorder=3,
                markeredgecolor=SURFACE, markeredgewidth=1.1)
    ends = {k: C[f"{k}@600"]["F1_mean"] for k, *_ in SERIES}
    gap = ends["S1"] - ends["THR"]; dy = 0 if abs(gap) > 1.8 else (1.8 - abs(gap)) / 2 * (1 if gap >= 0 else -1)
    for k, *_ in SERIES:
        ax.text(650, ends[k] + (dy if k == "S1" else -dy), f"{k} {ends[k]:.1f}", color=TEXT, fontsize=8.3, va="center")
    ax.set_xscale("log"); ax.set_xticks(SIZES); ax.set_xticklabels([str(n) for n in SIZES]); ax.minorticks_off()
    ax.set_xlim(22, 900); ax.set_ylim(lo, hi)
    ax.set_xlabel("Labeled local ECGs used for calibration (n)")
    ax.set_title(f"{title}\n", loc="left", color=TEXT, fontsize=10.5, pad=2)
    ax.text(0, 1.015, sub, transform=ax.transAxes, color=TEXT2, fontsize=8.3)
axes[0].set_ylabel("Macro-F1 on held-out test (official metric)")
axes[0].legend(loc="lower right", frameon=False, fontsize=8.3, labelcolor=TEXT)
fig.text(0.01, 0.012, "Frozen PULSE-7B, no retraining; all three datasets are out-of-domain for PULSE. Lines: mean of 20 random label draws; "
         "bands: 2.5–97.5th percentile. Each panel has its own y-range. CSN and G12EC are multi-label versions built from the ECGBench "
         "multiple-choice gold options; their B0 uses a CPSC-style prompt with that label list.", color=TEXT2, fontsize=7.2, wrap=True)
fig.tight_layout(rect=(0, 0.05, 1, 1))
out = os.path.join(ROOT, "outputs/clarity/learning_curves_3ood.png"); fig.savefig(out, facecolor=SURFACE); print(out)
