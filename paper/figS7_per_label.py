"""Supplementary Figure S7 — Per-label F1 (frozen PULSE-7B, test split): generated answer (B0) vs S1 fitted on 600 local labels.
CPSC: single S1 fit (results_m3.json); CSN/G12EC/SPH: ml_curve_*.json (per_label_F1)."""
import os, runpy
import numpy as np
S = runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "figstyle.py"))
plt, C, MM, J = S["plt"], S["C"], S["MM"], S["J"]
m3 = J("outputs/clarity/results_m3.json")["test"]
data = [("CPSC", m3["B0"]["per_label_F1"], m3["S1"]["per_label_F1"])]
for d, t in (("csn", "CSN"), ("g12", "G12EC"), ("sph", "SPH*")):
    pl = J(f"outputs/clarity/ml_curve_{d}.json")["results"]["per_label_F1"]; data.append((t, pl["B0"], pl["S1@600"]))
fig, axes = plt.subplots(1, 4, figsize=(183 * MM, 95 * MM), gridspec_kw=dict(width_ratios=[1, 1, 1.25, 1], wspace=0.75))
for k, (ax, (t, b0, s1)) in enumerate(zip(axes, data)):
    labs = sorted(b0, key=lambda L: s1[L] - b0[L]); y = np.arange(len(labs))
    for yy, L in zip(y, labs):
        col = C["up"] if s1[L] > b0[L] else C["down"]
        ax.plot([b0[L], s1[L]], [yy, yy], color=col, lw=1.1, alpha=0.8)
        ax.scatter([b0[L]], [yy], marker="X", s=14, color=C["B0"], zorder=3); ax.scatter([s1[L]], [yy], s=12, color=C["S1"], zorder=4)
    ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=5.6 if len(labs) > 15 else 6.4); ax.set_xlim(0, 100); ax.set_xticks([0, 50, 100]); ax.set_xlabel("F1")
    ax.set_title(t, loc="left", fontsize=8.2); ax.grid(axis="x", color=C["grid"], lw=0.5); pos = ax.get_position(); fig.text(pos.x0 - 0.075, pos.y1 + 0.035, "abcd"[k], fontsize=10, fontweight="bold")
from matplotlib.lines import Line2D
H = [Line2D([], [], marker="X", ls="", color=C["B0"], ms=4.5, label="Generated answer (B0)"), Line2D([], [], marker="o", ls="", color=C["S1"], ms=4, label="S1 (600 local labels)"),
     Line2D([], [], color=C["up"], lw=1.2, label="Improvement"), Line2D([], [], color=C["down"], lw=1.2, label="Deterioration")]
fig.legend(handles=H, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.0), fontsize=6.6)
fig.text(0.5, -0.1, "Labels sorted by change (largest gain at top). *SPH: pre-registered new site.", ha="center", fontsize=6.2, color="#555555")
S["save"](fig, "FigS7_per_label", supp=True)
