"""Supplementary Figure S1 — Clinical operating characteristics on CPSC test (1461 ECGs): generated answer (B0) vs S1 calibrated with 600 local
labels (mean and min-max over 20 cross-validation fold orders). STE is flagged: its probability AUROC is 0.52 and its S1
threshold is unstable. Data: outputs/clarity/stats_cpsc.json (clinical)"""
import os, runpy
import numpy as np
S = runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "figstyle.py"))
plt, C, MM, J = S["plt"], S["C"], S["MM"], S["J"]
cl = J("outputs/clarity/stats_cpsc.json")["clinical"]
order = ["NORM", "AF", "LBBB", "RBBB", "I-AVB", "PVC", "PAC", "STD", "STE"]
NAMES = {"NORM": "Normal", "AF": "Atrial fibrillation", "LBBB": "LBBB", "RBBB": "RBBB", "I-AVB": "1st-degree AV block",
         "PVC": "PVC", "PAC": "PAC", "STD": "ST depression", "STE": "ST elevation†"}
fig = plt.figure(figsize=(183 * MM, 62 * MM)); gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.75], wspace=0.35)

for k, (metric, title) in enumerate((("sens", "Sensitivity"), ("spec", "Specificity"))):
    ax = fig.add_subplot(gs[k]); ys = np.arange(len(order))[::-1]
    for y, L in zip(ys, order):
        b0 = cl["labels"][L]["B0"][metric]; m, lo, hi = cl["labels"][L]["S1 [mean, min, max]"][metric]
        col = C["up"] if m > b0 + 0.005 else (C["down"] if m < b0 - 0.005 else "#999999")
        ax.plot([b0, m], [y, y], color=col, lw=1.2, alpha=0.8)
        ax.plot([lo, hi], [y - 0.22, y - 0.22], color=C["S1"], lw=0.8, alpha=0.6)
        ax.scatter([b0], [y], marker="X", s=22, color=C["B0"], zorder=4); ax.scatter([m], [y], s=20, color=C["S1"], zorder=5)
    ax.set_yticks(ys); ax.set_yticklabels([NAMES[L] for L in order] if k == 0 else [], fontsize=6.8)
    ax.set_xlim((0, 1.02) if metric == "sens" else (0.4, 1.01)); ax.set_xlabel(title); ax.grid(axis="x", color=C["grid"], lw=0.5); ax.set_axisbelow(True)
    ax.axhspan(-0.5, 0.5, color="#FBEAEA", zorder=0)
    ax.set_title(title + " per diagnosis", loc="left", fontsize=8.5); S["panel_label"](ax, "ab"[k], x=-0.42 if k == 0 else -0.08)
    if k == 0:
        ax.scatter([], [], marker="X", s=22, color=C["B0"], label="Generated answer (B0)")
        ax.scatter([], [], s=20, color=C["S1"], label="S1, 600 labels (mean)")
        ax.plot([], [], color=C["S1"], lw=0.8, alpha=0.6, label="S1 range, 20 fold orders")
        ax.legend(loc="upper left", fontsize=6, bbox_to_anchor=(-0.02, 1.0))

ax = fig.add_subplot(gs[2]); wn = cl["wrongly_normal_per_1000"]
labels = ["Abnormal ECG\nlabelled normal\n(any NORM)", "Abnormal ECG\ncalled only\nnormal"]
x = np.arange(2); w = 0.36
ax.bar(x - w / 2, wn["B0"], w * 0.95, color=C["B0"], label="B0")
ax.bar(x + w / 2, wn["S1_mean"], w * 0.95, color=C["S1"], label="S1",
       yerr=[np.array(wn["S1_mean"]) - np.array(wn["S1_min"]), np.array(wn["S1_max"]) - np.array(wn["S1_mean"])],
       error_kw=dict(lw=0.7, capsize=2, ecolor="#333333"))
for i in range(2):
    ax.text(x[i] - w / 2, wn["B0"][i] + 3, f"{wn['B0'][i]:.0f}", ha="center", fontsize=6.3)
    ax.text(x[i] + w / 2, wn["S1_max"][i] + 3, f"{wn['S1_mean'][i]:.0f}", ha="center", fontsize=6.3)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6.2); ax.set_ylabel("Per 1000 ECGs"); ax.set_ylim(0, 195)
ax.legend(loc="upper right", fontsize=6.4); ax.set_title("Missed abnormal ECGs", loc="left", fontsize=8.5)
ax.grid(axis="y", color=C["grid"], lw=0.5); ax.set_axisbelow(True); S["panel_label"](ax, "c", x=-0.32)
fig.text(0.01, -0.12, "†ST elevation: Yes/No probability AUROC 0.52 (uninformative); S1 threshold unstable across folds (sensitivity 0.02–0.95) — "
         "not recommended for use. Green/red segments: improvement/deterioration vs B0.", fontsize=6.2, color="#555555")
S["save"](fig, "FigS1_clinical_cpsc", supp=True)
