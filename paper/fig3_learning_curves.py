"""Figure 3 — Performance across fitting-set sizes (frozen PULSE-7B; v3).
a-d Test macro-F1 against the number of labelled ECGs used for read-out fitting (n = 25-600; 20 repetitions per n, fixed test split):
    mean (line) and 2.5th-97.5th percentiles of repetitions (band; not confidence intervals). Horizontal lines: generated answer
    (B0) and P(yes) >= 0.5 (ZS), which use no local labels. Probe (dagger): layer and token selected once on the full 600-record pool.
e Probe minus S1 within each repetition: mean paired difference (top, percentage points) and repetitions with Probe > S1 (bottom).
(The n = 600 feature ablations, formerly panel f, are now Figure 2b,c.)
Data: outputs/clarity/baselines_{cpsc,csn,g12,sph}.json ("rows": one macro-F1 per method, n and repetition)"""
import os, runpy
import numpy as np
from matplotlib.lines import Line2D
S = runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "figstyle.py"))
plt, C, MM, SIZES = S["plt"], S["C"], S["MM"], S["SIZES"]
NAMES = {"cpsc": "CPSC 2018", "csn": "CSN", "g12": "G12EC", "sph": "SPH (pre-registered)"}
SHORT = {"cpsc": "CPSC", "csn": "CSN", "g12": "G12EC", "sph": "SPH"}
METH = [("THR", "THR", "s", (0, (5, 1.5, 1, 1.5)), "Threshold only (THR)"), ("S1", "S1", "o", "-", "S1 read-out"),
        ("PROBE", "PROBE-LLM", "^", (0, (4, 1.6)), "Hidden-state probe†")]

R = {}
for d, _ in S["DATASETS"]:
    rows = S["J"](f"outputs/clarity/baselines_{d}.json")["rows"]
    F = {(r["n"], r["rep"], r["method"]): r["F1"] for r in rows}
    assert len(F) == len(rows) == 6 * 7 * 20, d
    R[d] = F
reps = range(20)
pair = lambda d, a, b, n: np.array([R[d][(n, k, a)] - R[d][(n, k, b)] for k in reps])

fig = plt.figure(figsize=(183 * MM, 205 * MM))
outer = fig.add_gridspec(2, 1, height_ratios=[2.1, 0.86], hspace=0.38, top=0.935, bottom=0.06)
curves = outer[0].subgridspec(2, 2, hspace=0.36, wspace=0.22)
# ------------------------------------------------------------------ a-d: learning curves
for k, (d, _) in enumerate(S["DATASETS"]):
    ax = fig.add_subplot(curves[k // 2, k % 2])
    base = S["baselines"](d)["reference"]
    ax.axhline(base["B0"]["F1"], color=C["B0"], ls=(0, (4, 2)), lw=1.0, zorder=1)
    ax.axhline(base["ZS@0.5"]["F1"], color=C["ZS"], ls=(0, (1, 1.5)), lw=1.1, zorder=1)
    for col, key, mk, ls, lab in METH:
        v = np.array([[R[d][(n, r, key)] for r in reps] for n in SIZES])
        ax.fill_between(SIZES, np.percentile(v, 2.5, axis=1), np.percentile(v, 97.5, axis=1), color=C[col], alpha=0.13, lw=0, zorder=2)
        ax.plot(SIZES, v.mean(1), color=C[col], marker=mk, ms=3.2, lw=1.3, ls=ls, label=lab, zorder=3)
    ax.set_xscale("log"); ax.set_xticks(SIZES); ax.set_xticklabels([str(n) for n in SIZES]); ax.minorticks_off()
    ax.set_xlim(21, 700); ax.set_ylim(0, 80); ax.set_yticks(range(0, 81, 20))
    ax.grid(axis="y", color=C["grid"], lw=0.5); ax.set_axisbelow(True)
    ax.set_title(NAMES[d], loc="left", fontsize=8.5)
    if k % 2 == 0: ax.set_ylabel("Test macro-F1 (%)")
    if k >= 2: ax.set_xlabel("Labelled ECGs used for read-out fitting (n)")
    S["panel_label"](ax, "abcd"[k], x=-0.13 if k % 2 == 0 else -0.08, y=1.04)
    if k == 0: handles, labels = ax.get_legend_handles_labels()
handles += [Line2D([], [], color=C["B0"], ls=(0, (4, 2)), lw=1.0), Line2D([], [], color=C["ZS"], ls=(0, (1, 1.5)), lw=1.1)]
labels += ["Generated answer (B0)", "P(yes) ≥ 0.5 (ZS)"]
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=5, fontsize=7, frameon=False, handlelength=2.6, columnspacing=1.3)

bot = outer[1].subgridspec(1, 3, width_ratios=[0.1, 1, 0.1])
# ------------------------------------------------------------------ e: Probe - S1 paired matrix
ax = fig.add_subplot(bot[1])
dsets = [d for d, _ in S["DATASETS"]]
mean = np.array([[pair(d, "PROBE-LLM", "S1", n).mean() for n in SIZES] for d in dsets])
wins = np.array([[(pair(d, "PROBE-LLM", "S1", n) > 0).sum() for n in SIZES] for d in dsets])
ties = int(sum((pair(d, "PROBE-LLM", "S1", n) == 0).sum() for d in dsets for n in SIZES))
im = ax.imshow(wins, cmap="Greys", vmin=0, vmax=20, aspect="auto")
fmt = lambda x: f"{x:+.2f}" if abs(x) < 1 else f"{x:+.1f}"
for i in range(len(dsets)):
    for j in range(len(SIZES)):
        tc = "white" if wins[i, j] >= 13 else "#111111"
        ax.text(j, i - 0.16, fmt(mean[i, j]).replace("-", "−"), ha="center", va="center", fontsize=6.6, color=tc, fontweight="bold")
        ax.text(j, i + 0.2, f"{wins[i, j]}/20", ha="center", va="center", fontsize=6.0, color=tc)
ax.set_xticks(range(len(SIZES))); ax.set_xticklabels(SIZES); ax.set_yticks(range(len(dsets))); ax.set_yticklabels([SHORT[d] for d in dsets])
ax.set_xlabel("Labelled ECGs used for read-out fitting (n)")
ax.tick_params(length=0)
for sp in ax.spines.values(): sp.set_visible(False)
ax.set_title("Probe versus S1 across fitting-set sizes", loc="left", fontsize=8.5, pad=24)
ax.text(0, 1.03, "Top: mean Probe − S1 (percentage points)\nBottom: repetitions with Probe > S1 (of 20)",
        transform=ax.transAxes, fontsize=6.6, color="#444444", va="bottom", linespacing=1.25)
cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, ticks=[0, 10, 20])
cb.set_label("Repetitions with Probe > S1", fontsize=6.6); cb.ax.tick_params(labelsize=6.5, length=2)
S["panel_label"](ax, "e", x=-0.14, y=1.2)
print("ties Probe == S1:", ties)

S["save"](fig, "Fig3_learning_curves")
