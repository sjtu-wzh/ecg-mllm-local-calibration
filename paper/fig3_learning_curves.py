"""Figure 3 — Label-budget crossover on four out-of-domain datasets (frozen PULSE-7B).
Lines: mean macro-F1 over 20 random label draws; bands: 2.5-97.5th percentile of draws (fixed test split).
Data: outputs/clarity/baselines_{cpsc,csn,g12,sph}.json"""
import os, runpy
S = runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "figstyle.py"))
plt, C, MM, SIZES = S["plt"], S["C"], S["MM"], S["SIZES"]

fig, axes = plt.subplots(1, 4, figsize=(183 * MM, 58 * MM), sharex=True)
for k, (ax, (d, title)) in enumerate(zip(axes, S["DATASETS"])):
    r = S["baselines"](d); cur = r["curve"]
    ax.axhline(r["reference"]["B0"]["F1"], color=C["B0"], ls=(0, (4, 2)), lw=1.0, zorder=1)
    ax.axhline(r["reference"]["ZS@0.5"]["F1"], color=C["ZS"], ls=(0, (1, 1.5)), lw=1.0, zorder=1)
    for m, key, lab in (("THR", "THR", "Probability + threshold"), ("S1", "S1", "Probability + S1 calibration"),
                        ("PROBE", "PROBE-LLM", "Hidden-state linear probe")):
        mu = [cur[f"{key}@{n}"]["F1"][0] for n in SIZES]; lo = [cur[f"{key}@{n}"]["F1"][1] for n in SIZES]; hi = [cur[f"{key}@{n}"]["F1"][2] for n in SIZES]
        ax.fill_between(SIZES, lo, hi, color=C[m], alpha=0.14, lw=0, zorder=2)
        ax.plot(SIZES, mu, color=C[m], marker="o", ms=2.6, lw=1.3, label=lab, zorder=3)
    ax.set_xscale("log"); ax.set_xticks([25, 50, 100, 200, 600]); ax.set_xticklabels(["25", "50", "100", "200", "600"]); ax.minorticks_off()
    ax.set_title(title, loc="left", fontsize=8.5); ax.grid(axis="y", color=C["grid"], lw=0.5); ax.set_axisbelow(True)
    ax.set_xlabel("Labelled local ECGs (n)")
    ymin = min([cur[f"{m}@{n}"]["F1"][0] for m in ("THR", "S1", "PROBE-LLM") for n in SIZES] + [r["reference"]["B0"]["F1"]]) - 4
    ax.set_ylim(max(0, ymin), max(cur["PROBE-LLM@600"]["F1"][2], cur["S1@600"]["F1"][2]) + 3)  # every mean visible
    S["panel_label"](ax, "abcd"[k], x=-0.2)
axes[0].set_ylabel("Macro-F1 (held-out test)")
h, l = axes[0].get_legend_handles_labels()
from matplotlib.lines import Line2D
h += [Line2D([], [], color=C["B0"], ls=(0, (4, 2)), lw=1.0), Line2D([], [], color=C["ZS"], ls=(0, (1, 1.5)), lw=1.0)]
l += ["Generated answer (B0)", "Probability, P(yes) ≥ 0.5 (zero-shot)"]
fig.legend(h, l, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.07), handlelength=2.2, columnspacing=1.4)
fig.tight_layout(w_pad=1.2)
S["save"](fig, "Fig3_learning_curves")
