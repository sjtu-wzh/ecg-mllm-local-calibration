"""Supplementary Figure S8 — Robustness to image degradation and prompt wording (pre-registered Experiment 3; frozen PULSE-7B; SPH,
300-record pool subset, 800-record test subset; docs/prereg_robustness.md).
(The pre-registered contrasts R1-R4 are shown in Figure 4D.)
a Macro-F1 per arm: generated answer (B0), zero-shot probability, S1 fitted on the arm's own pool (matched), S1 fitted on clean images.
b,c Learning curves (THR and S1, mean of 20 draws) for the clean and degraded-image arms (b) and the paraphrased questions (c).
Data: outputs/clarity/robust_results.json (path can be overridden with ROBUST_RESULTS for testing)"""
import os, runpy
import numpy as np
S = runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "figstyle.py"))
plt, C, MM, ROOT = S["plt"], S["C"], S["MM"], S["ROOT"]
import json
R = json.load(open(os.environ.get("ROBUST_RESULTS") or os.path.join(ROOT, "outputs/clarity/robust_results.json")))
F = R["F1"]; ARMS = [("CLEAN", "Clean"), ("ROT", "Rotated"), ("PHOTO", "Phone photo"), ("SCAN", "Low-quality scan"),
                     ("QA", "Question QA"), ("QB", "Question QB"), ("P1", "Answer prompt P1")]
fig = plt.figure(figsize=(183 * MM, 118 * MM)); gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1], hspace=0.6, wspace=0.25)

ax = fig.add_subplot(gs[0, :]); x = np.arange(len(ARMS)); w = 0.2
bars = [("B0", "Generated answer (B0)", C["B0"]), ("ZS", "Zero-shot P(yes) ≥ 0.5", C["ZS"]),
        ("S1", "S1, fitted on the arm's own pool", C["S1"]), ("S1_clean->", "S1, fitted on clean images", "#7FA6CF")]
for k, (key, lab, col) in enumerate(bars):
    vals = []
    for a, _ in ARMS:
        if key == "B0": v = F.get(f"B0_{a}", F["B0_CLEAN"] if a in ("QA", "QB") else None)
        elif key == "S1_clean->": v = F.get(f"S1_clean->{a}")
        elif a == "P1": v = None
        else: v = F.get(f"{key}_{a}")
        vals.append(np.nan if v is None else v)
    ax.bar(x + (k - 1.5) * w, vals, w * 0.92, color=col, label=lab)
ax.axvline(3.5, color="#BBBBBB", lw=0.8, ls=(0, (3, 2)))
ax.text(1.5, 1.02, "Image degradation (prompts unchanged)", transform=ax.get_xaxis_transform(), ha="center", fontsize=7, color="#555555")
ax.text(5.0, 1.02, "Prompt wording (clean images)", transform=ax.get_xaxis_transform(), ha="center", fontsize=7, color="#555555")
ax.set_xticks(x); ax.set_xticklabels([l for _, l in ARMS], fontsize=7); ax.set_ylabel("Macro-F1 (800 test ECGs)")
ax.grid(axis="y", color=C["grid"], lw=0.5); ax.set_axisbelow(True); ax.legend(ncol=4, fontsize=6.4, loc="upper center", bbox_to_anchor=(0.5, -0.13))
S["panel_label"](ax, "a", x=-0.07, y=1.04)

ax = fig.add_subplot(gs[1, 0]); SIZES = [25, 50, 100, 200, 300]
cols = {"CLEAN": "#333333", "ROT": "#42949E", "PHOTO": "#B64342", "SCAN": "#9A4D8E"}
for a, col in cols.items():
    ax.plot(SIZES, [R["curve"][f"{a}|S1@{n}"][0] for n in SIZES], color=col, marker="o", ms=2.4, lw=1.2, label=dict(ARMS)[a])
    ax.plot(SIZES, [R["curve"][f"{a}|THR@{n}"][0] for n in SIZES], color=col, lw=0.9, ls=(0, (2, 1.5)))
ax.set_xscale("log"); ax.set_xticks(SIZES); ax.set_xticklabels(SIZES); ax.minorticks_off()
ax.set_xlabel("Labeled ECGs (n)"); ax.set_ylabel("Macro-F1"); ax.grid(axis="y", color=C["grid"], lw=0.5)
ax.set_title("Image degradation (solid S1, dashed threshold)", loc="left", fontsize=8.5); ax.legend(fontsize=6.2, loc="lower right")
S["panel_label"](ax, "b", x=-0.25); ylim = ax.get_ylim()
ax = fig.add_subplot(gs[1, 1])
for a, col in {"CLEAN": "#333333", "QA": "#42949E", "QB": "#B64342"}.items():
    ax.plot(SIZES, [R["curve"][f"{a}|S1@{n}"][0] for n in SIZES], color=col, marker="o", ms=2.4, lw=1.2, label=dict(ARMS)[a])
    ax.plot(SIZES, [R["curve"][f"{a}|THR@{n}"][0] for n in SIZES], color=col, lw=0.9, ls=(0, (2, 1.5)))
ax.set_xscale("log"); ax.set_xticks(SIZES); ax.set_xticklabels(SIZES); ax.minorticks_off(); ax.set_ylim(ylim)
ax.set_xlabel("Labeled ECGs (n)"); ax.grid(axis="y", color=C["grid"], lw=0.5)
ax.set_title("Question wording (solid S1, dashed threshold)", loc="left", fontsize=8.5); ax.legend(fontsize=6.2, loc="lower right")
S["panel_label"](ax, "c", x=-0.15)
S["save"](fig, os.environ.get("ROBUST_FIGNAME", "FigS8_robustness"), supp=True)
