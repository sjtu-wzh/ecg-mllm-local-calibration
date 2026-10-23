"""Supplementary Figure S4 — Where local calibration does not help.
a CODE-15% (in-domain for PULSE, native training prompts; 800 test items): learning curves never reach the generated answer.
b Zero-shot transfer of calibrators fitted on PTB-XL images (no CPSC labels) to the CPSC test split.
Data: code15_curve.json, results_transfer.json"""
import os, runpy
import numpy as np
S = runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "figstyle.py"))
TICKS = [25, 50, 100, 200, 400, 600]  # 300 omitted: collides with 400 on a log axis
plt, C, MM, J, SIZES = S["plt"], S["C"], S["MM"], S["J"], S["SIZES"]
c = J("outputs/clarity/code15_curve.json")["results"]; t = J("outputs/clarity/results_transfer.json")["cpsc_test"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(160 * MM, 62 * MM), gridspec_kw=dict(width_ratios=[1.1, 1], wspace=0.75))
for key, col, lab in (("THR", C["THR"], "Threshold only"), ("S1", C["S1"], "S1"), ("S1B0", C["COMBO"], "S1 + generated answer (exploratory)")):
    a1.fill_between(SIZES, [c["curve"][f"{key}@{n}"]["F1_p2.5"] for n in SIZES], [c["curve"][f"{key}@{n}"]["F1_p97.5"] for n in SIZES], color=col, alpha=0.13, lw=0)
    a1.plot(SIZES, [c["curve"][f"{key}@{n}"]["F1_mean"] for n in SIZES], color=col, marker="o", ms=2.8, label=lab)
a1.axhline(c["B0"]["F1"], color=C["B0"], ls=(0, (4, 2)), lw=1.0, label=f"Generated answer (B0) {c['B0']['F1']:.1f}")
a1.axhline(c["B1@0.5 (zero-shot)"]["F1"], color=C["ZS"], ls=(0, (1, 1.5)), lw=1.0, label="Zero-shot probability")
a1.set_xscale("log"); a1.set_xticks(TICKS); a1.set_xticklabels(TICKS); a1.minorticks_off(); a1.set_xlabel("Labelled ECGs (n)"); a1.set_ylabel("Macro-F1")
a1.legend(fontsize=6.2, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2); a1.set_title("CODE-15% (in-domain)", loc="left", fontsize=8.5); a1.grid(axis="y", color=C["grid"], lw=0.5)
S["panel_label"](a1, "a", x=-0.2)
V = [("B0 (official generation)", "Generated (B0)", C["B0"]), ("B1@0.5", "P(yes) ≥ 0.5", C["ZS"]), ("B1@src", "P(yes), PTB-XL thresholds", C["THR"]),
     ("S1", "S1 fitted on PTB-XL", C["S1"]), ("S1+EM", "S1 + EM prior shift", C["S1"]), ("S3", "E2 on PTB-XL", C["COMBO"]), ("S3+EM", "E2 + EM prior shift", C["COMBO"])]  # keys: internal codes (Table S6)
y = np.arange(len(V))[::-1]
a2.barh(y, [t[k]["F1"] for k, _, _ in V], color=[c_ for *_, c_ in V], height=0.62, alpha=0.9)
for yy, (k, _, _) in zip(y, V): a2.text(t[k]["F1"] - 0.4, yy, f"{t[k]['F1']:.1f}", va="center", ha="right", fontsize=6.3, color="white", fontweight="bold")
a2.axvline(t["B0 (official generation)"]["F1"], color=C["B0"], ls=(0, (4, 2)), lw=0.9)
a2.set_yticks(y); a2.set_yticklabels([l for _, l, _ in V], fontsize=6.6); a2.set_xlim(40, 64); a2.set_xlabel("CPSC test macro-F1")
a2.set_title("Transferred PTB-XL → CPSC", loc="left", fontsize=8.5); S["panel_label"](a2, "b", x=-0.62)
S["save"](fig, "FigS4_failures", supp=True)
