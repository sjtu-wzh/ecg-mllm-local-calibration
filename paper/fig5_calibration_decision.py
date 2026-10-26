"""Figure 5 — Calibration and decision trade-offs (frozen PULSE-7B; S1 and probe fitted on 600 local labels; formerly Figure 6).
a,b Reliability diagrams (all label x record pairs, 10 equal-width bins with >= 20 pairs) with macro ECE.
c   Decision curve for a diagnosis the generated answer already detects well (AF, CPSC): a control with little room for gain.
d-f Decision curves (net benefit vs threshold probability) for diagnoses where the generated answer under-detects.
    Decision curves for all evaluated diagnoses are summarised in the text and Figure S1; SPH subgroups moved to Figure S9.
Data: outputs/clarity/clinical_{cpsc,sph}.json"""
import os, runpy
import numpy as np
S = runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "figstyle.py"))
plt, C, MM, J = S["plt"], S["C"], S["MM"], S["J"]
R = {d: J(f"outputs/clarity/clinical_{d}.json") for d in ("cpsc", "sph")}
MC = {"RAW": (C["ZS"], "Raw P(yes)"), "S1": (C["S1"], "S1"), "PROBE": (C["PROBE"], "Hidden-state probe")}
fig = plt.figure(figsize=(183 * MM, 112 * MM)); gs = fig.add_gridspec(2, 3, hspace=0.62, wspace=0.42)

for k, (d, title) in enumerate((("cpsc", "CPSC"), ("sph", "SPH*"))):
    ax = fig.add_subplot(gs[0, k]); ax.plot([0, 1], [0, 1], color=C["grid"], lw=0.9)
    for m, (col, lab) in MC.items():
        rel = R[d]["reliability"][m]
        ax.plot([b["mean_pred"] for b in rel], [b["obs"] for b in rel], color=col, marker="o", ms=2.8, lw=1.2, label=lab if k == 0 else None)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed frequency")
    for q, m in enumerate(("RAW", "S1", "PROBE")):
        ax.text(0.97, 0.05 + 0.08 * (2 - q), f"ECE {R[d]['calibration'][m]['macro']['ECE']:.3f}", color=MC[m][0], ha="right", fontsize=6.4, transform=ax.transAxes,
                bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
    ax.set_title(f"Calibration: {title}", loc="left", fontsize=8.5)
    ax.grid(color=C["grid"], lw=0.5); ax.set_axisbelow(True); S["panel_label"](ax, "ab"[k], x=-0.27)

DCA = [("cpsc", "AF", "Atrial fibrillation, CPSC", gs[0, 2]), ("cpsc", "I-AVB", "1st-degree AV block, CPSC", gs[1, 0]),
       ("cpsc", "STD", "ST depression, CPSC", gs[1, 1]), ("sph", "STDEV", "ST deviation, SPH*", gs[1, 2])]
for k, (d, L, title, g) in enumerate(DCA):
    ax = fig.add_subplot(g); v = R[d]["dca"][L]; pt = np.array(v["pt"])
    ax.plot(pt, v["treat_all"], color="#BBBBBB", lw=0.9, ls=(0, (3, 2)), label="Treat all" if k == 0 else None)
    ax.axhline(0, color="#777777", lw=0.8, label="Treat none" if k == 0 else None)
    ax.plot(pt, v["B0"], color=C["B0"], lw=1.3, ls=(0, (4, 2)), label="Generated answer (B0)" if k == 0 else None)
    for m, (col, lab) in MC.items():
        ax.plot(pt, v[m], color=col, lw=1.3)
    top = max(max(v[m]) for m in ("RAW", "S1", "PROBE", "B0")); ax.set_ylim(-0.02, top * 1.15 + 0.005); ax.set_xlim(0, 0.5)
    ax.set_xlabel("Threshold probability"); ax.set_ylabel("Net benefit" if k < 2 else "")
    ax.set_title(title, loc="left", fontsize=8.5); ax.grid(color=C["grid"], lw=0.5); ax.set_axisbelow(True)
    ax.text(0.97, 0.93, f"prevalence {v['prevalence']:.2f}", transform=ax.transAxes, ha="right", fontsize=6.2, color="#555555")
    S["panel_label"](ax, "cdef"[k], x=-0.27)
    if L == "AF": ax.text(0.28, 0.3, "Control: the generated answer\nalready detects AF well", transform=ax.transAxes, ha="left", va="center", fontsize=6.2, color="#555555")
hs, ls = [], []
for a_ in fig.axes:
    h, l = a_.get_legend_handles_labels(); hs += h; ls += l
fig.legend(hs, ls, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.03), fontsize=6.6, handlelength=2.2)
fig.text(0.5, -0.07, "*SPH: pre-registered new site. S1 and probe fitted on 600 local labels; reliability bins with ≥ 20 label–record pairs.", ha="center", fontsize=6.2, color="#555555")
S["save"](fig, "Fig5_calibration_decision")
