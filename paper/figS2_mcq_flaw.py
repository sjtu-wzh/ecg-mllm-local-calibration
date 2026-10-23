"""Supplementary Figure S2 — ECGBench G12EC multiple-choice items can be solved without the image.
Accuracy on the fixed test split (1426 items): official PULSE-7B answer (B0), and conditional-logit read-outs fitted on n labelled
items: position bias only (POS), position + diagnosis-code priors + PULSE letter probabilities (S1-MCQ), and the IMAGE-BLIND
control (letter + code priors only; the model's letter probabilities removed). Data: mcq_curve_g12.json, mcq_blind_g12.json"""
import os, runpy
S = runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "figstyle.py"))
TICKS = [25, 50, 100, 200, 400, 600]  # 300 omitted: collides with 400 on a log axis
plt, C, MM, J, SIZES = S["plt"], S["C"], S["MM"], S["J"], S["SIZES"]
r = J("outputs/clarity/mcq_curve_g12.json")["results"]; bl = J("outputs/clarity/mcq_blind_g12.json")
fig, ax = plt.subplots(figsize=(110 * MM, 70 * MM))
for key, col, lab in (("POS", C["THR"], "Letter-position correction"), ("S1", C["S1"], "Code priors + PULSE letter probabilities")):
    m = [r["curve"][f"{key}@{n}"]["acc_mean"] for n in SIZES]
    ax.fill_between(SIZES, [r["curve"][f"{key}@{n}"]["p2.5"] for n in SIZES], [r["curve"][f"{key}@{n}"]["p97.5"] for n in SIZES], color=col, alpha=0.14, lw=0)
    ax.plot(SIZES, m, color=col, marker="o", ms=2.8, label=lab)
bn = [25, 100, 600]; bm = [bl["blind"][str(n)][0] for n in bn]
ax.errorbar(bn, bm, yerr=[[bm[i] - bl["blind"][str(n)][1] for i, n in enumerate(bn)], [bl["blind"][str(n)][2] - bm[i] for i, n in enumerate(bn)]],
            fmt="s--", color=C["PROBE"], ms=3.5, lw=1.2, capsize=2, label="IMAGE-BLIND: code priors only")
ax.axhline(r["B0"], color=C["B0"], ls=(0, (4, 2)), lw=1.0, label=f"PULSE generated answer (B0) {r['B0']:.1f}")
ax.axhline(12.5, color=C["ZS"], ls=(0, (1, 1.5)), lw=1.0, label="Chance (8 options) 12.5")
ax.set_xscale("log"); ax.set_xticks(TICKS); ax.set_xticklabels(TICKS); ax.minorticks_off(); ax.set_ylim(0, 100)
ax.set_xlabel("Labelled multiple-choice items (n)"); ax.set_ylabel("Accuracy (%)"); ax.grid(axis="y", color=C["grid"], lw=0.5)
ax.legend(loc="lower right", fontsize=6.3); ax.set_title("G12EC multiple-choice: solvable without the image", loc="left", fontsize=8.5)
S["save"](fig, "FigS2_mcq_flaw", supp=True)
