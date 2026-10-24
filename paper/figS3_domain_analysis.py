"""Supplementary Figure S3 — Pre-specified in-domain vs out-of-domain analysis (same CPSC 9-label prompt; frozen PULSE-7B).
a Gain G(n) = macro-F1(S1@n) - macro-F1(B0) over 8 labels (STE excluded) on CPSC (out-of-domain) and PTB-XL fold-10 images
  (in-domain; raw and re-weighted to CPSC case mix), 20 patient-grouped partitions, pooled cluster bootstrap 95% CI.
b Decomposition of G(600): read-out (a), re-ranking (c), operating point (b), at-least-one rule (e), macro over labels.
Data: outputs/clarity/domain_results.json (plan: preregistration_domain.json)"""
import os, runpy
import numpy as np
S = runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "figstyle.py"))
TICKS = [25, 50, 100, 200, 400, 600]  # 300 omitted: collides with 400 on a log axis
plt, C, MM, J, SIZES = S["plt"], S["C"], S["MM"], S["J"], S["SIZES"]
d = J("outputs/clarity/domain_results.json")
ARMS = [("CPSC", "CPSC (out-of-domain)", C["S1"]), ("PTB_raw", "PTB-XL (in-domain), raw", C["THR"]), ("PTB_std", "PTB-XL re-weighted to CPSC case mix", C["ZS"])]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(160 * MM, 62 * MM), gridspec_kw=dict(width_ratios=[1.2, 1], wspace=0.35))
for k, (key, lab, col) in enumerate(ARMS):
    cu = d["P1"]["curves"][key]; x = np.array(SIZES) * (1 + 0.04 * (k - 1))
    m = [cu[str(n)][0] for n in SIZES]; lo = [cu[str(n)][1] for n in SIZES]; hi = [cu[str(n)][2] for n in SIZES]
    a1.errorbar(x, m, yerr=[np.array(m) - lo, np.array(hi) - m], color=col, marker="o", ms=2.8, capsize=1.5, lw=1.1, label=lab)
a1.axhline(0, color="#333333", lw=0.8); a1.set_xscale("log"); a1.set_xticks(TICKS); a1.set_xticklabels(TICKS); a1.minorticks_off()
a1.set_xlabel("Labelled ECGs (n)"); a1.set_ylabel("Gain over B0 (Δ macro-F1)"); a1.legend(fontsize=6.3, loc="lower right")
a1.set_title("Gain on out-of-domain vs in-domain images", loc="left", fontsize=8.5); a1.grid(axis="y", color=C["grid"], lw=0.5); S["panel_label"](a1, "a", x=-0.22)
comp = [("a", "read-out"), ("c", "re-\nranking"), ("b", "operating\npoint"), ("e", "≥1-label\nrule")]
x = np.arange(len(comp)); w = 0.26
for k, (key, lab, col) in enumerate(ARMS):
    a2.bar(x + (k - 1) * w, [d["P3"][key][c] for c, _ in comp], w * 0.92, color=col, label=lab)
a2.axhline(0, color="#333333", lw=0.8); a2.set_xticks(x); a2.set_xticklabels([l for _, l in comp], fontsize=6.8)
a2.set_ylabel("Contribution to G(600) (Δ macro-F1)"); a2.set_title("Where the gain comes from", loc="left", fontsize=8.5)
a2.grid(axis="y", color=C["grid"], lw=0.5); a2.set_axisbelow(True); S["panel_label"](a2, "b", x=-0.22)
fig.text(0.01, -0.12, f"Pre-specified verdict (P2): inconclusive — CI of R_std wider than the pre-specified 0.8; R_std = {d['P2']['R_std'][0]:.2f} [{d['P2']['R_std'][1]:.2f}, {d['P2']['R_std'][2]:.2f}], "
         f"test-fold ESS ≥ {d['P2']['ESS_test_min']:.0f}.", fontsize=6.2, color="#555555")
S["save"](fig, "FigS3_domain_analysis", supp=True)
