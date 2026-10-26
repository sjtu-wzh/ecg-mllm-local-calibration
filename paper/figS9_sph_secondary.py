"""Supplementary Figure S9 — Pre-registered SPH validation: secondary analyses (frozen PULSE-7B; docs/prereg_sph.md).
(Formerly Figure 4b,c and Figure 6c; the confirmatory contrasts H1, H2, H3 and H6 are in Figure 4A.)
a H4: threshold-only minus S1 by label budget (mean over 20 draws); tested at n <= 100 (expect >= 0) and n >= 300 (expect < 0).
b H5: S1 minus hidden-state probe (paired per draw; mean and 2.5-97.5th percentile); non-inferiority margin -3 at n <= 200.
c Subgroups: S1 fitted on 600 local labels minus the generated answer (macro-F1), paired-bootstrap 95% CI.
Data: ml_curve_sph.json, baselines_sph.json, clinical_sph.json"""
import os, runpy
import numpy as np
S = runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "figstyle.py"))
plt, C, MM, J, SIZES = S["plt"], S["C"], S["MM"], S["J"], S["SIZES"]
P = J("outputs/clarity/ml_curve_sph.json")["results"]; B = J("outputs/clarity/baselines_sph.json")["results"]
fig = plt.figure(figsize=(183 * MM, 62 * MM)); gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.05], wspace=0.55)

ax = fig.add_subplot(gs[0]); c = P["curve"]
d = [c[f"THR@{n}"]["F1_mean"] - c[f"S1@{n}"]["F1_mean"] for n in SIZES]
ax.bar(range(len(SIZES)), d, color=[C["THR"] if v >= 0 else C["S1"] for v in d], width=0.7)
ax.axhline(0, color="#333333", lw=0.8)
ax.axvspan(-0.5, 2.5, color="#F2F2F2", zorder=0); ax.axvspan(3.5, 6.5, color="#F2F2F2", zorder=0)
ax.text(1, max(d) + 0.8, "tested:\nTHR ≥ S1", ha="center", fontsize=6.2); ax.text(5, max(d) + 0.8, "tested:\nS1 > THR", ha="center", fontsize=6.2)
ax.set_xticks(range(len(SIZES))); ax.set_xticklabels(SIZES, rotation=45); ax.set_xlabel("Labelled local ECGs (n)")
ax.set_ylabel("THR − S1 (macro-F1)"); ax.set_ylim(min(d) - 1.5, max(d) + 4)
ax.set_title("H4  label-budget rule", loc="left", fontsize=8.5); S["panel_label"](ax, "a", x=-0.3)

ax = fig.add_subplot(gs[1]); pk = "paired_F1_S1_minus_other [mean, p2.5, p97.5, draws S1 better]"
v = np.array([B[pk][f"S1-PROBE-LLM@{n}"][:3] for n in SIZES])
ax.axvspan(-0.5, 3.5, color="#F2F2F2", zorder=0); ax.axhline(0, color="#333333", lw=0.8)
ax.axhline(-3, color=C["down"], lw=0.8, ls=(0, (3, 2))); ax.text(-0.3, -3.6, "margin −3", ha="left", va="top", fontsize=6.2, color=C["down"])
ax.errorbar(range(len(SIZES)), v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt="o", ms=3.2, color=C["S1"], ecolor=C["S1"], elinewidth=1, capsize=2)
ax.text(1.5, v[:, 2].max() + 1, "tested (n ≤ 200)", ha="center", fontsize=6.2)
ax.set_xticks(range(len(SIZES))); ax.set_xticklabels(SIZES, rotation=45); ax.set_xlabel("Labelled local ECGs (n)")
ax.set_ylabel("S1 − hidden-state probe (macro-F1)"); ax.set_title("H5  non-inferiority to probe", loc="left", fontsize=8.5)
S["panel_label"](ax, "b", x=-0.3)

ax = fig.add_subplot(gs[2]); sg = J("outputs/clarity/clinical_sph.json")["subgroups"]
names = {"female": "Female", "male": "Male", "age<60": "Age < 60", "age>=60": "Age ≥ 60"}
for y, (g, v) in enumerate(list(sg.items())[::-1]):
    m, lo, hi = v["diff"]
    ax.plot([lo, hi], [y, y], color=C["S1"], lw=1.6); ax.scatter([m], [y], s=22, color=C["S1"], zorder=4)
    ax.text(19, y, f"{m:+.1f}  (n={v['n']})", va="center", fontsize=6.2)
ax.set_yticks(range(len(sg))); ax.set_yticklabels([names[g] for g in list(sg)[::-1]], fontsize=7)
ax.axvline(0, color="#333333", lw=0.8); ax.set_xlim(-2, 27); ax.set_xticks([0, 5, 10, 15])
ax.set_xlabel("S1 − generated answer (Δ macro-F1)"); ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
ax.set_title("Subgroups: S1 − B0", loc="left", fontsize=8.5); S["panel_label"](ax, "c", x=-0.3)

S["save"](fig, "FigS9_sph_secondary", supp=True)
