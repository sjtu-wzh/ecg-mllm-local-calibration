"""Figure 4 — Pre-registered validation, replication, boundary conditions and robustness (v1; replaces the former Figures 4 and 5).
Every row is a difference in test macro-F1 with a 95% paired-bootstrap percentile interval over test records (B = 2000; point =
bootstrap mean), and every effect appears once. S1 is fitted on 600 local development records unless stated otherwise.
A SPH, pre-registered (docs/prereg_sph.md): H1, H2, H3, H6 (open marker = not supported). H4/H5 are in Table 3 and Figure S9.
B Development datasets (PULSE) and replication (MedGemma): S1 - generated answer (B0). SPH rows are in A.
C Boundary conditions: PULSE in-domain with its training prompt (CODE-15%; 300 + 300 development records), a general model
  (Qwen2-VL, CPSC) and an S1 calibrator fitted on PTB-XL images and applied to CPSC without CPSC labels.
D Robustness, pre-registered (docs/prereg_robustness.md): SPH subset (300 pool / 800 test records), S1 only; R1 S1 fitted on
  the degraded arm, R2 S1 fitted on clean images, R3 paraphrased Yes/No question, R4 alternative answer prompt for B0.
Dashed line: smallest effect of interest (3 points) for the contrasts whose rule includes it (H1, H3, H6, R1, R3, R4).
Data: ml_curve_sph, mg_ml_curve_sph, pulse/mg/qwen_bootstrap_cpsc, ml_curve_{csn,g12}, mg_ml_curve_csn, code15_bootstrap,
      stats_cpsc (bootstrap_transfer), robust_results"""
import os, runpy
S = runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "figstyle.py"))
plt, C, MM, J = S["plt"], S["C"], S["MM"], S["J"]
P = J("outputs/clarity/ml_curve_sph.json")["results"]["bootstrap"]; M = J("outputs/clarity/mg_ml_curve_sph.json")["results"]["bootstrap"]
pb, mb, qb = (J(f"outputs/clarity/{m}_bootstrap_cpsc.json")["bootstrap"] for m in ("pulse", "mg", "qwen"))
ml = {d: J(f"outputs/clarity/ml_curve_{d}.json")["results"]["bootstrap"] for d in ("csn", "g12")}
mg_csn = J("outputs/clarity/mg_ml_curve_csn.json")["results"]["bootstrap"]
c15 = J("outputs/clarity/code15_bootstrap.json")["S1@600-B0"]
tr = J("outputs/clarity/stats_cpsc.json")["bootstrap_transfer"]["cpsc_test"]["S1 - B0 (official generation)"][:3]
RH = J("outputs/clarity/robust_results.json")["hypotheses"]
QW, CODE, TR = "#9A9A9A", C["B0"], C["THR"]

# (label, [mean, lo, hi], colour, supported?, sesoi?) ; header rows: (text, None) block, (text, "sub") sub-block
LEFT = [("A  SPH, pre-registered (2312 test ECGs)", None),
        ("H1  PULSE: S1 − B0", P["S1@600-B0"], C["S1"], True, True),
        ("H2  PULSE: ZS − B0", P["ZS-B0"], C["ZS"], False, False),
        ("H3  PULSE: S1 − ZS", P["S1@600-ZS"], C["S1"], True, True),
        ("H6  MedGemma: S1 − B0", M["S1@600-B0"], C["MedGemma"], True, True),
        ("B  Development datasets and replication (S1 − B0)", None),
        ("PULSE · CPSC", pb["S1@600-B0"], C["S1"], True, False),
        ("PULSE · CSN", ml["csn"]["S1@600-B0"], C["S1"], True, False),
        ("PULSE · G12EC", ml["g12"]["S1@600-B0"], C["S1"], True, False),
        ("MedGemma · CPSC", mb["S1@600-B0"], C["MedGemma"], True, False),
        ("MedGemma · CSN", mg_csn["S1@600-B0"], C["MedGemma"], True, False),
        ("C  Boundary conditions (S1 − B0)", None),
        ("PULSE · CODE-15%, in-domain", c15, CODE, True, False),
        ("Qwen2-VL · CPSC", qb["S1@600-B0"], QW, True, False),
        ("PULSE · PTB-XL → CPSC transfer", tr, TR, True, False)]
RIGHT = [("D  Robustness, pre-registered (SPH subset; S1 − B0)", None), ("Image degradation", "sub")]
for arm, name in (("ROT", "Rotated"), ("PHOTO", "Phone photo"), ("SCAN", "Low-quality scan")):
    for r in ("R1", "R2"):
        h = next(v for k, v in RH.items() if k.startswith(f"{r} {arm}:"))
        RIGHT.append((f"{r}  {name}" + (", clean-fitted S1" if r == "R2" else ""), h["estimate"], C["S1"], h["verdict"] == "SUPPORTED", "pt>=3" in h["rule"]))
RIGHT.append(("Prompt wording (clean images)", "sub"))
for k, lab in (("R3 QA:", "R3  Question QA"), ("R3 QB:", "R3  Question QB"), ("R4:", "R4  Answer prompt P1 (B0)")):
    h = next(v for kk, v in RH.items() if kk.startswith(k))
    RIGHT.append((lab, h["estimate"], C["S1"], h["verdict"] == "SUPPORTED", "pt>=3" in h["rule"]))

from matplotlib.transforms import blended_transform_factory as blend
fig = plt.figure(figsize=(183 * MM, 98 * MM))
TOP, BOT = 0.965, 0.175


def forest(rows, x0, x1, xhead, nrows):
    ax = fig.add_axes([x0, BOT, x1 - x0, TOP - BOT]); hd = blend(fig.transFigure, ax.transData)
    y, ticks, labels, sesoi = 0, [], [], []
    for row in rows:
        if row[1] is None or row[1] == "sub":
            if y > 0 and row[1] is None: y += 0.4
            ax.text(xhead if row[1] is None else x0 - 0.008, y, row[0], transform=hd, ha="left" if row[1] is None else "right", va="center",
                    fontsize=7 if row[1] is None else 6.4, fontweight="bold" if row[1] is None else "normal",
                    fontstyle="normal" if row[1] is None else "italic", color="#222222" if row[1] is None else "#666666", zorder=6,
                    bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="none") if row[1] is None else None)
            y += 1; continue
        lab, (m, lo, hi), col, ok, ses = row
        ax.plot([lo, hi], [y, y], color=col, lw=1.6, solid_capstyle="butt")
        ax.scatter([m], [y], s=22, color=col if ok else "white", edgecolor=col, lw=1.2, zorder=4)
        ax.text(1.03, y, f"{m:+.1f} [{lo:.1f}, {hi:.1f}]".replace("-", "−"), transform=ax.get_yaxis_transform(), va="center", fontsize=6.2)
        if ses: sesoi.append(y)
        ticks.append(y); labels.append(lab); y += 1
    for s_ in sesoi: ax.plot([3, 3], [s_ - 0.45, s_ + 0.45], color="#8C8C8C", lw=0.8, ls=(0, (2.2, 1.4)), zorder=1)
    ax.axvline(0, color="#333333", lw=0.8, zorder=0)
    ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=6.6); ax.set_ylim(nrows - 0.5, -0.6); ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False); ax.grid(axis="x", color=C["grid"], lw=0.5); ax.set_axisbelow(True)
    ax.set_xlim(-8, 32); ax.set_xticks([0, 10, 20, 30]); ax.set_xlabel("Δ macro-F1 versus comparator (95% CI)", fontsize=7)
    return y


n = 15 + 2 * 0.4
forest(LEFT, 0.215, 0.385, 0.005, n)
ax = forest(RIGHT, 0.735, 0.885, 0.53, n)
F = J("outputs/clarity/robust_results.json")["F1"]
fig.axes[-1].text(-0.95, n - 2.2, f"Clean reference (same 800 test ECGs):\nB0 {F['B0_CLEAN']:.1f}; S1 {F['S1_CLEAN']:.1f} macro-F1",
                  transform=fig.axes[-1].get_yaxis_transform(), fontsize=6.3, color="#555555", va="center", linespacing=1.4, zorder=6, bbox=dict(boxstyle="square,pad=0.2", facecolor="white", edgecolor="none"))
fig.text(0.005, 0.005, "B0, generated answer; ZS, P(yes) ≥ 0.5. Filled marker: supported or descriptive; open: pre-registered hypothesis not supported. "
         "Dashed segment: smallest effect of interest (3 points),\nshown where the pre-registered rule includes it. S1 fitted on 600 local "
         "development records, except CODE-15% (300 + 300), PTB-XL → CPSC (PTB-XL records only; no CPSC labels) and D (300).",
         fontsize=6, color="#555555", va="bottom", linespacing=1.4)
S["save"](fig, "Fig4_validation_boundaries")
