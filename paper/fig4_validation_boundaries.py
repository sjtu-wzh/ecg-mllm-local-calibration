"""Figure 4 — Pre-registered validation, replication, boundary conditions and robustness (v2).
Every row is a difference in test macro-F1 with a 95% paired-bootstrap percentile interval over test records; every effect appears once.
Point estimates follow each analysis's own implementation: A-C bootstrap mean (B = 2000; PTB-XL -> CPSC transfer B = 1000, from
clarity/stats_cpsc.py); D and E observed difference on the test records (B = 2000, clarity/robust_eval.py).
A SPH, pre-registered (docs/prereg_sph.md): H1, H2, H3, H6 with their verdicts; H4/H5 are in Table 3 and Figure S9.
B Development datasets (PULSE) and replication (MedGemma), grouped by dataset: S1 - generated answer (B0). SPH rows are in A.
C Boundary conditions: a general model (Qwen2-VL, CPSC), PULSE in its training domain with its training prompt (CODE-15%;
  300 + 300 development records) and an S1 calibrator fitted on PTB-XL images and applied to CPSC without CPSC labels.
D Robustness, pre-registered (docs/prereg_robustness.md): SPH subset (300 fitting / 800 test records), PULSE S1 only. R1 S1 fitted
  on the perturbed images, R2 S1 fitted on clean images (both minus B0 on the perturbed images); R3 paraphrased Yes/No question
  (S1 refitted) minus clean B0; R4 original S1 minus B0 generated with an alternative answer prompt (only B0 changes).
E Prompt mismatch (descriptive): S1 fitted on the original question and applied to a paraphrase, minus S1 refitted on that
  paraphrase; both evaluated on the same paraphrase.
Dashed segment: smallest effect of interest (3 points) for contrasts whose pre-registered rule includes it (H1, H6, R1, R3, R4).
Data: ml_curve_sph, mg_ml_curve_sph, {pulse,mg,qwen}_bootstrap_cpsc, ml_curve_{csn,g12}, mg_ml_curve_csn, code15_bootstrap,
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
RB = J("outputs/clarity/robust_results.json"); RH, RD = RB["hypotheses"], RB["descriptive"]
rh = lambda prefix: RH[next(k for k in RH if k.startswith(prefix))]
R1C, R2C, MIS = C["S1"], "#7FA6CF", "#6E6E6E"

# row: (label, descriptor (text or (text, color)) or None, [(estimate, color, sesoi)], status or None); "gap" inserts space between groups
A = [("H1  PULSE-7B: S1 − B0", ("supported", "#444444"), [(P["S1@600-B0"], C["S1"], True)], None),
     ("H2  PULSE-7B: ZS − B0", ("not supported", C["down"]), [(P["ZS-B0"], C["ZS"], False)], None),
     ("H3  PULSE-7B: S1 − ZS", ("supported", "#444444"), [(P["S1@600-ZS"], C["S1"], False)], None),
     ("H6  MedGemma: S1 − B0", ("supported", "#444444"), [(M["S1@600-B0"], C["MedGemma"], True)], None)]
B = [("CPSC · PULSE-7B", None, [(pb["S1@600-B0"], C["S1"], False)], None),
     ("CPSC · MedGemma", None, [(mb["S1@600-B0"], C["MedGemma"], False)], None), "gap",
     ("CSN · PULSE-7B", None, [(ml["csn"]["S1@600-B0"], C["S1"], False)], None),
     ("CSN · MedGemma", None, [(mg_csn["S1@600-B0"], C["MedGemma"], False)], None), "gap",
     ("G12EC · PULSE-7B", None, [(ml["g12"]["S1@600-B0"], C["S1"], False)], None)]
Cb = [("Qwen2-VL · CPSC", "general model: limited gain", [(qb["S1@600-B0"], "#9A9A9A", False)], None),
      ("PULSE-7B · CODE-15%", "training domain and prompt", [(c15, C["B0"], False)], None),
      ("PULSE-7B · PTB-XL → CPSC", "no target labels; CI includes 0", [(tr, C["THR"], False)], None)]
D = []
for arm, name in (("ROT", "Rotated"), ("PHOTO", "Simulated phone photo"), ("SCAN", "Simulated low-quality scan")):
    D.append((name, None, [(rh(f"R1 {arm}:")["estimate"], R1C, True), (rh(f"R2 {arm}:")["estimate"], R2C, False)], None))
D += ["gap", ("Question paraphrase A (R3)", "S1 refitted on the paraphrase", [(rh("R3 QA:")["estimate"], R1C, True)], None),
      ("Question paraphrase B (R3)", "S1 refitted on the paraphrase", [(rh("R3 QB:")["estimate"], R1C, True)], None),
      ("Answer prompt P1 (R4)", "original S1; B0 with answer prompt P1", [(rh("R4:")["estimate"], R1C, True)], None)]
assert all(h["verdict"] == "SUPPORTED" for h in RH.values()) and len(RH) == 9
DM = [("Question paraphrase A", None, [(RD["S1_clean->QA - S1_QA"], MIS, False)], None),
      ("Question paraphrase B", None, [(RD["S1_clean->QB - S1_QB"], MIS, False)], None)]

FS_LAB, FS_VAL, FS_T = 7.3, 7.0, 8.2
fig = plt.figure(figsize=(183 * MM, 150 * MM))
UNIT, GAP_T = 0.036, 0.07   # figure-height fraction per row; space above each block for its title and subtitle


def extent(rows):
    return sum(0.45 if r == "gap" else (1.35 if r[1] or len(r[2]) > 1 else 1.0) for r in rows)


def block(rows, x0, x1, ytop, title, sub, xval=1.03, xstat=None, xlabel=False):
    h = extent(rows) * UNIT; ax = fig.add_axes([x0, ytop - GAP_T - h, x1 - x0, h])
    y = 0.0
    for r in rows:
        if r == "gap": y += 0.45; continue
        lab, desc, pts, status = r; step = 1.35 if desc or len(pts) > 1 else 1.0; yc = y + step / 2
        ax.text(-0.03, yc - (0.15 if desc else 0), lab, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=FS_LAB)
        if desc:
            dt, dc = desc if isinstance(desc, tuple) else (desc, "#666666")
            ax.text(-0.03, yc + 0.33, dt, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=6.4, color=dc, fontstyle="italic",
                    fontweight="bold" if dc == C["down"] else "normal")
        offs = [0.0] if len(pts) == 1 else [-0.25, 0.25]
        for (est, col, ses), o in zip(pts, offs):
            m, lo, hi = est[:3]
            ax.plot([lo, hi], [yc + o, yc + o], color=col, lw=1.7, solid_capstyle="butt", zorder=3)
            ax.scatter([m], [yc + o], s=24, color=col, zorder=4, edgecolor="white", linewidth=0.4)
            if ses: ax.plot([3, 3], [yc + o - 0.33, yc + o + 0.33], color="#8C8C8C", lw=0.8, ls=(0, (2.2, 1.4)), zorder=2)
            ax.text(xval, yc + o, f"{m:+.1f} [{lo:.1f}, {hi:.1f}]".replace("-", "−"), transform=ax.get_yaxis_transform(), va="center",
                    fontsize=FS_VAL if len(pts) == 1 else 6.6, color="#222222" if len(pts) == 1 or col == R1C else "#4F7BAA")
        if status:
            ax.text(xstat, yc, status, transform=ax.get_yaxis_transform(), va="center", fontsize=6.4,
                    color="#222222" if status == "Supported" else C["down"], fontweight="normal" if status == "Supported" else "bold")
        y += step
    ax.set_ylim(y, 0); ax.set_yticks([]); ax.axvline(0, color="#333333", lw=0.8, zorder=1)
    ax.set_xlim(-10, 32); ax.set_xticks([-10, 0, 10, 20, 30]); ax.tick_params(axis="x", labelsize=6.6, length=2.5)
    ax.spines["left"].set_visible(False); ax.grid(axis="x", color=C["grid"], lw=0.5); ax.set_axisbelow(True)
    if xlabel: ax.set_xlabel("Δ macro-F1 (percentage points)", fontsize=7.3)
    lx = x0 - 0.205
    fig.text(lx, ytop - 0.022, title, fontsize=FS_T, fontweight="bold", va="center")
    fig.text(lx, ytop - 0.050, sub, fontsize=6.6, color="#555555", va="center")
    return ax, ytop - GAP_T - h - 0.035


# left column: A, B, C
L0, L1 = 0.215, 0.37
_, y = block(A, L0, L1, 0.995, "A  SPH validation, pre-registered", "2312 test ECGs; verdict under each row from its pre-registered rule")
_, y = block(B, L0, L1, y, "B  Development datasets and replication", "S1 − B0; each dataset fitted on its own 600 development ECGs")
block(Cb, L0, L1, y, "C  Boundary conditions", "S1 − B0; three different conditions, not one failure mode", xlabel=True)
# right column: D, E
R0, R1_ = 0.735, 0.885
ax_d, y = block(D, R0, R1_, 0.995, "D  Robustness, pre-registered", "S1 − B0; PULSE-7B; 300 fitting / 800 test SPH ECGs; all 9 rules met")
ax_d.scatter([], [], s=24, color=R1C, label="Image rows: fitted on perturbed images (R1)")
ax_d.scatter([], [], s=24, color=R2C, label="Image rows: fitted on clean images (R2)")
ax_d.legend(loc="upper left", bbox_to_anchor=(-1.38, -0.12), fontsize=6.6, handletextpad=0.3, labelspacing=0.5, borderaxespad=0)
_, y = block(DM, R0, R1_, y - 0.075, "E  Effect of prompt mismatch (descriptive)", "Original-prompt fit − matched-prompt refit (same paraphrase at test)", xlabel=True)
fig.text(R0 - 0.205, y - 0.04, "Dashed segment: smallest effect of interest (3 points),\nshown where the pre-registered rule includes it.\n"
         "B0, generated answer; ZS, P(yes) ≥ 0.5.", fontsize=6.4, color="#555555", va="top", linespacing=1.5)
S["save"](fig, "Fig4_validation_boundaries")
