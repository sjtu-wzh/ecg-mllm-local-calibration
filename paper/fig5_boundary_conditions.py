"""Figure 5 — When does local calibration help? Gain = S1 (600 local labels) minus the generated answer (B0), paired bootstrap 95% CI.
Conditions: ECG-trained models on out-of-domain sites; PULSE in-domain with its training prompt (CODE-15%); a general VLM
(Qwen2-VL, CPSC); a calibrator transferred across sites without local labels (PTB-XL -> CPSC).
b Each condition placed by how good the generated answer is (x) and how well the Yes/No probability ranks cases (y);
  marker area = gain. Gains appear only where the model 'knows' (high AUROC) but does not 'say' (low B0).
Data: *_bootstrap*.json, ml_curve_*.json, mg_*curve*.json, code15_bootstrap.json, qwen_*.json, bootstrap_transfer / results_transfer"""
import os, runpy
import numpy as np
from sklearn.metrics import roc_auc_score
S = runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "figstyle.py"))
plt, C, MM, J = S["plt"], S["C"], S["MM"], S["J"]

def cpsc_auc(path):
    import json
    L = ["NORM", "AF", "I-AVB", "LBBB", "RBBB", "PAC", "PVC", "STD", "STE"]
    gold = {x["id"]: x["conversations"][1]["value"] for x in J("data/ECGBench/cpsc-test.json")}
    test = J("data/splits/cpsc.json")["test"]
    P = {json.loads(l)["id"]: json.loads(l)["probs"] for l in open(os.path.join(S["ROOT"], path))}
    return float(np.mean([roc_auc_score([l in gold[i] for i in test], [P[i][l] for i in test]) for l in L]))

pb, mb, qb = J("outputs/clarity/pulse_bootstrap_cpsc.json"), J("outputs/clarity/mg_bootstrap_cpsc.json"), J("outputs/clarity/qwen_bootstrap_cpsc.json")
ml = {d: J(f"outputs/clarity/ml_curve_{d}.json")["results"] for d in ("csn", "g12", "sph")}
mg = {d: J(f"outputs/clarity/mg_ml_curve_{d}.json")["results"] for d in ("csn", "sph")}
c15 = J("outputs/clarity/code15_bootstrap.json"); tr = J("outputs/clarity/stats_cpsc.json")["bootstrap_transfer"]["cpsc_test"]["S1 - B0 (official generation)"]
auc_p, auc_m, auc_q = cpsc_auc("outputs/clarity/b1_cpsc_all.jsonl"), cpsc_auc("outputs/clarity/mg_b1_cpsc_all.jsonl"), cpsc_auc("outputs/clarity/qwen_b1_cpsc_all.jsonl")
G_OOD, G_ID, G_GEN, G_TR = "ECG-trained, out-of-domain", "ECG-trained, in-domain + training prompt", "General VLM", "Calibrator transferred across sites"
rows = [  # label, group, model colour, [mean, lo, hi], B0, AUROC
    ("PULSE · CPSC", G_OOD, C["S1"], pb["bootstrap"]["S1@600-B0"], pb["point"]["B0"], auc_p),
    ("PULSE · CSN", G_OOD, C["S1"], ml["csn"]["bootstrap"]["S1@600-B0"], ml["csn"]["B0"][0], ml["csn"]["prob_macroAUC"]),
    ("PULSE · G12EC", G_OOD, C["S1"], ml["g12"]["bootstrap"]["S1@600-B0"], ml["g12"]["B0"][0], ml["g12"]["prob_macroAUC"]),
    ("PULSE · SPH*", G_OOD, C["S1"], ml["sph"]["bootstrap"]["S1@600-B0"], ml["sph"]["B0"][0], ml["sph"]["prob_macroAUC"]),
    ("MedGemma · CPSC", G_OOD, C["MedGemma"], mb["bootstrap"]["S1@600-B0"], mb["point"]["B0"], auc_m),
    ("MedGemma · CSN", G_OOD, C["MedGemma"], mg["csn"]["bootstrap"]["S1@600-B0"], mg["csn"]["B0"][0], mg["csn"]["prob_macroAUC"]),
    ("MedGemma · SPH*", G_OOD, C["MedGemma"], mg["sph"]["bootstrap"]["S1@600-B0"], mg["sph"]["B0"][0], mg["sph"]["prob_macroAUC"]),
    ("PULSE · CODE-15%", G_ID, C["B0"], c15["S1@600-B0"], c15["point"]["B0"], c15["prob_macroAUC_multilabel"]),
    ("Qwen2-VL · CPSC", G_GEN, "#9A9A9A", qb["bootstrap"]["S1@600-B0"], qb["point"]["B0"], auc_q),
    ("PULSE · PTB-XL→CPSC", G_TR, C["THR"], tr[:3], pb["point"]["B0"], auc_p),
]
fig = plt.figure(figsize=(183 * MM, 72 * MM)); gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], wspace=0.35)
ax = fig.add_subplot(gs[0]); y, ticks, labels, prev = 0, [], [], None
for lab, grp, col, (m, lo, hi), b0, auc in rows[::-1]:
    if prev is not None and grp != prev: y += 0.6
    ax.plot([lo, hi], [y, y], color=col, lw=1.6); ax.scatter([m], [y], s=22, color=col, zorder=4)
    ax.text(31.5, y, f"{m:+.1f} [{lo:.1f}, {hi:.1f}]", va="center", fontsize=6)
    ticks.append(y); labels.append(lab); prev = grp; y += 1
ax.axvline(0, color="#333333", lw=0.8)
ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=6.6); ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
ax.set_xlim(-8, 42); ax.set_xticks([0, 10, 20, 30]); ax.set_xlabel("Gain: S1 (600 labels) − generated answer, Δ macro-F1 (95% CI)")
grp_y = {}
for t, (lab, grp, *_ ) in zip(ticks, rows[::-1]): grp_y.setdefault(grp, []).append(t)
for grp, ys in grp_y.items():
    ax.text(-7.6, max(ys) + 0.5, grp, fontsize=6.3, fontstyle="italic", color="#555555", va="bottom", zorder=5,
            bbox=dict(boxstyle="square,pad=0.1", facecolor="white", edgecolor="none"))
ax.set_title("Gain of local calibration by condition", loc="left", fontsize=8.5, pad=14); S["panel_label"](ax, "a", x=-0.42)

POS = {"PULSE · CSN": (39.3, 0.962, "center"), "PULSE · SPH*": (51.0, 0.903, "left"), "PULSE · CPSC": (61.0, 0.845, "left"),
       "PULSE · G12EC": (32.4, 0.805, "center"), "MedGemma · SPH*": (21.2, 0.868, "center"), "MedGemma · CSN": (11.5, 0.77, "left"),
       "MedGemma · CPSC": (27.5, 0.714, "left"), "PULSE · CODE-15%": (88.0, 0.932, "right"), "Qwen2-VL · CPSC": (23.0, 0.53, "left")}
ax = fig.add_subplot(gs[1])
for lab, grp, col, (m, lo, hi), b0, auc in rows:
    if grp == G_TR: continue
    ax.scatter([b0], [auc], s=max(m, 0.5) * 9 + 6, color=col, alpha=0.75 if m > 0 else 0.25, edgecolor=col if m > 0 else C["down"], lw=0.9, zorder=3)
    tx, ty, ha = POS[lab]
    ax.text(tx, ty, lab.replace(" · ", " "), fontsize=5.6, color="#333333", va="center", ha=ha)
for g in (10, 20):
    ax.scatter([], [], s=g * 9 + 6, color="#BBBBBB", label=f"+{g}")
ax.scatter([], [], s=20, color="white", edgecolor=C["down"], lw=0.9, label="negative")
ax.legend(title="Gain (F1)", title_fontsize=6.5, loc="lower right", fontsize=6, labelspacing=1.1, borderpad=0.8)
ax.set_xlabel("Generated answer (B0) macro-F1"); ax.set_ylabel("Probability macro-AUROC"); ax.set_xlim(10, 95); ax.set_ylim(0.48, 1.0)
ax.grid(color=C["grid"], lw=0.5); ax.set_axisbelow(True)
ax.text(31, 0.655, "upper left = model knows (high AUROC)\nbut does not say (low B0)\n→ largest gains", fontsize=6.2, color=C["S1"], va="center")
ax.set_title("Knowing (AUROC) vs saying (B0)", loc="left", fontsize=8.5); S["panel_label"](ax, "b", x=-0.22)
fig.text(0.01, -0.09, "*SPH: pre-registered new site. In-domain = CODE-15% with PULSE's training prompt. PTB-XL→CPSC: calibrator fitted on PTB-XL images, no CPSC labels.",
         fontsize=6.2, color="#555555")
S["save"](fig, "Fig5_boundary_conditions")
