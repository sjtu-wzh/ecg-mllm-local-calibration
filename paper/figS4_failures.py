"""Supplementary Figure S4 — Where local calibration helped little or not at all.
a CODE-15% (in-domain for PULSE, native training prompts; 800 test items): learning curves never reach the generated answer.
b Zero-shot transfer of calibrators fitted on PTB-XL images (no CPSC labels) to the CPSC test split.
c Descriptive overview (formerly Figure 5b): generated-answer macro-F1 (x) against probability macro-AUROC (y); marker area = gain
  (S1 - B0, bootstrap mean). The gain contains B0, so x and marker size share a term; models, label sets and case mix differ.
  Not a validated deployment rule. Data: code15_curve.json, results_transfer.json, *_bootstrap_cpsc.json, ml_curve_*.json,
  mg_ml_curve_*.json, code15_bootstrap.json, {,mg_,qwen_}b1_cpsc_all.jsonl"""
import os, runpy
import numpy as np
S = runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "figstyle.py"))
TICKS = [25, 50, 100, 200, 400, 600]  # 300 omitted: collides with 400 on a log axis
plt, C, MM, J, SIZES = S["plt"], S["C"], S["MM"], S["J"], S["SIZES"]
c = J("outputs/clarity/code15_curve.json")["results"]; t = J("outputs/clarity/results_transfer.json")["cpsc_test"]
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(183 * MM, 64 * MM), gridspec_kw=dict(width_ratios=[1.0, 0.85, 1.05], wspace=1.05))
for key, col, lab in (("THR", C["THR"], "Threshold only"), ("S1", C["S1"], "S1"), ("S1B0", C["COMBO"], "S1 + generated answer (exploratory)")):
    a1.fill_between(SIZES, [c["curve"][f"{key}@{n}"]["F1_p2.5"] for n in SIZES], [c["curve"][f"{key}@{n}"]["F1_p97.5"] for n in SIZES], color=col, alpha=0.13, lw=0)
    a1.plot(SIZES, [c["curve"][f"{key}@{n}"]["F1_mean"] for n in SIZES], color=col, marker="o", ms=2.8, label=lab)
a1.axhline(c["B0"]["F1"], color=C["B0"], ls=(0, (4, 2)), lw=1.0, label=f"Generated answer (B0) {c['B0']['F1']:.1f}")
a1.axhline(c["B1@0.5 (zero-shot)"]["F1"], color=C["ZS"], ls=(0, (1, 1.5)), lw=1.0, label="Zero-shot probability")
a1.set_xscale("log"); a1.set_xticks(TICKS); a1.set_xticklabels(TICKS); a1.minorticks_off(); a1.set_xlabel("Labelled ECGs (n)"); a1.set_ylabel("Macro-F1")
a1.legend(fontsize=6.2, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2); a1.set_title("CODE-15% (in-domain)", loc="left", fontsize=8.5); a1.grid(axis="y", color=C["grid"], lw=0.5)
S["panel_label"](a1, "a", x=-0.2)
E3C = "#C48DBC"   # E3 shares the E2 family colour, lighter
V = [("B0 (official generation)", "Generated (B0)", C["B0"]), ("B1@0.5", "ZS, P(yes) ≥ 0.5 (no fitting)", C["ZS"]), ("B1@src", "THR (PTB-XL thresholds)", C["THR"]),
     ("S1", "S1 fitted on PTB-XL", C["S1"]), ("S1+EM", "S1 + EM prior shift", C["S1"]), ("S4", "E3 fitted on PTB-XL", E3C), ("S4+EM", "E3 + EM prior shift", E3C),
     ("S3", "E2 fitted on PTB-XL", C["COMBO"]), ("S3+EM", "E2 + EM prior shift", C["COMBO"])]  # keys: internal codes (Table S6)
y = np.arange(len(V))[::-1]
bars = a2.barh(y, [t[k]["F1"] for k, _, _ in V], color=[c_ for *_, c_ in V], height=0.62, alpha=0.9)
for b, (k, _, _) in zip(bars, V):
    if k.endswith("+EM"): b.set_hatch("////"); b.set_edgecolor("white"); b.set_linewidth(0)
for yy, (k, _, _) in zip(y, V): a2.text(t[k]["F1"] + 0.4, yy, f"{t[k]['F1']:.1f}", va="center", ha="left", fontsize=6.3, color="#222222", zorder=5, bbox=dict(fc="white", ec="none", pad=0.4))
a2.axvline(t["B0 (official generation)"]["F1"], color=C["B0"], ls=(0, (4, 2)), lw=0.9)
a2.set_yticks(y); a2.set_yticklabels([l for _, l, _ in V], fontsize=6.6); a2.set_xlim(40, 66); a2.set_xlabel("CPSC test macro-F1")
a2.set_title("Transferred PTB-XL → CPSC", loc="left", fontsize=8.5); S["panel_label"](a2, "b", x=-0.62)

# ---------- c: descriptive overview of conditions (former Figure 5b)
import json
from sklearn.metrics import roc_auc_score
def cpsc_auc(path):
    L = ["NORM", "AF", "I-AVB", "LBBB", "RBBB", "PAC", "PVC", "STD", "STE"]
    gold = {x["id"]: x["conversations"][1]["value"] for x in J("data/ECGBench/cpsc-test.json")}
    test = J("data/splits/cpsc.json")["test"]
    P = {json.loads(l)["id"]: json.loads(l)["probs"] for l in open(os.path.join(S["ROOT"], path))}
    return float(np.mean([roc_auc_score([l in gold[i] for i in test], [P[i][l] for i in test]) for l in L]))
pb, mb, qb = (J(f"outputs/clarity/{m}_bootstrap_cpsc.json") for m in ("pulse", "mg", "qwen"))
ml = {d: J(f"outputs/clarity/ml_curve_{d}.json")["results"] for d in ("csn", "g12", "sph")}
mg = {d: J(f"outputs/clarity/mg_ml_curve_{d}.json")["results"] for d in ("csn", "sph")}
c15 = J("outputs/clarity/code15_bootstrap.json")
PTS = [("PULSE CPSC", C["S1"], pb["bootstrap"]["S1@600-B0"][0], pb["point"]["B0"], cpsc_auc("outputs/clarity/b1_cpsc_all.jsonl"), (62.0, 0.845, "left")),
       ("PULSE CSN", C["S1"], ml["csn"]["bootstrap"]["S1@600-B0"][0], ml["csn"]["B0"][0], ml["csn"]["prob_macroAUC"], (39.3, 0.962, "center")),
       ("PULSE G12EC", C["S1"], ml["g12"]["bootstrap"]["S1@600-B0"][0], ml["g12"]["B0"][0], ml["g12"]["prob_macroAUC"], (36.5, 0.818, "left")),
       ("PULSE SPH", C["S1"], ml["sph"]["bootstrap"]["S1@600-B0"][0], ml["sph"]["B0"][0], ml["sph"]["prob_macroAUC"], (52.0, 0.893, "left")),
       ("MedGemma CPSC", C["MedGemma"], mb["bootstrap"]["S1@600-B0"][0], mb["point"]["B0"], cpsc_auc("outputs/clarity/mg_b1_cpsc_all.jsonl"), (28.0, 0.714, "left")),
       ("MedGemma CSN", C["MedGemma"], mg["csn"]["bootstrap"]["S1@600-B0"][0], mg["csn"]["B0"][0], mg["csn"]["prob_macroAUC"], (14.0, 0.762, "center")),
       ("MedGemma SPH", C["MedGemma"], mg["sph"]["bootstrap"]["S1@600-B0"][0], mg["sph"]["B0"][0], mg["sph"]["prob_macroAUC"], (19.0, 0.88, "center")),
       ("PULSE CODE-15%", C["B0"], c15["S1@600-B0"][0], c15["point"]["B0"], c15["prob_macroAUC_multilabel"], (85.2, 0.925, "center")),
       ("Qwen2-VL CPSC", "#9A9A9A", qb["bootstrap"]["S1@600-B0"][0], qb["point"]["B0"], cpsc_auc("outputs/clarity/qwen_b1_cpsc_all.jsonl"), (19.7, 0.565, "center"))]
for lab, col, m, b0, auc, (tx, ty, ha) in PTS:
    a3.scatter([b0], [auc], s=max(m, 0.5) * 4.5 + 5, color=col, alpha=0.75 if m > 0 else 0.25, edgecolor=col if m > 0 else C["down"], lw=0.9, zorder=3)
    a3.text(tx, ty, lab, fontsize=5.4, color="#333333", va="center", ha=ha)
for g in (10, 20): a3.scatter([], [], s=g * 4.5 + 5, color="#BBBBBB", label=f"+{g}")
a3.scatter([], [], s=16, color="white", edgecolor=C["down"], lw=0.9, label="negative")
a3.legend(title="S1 − B0", title_fontsize=6, loc="lower right", fontsize=5.8, labelspacing=1.0, borderpad=0.6)
a3.set_xlabel("Generated answer (B0) macro-F1"); a3.set_ylabel("Probability macro-AUROC"); a3.set_xlim(0, 95); a3.set_ylim(0.48, 1.0)
a3.grid(color=C["grid"], lw=0.5); a3.set_axisbelow(True)
a3.set_title("Conditions overview (descriptive)", loc="left", fontsize=8.5); S["panel_label"](a3, "c", x=-0.3)
S["save"](fig, "FigS4_failures", supp=True)
