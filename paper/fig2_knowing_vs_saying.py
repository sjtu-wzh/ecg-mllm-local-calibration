"""Figure 2 — Frozen PULSE-7B knows more than it says.
a ROC of the raw Yes/No probability on CPSC test (n = 1461) for three labels; the generated answer (B0) and the S1-calibrated
  decision (fitted on 600 dev records) are single operating points.  Data: b1_cpsc_test.jsonl, pulse7b/cpsc-test answers, preds_test.json
b Macro-F1 of the generated answer vs read-outs of the same frozen model on four out-of-domain sets (S1 / probe: 600 labels,
  mean and 2.5-97.5th percentile of 20 draws).  Data: baselines_*.json
c Where the diagnostic information lives: pool cross-validated macro-F1 of linear probes on LLM hidden states (last prompt
  token) by layer, vs CLIP vision features.  Data: baselines_*.json (stage 1, pool only)"""
import json, os, runpy
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
S = runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "figstyle.py"))
plt, C, MM, ROOT, J = S["plt"], S["C"], S["MM"], S["ROOT"], S["J"]
LABELS = ["NORM", "AF", "I-AVB", "LBBB", "RBBB", "PAC", "PVC", "STD", "STE"]
NAMES = {"I-AVB": "1st-degree AV block", "STD": "ST depression", "PVC": "Premature ventricular contraction"}

gold = {x["id"]: [l for l in LABELS if l in x["conversations"][1]["value"]] for x in J("data/ECGBench/cpsc-test.json")}
test = J("data/splits/cpsc.json")["test"]
P = {json.loads(l)["id"]: json.loads(l)["probs"] for l in open(os.path.join(ROOT, "outputs/clarity/b1_cpsc_test.jsonl"))}
b0 = {json.loads(l)["question_id"]: [t for t in LABELS if t in json.loads(l)["text"]] for l in open(os.path.join(ROOT, "outputs/pulse7b/cpsc-test/answers.jsonl"))}
s1 = J("outputs/clarity/preds_test.json")["S1"]

fig = plt.figure(figsize=(183 * S["MM"], 62 * S["MM"]))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.45, 1.0], wspace=0.5)

# ---- a: ROC
ax = fig.add_subplot(gs[0])
ax.plot([0, 1], [0, 1], color=C["grid"], lw=0.8)
shades = {"I-AVB": "#0F4D92", "STD": "#42949E", "PVC": "#7884B4"}
for L, col in shades.items():
    y = np.array([L in gold[i] for i in test]); s = np.array([P[i][L] for i in test])
    fpr, tpr, _ = roc_curve(y, s); ax.plot(fpr, tpr, color=col, lw=1.2, label=f"{L}  AUROC {roc_auc_score(y, s):.2f}")
    for pred, mk, face in ((b0, "X", C["B0"]), (s1, "o", "white")):
        p = np.array([L in pred[i] for i in test])
        ax.scatter([(p & ~y).sum() / (~y).sum()], [(p & y).sum() / y.sum()], marker=mk, s=28 if mk == "X" else 22, color=face,
                   edgecolor=col if mk == "o" else "white", linewidth=1.0, zorder=5)
ax.scatter([], [], marker="X", s=28, color=C["B0"], label="Generated answer (B0)")
ax.scatter([], [], marker="o", s=22, color="white", edgecolor=C["S1"], label="S1 decision (600 labels)")
ax.set_xlim(-0.02, 1); ax.set_ylim(0, 1.02); ax.set_xlabel("False-positive rate"); ax.set_ylabel("True-positive rate")
ax.set_title("CPSC: ROC of P(yes)", loc="left", fontsize=8.5)
ax.legend(loc="lower right", fontsize=6.2, handletextpad=0.3, borderaxespad=0.2)
S["panel_label"](ax, "a", x=-0.24)

# ---- b: grouped bars
ax = fig.add_subplot(gs[1])
bars = [("B0", "Generated answer", C["B0"]), ("ZS", "Probability (zero-shot)", C["ZS"]),
        ("S1", "S1 calibration", C["S1"]), ("PROBE", "Hidden-state probe", C["PROBE"])]
x = np.arange(4); w = 0.2
for k, (key, lab, col) in enumerate(bars):
    vals, err = [], [[], []]
    for d, _ in S["DATASETS"]:
        r = S["baselines"](d); cur = r["curve"]
        if key == "B0": v, lo, hi = r["reference"]["B0"]["F1"], None, None
        elif key == "ZS": v, lo, hi = r["reference"]["ZS@0.5"]["F1"], None, None
        else: v, lo, hi = cur[("S1" if key == "S1" else "PROBE-LLM") + "@600"]["F1"]
        vals.append(v); err[0].append(0 if lo is None else v - lo); err[1].append(0 if hi is None else hi - v)
    ax.bar(x + (k - 1.5) * w, vals, w * 0.92, color=col, label=lab, yerr=err if key in ("S1", "PROBE") else None,
           error_kw=dict(lw=0.7, capsize=1.5, ecolor="#333333"))
ax.set_xticks(x); ax.set_xticklabels(["CPSC", "CSN", "G12EC", "SPH*"]); ax.set_ylabel("Macro-F1 (held-out test)")
ax.set_ylim(0, 78); ax.grid(axis="y", color=C["grid"], lw=0.5); ax.set_axisbelow(True)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=6.4, columnspacing=0.8, handlelength=1.2)
S["panel_label"](ax, "b", x=-0.14)

# ---- c: layer-wise probing
ax = fig.add_subplot(gs[2])
mk = {"cpsc": "o", "csn": "s", "g12": "^", "sph": "D"}
for d, title in S["DATASETS"]:
    st = S["baselines"](d)["stage1_pool_oof_F1"]
    ys = [st[f"llm_L{L}_last"] for L in (8, 16, 24, 32)] + [st["clip_patch"]]
    ax.plot([0, 1, 2, 3], ys[:4], color=C["PROBE"], marker=mk[d], ms=3, lw=1.0, alpha=0.9)
    ax.plot([3, 4.2], [ys[3], ys[4]], color=C["CLIP"], ls=(0, (2, 2)), lw=0.8)
    ax.scatter([4.2], [ys[4]], marker=mk[d], s=10, color=C["CLIP"], edgecolor="#777777", lw=0.5, zorder=4)
    ax.text(3.1, ys[3], d.upper() if d != "g12" else "G12EC", fontsize=6, va="center", color="#333333")
ax.set_xticks([0, 1, 2, 3, 4.2]); ax.set_xticklabels(["L8", "L16", "L24", "L32", "CLIP"])
ax.set_xlabel("LLM layer (last token) or CLIP"); ax.set_ylabel("Probe macro-F1 (pool CV, n = 600)")
ax.set_xlim(-0.3, 4.6); ax.grid(axis="y", color=C["grid"], lw=0.5); ax.set_axisbelow(True)
ax.set_title("Information by layer", loc="left", fontsize=8.5)
S["panel_label"](ax, "c", x=-0.24)
fig.text(0.5, -0.12, "*SPH: new site, analyses pre-registered. Error bars: 2.5–97.5th percentile over 20 label draws.", ha="center", fontsize=6.5, color="#555555")
S["save"](fig, "Fig2_knowing_vs_saying")
