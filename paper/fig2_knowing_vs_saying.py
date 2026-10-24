"""Figure 2 — Generated answers versus read-outs of frozen PULSE-7B (v2).
a CPSC test split (n = 1461), three diagnoses: ROC of the raw Yes/No probability (solid) and of the S1 read-out (dashed; logistic
  regression on all nine label log-odds, one fit on the 600 development records, as in clarity/verify_stack.py). Markers: generated
  answer (B0) and the S1 decision (out-of-fold threshold plus the at-least-one-label rule, so it can lie slightly off the S1 curve).
b Held-out macro-F1 of four read-outs on four datasets. S1 and probe: mean and 2.5th-97.5th percentiles of 20 repetitions on the same
  600-record development pool (record order and cross-validation folds differ); B0 and P(yes) >= 0.5 use no local labels.
c Exploratory development-pool feature screening (5-fold out-of-fold macro-F1 with thresholds chosen on the same predictions; optimistic).
  LLM: last prompt token by layer; CLIP: patch-token mean of the resized whole image (a different input view), shown separately.
Data: b1_cpsc_{dev,test}.jsonl, pulse7b/cpsc-test answers, preds_test.json, baselines_*.json"""
import json, os, runpy
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
S = runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "figstyle.py"))
plt, C, MM, ROOT, J = S["plt"], S["C"], S["MM"], S["ROOT"], S["J"]
LABELS = ["NORM", "AF", "I-AVB", "LBBB", "RBBB", "PAC", "PVC", "STD", "STE"]
SHOW = ["I-AVB", "STD", "PVC"]

gold = {x["id"]: [l for l in LABELS if l in x["conversations"][1]["value"]] for x in J("data/ECGBench/cpsc-test.json")}
split = J("data/splits/cpsc.json")
rd = lambda p: {json.loads(l)["id"]: json.loads(l)["probs"] for l in open(os.path.join(ROOT, p))}
P = {**rd("outputs/clarity/b1_cpsc_dev.jsonl"), **rd("outputs/clarity/b1_cpsc_test.jsonl")}
b0 = {json.loads(l)["question_id"]: [t for t in LABELS if t in json.loads(l)["text"]] for l in open(os.path.join(ROOT, "outputs/pulse7b/cpsc-test/answers.jsonl"))}
s1_dec = J("outputs/clarity/preds_test.json")["S1"]
dev, test = [i for i in split["dev"] if i in P], [i for i in split["test"] if i in P]
logit = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
X = lambda ids: np.column_stack([logit(np.array([P[i][L] for i in ids])) for L in LABELS])
Xd, Xt = X(dev), X(test)

fig = plt.figure(figsize=(183 * MM, 150 * MM))
outer = fig.add_gridspec(2, 1, height_ratios=[1, 1.08], hspace=0.62)
top = outer[0].subgridspec(1, 3, wspace=0.28)
# ------------------------------------------------------------------ a: three ROC panels
for k, L in enumerate(SHOW):
    ax = fig.add_subplot(top[k])
    y = np.array([L in gold[i] for i in test]); raw = np.array([P[i][L] for i in test])
    s1 = LogisticRegression(C=1.0, max_iter=2000).fit(Xd, np.array([L in gold[i] for i in dev])).predict_proba(Xt)[:, 1]
    ax.plot([0, 1], [0, 1], color=C["grid"], lw=0.8)
    f, t, _ = roc_curve(y, raw); ax.plot(f, t, color=C["ZS"] if False else "#6E6E6E", lw=1.3, label="Raw P(yes)")
    f, t, _ = roc_curve(y, s1); ax.plot(f, t, color=C["S1"], lw=1.3, ls=(0, (4, 1.6)), label="S1 read-out")
    for pred, mk, face, ec, lab in ((b0, "X", "#222222", "#222222", "Generated answer (B0)"), (s1_dec, "o", "white", C["S1"], "S1 decision")):
        p = np.array([L in pred[i] for i in test])
        ax.scatter([(p & ~y).sum() / (~y).sum()], [(p & y).sum() / y.sum()], marker=mk, s=40 if mk == "X" else 36, color=face, edgecolor=ec, linewidth=0.6 if mk == "X" else 1.2, zorder=5, label=lab)
    ax.set_title(f"{L}\nAUROC: raw {roc_auc_score(y, raw):.2f}; S1 {roc_auc_score(y, s1):.2f}", loc="left", fontsize=7.8, linespacing=1.3)
    ax.set_xlim(-0.045, 1); ax.set_ylim(0, 1.02); ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1]); ax.grid(color=C["grid"], lw=0.4); ax.set_axisbelow(True)
    ax.set_xlabel("False-positive rate")
    if k == 0:
        ax.set_ylabel("Sensitivity"); S["panel_label"](ax, "a", x=-0.36, y=1.08)
        handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.975), ncol=4, fontsize=7, frameon=False, handlelength=2.2, columnspacing=1.6)

# ------------------------------------------------------------------ b: held-out macro-F1
bot = outer[1].subgridspec(1, 2, width_ratios=[1.22, 1], wspace=0.32)
ax = fig.add_subplot(bot[0])
bars = [("B0", "Generated answer", C["B0"]), ("ZS", "P(yes) ≥ 0.5", C["ZS"]),
        ("S1", "S1 read-out", C["S1"]), ("PROBE", "Hidden-state probe", C["PROBE"])]
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
ax.set_xticks(x); ax.set_xticklabels(["CPSC", "CSN", "G12EC", "SPH"]); ax.set_ylabel("Macro-F1 (%)")
ax.set_ylim(0, 80); ax.grid(axis="y", color=C["grid"], lw=0.5); ax.set_axisbelow(True)
ax.set_title("Held-out performance", loc="left", fontsize=8.5, pad=26)
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=6.8, columnspacing=1.2, handlelength=1.1, handletextpad=0.4, borderaxespad=0.2)
S["panel_label"](ax, "b", x=-0.15, y=1.2)

# ------------------------------------------------------------------ c: development-pool feature screening
ax = fig.add_subplot(bot[1])
STY = {"cpsc": ("o", "-", "#0F4D92", "CPSC"), "csn": ("s", (0, (4, 1.5)), "#42949E", "CSN"),
       "g12": ("^", (0, (1, 1.2)), "#B64342", "G12EC"), "sph": ("D", (0, (5, 1.5, 1, 1.5)), "#9A4D8E", "SPH")}
XC = 4.9; jit = {"cpsc": -0.18, "csn": -0.06, "g12": 0.06, "sph": 0.18}
for d, _ in S["DATASETS"]:
    st = S["baselines"](d)["stage1_pool_oof_F1"]; mk, ls, col, nm = STY[d]
    ax.plot([0, 1, 2, 3], [st[f"llm_L{L}_last"] for L in (8, 16, 24, 32)], color=col, ls=ls, marker=mk, ms=3.2, lw=1.1, label=nm)
    ax.scatter([XC + jit[d]], [st["clip_patch"]], marker=mk, s=14, color=col, zorder=4)
ax.axvline(4.0, color="#BBBBBB", lw=0.7, ls=(0, (2, 2)))
ax.set_xticks([0, 1, 2, 3, XC]); ax.set_xticklabels(["L8", "L16", "L24", "L32", "CLIP"])
ax.text(1.5, -0.13, "LLM layer (last prompt token)", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7)
ax.text(XC, -0.13, "patch mean", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7)
ax.set_xlim(-0.4, 5.5); ax.set_ylim(20, 80); ax.set_ylabel("Screening macro-F1 (%)")
ax.grid(axis="y", color=C["grid"], lw=0.5); ax.set_axisbelow(True)
ax.set_title("Development-pool feature screening", loc="left", fontsize=8.5, pad=26)
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=6.8, columnspacing=1.2, handlelength=2.0, handletextpad=0.4, borderaxespad=0.2)
S["panel_label"](ax, "c", x=-0.2, y=1.2)
S["save"](fig, "Fig2_knowing_vs_saying")
