"""Figure 2 — Read-outs re-rank cases: ROC analysis and feature ablations (frozen PULSE-7B; v3).
a CPSC test split (n = 1461), three diagnoses: ROC of the raw Yes/No probability (solid) and of the S1 read-out (dashed; logistic
  regression on all nine label log-odds, one fit on the 600 development records, as in clarity/verify_stack.py). Markers: generated
  answer (B0) and the S1 decision (out-of-fold threshold plus the at-least-one-label rule, so it can lie slightly off the S1 curve).
b Cross-label ablation at n = 600: S1 (all label log-odds) minus S1-uni (the label's own log-odds), paired within repetitions.
c COMBO (hidden state plus all label log-odds) minus the hidden-state probe at n = 600, paired within repetitions.
  b, c: mean and 2.5th-97.5th percentiles of 20 repetitions (record order and cross-validation folds differ).
Data: b1_cpsc_{dev,test}.jsonl, pulse7b/cpsc-test answers, preds_test.json, baselines_*.json ("rows")"""
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

fig = plt.figure(figsize=(183 * MM, 128 * MM))
outer = fig.add_gridspec(2, 1, height_ratios=[1, 0.62], hspace=0.62)
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

# ------------------------------------------------------------------ b, c: feature ablations at n = 600
SHORT = {"cpsc": "CPSC", "csn": "CSN", "g12": "G12EC", "sph": "SPH"}
R = {d: {(r["n"], r["rep"], r["method"]): r["F1"] for r in J(f"outputs/clarity/baselines_{d}.json")["rows"]} for d, _ in S["DATASETS"]}
pair = lambda d, a, b: np.array([R[d][(600, k, a)] - R[d][(600, k, b)] for k in range(20)])
bot = outer[1].subgridspec(1, 2, wspace=0.75)
sg = lambda x: f"{x + 0:+.1f}" if round(x, 1) != 0 else "0.0"   # no "-0.0"
PAN = [("S1", "S1-uni", C["S1"], "Cross-label ablation: S1 − S1-uni", "all label log-odds vs the label's own log-odds"),
       ("COMBO", "PROBE-LLM", C["COMBO"], "Adding label log-odds: COMBO − probe", "hidden state + all label log-odds vs hidden state")]
for k, (a, b, col, title, sub) in enumerate(PAN):
    ax = fig.add_subplot(bot[k]); dsets = [d for d, _ in S["DATASETS"]]
    for y, d in enumerate(dsets):
        v = pair(d, a, b); lo, hi = np.percentile(v, [2.5, 97.5])
        ax.plot([lo, hi], [y, y], color=col, lw=1.5, solid_capstyle="butt"); ax.scatter([v.mean()], [y], s=20, color=col, zorder=3)
        ax.text(1.02, y, f"{sg(v.mean())} [{lo:.1f}, {hi:.1f}]".replace("-", "−"), transform=ax.get_yaxis_transform(), va="center", fontsize=6.3)
    ax.axvline(0, color="#333333", lw=0.8, zorder=0)
    ax.set_yticks(range(4)); ax.set_yticklabels([SHORT[d] for d in dsets]); ax.set_ylim(3.6, -0.6); ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False); ax.grid(axis="x", color=C["grid"], lw=0.5); ax.set_axisbelow(True)
    ax.set_xlim(-2, 14); ax.set_xticks([0, 5, 10]); ax.set_xlabel("Paired difference at n = 600 (percentage points)")
    ax.set_title(title, loc="left", fontsize=8.5, pad=16)
    ax.text(0, 1.03, sub, transform=ax.transAxes, fontsize=6.6, color="#555555", va="bottom")
    S["panel_label"](ax, "bc"[k], x=-0.2, y=1.16)
S["save"](fig, "Fig2_knowing_vs_saying")
