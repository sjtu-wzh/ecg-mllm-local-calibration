"""Saved, re-runnable versions of statistics that were first computed interactively (CPSC, PULSE-7B).
  A  paired bootstrap of macro-F1 differences      -> outputs/clarity/bootstrap.txt, bootstrap_transfer.txt
  B  control: the S1 regression on B0's 0/1 outputs -> stats_cpsc.json["control_b0_regression"]
  C  S1 coefficients (fit on the 600 dev records)    -> stats_cpsc.json["s1_coefficients"]
  D  clinical translation, B0 vs S1@600 over 20 CV fold orders (S1 is the dev-selectable headline;
     the earlier single-run S3 translation is superseded)   -> stats_cpsc.json["clinical"], clinical_s1.txt
Bootstrap: resample test records with replacement, B = 1000, numpy default_rng(0), two-sided p = 2 min(P(d<=0), P(d>=0)).
"""
import json, os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs/clarity")
LABELS = ["NORM", "AF", "I-AVB", "LBBB", "RBBB", "PAC", "PVC", "STD", "STE"]
J = lambda p: json.load(open(os.path.join(ROOT, p)))
gold = {x["id"]: [l for l in LABELS if l in x["conversations"][1]["value"]] for x in J("data/ECGBench/cpsc-test.json")}
split = J("data/splits/cpsc.json"); dev, test = split["dev"], split["test"]
b0 = {r["question_id"]: [t for t in LABELS if t in r["text"]]
      for r in map(json.loads, open(os.path.join(ROOT, "outputs/pulse7b/cpsc-test/answers.jsonl")))}
res = {}


def mat(pred, ids):
    return np.array([[L in pred[i] for L in LABELS] for i in ids])


def macro_f1(G, M, idx):
    return np.mean([f1_score(G[idx, j], M[idx, j], zero_division=0) for j in range(len(LABELS))])


def paired_bootstrap(preds, ids, pairs, B=1000, seed=0):
    G = mat(gold, ids); M = {v: mat(preds[v], ids) for v in {x for p in pairs for x in p}}
    rng = np.random.default_rng(seed); d = {p: [] for p in pairs}; n = len(ids)
    for _ in range(B):
        idx = rng.integers(0, n, n); f = {v: macro_f1(G, M[v], idx) for v in M}
        for a, b in pairs: d[(a, b)].append(100 * (f[a] - f[b]))
    out = {}
    for (a, b), x in d.items():
        x = np.array(x); lo, hi = np.percentile(x, [2.5, 97.5]); p = 2 * min((x <= 0).mean(), (x >= 0).mean())
        out[f"{a} - {b}"] = [round(float(x.mean()), 2), round(float(lo), 2), round(float(hi), 2), round(float(p), 3)]
    return out


# ---------------- A. bootstrap
pt = J("outputs/clarity/preds_test.json"); ids = list(pt["B0"])
A1 = paired_bootstrap(pt, ids, [("S1", "B0"), ("S3", "S1"), ("S4", "S1"), ("S3", "S4"), ("S3", "B0")])
lines = [f"paired bootstrap on CPSC test (n={len(ids)}, B=1000), macro-F1 difference"] + \
        [f"  {k}: {v[0]:+.1f}  95% CI [{v[1]:+.1f}, {v[2]:+.1f}]  p~{v[3]:.3f}" for k, v in A1.items()]
open(os.path.join(OUT, "bootstrap.txt"), "w").write("\n".join(lines) + "\n")
ptr = J("outputs/clarity/preds_transfer.json"); B0k = "B0 (official generation)"; lines = []; A2 = {}
for s in ("cpsc_test", "cpsc_all"):
    ids_s = list(ptr[s][B0k]); A2[s] = paired_bootstrap(ptr[s], ids_s, [(v, B0k) for v in ("B1@src", "B1@0.5", "S1", "S3")])
    lines.append(f"{s} (n={len(ids_s)}): macro-F1 difference vs B0, paired bootstrap B=1000")
    lines += [f"  {k.split(' - ')[0]:8s} {v[0]:+5.1f}  95% CI [{v[1]:+.1f}, {v[2]:+.1f}]  p~{v[3]:.3f}" for k, v in A2[s].items()]
open(os.path.join(OUT, "bootstrap_transfer.txt"), "w").write("\n".join(lines) + "\n")
res["bootstrap_fewshot"], res["bootstrap_transfer"] = A1, A2

# ---------------- shared S1 pieces
P = {r["id"]: r["probs"] for f in ("b1_cpsc_dev.jsonl", "b1_cpsc_test.jsonl") for r in map(json.loads, open(os.path.join(OUT, f)))}
logit = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
X = lambda s: np.column_stack([logit(np.array([P[i][L] for i in s])) for L in LABELS])
GRID = np.round(np.arange(0.02, 0.97, 0.01), 2)
best_thr = lambda y, s: float(max(GRID, key=lambda t: f1_score(y, s >= t, zero_division=0)))
Yd = {L: np.array([L in gold[i] for i in dev]) for L in LABELS}


def fit_predict(Xd, Xt, seed):
    """S1 protocol: per-label LR (C=1) on dev, 5-fold OOF threshold, at-least-one rule on test."""
    sc = {}
    for L in LABELS:
        oof = cross_val_predict(LogisticRegression(max_iter=2000), Xd, Yd[L], method="predict_proba",
                                cv=StratifiedKFold(5, shuffle=True, random_state=seed))[:, 1]
        sc[L] = (LogisticRegression(max_iter=2000).fit(Xd, Yd[L]).predict_proba(Xt)[:, 1], best_thr(Yd[L], oof))
    pred = {}
    for n, i in enumerate(test):
        lab = [L for L in LABELS if sc[L][0][n] >= sc[L][1]]
        pred[i] = lab or [max(LABELS, key=lambda L: sc[L][0][n] / sc[L][1])]
    return pred


# ---------------- B. control: same regression on B0 0/1 outputs
Xb = lambda s: np.array([[float(L in b0[i]) for L in LABELS] for i in s])
ctrl = fit_predict(Xb(dev), Xb(test), seed=0)
f = lambda pr: round(100 * np.mean([f1_score([L in gold[i] for i in test], [L in pr[i] for i in test], zero_division=0) for L in LABELS]), 1)
res["control_b0_regression"] = {"macroF1": f(ctrl), "B0": f(b0), "note": "S1 protocol with B0 0/1 indicators as features"}

# ---------------- C. S1 coefficients
Xd = X(dev)
res["s1_coefficients"] = {L: dict(zip(["intercept"] + LABELS, np.round(np.r_[m.intercept_, m.coef_[0]], 3).tolist()))
                          for L in LABELS for m in [LogisticRegression(max_iter=2000).fit(Xd, Yd[L])]}

# ---------------- D. clinical translation, S1@600 over 20 CV fold orders
Xt = X(test); n = len(test)


def op(pred, L):
    y = np.array([L in gold[i] for i in test]); p = np.array([L in pred[i] for i in test])
    tp, fp, fn, tn = (y & p).sum(), (~y & p).sum(), (y & ~p).sum(), (~y & ~p).sum()
    return {"sens": tp / (tp + fn), "spec": tn / (tn + fp), "ppv": tp / (tp + fp) if tp + fp else np.nan, "tp": tp, "fp": fp}


def wrongly_normal(pred):
    a = sum("NORM" in pred[i] and "NORM" not in gold[i] for i in test)          # NORM called on an abnormal record
    b = sum(pred[i] == ["NORM"] and "NORM" not in gold[i] for i in test)        # abnormal record called ONLY normal
    return 1000 * a / n, 1000 * b / n


reps = [fit_predict(Xd, Xt, seed=r) for r in range(20)]
clin = {"definition_wrongly_normal": {"A": "prediction contains NORM, gold has no NORM",
                                      "B": "prediction is exactly [NORM], gold has no NORM"},
        "per_1000_basis": f"{n} CPSC test records", "labels": {}}
wn0 = wrongly_normal(b0); wn1 = np.array([wrongly_normal(p) for p in reps])
clin["wrongly_normal_per_1000"] = {"B0": [round(wn0[0], 1), round(wn0[1], 1)],
                                   "S1_mean": wn1.mean(0).round(1).tolist(), "S1_min": wn1.min(0).round(1).tolist(), "S1_max": wn1.max(0).round(1).tolist()}
for L in LABELS:
    o0 = op(b0, L); o1 = [op(p, L) for p in reps]
    agg = lambda k: [round(float(np.nanmean([o[k] for o in o1])), 3), round(float(np.nanmin([o[k] for o in o1])), 3), round(float(np.nanmax([o[k] for o in o1])), 3)]
    clin["labels"][L] = {"prevalence": int(sum(L in gold[i] for i in test)),
                         "B0": {k: round(float(o0[k]), 3) for k in ("sens", "spec", "ppv")},
                         "S1 [mean, min, max]": {k: agg(k) for k in ("sens", "spec", "ppv")},
                         "delta_TP_per_1000": round(1000 * (np.mean([o["tp"] for o in o1]) - o0["tp"]) / n, 1),
                         "delta_FP_per_1000": round(1000 * (np.mean([o["fp"] for o in o1]) - o0["fp"]) / n, 1)}
res["clinical"] = clin
res["s1_macroF1_over_20_fold_orders"] = [f(p) for p in reps]
json.dump(res, open(os.path.join(OUT, "stats_cpsc.json"), "w"), indent=1, default=float)

L = [f"Clinical translation on CPSC test (n={n}); B0 vs S1@600 (mean [min-max] over 20 CV fold orders); per 1000 ECGs",
     f"Wrongly normal (A: NORM called on abnormal record): B0 {wn0[0]:.1f}  S1 {wn1[:,0].mean():.1f} [{wn1[:,0].min():.1f}-{wn1[:,0].max():.1f}]",
     f"Wrongly normal (B: abnormal record called only NORM): B0 {wn0[1]:.1f}  S1 {wn1[:,1].mean():.1f} [{wn1[:,1].min():.1f}-{wn1[:,1].max():.1f}]",
     f"{'label':6s} {'prev':>4s} | {'sens B0':>7s} {'S1':>17s} | {'spec B0':>7s} {'S1':>17s} | {'PPV B0':>6s} {'S1':>17s} | dTP dFP /1000"]
for lab, c in clin["labels"].items():
    s = c["S1 [mean, min, max]"]; fm = lambda v: f"{v[0]:.2f} [{v[1]:.2f}-{v[2]:.2f}]"
    L.append(f"{lab:6s} {c['prevalence']:4d} | {c['B0']['sens']:7.2f} {fm(s['sens']):>17s} | {c['B0']['spec']:7.2f} {fm(s['spec']):>17s} | "
             f"{c['B0']['ppv']:6.2f} {fm(s['ppv']):>17s} | {c['delta_TP_per_1000']:+.1f} {c['delta_FP_per_1000']:+.1f}")
open(os.path.join(OUT, "clinical_s1.txt"), "w").write("\n".join(L) + "\n")
print(open(os.path.join(OUT, "bootstrap.txt")).read()); print(open(os.path.join(OUT, "bootstrap_transfer.txt")).read())
print("control:", res["control_b0_regression"]); print("S1 F1 over fold orders:", res["s1_macroF1_over_20_fold_orders"])
print("coef I-AVB self:", res["s1_coefficients"]["I-AVB"]["I-AVB"], " STD<-I-AVB:", res["s1_coefficients"]["STD"]["I-AVB"],
      " NORM self/STE:", res["s1_coefficients"]["NORM"]["NORM"], res["s1_coefficients"]["NORM"]["STE"])
print("\n".join(L))
