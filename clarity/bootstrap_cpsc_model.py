"""Paired bootstrap on the CPSC test split for any base model's B0 / B1 outputs (same protocol as ml_curve.py):
S1@600 = per-label logistic regression on all 9 label logits fitted on the 600 dev records, 5-fold OOF thresholds,
at-least-one rule; ZS = raw P(yes) >= 0.5 with at-least-one. B = 2000 record resamples, default_rng(0).
Usage: python -I clarity/bootstrap_cpsc_model.py --b1 outputs/clarity/mg_b1_cpsc_all.jsonl --b0 outputs/medgemma/cpsc/answers.jsonl --out outputs/clarity/mg_bootstrap_cpsc.json"""
import argparse, json, os, runpy
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS = ["NORM", "AF", "I-AVB", "LBBB", "RBBB", "PAC", "PVC", "STD", "STE"]
ap = argparse.ArgumentParser(); ap.add_argument("--b1", required=True); ap.add_argument("--b0", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
cfa = runpy.run_path(os.path.join(ROOT, "scripts/score_ecgbench.py"), run_name="lib")["compute_f1_auc"]
gold = {x["id"]: [l for l in LABELS if l in x["conversations"][1]["value"]] for x in json.load(open(os.path.join(ROOT, "data/ECGBench/cpsc-test.json")))}
sp = json.load(open(os.path.join(ROOT, "data/splits/cpsc.json"))); dev, test = sp["dev"], sp["test"]
P = {r["id"]: r["probs"] for r in map(json.loads, open(os.path.join(ROOT, a.b1)))}
B0 = {r["question_id"]: [l for l in LABELS if l in r["text"]] for r in map(json.loads, open(os.path.join(ROOT, a.b0)))}
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
X = lambda s: np.column_stack([lg(np.array([P[i][L] for i in s])) for L in LABELS])
GRID = np.round(np.arange(0.02, 0.97, 0.01), 2)
Xd, Xt, sc = X(dev), X(test), {}
for L in LABELS:
    y = np.array([L in gold[i] for i in dev])
    oof = cross_val_predict(LogisticRegression(max_iter=2000), Xd, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    thr = float(max(GRID, key=lambda t: f1_score(y, oof >= t, zero_division=0)))
    sc[L] = (LogisticRegression(max_iter=2000).fit(Xd, y).predict_proba(Xt)[:, 1], thr)


def decide(scores):
    out = {}
    for n, i in enumerate(test):
        lab = [L for L in LABELS if scores[L][0][n] >= scores[L][1]]
        out[i] = lab or [max(LABELS, key=lambda L: scores[L][0][n] / scores[L][1])]
    return out


s1, zs, b0 = decide(sc), decide({L: (np.array([P[i][L] for i in test]), 0.5) for L in LABELS}), {i: B0[i] for i in test}
f = lambda pr, ids: 100 * cfa([pr[i] for i in ids], [gold[i] for i in ids])[0]
rng = np.random.default_rng(0); bs = {"S1@600-B0": [], "S1@600-ZS": [], "ZS-B0": []}
for _ in range(2000):
    ids = [test[j] for j in rng.integers(0, len(test), len(test))]
    a1, az, ab = f(s1, ids), f(zs, ids), f(b0, ids)
    bs["S1@600-B0"].append(a1 - ab); bs["S1@600-ZS"].append(a1 - az); bs["ZS-B0"].append(az - ab)
res = {"point": {"B0": f(b0, test), "ZS": f(zs, test), "S1@600": f(s1, test)},
       "bootstrap": {k: [round(float(np.mean(v)), 2), round(float(np.percentile(v, 2.5)), 2), round(float(np.percentile(v, 97.5)), 2)] for k, v in bs.items()}}
json.dump(res, open(os.path.join(ROOT, a.out), "w"), indent=1); print(json.dumps(res, indent=1))
