"""CPSC learning curve for another base model (default: Qwen2-VL-7B-Instruct; --b1/--b0/--out for others, e.g. MedGemma), same protocol as clarity/learning_curve.py:
fixed split data/splits/cpsc.json (dev pool 600 / test 1461), 20 random draws per n, THR and S1 calibrators,
OOF thresholds, at-least-one rule. Fallback for a label with < 2 positives/negatives: raw P(yes) >= 0.5
(PULSE used PTB-XL thresholds; none exist for Qwen). Also reports B0 (Qwen generation), B1@0.5 and probability AUC."""
import argparse, json, os, runpy
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS = ["NORM", "AF", "I-AVB", "LBBB", "RBBB", "PAC", "PVC", "STD", "STE"]
SIZES = [25, 50, 100, 200, 300, 400, 600]
GRID = np.round(np.arange(0.02, 0.97, 0.01), 2)
ap = argparse.ArgumentParser()
ap.add_argument("--b1", default="outputs/clarity/qwen_b1_cpsc_all.jsonl")
ap.add_argument("--b0", default="outputs/qwen2vl/cpsc-test/answers.jsonl")
ap.add_argument("--out", default="outputs/clarity/qwen_curve_cpsc.json")
args = ap.parse_args()
compute_f1_auc = runpy.run_path(os.path.join(ROOT, "scripts/score_ecgbench.py"), run_name="lib")["compute_f1_auc"]

gold = {x["id"]: [l for l in LABELS if l in x["conversations"][1]["value"]]
        for x in json.load(open(os.path.join(ROOT, "data/ECGBench/cpsc-test.json")))}
split = json.load(open(os.path.join(ROOT, "data/splits/cpsc.json")))
P = {r["id"]: r["probs"] for r in map(json.loads, open(os.path.join(ROOT, args.b1)))}
B0 = {r["question_id"]: [l for l in LABELS if l in r["text"]]
      for r in map(json.loads, open(os.path.join(ROOT, args.b0)))}
pool, test = split["dev"], split["test"]
assert all(i in P and i in B0 for i in pool + test), "Qwen outputs incomplete"


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4); return np.log(p / (1 - p))


X = lambda ids: np.column_stack([logit([P[i][L] for i in ids]) for L in LABELS])
Xte, raw_te = X(test), {L: np.array([P[i][L] for i in test]) for L in LABELS}
best_thr = lambda y, s: float(max(GRID, key=lambda t: f1_score(y, s >= t, zero_division=0)))


def run(method, sample, seed):
    out = {}
    for L in LABELS:
        y = np.array([L in gold[i] for i in sample]); pos, neg = int(y.sum()), int((~y).sum())
        if pos < 2 or neg < 2: out[L] = (raw_te[L], 0.5); continue
        if method == "THR": out[L] = (raw_te[L], best_thr(y, np.array([P[i][L] for i in sample]))); continue
        Xs = X(sample); k = min(5, pos, neg)
        oof = cross_val_predict(LogisticRegression(max_iter=2000), Xs, y, cv=StratifiedKFold(k, shuffle=True, random_state=seed),
                                method="predict_proba")[:, 1]
        out[L] = (LogisticRegression(max_iter=2000).fit(Xs, y).predict_proba(Xte)[:, 1], best_thr(y, oof))
    return out


def score(pred):
    f1, auc, hl = compute_f1_auc([pred[i] for i in test], [gold[i] for i in test]); return 100 * f1, 100 * auc, 100 * hl


def decide(sc):
    pred = {}
    for n, i in enumerate(test):
        lab = [L for L in LABELS if sc[L][0][n] >= sc[L][1]]
        pred[i] = lab or [max(LABELS, key=lambda L: sc[L][0][n] / sc[L][1])]
    return pred


res = {"B0": score({i: B0[i] for i in test}), "B1@0.5": score(decide({L: (raw_te[L], 0.5) for L in LABELS})),
       "prob_AUC_per_label": {L: round(roc_auc_score([L in gold[i] for i in test], raw_te[L]), 3) for L in LABELS},
       "B0_mean_labels_per_answer": float(np.mean([len(B0[i]) for i in test]))}
res["prob_macroAUC"] = float(np.mean(list(res["prob_AUC_per_label"].values())))
rng = np.random.default_rng(42); rows = []
for n in SIZES:
    for r in range(20):
        sample = list(rng.choice(pool, size=n, replace=False))
        for m in ("THR", "S1"):
            f1, auc, hl = score(decide(run(m, sample, r))); rows.append({"n": n, "rep": r, "method": m, "F1": f1, "AUC": auc, "HL": hl})
    print(f"n={n} done", flush=True)
res["curve"] = {f"{m}@{n}": {k: round(float(f(np.array([x['F1'] for x in rows if x['method'] == m and x['n'] == n]))), 1)
                              for k, f in (("F1_mean", np.mean), ("F1_sd", np.std), ("p2.5", lambda v: np.percentile(v, 2.5)),
                                           ("p97.5", lambda v: np.percentile(v, 97.5)))}
                for m in ("THR", "S1") for n in SIZES}
json.dump({"results": res, "rows": rows}, open(os.path.join(ROOT, args.out), "w"), indent=1)
print(json.dumps(res, indent=1))
