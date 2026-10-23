"""Learning curve on the multi-label versions of G12EC / CSN (data/derived/{g12,csn}_ml.json), out-of-domain for PULSE.
Written before the B0/B1 outputs existed. Same protocol as clarity/learning_curve.py (CPSC):
  split   : random 600 pool / rest test (seed 42; redraw until every label has >= 5 positives in pool and test)
  B0      : PULSE generation for the CPSC-style prompt with this label list, parsed by EXACT code tokens
            (substring match would read 'AF' inside 'AFL')
  ZS      : raw P(yes) >= 0.5, at-least-one rule (no labels)
  BLIND   : image-blind ceiling = predict every label for every record (best constant macro-F1 per label)
  THR, S1 : 20 random draws per n; S1 = per-label logistic regression (C = 1) on all label logits;
            OOF thresholds (k = min(5, pos, neg)); fallback raw P >= 0.5 if < 2 positives/negatives; at-least-one rule
Metric: official macro-F1 (scripts/score_ecgbench.py compute_f1_auc). Paired bootstrap (B = 2000) for S1@600 vs B0 / ZS."""
import argparse, json, os, re, runpy
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIZES = [25, 50, 100, 200, 300, 400, 600]
GRID = np.round(np.arange(0.02, 0.97, 0.01), 2)
ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True, choices=["g12", "csn", "sph"])
ap.add_argument("--model", default="pulse", choices=["pulse", "medgemma"]); a = ap.parse_args()
B0_PATH = {"pulse": f"outputs/pulse7b/{a.data}-ml/answers.jsonl", "medgemma": f"outputs/medgemma/{a.data}ml/answers.jsonl"}[a.model]
B1_PATH = {"pulse": f"outputs/clarity/b1_{a.data}ml_all.jsonl", "medgemma": f"outputs/clarity/mg_b1_{a.data}ml_all.jsonl"}[a.model]
OUT_PATH = {"pulse": f"outputs/clarity/ml_curve_{a.data}.json", "medgemma": f"outputs/clarity/mg_ml_curve_{a.data}.json"}[a.model]
compute_f1_auc = runpy.run_path(os.path.join(ROOT, "scripts/score_ecgbench.py"), run_name="lib")["compute_f1_auc"]

D = json.load(open(os.path.join(ROOT, f"data/derived/{a.data}_ml.json")))
LABELS = list(D["labels"]); LS = set(LABELS)
gold = {x["id"]: x["conversations"][1]["value"].split(";") for x in D["records"]}
codes = lambda t: [c for c in LABELS if c in set(re.findall(r"[A-Za-z0-9]+", t))]
B0 = {r["question_id"]: codes(r["text"]) for r in map(json.loads, open(os.path.join(ROOT, B0_PATH)))}
P = {r["id"]: r["probs"] for r in map(json.loads, open(os.path.join(ROOT, B1_PATH)))}
ids = sorted(gold); assert all(i in P and i in B0 for i in ids), "outputs incomplete"

sp_path = os.path.join(ROOT, f"data/splits/{a.data}_ml.json")
if not os.path.exists(sp_path):
    k = 0
    while True:
        perm = np.random.default_rng(42 + k).permutation(ids); pool, test = sorted(perm[:600]), sorted(perm[600:])
        if all(sum(L in gold[i] for i in s) >= 5 for L in LABELS for s in (pool, test)): break
        k += 1
    json.dump({"pool": pool, "test": test, "redraws": k}, open(sp_path, "w"), indent=1)
sp = json.load(open(sp_path)); pool, test = sp["pool"], sp["test"]


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4); return np.log(p / (1 - p))


X = lambda s: np.column_stack([logit([P[i][L] for i in s]) for L in LABELS])
Xte, raw = X(test), {L: np.array([P[i][L] for i in test]) for L in LABELS}
best_thr = lambda y, s: float(max(GRID, key=lambda t: f1_score(y, s >= t, zero_division=0)))


def fit(method, sample, seed):
    out = {}
    for L in LABELS:
        y = np.array([L in gold[i] for i in sample]); pos, neg = int(y.sum()), int((~y).sum())
        if pos < 2 or neg < 2: out[L] = (raw[L], 0.5); continue
        if method == "THR": out[L] = (raw[L], best_thr(y, np.array([P[i][L] for i in sample]))); continue
        Xs = X(sample)
        oof = cross_val_predict(LogisticRegression(max_iter=2000), Xs, y, method="predict_proba",
                                cv=StratifiedKFold(min(5, pos, neg), shuffle=True, random_state=seed))[:, 1]
        out[L] = (LogisticRegression(max_iter=2000).fit(Xs, y).predict_proba(Xte)[:, 1], best_thr(y, oof))
    return out


def decide(sc):
    pred = {}
    for n, i in enumerate(test):
        lab = [L for L in LABELS if sc[L][0][n] >= sc[L][1]]
        pred[i] = lab or [max(LABELS, key=lambda L: sc[L][0][n] / sc[L][1])]
    return pred


def metrics(pred):
    f1, auc, hl = compute_f1_auc([pred[i] for i in test], [gold[i] for i in test]); return [100 * f1, 100 * auc, 100 * hl]


per_label = lambda pred: {L: round(100 * f1_score([L in gold[i] for i in test], [L in pred[i] for i in test], zero_division=0), 1) for L in LABELS}
b0 = {i: B0[i] for i in test}; zs = decide({L: (raw[L], 0.5) for L in LABELS}); blind = {i: list(LABELS) for i in test}
res = {"data": a.data, "n_labels": len(LABELS), "n_pool": len(pool), "n_test": len(test), "split_redraws": sp["redraws"],
       "B0": metrics(b0), "ZS": metrics(zs), "BLIND": metrics(blind),
       "B0_empty": sum(not v for v in b0.values()), "B0_labels_per_answer": float(np.mean([len(v) for v in b0.values()])),
       "prob_AUC": {L: round(roc_auc_score([L in gold[i] for i in test], raw[L]), 3) for L in LABELS}}
res["prob_macroAUC"] = float(np.mean(list(res["prob_AUC"].values())))
rng = np.random.default_rng(42); rows = []
for n in SIZES:
    for r in range(20):
        sample = list(rng.choice(pool, size=n, replace=False))
        for m in ("THR", "S1"):
            f1, auc, hl = metrics(decide(fit(m, sample, r))); rows.append({"n": n, "rep": r, "method": m, "F1": f1, "AUC": auc, "HL": hl})
    print(f"n={n} done", flush=True)
res["curve"] = {}
for m in ("THR", "S1"):
    for n in SIZES:
        v = np.array([x["F1"] for x in rows if x["method"] == m and x["n"] == n])
        res["curve"][f"{m}@{n}"] = {"F1_mean": round(v.mean(), 1), "F1_sd": round(v.std(), 1), "p2.5": round(np.percentile(v, 2.5), 1),
                                    "p97.5": round(np.percentile(v, 97.5), 1), "draws_beating_B0": int((v > res["B0"][0]).sum())}
s1 = decide(fit("S1", pool, 0))
res["per_label_F1"] = {"B0": per_label(b0), "ZS": per_label(zs), "S1@600": per_label(s1)}
G = {i: set(gold[i]) for i in test}


def boot(pa, pb, B=2000):
    brng = np.random.default_rng(0); out = []
    for _ in range(B):
        k = brng.integers(0, len(test), len(test)); s = [test[j] for j in k]
        fa = compute_f1_auc([pa[i] for i in s], [gold[i] for i in s])[0]; fb = compute_f1_auc([pb[i] for i in s], [gold[i] for i in s])[0]
        out.append(100 * (fa - fb))
    return [round(float(np.mean(out)), 2), round(float(np.percentile(out, 2.5)), 2), round(float(np.percentile(out, 97.5)), 2)]


res["bootstrap"] = {"S1@600-B0": boot(s1, b0), "S1@600-ZS": boot(s1, zs), "ZS-B0": boot(zs, b0)}
json.dump({"results": res, "rows": rows}, open(os.path.join(ROOT, OUT_PATH), "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "per_label_F1"}, indent=1))
