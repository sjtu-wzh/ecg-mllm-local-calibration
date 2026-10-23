"""CODE-15% learning curve (in-domain for PULSE). ECGBench's code15-test mixes two tasks on disjoint images:
  bin   : 700 'normal vs abnormal' questions (gold NORM / ABNORMAL)
  multi : 700 multi-label questions over 1dAVb, RBBB, LBBB, SB, ST, AF
Split (fixed, stratified, per task): 300 dev-pool / 400 test. A budget of n labels is split evenly over the tasks.
Metric: the official combined macro-F1/AUC/HL over all test records (classes NORM, ABNORMAL + 6 labels),
plus per-task numbers. Every image has 7 Yes/No probabilities (NORM + 6 labels) from clarity/label_prob.py.
Pre-registered rules: THR = per-class threshold on the raw probability; S1 = logistic regression on all 7 logits;
thresholds from out-of-fold predictions; a class with < 2 positives or negatives falls back to raw probability >= 0.5;
multi-task records get at least one label; LR C = 1.0; 20 random draws per n.
S1B0 (exploratory, added after seeing THR/S1): S1 features + the official generated answer (B0) as 0/1 features."""
import collections, json, os, runpy
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAB6 = ["1dAVb", "RBBB", "LBBB", "SB", "ST", "AF"]
Q7 = ["NORM"] + LAB6
SIZES = [25, 50, 100, 200, 300, 400, 600]
REPEATS = 20
GRID = np.round(np.arange(0.02, 0.97, 0.01), 2)
compute_f1_auc = runpy.run_path(os.path.join(ROOT, "scripts/score_ecgbench.py"), run_name="lib")["compute_f1_auc"]

# ---- gold, task, split ----
data = json.load(open(os.path.join(ROOT, "data/ECGBench/code15-test.json")))
gold, task = {}, {}
for x in data:
    v = x["conversations"][1]["value"]
    if v in ("NORM", "ABNORMAL"):
        gold[x["id"]], task[x["id"]] = [v], "bin"
    else:
        gold[x["id"]], task[x["id"]] = [L for L in LAB6 if L in v], "multi"

split_path = os.path.join(ROOT, "data/splits/code15.json")
if not os.path.exists(split_path):
    sp = {"dev": [], "test": []}
    for t in ("bin", "multi"):
        ids = sorted(i for i in task if task[i] == t)
        strata = [";".join(gold[i]) for i in ids]
        c = collections.Counter(strata)
        strata = [s if c[s] >= 2 else "rare" for s in strata]
        dev, test = train_test_split(ids, train_size=300, stratify=strata, random_state=42)
        sp["dev"] += sorted(dev); sp["test"] += sorted(test)
    json.dump(sp, open(split_path, "w"), indent=1)
sp = json.load(open(split_path))
pool = {t: [i for i in sp["dev"] if task[i] == t] for t in ("bin", "multi")}
test = sp["test"]
test_t = {t: [i for i in test if task[i] == t] for t in ("bin", "multi")}

# ---- B0: official generation, parsed exactly like evaluate_ecgbench.eval_code15_test ----
def parse_b0(text):
    if "Answer:" in text:
        text = text.split("Answer:")[-1]
    if "NORM" in text and "ABNORMAL" not in text:
        return ["NORM"] + [l for l in LAB6 if l in text]
    if "ABNORMAL" in text:
        return ["ABNORMAL"] + [l for l in LAB6 if l in text]
    return [l for l in LAB6 if l in text]


b0 = {r["question_id"]: parse_b0(r["text"])
      for r in map(json.loads, open(os.path.join(ROOT, "outputs/pulse7b/code15-test/answers.jsonl")))}
P = {r["id"]: r["probs"] for r in map(json.loads, open(os.path.join(ROOT, "outputs/clarity/b1_code15_all.jsonl")))}


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


X = lambda ids: np.column_stack([logit([P[i][q] for i in ids]) for q in Q7])
XB = lambda ids: np.hstack([X(ids), np.array([[float(c in b0[i]) for c in Q7] for i in ids])])
FEATS = {"S1": X, "S1B0": XB}
Xtest = {m: {t: f(test_t[t]) for t in test_t} for m, f in FEATS.items()}


def best_thr(y, s):
    return float(max(GRID, key=lambda t: f1_score(y, s >= t, zero_division=0)))


def fit_class(method, ids, y, q, Xte, seed):
    """Returns (test scores, threshold) for one binary target q."""
    raw_te = np.array([P[i][q] for i in test_t["bin" if q == "NORM" else "multi"]])
    pos, neg = int(y.sum()), int((~y).sum())
    if pos < 2 or neg < 2:
        return raw_te, 0.5
    if method == "THR":
        return raw_te, best_thr(y, np.array([P[i][q] for i in ids]))
    Xs = FEATS[method](ids); k = min(5, pos, neg)
    oof = cross_val_predict(LogisticRegression(max_iter=2000), Xs, y,
                            cv=StratifiedKFold(k, shuffle=True, random_state=seed), method="predict_proba")[:, 1]
    return LogisticRegression(max_iter=2000).fit(Xs, y).predict_proba(Xte)[:, 1], best_thr(y, oof)


def predict(method, s_bin, s_multi, seed):
    """Predictions for every test record, from labeled samples of each task."""
    pred = {}
    yb = np.array(["NORM" in gold[i] for i in s_bin])
    Xte = Xtest.get(method, Xtest["S1"])
    sc, thr = fit_class(method, s_bin, yb, "NORM", Xte["bin"], seed)
    for n, i in enumerate(test_t["bin"]):
        pred[i] = ["NORM"] if sc[n] >= thr else ["ABNORMAL"]
    per = {L: fit_class(method, s_multi, np.array([L in gold[i] for i in s_multi]), L, Xte["multi"], seed) for L in LAB6}
    for n, i in enumerate(test_t["multi"]):
        lab = [L for L in LAB6 if per[L][0][n] >= per[L][1]]
        pred[i] = lab or [max(LAB6, key=lambda L: per[L][0][n] / per[L][1])]
    return pred


def metrics(pred):
    f1, auc, hl = compute_f1_auc([pred[i] for i in test], [gold[i] for i in test])
    acc_bin = np.mean([pred[i] == gold[i] for i in test_t["bin"]])
    mf1 = np.mean([f1_score([L in gold[i] for i in test_t["multi"]], [L in pred[i] for i in test_t["multi"]],
                            zero_division=0) for L in LAB6])
    return {"F1": 100 * f1, "AUC": 100 * auc, "HL": 100 * hl, "bin_acc": 100 * acc_bin, "multi_F1": 100 * mf1}


res = {"split": {"dev_pool": {t: len(v) for t, v in pool.items()}, "test": {t: len(v) for t, v in test_t.items()}}}
res["B0"] = metrics({i: b0[i] for i in test})
zs = {}
for n, i in enumerate(test_t["bin"]):
    zs[i] = ["NORM"] if P[i]["NORM"] >= 0.5 else ["ABNORMAL"]
for i in test_t["multi"]:
    lab = [L for L in LAB6 if P[i][L] >= 0.5]
    zs[i] = lab or [max(LAB6, key=lambda L: P[i][L])]
res["B1@0.5 (zero-shot)"] = metrics(zs)

rng = np.random.default_rng(42)
rows = []
for n in SIZES:
    nb = n // 2; nm = n - nb
    for r in range(REPEATS):
        s_bin = list(rng.choice(pool["bin"], size=nb, replace=False))
        s_multi = list(rng.choice(pool["multi"], size=nm, replace=False))
        for m in ("THR", "S1", "S1B0"):
            rows.append({"n": n, "rep": r, "method": m, **metrics(predict(m, s_bin, s_multi, seed=r))})
    print(f"n={n} done", flush=True)

summ = {}
for m in ("THR", "S1", "S1B0"):
    for n in SIZES:
        v = [x for x in rows if x["method"] == m and x["n"] == n]
        f = np.array([x["F1"] for x in v])
        summ[f"{m}@{n}"] = {"F1_mean": round(f.mean(), 1), "F1_sd": round(f.std(), 1),
                            "F1_p2.5": round(np.percentile(f, 2.5), 1), "F1_p97.5": round(np.percentile(f, 97.5), 1),
                            **{k: round(np.mean([x[k] for x in v]), 1) for k in ("AUC", "HL", "bin_acc", "multi_F1")}}
res["curve"] = summ
json.dump({"results": res, "rows": rows}, open(os.path.join(ROOT, "outputs/clarity/code15_curve.json"), "w"), indent=1)

print("split:", res["split"])
for k in ("B0", "B1@0.5 (zero-shot)"):
    r = res[k]
    print(f"{k:22s} F1 {r['F1']:.1f} AUC {r['AUC']:.1f} HL {r['HL']:.1f} | bin acc {r['bin_acc']:.1f} | multi F1 {r['multi_F1']:.1f}")
print(f"{'method@n':9s} {'F1 mean±sd':>12s} {'[2.5,97.5]':>14s} {'AUC':>5s} {'HL':>5s} {'binAcc':>6s} {'multiF1':>7s}")
for k, s in summ.items():
    print(f"{k:9s} {s['F1_mean']:6.1f}±{s['F1_sd']:<4.1f} [{s['F1_p2.5']:5.1f},{s['F1_p97.5']:5.1f}] "
          f"{s['AUC']:5.1f} {s['HL']:5.1f} {s['bin_acc']:6.1f} {s['multi_F1']:7.1f}")
