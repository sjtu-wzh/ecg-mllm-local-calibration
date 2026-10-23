"""Experiment 1: same-label-budget baselines vs S1 (frozen PULSE-7B), on CPSC (9 labels) and the CSN / G12EC multi-label sets.
Protocol (fixed before any probe result was seen):
  Stage 1 (pool only, no test labels): 5-fold OOF macro-F1 on the 600-record pool for every saved feature
           (8 LLM features, 2 CLIP features) -> pick the best LLM feature and the best CLIP feature per dataset.
  Stage 2 (learning curve): n in {25,50,100,200,300,400,600}, 20 random draws from the pool (same rng as learning_curve.py),
           evaluated on the fixed test split. Methods:
    THR       per-label threshold on raw P(yes)
    S1-uni    per-label logistic regression on its OWN logit only (Platt scaling)
    S1        per-label logistic regression on ALL label logits (the proposed calibration)
    PROBE-LLM / PROBE-CLIP  standardised features -> L2 logistic regression; C in {1e-3,1e-2,1e-1,1} chosen per label by
              inner stratified CV (k = min(3,pos,neg), log-loss) inside the n training records
    COMBO     PROBE-LLM features + all label logits
  Common rules: thresholds from 5-fold OOF predictions (k = min(5,pos,neg)) maximising F1; labels with < 2 positives or
  negatives fall back to raw P(yes) >= 0.5 (probability methods) or predict-negative (probes); at least one label per record.
  Metrics: official macro-F1 (scripts/score_ecgbench.py compute_f1_auc), mean AUROC and AUPRC of the scores.
Output: outputs/clarity/baselines_{data}.json"""
import argparse, json, os, re, runpy, warnings
import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIZES = [25, 50, 100, 200, 300, 400, 600]
REPEATS = 20
GRID = np.round(np.arange(0.02, 0.97, 0.01), 2)
CS = [1e-3, 1e-2, 1e-1, 1.0]
ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True, choices=["cpsc", "csn", "g12", "sph"])
ap.add_argument("--jobs", type=int, default=8); ap.add_argument("--quick", action="store_true", help="2 draws, n in {100,600}")
a = ap.parse_args()
if a.quick: SIZES, REPEATS = [100, 600], 2
cfa = runpy.run_path(os.path.join(ROOT, "scripts/score_ecgbench.py"), run_name="lib")["compute_f1_auc"]
jl = lambda p: [json.loads(l) for l in open(os.path.join(ROOT, p)) if l.strip()]

# ---------------- data
if a.data == "cpsc":
    LABELS = ["NORM", "AF", "I-AVB", "LBBB", "RBBB", "PAC", "PVC", "STD", "STE"]
    gold = {x["id"]: [l for l in LABELS if l in x["conversations"][1]["value"]] for x in json.load(open(os.path.join(ROOT, "data/ECGBench/cpsc-test.json")))}
    sp = json.load(open(os.path.join(ROOT, "data/splits/cpsc.json"))); pool, test = sp["dev"], sp["test"]
    P = {r["id"]: r["probs"] for r in jl("outputs/clarity/b1_cpsc_dev.jsonl") + jl("outputs/clarity/b1_cpsc_test.jsonl")}
    B0 = {r["question_id"]: [l for l in LABELS if l in r["text"]] for r in jl("outputs/pulse7b/cpsc-test/answers.jsonl")}
    feat_file, fallback_zs = "outputs/features/pulse_cpsc.npz", True
else:
    D = json.load(open(os.path.join(ROOT, f"data/derived/{a.data}_ml.json"))); LABELS = list(D["labels"])
    gold = {x["id"]: x["conversations"][1]["value"].split(";") for x in D["records"]}
    sp = json.load(open(os.path.join(ROOT, f"data/splits/{a.data}_ml.json"))); pool, test = sp["pool"], sp["test"]
    P = {r["id"]: r["probs"] for r in jl(f"outputs/clarity/b1_{a.data}ml_all.jsonl")}
    codes = lambda t: [c for c in LABELS if c in set(re.findall(r"[A-Za-z0-9]+", t))]
    B0 = {r["question_id"]: codes(r["text"]) for r in jl(f"outputs/pulse7b/{a.data}-ml/answers.jsonl")}
    feat_file = f"outputs/features/pulse_{a.data}ml.npz"
_npz = np.load(os.path.join(ROOT, feat_file)); F = {k: _npz[k] for k in _npz.files}; _npz.close()  # in memory (picklable for joblib)
fidx = {i: n for n, i in enumerate(F["ids"])}
FEATS = [k for k in F if k != "ids"]
Y = lambda ids: np.array([[L in gold[i] for L in LABELS] for i in ids])
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
LOGIT = lambda ids: np.column_stack([lg(np.array([P[i][L] for i in ids])) for L in LABELS])
RAW = lambda ids: np.column_stack([np.array([P[i][L] for i in ids]) for L in LABELS])
FX = lambda f, ids: F[f][[fidx[i] for i in ids]].astype(np.float32)
Yte, Lte, Rte = Y(test), LOGIT(test), RAW(test)
best_thr = lambda y, s: float(GRID[np.argmax([f1_score(y, s >= t, zero_division=0) for t in GRID])])


def lr_fit_predict(Xtr, y, Xte, C):
    m = LogisticRegression(C=C, max_iter=3000).fit(Xtr, y)
    return m.predict_proba(Xte)[:, 1]


def choose_C(X, y, seed):
    k = min(3, int(y.sum()), int((~y).sum()))
    if k < 2: return 1e-2
    best, bl = CS[0], np.inf
    for C in CS:
        oof = np.zeros(len(y))
        for tr, va in StratifiedKFold(k, shuffle=True, random_state=seed).split(X, y):
            oof[va] = lr_fit_predict(X[tr], y[tr], X[va], C)
        l = log_loss(y, np.clip(oof, 1e-6, 1 - 1e-6))
        if l < bl: best, bl = C, l
    return best


def fit_label(kind, Xtr, y, Xte, seed, standardize):
    """returns (test scores, threshold) for one label; kind 'fixedC' (C=1) or 'cvC'."""
    pos, neg = int(y.sum()), int((~y).sum())
    if standardize:
        sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    C = 1.0 if kind == "fixedC" else choose_C(Xtr, y, seed)
    k = min(5, pos, neg); oof = np.zeros(len(y))
    for tr, va in StratifiedKFold(k, shuffle=True, random_state=seed).split(Xtr, y):
        oof[va] = lr_fit_predict(Xtr[tr], y[tr], Xtr[va], C)
    return lr_fit_predict(Xtr, y, Xte, C), best_thr(y, oof)


def method_scores(method, sample, seed, best):
    Ytr = Y(sample); S = np.zeros_like(Rte); T = np.full(len(LABELS), 0.5)
    for j in range(len(LABELS)):
        y = Ytr[:, j]; pos, neg = int(y.sum()), int((~y).sum())
        if pos < 2 or neg < 2:
            S[:, j], T[j] = (Rte[:, j], 0.5) if method in ("THR", "S1", "S1-uni", "COMBO") else (np.zeros(len(test)), 0.5)
            continue
        if method == "THR":
            S[:, j], T[j] = Rte[:, j], best_thr(y, RAW(sample)[:, j])
        elif method == "S1":
            S[:, j], T[j] = fit_label("fixedC", LOGIT(sample), y, Lte, seed, False)
        elif method == "S1-uni":
            S[:, j], T[j] = fit_label("fixedC", LOGIT(sample)[:, [j]], y, Lte[:, [j]], seed, False)
        elif method == "PROBE-LLM":
            S[:, j], T[j] = fit_label("cvC", FX(best["llm"], sample), y, FX(best["llm"], test), seed, True)
        elif method == "PROBE-CLIP":
            S[:, j], T[j] = fit_label("cvC", FX(best["clip"], sample), y, FX(best["clip"], test), seed, True)
        elif method == "COMBO":
            Xtr = np.hstack([FX(best["llm"], sample), LOGIT(sample)]); Xte = np.hstack([FX(best["llm"], test), Lte])
            S[:, j], T[j] = fit_label("cvC", Xtr, y, Xte, seed, True)
    return S, T


def evaluate(S, T):
    Pr = S >= T[None, :]
    empty = ~Pr.any(1); Pr[empty, np.argmax(S[empty] / T[None, :], 1)] = True
    pred = [[LABELS[j] for j in np.where(r)[0]] for r in Pr]
    f1 = 100 * cfa(pred, [gold[i] for i in test])[0]
    au = [roc_auc_score(Yte[:, j], S[:, j]) for j in range(len(LABELS)) if 0 < Yte[:, j].sum() < len(test)]
    ap_ = [average_precision_score(Yte[:, j], S[:, j]) for j in range(len(LABELS)) if Yte[:, j].sum() > 0]
    return {"F1": f1, "AUROC": 100 * float(np.mean(au)), "AUPRC": 100 * float(np.mean(ap_))}


# ---------------- Stage 1: feature selection on the pool only
def pool_oof_f1(f):
    X, Yp = FX(f, pool), Y(pool); pred = np.zeros_like(Yp)
    for j in range(len(LABELS)):
        y = Yp[:, j]
        if min(y.sum(), (~y).sum()) < 5: continue
        oof = np.zeros(len(y)); sc = StandardScaler()
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
            s = sc.fit(X[tr]); C = choose_C(s.transform(X[tr]), y[tr], 0)
            oof[va] = lr_fit_predict(s.transform(X[tr]), y[tr], s.transform(X[va]), C)
        pred[:, j] = oof >= best_thr(y, oof)  # optimistic threshold, identical for all features (used for ranking only)
    return float(np.mean([f1_score(Yp[:, j], pred[:, j], zero_division=0) for j in range(len(LABELS))])) * 100


stage1 = dict(zip(FEATS, Parallel(n_jobs=a.jobs)(delayed(pool_oof_f1)(f) for f in FEATS)))
best = {"llm": max([f for f in FEATS if f.startswith("llm")], key=stage1.get), "clip": max([f for f in FEATS if f.startswith("clip")], key=stage1.get)}
print("stage 1 (pool OOF macro-F1):", {k: round(v, 1) for k, v in sorted(stage1.items(), key=lambda kv: -kv[1])}, "->", best, flush=True)

# ---------------- Stage 2: learning curves
METHODS = ["THR", "S1-uni", "S1", "PROBE-LLM", "PROBE-CLIP", "COMBO"]
rng = np.random.default_rng(42)
draws = [(n, r, list(rng.choice(pool, size=n, replace=False))) for n in SIZES for r in range(REPEATS)]


def run_draw(n, r, sample):
    return [{"n": n, "rep": r, "method": m, **evaluate(*method_scores(m, sample, r, best))} for m in METHODS]


rows = [x for res in Parallel(n_jobs=a.jobs, verbose=5)(delayed(run_draw)(*d) for d in draws) for x in res]
B0m = np.array([[float(L in B0[i]) for L in LABELS] for i in test])  # B0 is scored as generated (no at-least-one rule)
ref = {"B0": {"F1": 100 * cfa([B0[i] for i in test], [gold[i] for i in test])[0],
              "AUROC": 100 * float(np.mean([roc_auc_score(Yte[:, j], B0m[:, j]) for j in range(len(LABELS)) if 0 < Yte[:, j].sum() < len(test)]))},
       "ZS@0.5": evaluate(Rte, np.full(len(LABELS), 0.5))}
summ = {}
for m in METHODS:
    for n in SIZES:
        v = [x for x in rows if x["method"] == m and x["n"] == n]
        summ[f"{m}@{n}"] = {k: [round(float(np.mean([x[k] for x in v])), 2), round(float(np.percentile([x[k] for x in v], 2.5)), 2),
                                round(float(np.percentile([x[k] for x in v], 97.5)), 2)] for k in ("F1", "AUROC", "AUPRC")}
paired = {}
for other in ("PROBE-LLM", "PROBE-CLIP", "S1-uni", "COMBO"):
    for n in SIZES:
        d = [next(x["F1"] for x in rows if x["method"] == "S1" and x["n"] == n and x["rep"] == r) -
             next(x["F1"] for x in rows if x["method"] == other and x["n"] == n and x["rep"] == r) for r in range(REPEATS)]
        paired[f"S1-{other}@{n}"] = [round(float(np.mean(d)), 2), round(float(np.percentile(d, 2.5)), 2), round(float(np.percentile(d, 97.5)), 2),
                                     int(sum(x > 0 for x in d))]
out = {"data": a.data, "labels": LABELS, "n_pool": len(pool), "n_test": len(test), "stage1_pool_oof_F1": stage1, "selected": best,
       "reference": ref, "curve": summ, "paired_F1_S1_minus_other [mean, p2.5, p97.5, draws S1 better]": paired}
json.dump({"results": out, "rows": rows}, open(os.path.join(ROOT, f"outputs/clarity/baselines_{a.data}{'_quick' if a.quick else ''}.json"), "w"), indent=1)
print(json.dumps({k: out[k] for k in ("selected", "reference")}, indent=1))
for k, v in summ.items(): print(f"{k:16s} F1 {v['F1']}  AUROC {v['AUROC'][0]}  AUPRC {v['AUPRC'][0]}")
for k, v in paired.items(): print(k, v)
