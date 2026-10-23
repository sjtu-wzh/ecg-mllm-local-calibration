"""Experiment 4: calibration and clinical utility of the read-outs of frozen PULSE-7B (CPSC and the pre-registered SPH site).
Read-outs scored on the fixed test split, all fitted on the full 600-record pool:
  RAW   raw Yes/No probability P(yes) (no fitting)
  S1    per-label logistic regression on all label logits (C = 1)
  PROBE per-label L2 logistic regression on the pool-selected PULSE hidden state (standardised; C by inner CV, as baselines.py)
  B0    generated answer (binary decision; one operating point)
Outputs (outputs/clarity/clinical_{data}.json):
  calibration  ECE (10 equal-width bins), Brier and AUROC per label and macro; pooled reliability curve (label x record pairs)
  dca          net benefit NB(pt) = TP/N - FP/N * pt/(1-pt), pt in 0.01..0.50, for key labels; treat-all / treat-none
  operating    threshold reaching sensitivity >= 0.90 on pool out-of-fold predictions, applied to test: sens, spec, PPV, NPV
               with paired bootstrap 95% CI (B = 1000)
  subgroups    (SPH only) macro-F1 of B0 and S1@600 decisions by sex and age group, with bootstrap CI of the difference"""
import argparse, json, os, re, runpy, warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True, choices=["cpsc", "sph"]); a = ap.parse_args()
J = lambda p: json.load(open(os.path.join(ROOT, p)))
jl = lambda p: [json.loads(l) for l in open(os.path.join(ROOT, p)) if l.strip()]
cfa = runpy.run_path(os.path.join(ROOT, "scripts/score_ecgbench.py"), run_name="lib")["compute_f1_auc"]
GRID = np.round(np.arange(0.02, 0.97, 0.01), 2); CS = [1e-3, 1e-2, 1e-1, 1.0]; PT = np.round(np.arange(0.01, 0.51, 0.01), 2)

if a.data == "cpsc":
    LABELS = ["NORM", "AF", "I-AVB", "LBBB", "RBBB", "PAC", "PVC", "STD", "STE"]; KEY = ["AF", "I-AVB", "STD", "PVC"]
    gold = {x["id"]: [l for l in LABELS if l in x["conversations"][1]["value"]] for x in J("data/ECGBench/cpsc-test.json")}
    sp = J("data/splits/cpsc.json"); pool, test = sp["dev"], sp["test"]
    P = {r["id"]: r["probs"] for r in jl("outputs/clarity/b1_cpsc_dev.jsonl") + jl("outputs/clarity/b1_cpsc_test.jsonl")}
    B0 = {r["question_id"]: [l for l in LABELS if l in r["text"]] for r in jl("outputs/pulse7b/cpsc-test/answers.jsonl")}
    feat = "outputs/features/pulse_cpsc.npz"; meta = None
else:
    D = J("data/derived/sph_ml.json"); LABELS = list(D["labels"]); KEY = ["AF", "SB", "STDEV", "PVC"]
    gold = {x["id"]: x["conversations"][1]["value"].split(";") for x in D["records"]}
    meta = {x["id"]: (x["sex"], int(x["age"])) for x in D["records"]}
    sp = J("data/splits/sph_ml.json"); pool, test = sp["pool"], sp["test"]
    P = {r["id"]: r["probs"] for r in jl("outputs/clarity/b1_sphml_all.jsonl")}
    codes = lambda t: [c for c in LABELS if c in set(re.findall(r"[A-Za-z0-9]+", t))]
    B0 = {r["question_id"]: codes(r["text"]) for r in jl("outputs/pulse7b/sph-ml/answers.jsonl")}
    feat = "outputs/features/pulse_sphml.npz"
best = J(f"outputs/clarity/baselines_{a.data}.json")["results"]["selected"]["llm"]
_n = np.load(os.path.join(ROOT, feat)); fidx = {i: k for k, i in enumerate(_n["ids"])}; FE = _n[best].astype(np.float32)
Y = lambda ids: np.array([[L in gold[i] for L in LABELS] for i in ids])
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
RAW = lambda ids: np.array([[P[i][L] for L in LABELS] for i in ids])
Yp, Yt = Y(pool), Y(test); Rp, Rt = RAW(pool), RAW(test)
Xf_p, Xf_t = FE[[fidx[i] for i in pool]], FE[[fidx[i] for i in test]]


def choose_C(X, y):
    k = min(3, int(y.sum()), int((~y).sum())); best_c, bl = 1e-2, np.inf
    for C in CS:
        oof = cross_val_predict(LogisticRegression(C=C, max_iter=3000), X, y, cv=StratifiedKFold(k, shuffle=True, random_state=0), method="predict_proba")[:, 1]
        l = log_loss(y, np.clip(oof, 1e-6, 1 - 1e-6))
        if l < bl: best_c, bl = C, l
    return best_c


# probabilities on test + OOF probabilities on pool (for operating-point selection)
prob = {"RAW": Rt}; oofp = {"RAW": Rp}
for m in ("S1", "PROBE"):
    prob[m] = np.zeros_like(Rt); oofp[m] = np.zeros_like(Rp)
for j in range(len(LABELS)):
    y = Yp[:, j]; cv = StratifiedKFold(5, shuffle=True, random_state=0)
    Xs_p, Xs_t = lg(Rp), lg(Rt)
    oofp["S1"][:, j] = cross_val_predict(LogisticRegression(max_iter=3000), Xs_p, y, cv=cv, method="predict_proba")[:, 1]
    prob["S1"][:, j] = LogisticRegression(max_iter=3000).fit(Xs_p, y).predict_proba(Xs_t)[:, 1]
    sc = StandardScaler().fit(Xf_p); Fp, Ft = sc.transform(Xf_p), sc.transform(Xf_t); C = choose_C(Fp, y)
    oofp["PROBE"][:, j] = cross_val_predict(LogisticRegression(C=C, max_iter=3000), Fp, y, cv=cv, method="predict_proba")[:, 1]
    prob["PROBE"][:, j] = LogisticRegression(C=C, max_iter=3000).fit(Fp, y).predict_proba(Ft)[:, 1]
B0m = np.array([[L in B0[i] for L in LABELS] for i in test])


def ece(y, p, bins=10):
    e = np.linspace(0, 1, bins + 1); idx = np.clip(np.digitize(p, e) - 1, 0, bins - 1)
    return float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any()))


res = {"data": a.data, "n_test": len(test), "probe_feature": best, "calibration": {}, "reliability": {}, "dca": {}, "operating": {}}
for m in ("RAW", "S1", "PROBE"):
    pl = {L: {"ECE": ece(Yt[:, j], prob[m][:, j]), "Brier": float(np.mean((prob[m][:, j] - Yt[:, j]) ** 2)),
              "AUROC": float(roc_auc_score(Yt[:, j], prob[m][:, j]))} for j, L in enumerate(LABELS)}
    res["calibration"][m] = {"per_label": pl, "macro": {k: float(np.mean([v[k] for v in pl.values()])) for k in ("ECE", "Brier", "AUROC")}}
    y, p = Yt.ravel(), prob[m].ravel(); e = np.linspace(0, 1, 11); idx = np.clip(np.digitize(p, e) - 1, 0, 9)
    res["reliability"][m] = [{"bin": b, "mean_pred": float(p[idx == b].mean()), "obs": float(y[idx == b].mean()), "n": int((idx == b).sum())}
                             for b in range(10) if (idx == b).sum() >= 20]
for L in KEY:
    j = LABELS.index(L); y = Yt[:, j]; n = len(y); prev = y.mean()
    nb = lambda pred, pt: (pred & y).sum() / n - (pred & ~y).sum() / n * pt / (1 - pt)
    res["dca"][L] = {"pt": PT.tolist(), "prevalence": float(prev),
                     "treat_all": [float(prev - (1 - prev) * pt / (1 - pt)) for pt in PT], "treat_none": [0.0] * len(PT),
                     "B0": [float(nb(B0m[:, j], pt)) for pt in PT],
                     **{m: [float(nb(prob[m][:, j] >= pt, pt)) for pt in PT] for m in ("RAW", "S1", "PROBE")}}

rng = np.random.default_rng(0); BI = [rng.integers(0, len(test), len(test)) for _ in range(1000)]


def opstats(y, d):
    tp, fp, fn, tn = (d & y).sum(), (d & ~y).sum(), (~d & y).sum(), (~d & ~y).sum()
    return [tp / max(tp + fn, 1), tn / max(tn + fp, 1), tp / max(tp + fp, 1), tn / max(tn + fn, 1)]


for L in KEY:
    j = LABELS.index(L); y = Yt[:, j]; out = {}
    for m in ("B0", "RAW", "S1", "PROBE"):
        if m == "B0": d = B0m[:, j]; thr = None
        else:
            yp, sp_ = Yp[:, j], oofp[m][:, j]
            cand = [t for t in np.round(np.arange(0.001, 1.0, 0.001), 3) if (sp_[yp] >= t).mean() >= 0.90]
            thr = max(cand) if cand else 0.0; d = prob[m][:, j] >= thr
        point = opstats(y, d); boots = np.array([opstats(y[b], d[b]) for b in BI])
        out[m] = {"threshold": thr, **{k: [round(float(point[q]), 3), round(float(np.percentile(boots[:, q], 2.5)), 3), round(float(np.percentile(boots[:, q], 97.5)), 3)]
                                        for q, k in enumerate(("sens", "spec", "PPV", "NPV"))}}
    res["operating"][L] = out

if meta:
    s1d = np.zeros_like(Yt)
    thr = [GRID[np.argmax([f1_score(Yp[:, j], oofp["S1"][:, j] >= t, zero_division=0) for t in GRID])] for j in range(len(LABELS))]
    s1d = prob["S1"] >= np.array(thr)[None, :]
    empty = ~s1d.any(1); s1d[empty, np.argmax(prob["S1"][empty] / np.array(thr)[None, :], 1)] = True
    s1p = {i: [L for j, L in enumerate(LABELS) if s1d[k, j]] for k, i in enumerate(test)}
    groups = {"female": lambda i: meta[i][0] == "F", "male": lambda i: meta[i][0] == "M", "age<60": lambda i: meta[i][1] < 60, "age>=60": lambda i: meta[i][1] >= 60}
    f = lambda pr, ids: 100 * cfa([pr[i] for i in ids], [gold[i] for i in ids])[0]
    res["subgroups"] = {}
    for g, fn in groups.items():
        ids = [i for i in test if fn(i)]; brng = np.random.default_rng(1); diffs = []
        for _ in range(1000):
            s = [ids[k] for k in brng.integers(0, len(ids), len(ids))]; diffs.append(f(s1p, s) - f(B0, s))
        res["subgroups"][g] = {"n": len(ids), "B0": f(B0, ids), "S1@600": f(s1p, ids),
                               "diff": [round(float(np.mean(diffs)), 2), round(float(np.percentile(diffs, 2.5)), 2), round(float(np.percentile(diffs, 97.5)), 2)]}
json.dump(res, open(os.path.join(ROOT, f"outputs/clarity/clinical_{a.data}.json"), "w"), indent=1)
print(a.data, "probe feature", best)
for m, v in res["calibration"].items(): print(f"  {m:6s} macro ECE {v['macro']['ECE']:.3f}  Brier {v['macro']['Brier']:.4f}  AUROC {v['macro']['AUROC']:.3f}")
for L, o in res["operating"].items(): print("  ", L, {m: (o[m]["sens"][0], o[m]["spec"][0]) for m in o})
if meta: print("  subgroups", {g: (v["n"], round(v["B0"], 1), round(v["S1@600"], 1), v["diff"]) for g, v in res["subgroups"].items()})
