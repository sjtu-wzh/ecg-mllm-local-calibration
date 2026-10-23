"""Experiment 3 (robustness): analysis. Written and frozen (docs/prereg_robustness.md) before the model outputs existed.
Frozen PULSE-7B on SPH; subsets data/splits/sph_robust.json (300 pool, 800 test). Same read-out rules as clarity/ml_curve.py:
S1 = per-label logistic regression (C = 1) on all 12 label logits; thresholds from 5-fold OOF predictions maximising F1;
at least one label per record; B0 parsed by exact code tokens; official macro-F1 (scripts/score_ecgbench.py).

Arms (all evaluated on the same 800 test records):
  CLEAN           clean images, original prompts (existing outputs: outputs/pulse7b/sph-ml, outputs/clarity/b1_sphml_all.jsonl)
  ROT/PHOTO/SCAN  perturbed images (pool and test), original prompts
  QA/QB           clean images, paraphrased Yes/No question
  P1              clean images, paraphrased generated-answer prompt (B0 only)
Read-outs per arm a: B0_a; ZS_a (P >= 0.5); S1_a (fitted on the arm's own 300 pool records = 'matched');
S1_clean->a (fitted on the 300 clean pool records, applied unchanged to arm a; 'unmatched').
Pre-registered contrasts (paired bootstrap over test records, B = 2000, default_rng(0)):
  R1 (per image condition c)  S1_c − B0_c                 supported if 95% CI lower > 0 and point >= 3
  R2 (per image condition c)  S1_clean->c − B0_c          supported if CI lower > 0
  R3 (per paraphrase q)       S1_q − B0_CLEAN             supported if CI lower > 0 and point >= 3
  R4                          S1_CLEAN − B0_P1            supported if CI lower > 0 and point >= 3
Descriptive: B0_c − B0_CLEAN, S1_c − S1_CLEAN, S1_clean->q − S1_q, B0_P1 − B0_CLEAN (bootstrap CI); macro AUROC of P(yes);
learning curves n in {25, 50, 100, 200, 300} x 20 draws (THR, S1) per arm.
Output: outputs/clarity/robust_results.json
"""
import json, os, re, runpy
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
J = lambda p: os.path.join(ROOT, p)
compute_f1_auc = runpy.run_path(J("scripts/score_ecgbench.py"), run_name="lib")["compute_f1_auc"]
D = json.load(open(J("data/derived/sph_ml.json"))); LABELS = list(D["labels"])
gold = {x["id"]: x["conversations"][1]["value"].split(";") for x in D["records"]}
sp = json.load(open(J("data/splits/sph_robust.json"))); pool, test = sp["pool"], sp["test"]
GRID = np.round(np.arange(0.02, 0.97, 0.01), 2); SIZES = [25, 50, 100, 200, 300]
codes = lambda t: [c for c in LABELS if c in set(re.findall(r"[A-Za-z0-9]+", t))]
rd = lambda p: [json.loads(l) for l in open(J(p)) if l.strip()]

B1 = {"CLEAN": "outputs/clarity/b1_sphml_all.jsonl", **{a: f"outputs/robust/b1_{a}.jsonl" for a in ("ROT", "PHOTO", "SCAN", "QA", "QB")}}
B0 = {"CLEAN": "outputs/pulse7b/sph-ml/answers.jsonl", **{a: f"outputs/robust/b0_{a}.jsonl" for a in ("ROT", "PHOTO", "SCAN", "P1")}}
DRY = os.environ.get("ROBUST_DRYRUN")  # code check only: every arm reads the CLEAN outputs, results go to $ROBUST_DRYRUN
if DRY: B1 = {a: B1["CLEAN"] for a in B1}; B0 = {a: B0["CLEAN"] for a in B0}
P = {a: {r["id"]: r["probs"] for r in rd(p)} for a, p in B1.items()}
G0 = {a: {r["question_id"]: codes(r["text"]) for r in rd(p)} for a, p in B0.items()}
for a in P: assert all(i in P[a] for i in pool + test), f"B1 {a} incomplete"
for a in G0: assert all(i in G0[a] for i in test), f"B0 {a} incomplete"

logit = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
X = lambda a, s: np.column_stack([logit(np.array([P[a][i][L] for i in s])) for L in LABELS])
best_thr = lambda y, s: float(max(GRID, key=lambda t: f1_score(y, s >= t, zero_division=0)))


def fit(method, fit_arm, sample, test_arm, seed=0):
    Xte = X(test_arm, test); raw = {L: np.array([P[test_arm][i][L] for i in test]) for L in LABELS}; out = {}
    for L in LABELS:
        y = np.array([L in gold[i] for i in sample]); pos, neg = int(y.sum()), int((~y).sum())
        if pos < 2 or neg < 2: out[L] = (raw[L], 0.5); continue
        if method == "THR": out[L] = (raw[L], best_thr(y, np.array([P[fit_arm][i][L] for i in sample]))); continue
        Xs = X(fit_arm, sample)
        oof = cross_val_predict(LogisticRegression(max_iter=2000), Xs, y, method="predict_proba",
                                cv=StratifiedKFold(min(5, pos, neg), shuffle=True, random_state=seed))[:, 1]
        out[L] = (LogisticRegression(max_iter=2000).fit(Xs, y).predict_proba(Xte)[:, 1], best_thr(y, oof))
    return out


def decide(sc):
    return {i: [L for L in LABELS if sc[L][0][n] >= sc[L][1]] or [max(LABELS, key=lambda L: sc[L][0][n] / sc[L][1])] for n, i in enumerate(test)}


f1 = lambda pred, s=test: 100 * compute_f1_auc([pred[i] for i in s], [gold[i] for i in s])[0]
zs = lambda a: decide({L: (np.array([P[a][i][L] for i in test]), 0.5) for L in LABELS})
auc = lambda a: float(np.mean([roc_auc_score([L in gold[i] for i in test], [P[a][i][L] for i in test]) for L in LABELS]))


def boot(pa, pb, B=2000):
    r = np.random.default_rng(0); out = []
    for _ in range(B):
        s = [test[j] for j in r.integers(0, len(test), len(test))]; out.append(f1(pa, s) - f1(pb, s))
    return [round(f1(pa) - f1(pb), 2), round(float(np.percentile(out, 2.5)), 2), round(float(np.percentile(out, 97.5)), 2)]


b0 = {a: {i: G0[a][i] for i in test} for a in G0}
s1 = {a: decide(fit("S1", a, pool, a)) for a in P}
s1_from_clean = {a: decide(fit("S1", "CLEAN", pool, a)) for a in P if a != "CLEAN"}
res = {"n_pool": len(pool), "n_test": len(test),
       "F1": {**{f"B0_{a}": round(f1(b0[a]), 2) for a in b0}, **{f"ZS_{a}": round(f1(zs(a)), 2) for a in P},
              **{f"S1_{a}": round(f1(s1[a]), 2) for a in P}, **{f"S1_clean->{a}": round(f1(s1_from_clean[a]), 2) for a in s1_from_clean}},
       "prob_macroAUC": {a: round(auc(a), 4) for a in P},
       "B0_labels_per_answer": {a: round(float(np.mean([len(v) for v in b0[a].values()])), 2) for a in b0}}
H, DESC = {}, {}
for c in ("ROT", "PHOTO", "SCAN"):
    H[f"R1 {c}: S1_{c} - B0_{c}"] = (boot(s1[c], b0[c]), "lo>0 & pt>=3")
    H[f"R2 {c}: S1_clean->{c} - B0_{c}"] = (boot(s1_from_clean[c], b0[c]), "lo>0")
    DESC[f"B0_{c} - B0_CLEAN"] = boot(b0[c], b0["CLEAN"]); DESC[f"S1_{c} - S1_CLEAN"] = boot(s1[c], s1["CLEAN"])
    DESC[f"S1_clean->{c} - S1_{c}"] = boot(s1_from_clean[c], s1[c])
for q in ("QA", "QB"):
    H[f"R3 {q}: S1_{q} - B0_CLEAN"] = (boot(s1[q], b0["CLEAN"]), "lo>0 & pt>=3")
    DESC[f"S1_{q} - S1_CLEAN"] = boot(s1[q], s1["CLEAN"]); DESC[f"S1_clean->{q} - S1_{q}"] = boot(s1_from_clean[q], s1[q])
H["R4: S1_CLEAN - B0_P1"] = (boot(s1["CLEAN"], b0["P1"]), "lo>0 & pt>=3")
DESC["B0_P1 - B0_CLEAN"] = boot(b0["P1"], b0["CLEAN"])
res["hypotheses"] = {k: {"estimate": v, "rule": rule, "verdict": "SUPPORTED" if v[1] > 0 and (rule == "lo>0" or v[0] >= 3) else "NOT SUPPORTED"}
                     for k, (v, rule) in H.items()}
res["descriptive"] = DESC
rng = np.random.default_rng(42); curve = {}
for a in P:
    for m in ("THR", "S1"):
        for n in SIZES:
            v = [f1(decide(fit(m, a, list(rng.choice(pool, n, replace=False)), a, seed=r))) for r in range(2 if DRY else 20)]
            curve[f"{a}|{m}@{n}"] = [round(float(np.mean(v)), 2), round(float(np.percentile(v, 2.5)), 2), round(float(np.percentile(v, 97.5)), 2)]
    print(f"curve {a} done", flush=True)
res["curve"] = curve
json.dump(res, open(DRY or J("outputs/clarity/robust_results.json"), "w"), indent=1)
for k, v in res["hypotheses"].items(): print(f"{v['verdict']:14s} {k}: {v['estimate']}")
print(json.dumps({k: res[k] for k in ("F1", "prob_macroAUC", "descriptive")}, indent=1))
