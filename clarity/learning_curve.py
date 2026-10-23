"""Learning curve: how many labeled target (CPSC) ECGs does local calibration need?

For n in SIZES, draw n records at random from the CPSC dev pool (600), fit a calibrator on them only,
evaluate on the fixed CPSC test split (1461). REPEATS random draws per n (at n = 600 the repeats only vary the CV folds).
Pre-registered rules:
  * methods: THR (per-label threshold on raw B1 probability), S1 (all label probabilities), S3 (+ all evidence)
  * thresholds from out-of-fold predictions inside the n samples (k = min(5, #pos, #neg) folds)
  * a label with < 2 positives or < 2 negatives in the sample falls back to the zero-shot rule
    (raw B1 probability with the threshold chosen on PTB-XL)
  * at least one label per CPSC record; logistic regression C = 1.0
S1B0 (exploratory, added after the CODE-15% result): S1 features + the official generated answer (B0) as 0/1 features.
"""
import json, os, runpy
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS = ["NORM", "AF", "I-AVB", "LBBB", "RBBB", "PAC", "PVC", "STD", "STE"]
EVID = ["irregular", "no_p", "long_pr", "wide_qrs", "rsr_v1", "notched_r_lat",
        "prem_atrial", "prem_vent", "st_elev", "st_dep", "t_inv"]
SIZES = [25, 50, 100, 200, 300, 400, 600]
REPEATS = 20
compute_f1_auc = runpy.run_path(os.path.join(ROOT, "scripts/score_ecgbench.py"), run_name="lib")["compute_f1_auc"]
J = lambda p: json.load(open(os.path.join(ROOT, p)))


def jsonl(p):
    return {r["id"]: r["probs"] for r in map(json.loads, open(os.path.join(ROOT, p)))}


gold = {x["id"]: [l for l in LABELS if l in x["conversations"][1]["value"]] for x in J("data/ECGBench/cpsc-test.json")}
split = J("data/splits/cpsc.json")
P = {**jsonl("outputs/clarity/b1_cpsc_dev.jsonl"), **jsonl("outputs/clarity/b1_cpsc_test.jsonl")}
E = {**jsonl("outputs/clarity/ev_cpsc_dev.jsonl"), **jsonl("outputs/clarity/ev_cpsc_test.jsonl")}
pool, test = split["dev"], split["test"]

# zero-shot fallback thresholds: raw B1 probability, thresholds chosen on PTB-XL fit split
ptb_lab = J("data/ptbxl_meta/ptbxl_cpsc_labels.json")["labels"]
Pptb = jsonl("outputs/clarity/b1_ptbxl_all.jsonl")
ptb_fit = J("data/splits/ptbxl.json")["fit"]
GRID = np.round(np.arange(0.02, 0.97, 0.01), 2)


def best_thr(y, s):
    return float(max(GRID, key=lambda t: f1_score(y, s >= t, zero_division=0)))


ZS_THR = {L: best_thr(np.array([L in ptb_lab[i] for i in ptb_fit]), np.array([Pptb[i][L] for i in ptb_fit]))
          for L in LABELS}


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


B0 = {r["question_id"]: [t for t in LABELS if t in r["text"]]
      for r in map(json.loads, open(os.path.join(ROOT, "outputs/pulse7b/cpsc-test/answers.jsonl")))}


def feats(method, ids):
    lab = np.column_stack([logit([P[i][L] for i in ids]) for L in LABELS])
    if method == "S1":
        return lab
    if method == "S1B0":
        return np.hstack([lab, np.array([[float(L in B0[i]) for L in LABELS] for i in ids])])
    return np.hstack([lab, np.column_stack([logit([E[i][k] for i in ids]) for k in EVID])])


Xtest = {m: feats(m, test) for m in ("S1", "S3", "S1B0")}
raw_test = {L: np.array([P[i][L] for i in test]) for L in LABELS}
Ytest = np.array([[L in gold[i] for L in LABELS] for i in test])


def run(method, sample, seed):
    """Returns per-label (test score, threshold) and how many labels fell back to zero-shot."""
    y_all = {L: np.array([L in gold[i] for i in sample]) for L in LABELS}
    Xs = feats(method, sample) if method != "THR" else None
    out, n_fallback = {}, 0
    for L in LABELS:
        y = y_all[L]; pos, neg = int(y.sum()), int((~y).sum())
        if pos < 2 or neg < 2:
            out[L] = (raw_test[L], ZS_THR[L]); n_fallback += 1
            continue
        if method == "THR":
            out[L] = (raw_test[L], best_thr(y, np.array([P[i][L] for i in sample])))
            continue
        k = min(5, pos, neg)
        oof = cross_val_predict(LogisticRegression(max_iter=2000), Xs, y,
                                cv=StratifiedKFold(k, shuffle=True, random_state=seed), method="predict_proba")[:, 1]
        clf = LogisticRegression(max_iter=2000).fit(Xs, y)
        out[L] = (clf.predict_proba(Xtest[method])[:, 1], best_thr(y, oof))
    return out, n_fallback


def evaluate(sc):
    pred = []
    for n in range(len(test)):
        lab = [L for L in LABELS if sc[L][0][n] >= sc[L][1]]
        if not lab:
            lab = [max(LABELS, key=lambda L: sc[L][0][n] / sc[L][1])]
        pred.append(lab)
    f1, auc, hl = compute_f1_auc(pred, [gold[i] for i in test])
    return 100 * f1, 100 * auc, 100 * hl


rng = np.random.default_rng(42)
rows = []
for n in SIZES:
    reps = REPEATS  # at n = 600 the draws differ only in order, i.e. in CV folds -> threshold-selection variance
    for r in range(reps):
        sample = list(rng.choice(pool, size=n, replace=False))
        for m in ("THR", "S1", "S3", "S1B0"):
            sc, nf = run(m, sample, seed=r)
            f1, auc, hl = evaluate(sc)
            rows.append({"n": n, "rep": r, "method": m, "F1": f1, "AUC": auc, "HL": hl, "fallback_labels": nf})
    print(f"n={n} done", flush=True)

summary = {}
for m in ("THR", "S1", "S3", "S1B0"):
    for n in SIZES:
        v = [r for r in rows if r["method"] == m and r["n"] == n]
        f = np.array([r["F1"] for r in v])
        summary[f"{m}@{n}"] = {"reps": len(v), "F1_mean": round(f.mean(), 1), "F1_sd": round(f.std(), 1),
                               "F1_p2.5": round(np.percentile(f, 2.5), 1), "F1_p97.5": round(np.percentile(f, 97.5), 1),
                               "AUC_mean": round(np.mean([r["AUC"] for r in v]), 1),
                               "HL_mean": round(np.mean([r["HL"] for r in v]), 1),
                               "fallback_mean": round(np.mean([r["fallback_labels"] for r in v]), 1)}
json.dump({"rows": rows, "summary": summary, "zero_shot_thresholds": ZS_THR},
          open(os.path.join(ROOT, "outputs/clarity/learning_curve.json"), "w"), indent=1)
print(f"{'method@n':10s} {'F1 mean±sd':>12s} {'[2.5,97.5]':>14s} {'AUC':>5s} {'HL':>5s} fallback")
for k, s in summary.items():
    print(f"{k:10s} {s['F1_mean']:6.1f}±{s['F1_sd']:<4.1f} [{s['F1_p2.5']:5.1f},{s['F1_p97.5']:5.1f}] "
          f"{s['AUC_mean']:5.1f} {s['HL_mean']:5.1f} {s['fallback_mean']}")
