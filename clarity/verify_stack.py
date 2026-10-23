"""M3 (+M4) ablation on CPSC: clinically linked evidence verification over B1 label probabilities.

Variants (all fitted / thresholded on dev only; dev thresholds come from 5-fold out-of-fold predictions):
  B1        raw label probability, per-label threshold
  S1        logistic stack over all 9 label probabilities (cross-label context, no evidence)
  S2        CLARITY-M3: label probability + only its clinically linked evidence probabilities
  S3        unconstrained: all 9 label + all evidence probabilities
  H         hybrid: label probability + the official generated answer (B0) for that label
  S2H       CLARITY-M3 + B0: label probability + linked evidence + B0
  S4        S1 + only the clinically linked evidence (S1 vs S4 vs S3 isolates evidence and the constraint)
  S3+M4     S3 only for uncertain cases (S1 score near its threshold, delta chosen on dev), S1 elsewhere
Evidence variants are skipped until outputs/clarity/ev_cpsc_*.jsonl exist.
Paper naming (Table S6): S2 -> E1, S2H -> E1 + B0, S3 -> E2, S4 -> E3; keys in results_m3.json keep the codes above.
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
LINKS = {  # clinical prior: which evidence may support / refute each diagnosis
    "NORM": EVID, "AF": ["irregular", "no_p"], "I-AVB": ["long_pr"], "LBBB": ["wide_qrs", "notched_r_lat"],
    "RBBB": ["wide_qrs", "rsr_v1"], "PAC": ["prem_atrial", "irregular"], "PVC": ["prem_vent", "irregular"],
    "STD": ["st_dep", "t_inv"], "STE": ["st_elev"],
}
compute_f1_auc = runpy.run_path(os.path.join(ROOT, "scripts/score_ecgbench.py"), run_name="lib")["compute_f1_auc"]
split = json.load(open(os.path.join(ROOT, "data/splits/cpsc.json")))
gold = {x["id"]: [l for l in LABELS if l in x["conversations"][1]["value"]]
        for x in json.load(open(os.path.join(ROOT, "data/ECGBench/cpsc-test.json")))}
b0 = {json.loads(l)["question_id"]: [t for t in LABELS if t in json.loads(l)["text"]]
      for l in open(os.path.join(ROOT, "outputs/pulse7b/cpsc-test/answers.jsonl"))}


def load(prefix, part):
    p = os.path.join(ROOT, f"outputs/clarity/{prefix}_cpsc_{part}.jsonl")
    return {r["id"]: r["probs"] for r in map(json.loads, open(p))} if os.path.exists(p) else {}


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


P = {"dev": load("b1", "dev"), "test": load("b1", "test")}
E = {"dev": load("ev", "dev"), "test": load("ev", "test")}
HAS_EV = all(len(E[k]) == len(split[k]) for k in E)
ids = {k: [i for i in split[k] if i in P[k] and (i in E[k] or not HAS_EV)] for k in ("dev", "test")}
Y = {k: {L: np.array([L in gold[i] for i in ids[k]]) for L in LABELS} for k in ids}


def features(variant, L, part):
    lab = lambda keys: np.column_stack([logit([P[part][i][k] for i in ids[part]]) for k in keys])
    ev = lambda keys: np.column_stack([logit([E[part][i][k] for i in ids[part]]) for k in keys])
    gen = lambda: np.array([[float(L in b0[i])] for i in ids[part]])  # B0 generated answer, 0/1
    if variant == "S1":
        return lab(LABELS)
    if variant == "H":
        return np.hstack([lab([L]), gen()])
    if variant == "S2":
        return np.hstack([lab([L]), ev(LINKS[L])])
    if variant == "S2H":
        return np.hstack([lab([L]), ev(LINKS[L]), gen()])
    if variant == "S3":
        return np.hstack([lab(LABELS), ev(EVID)])
    if variant == "S4":  # S1 + only the clinically linked evidence (clean test of the clinical constraint)
        return np.hstack([lab(LABELS), ev(LINKS[L])])
    raise ValueError(variant)


def best_thr(y, s):
    grid = np.round(np.arange(0.02, 0.97, 0.01), 2)
    return float(max(grid, key=lambda t: f1_score(y, s >= t, zero_division=0)))


def fit_variant(variant):
    """Returns per-label (dev OOF score, test score, threshold)."""
    out = {}
    for L in LABELS:
        if variant == "B1":
            sd = np.array([P["dev"][i][L] for i in ids["dev"]]); st = np.array([P["test"][i][L] for i in ids["test"]])
        else:
            Xd, Xt, y = features(variant, L, "dev"), features(variant, L, "test"), Y["dev"][L]
            clf = LogisticRegression(C=1.0, max_iter=2000)
            cv = StratifiedKFold(5, shuffle=True, random_state=0)
            sd = cross_val_predict(clf, Xd, y, cv=cv, method="predict_proba")[:, 1]
            st = clf.fit(Xd, y).predict_proba(Xt)[:, 1]
        out[L] = (sd, st, best_thr(Y["dev"][L], sd))
    return out


def to_pred(scores, part, at_least_one=True):
    k = 0 if part == "dev" else 1
    pred = {}
    for n, i in enumerate(ids[part]):
        lab = [L for L in LABELS if scores[L][k][n] >= scores[L][2]]
        if at_least_one and not lab:
            lab = [max(LABELS, key=lambda L: scores[L][k][n] / scores[L][2])]
        pred[i] = lab
    return pred


def report(pred, part):
    f1, auc, hl = compute_f1_auc([pred[i] for i in ids[part]], [gold[i] for i in ids[part]])
    per = {L: round(100 * f1_score(Y[part][L], [L in pred[i] for i in ids[part]], zero_division=0), 1) for L in LABELS}
    return {"F1": round(100 * f1, 1), "AUC": round(100 * auc, 1), "HL": round(100 * hl, 1), "per_label_F1": per}


res = {"n_dev": len(ids["dev"]), "n_test": len(ids["test"]), "dev_oof": {}, "test": {}}
res["dev_oof"]["B0"] = report(b0, "dev"); res["test"]["B0"] = report(b0, "test")
variants = ("B1", "S1", "H") + (("S2", "S3", "S2H", "S4") if HAS_EV else ())
fits = {v: fit_variant(v) for v in variants}
test_preds = {"B0": {i: b0[i] for i in ids["test"]}}
for v, sc in fits.items():
    res["dev_oof"][v] = report(to_pred(sc, "dev"), "dev")
    test_preds[v] = to_pred(sc, "test")
    res["test"][v] = report(test_preds[v], "test")

# M4: route only uncertain cases (S1 score near its threshold) to S3; delta chosen on dev OOF F1
if HAS_EV:
    base, ver = fits["S1"], fits["S3"]

    def gated(part, delta):
        k = 0 if part == "dev" else 1
        unc = np.zeros(len(ids[part]), bool)
        for M in LABELS:
            unc |= np.abs(base[M][k] - base[M][2]) < delta
        # score / threshold puts both sources on one scale where >= 1 means positive
        mixed = {L: (lambda x: (x, x, 1.0))(np.where(unc, ver[L][k] / ver[L][2], base[L][k] / base[L][2]))
                 for L in LABELS}
        return mixed, int(unc.sum())

    best = None
    for delta in (0.05, 0.1, 0.15, 0.2, 0.3):
        sc, n_unc = gated("dev", delta)
        f = report(to_pred(sc, "dev"), "dev")["F1"]
        res["dev_oof"][f"S3+M4(d={delta})"] = {"F1": f, "uncertain_frac": round(n_unc / len(ids["dev"]), 2)}
        if best is None or f > best[1]:
            best = (delta, f)
    sc, n_unc = gated("test", best[0])
    key = f"S3+M4(d={best[0]})"
    test_preds[key] = to_pred(sc, "test")
    res["test"][key] = dict(report(test_preds[key], "test"), uncertain_frac=round(n_unc / len(ids["test"]), 2))

os.makedirs(os.path.join(ROOT, "outputs/clarity"), exist_ok=True)
json.dump(res, open(os.path.join(ROOT, "outputs/clarity/results_m3.json"), "w"), indent=1)
json.dump(test_preds, open(os.path.join(ROOT, "outputs/clarity/preds_test.json"), "w"))
print(f"{'variant':24s} {'devF1':>6s} | {'F1':>5s} {'AUC':>5s} {'HL':>5s}  per-label F1 (test)")
for v in res["test"]:
    d = res["dev_oof"].get(v, {}).get("F1", "")
    t = res["test"][v]
    print(f"{v:24s} {str(d):>6s} | {t['F1']:5.1f} {t['AUC']:5.1f} {t['HL']:5.1f}  {t['per_label_F1']}")
