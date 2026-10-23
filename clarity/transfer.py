"""Zero-shot transfer: calibrators are fitted on PTB-XL images only (fit split, thresholds from 5-fold OOF),
then applied unchanged to CPSC. No CPSC label is used anywhere.

Pre-registered variants (all reported, none selected on CPSC):
  B1@0.5     raw Yes/No probability, threshold 0.5
  B1@src     raw probability, per-label thresholds chosen on PTB-XL
  S1/S4/S3   logistic calibrators (as in verify_stack.py) fitted on PTB-XL
  +EM        label-prior shift correction estimated by EM on the *unlabeled* target images
             (Saerens et al. 2002), source thresholds kept
  S3(STE raw) sensitivity analysis: STE from B1@0.5 because the PTB-XL STE mapping is approximate
Paper naming (Table S6): S3 -> E2, S4 -> E3; result keys keep the codes above.
Decision rule on CPSC: at least one label per record (the CPSC task asks to classify into these classes).
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
LINKS = {"NORM": EVID, "AF": ["irregular", "no_p"], "I-AVB": ["long_pr"], "LBBB": ["wide_qrs", "notched_r_lat"],
         "RBBB": ["wide_qrs", "rsr_v1"], "PAC": ["prem_atrial", "irregular"], "PVC": ["prem_vent", "irregular"],
         "STD": ["st_dep", "t_inv"], "STE": ["st_elev"]}
compute_f1_auc = runpy.run_path(os.path.join(ROOT, "scripts/score_ecgbench.py"), run_name="lib")["compute_f1_auc"]
J = lambda p: json.load(open(os.path.join(ROOT, p)))


def jsonl(p):
    p = os.path.join(ROOT, p)
    return {r["id"]: r["probs"] for r in map(json.loads, open(p))}


# ---- data ----
ptb = J("data/ptbxl_meta/ptbxl_cpsc_labels.json")["labels"]
psplit = J("data/splits/ptbxl.json")
cpsc_gold = {x["id"]: [l for l in LABELS if l in x["conversations"][1]["value"]] for x in J("data/ECGBench/cpsc-test.json")}
csplit = J("data/splits/cpsc.json")
b0 = {json.loads(l)["question_id"]: [t for t in LABELS if t in json.loads(l)["text"]]
      for l in open(os.path.join(ROOT, "outputs/pulse7b/cpsc-test/answers.jsonl"))}
P = {"ptb": jsonl("outputs/clarity/b1_ptbxl_all.jsonl"),
     "cpsc": {**jsonl("outputs/clarity/b1_cpsc_dev.jsonl"), **jsonl("outputs/clarity/b1_cpsc_test.jsonl")}}
E = {"ptb": jsonl("outputs/clarity/ev_ptbxl_all.jsonl"),
     "cpsc": {**jsonl("outputs/clarity/ev_cpsc_dev.jsonl"), **jsonl("outputs/clarity/ev_cpsc_test.jsonl")}}
SETS = {  # name: (domain, ids, gold)
    "ptb_fit": ("ptb", psplit["fit"], ptb),
    "ptb_heldout": ("ptb", psplit["heldout"], ptb),
    "cpsc_test": ("cpsc", csplit["test"], cpsc_gold),
    "cpsc_all": ("cpsc", sorted(cpsc_gold), cpsc_gold),
}
for name, (dom, ids, gold) in list(SETS.items()):
    have = [i for i in ids if i in P[dom] and i in E[dom]]
    if len(have) < len(ids):
        print(f"WARNING {name}: only {len(have)}/{len(ids)} records have probabilities")
    SETS[name] = (dom, have, gold)


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def X(variant, L, s):
    dom, ids, _ = SETS[s]
    lab = lambda keys: np.column_stack([logit([P[dom][i][k] for i in ids]) for k in keys])
    ev = lambda keys: np.column_stack([logit([E[dom][i][k] for i in ids]) for k in keys])
    return {"S1": lambda: lab(LABELS), "S4": lambda: np.hstack([lab(LABELS), ev(LINKS[L])]),
            "S3": lambda: np.hstack([lab(LABELS), ev(EVID)])}[variant]()


def y(L, s):
    _, ids, gold = SETS[s]
    return np.array([L in gold[i] for i in ids])


def best_thr(yy, s):
    grid = np.round(np.arange(0.02, 0.97, 0.01), 2)
    return float(max(grid, key=lambda t: f1_score(yy, s >= t, zero_division=0)))


def em_prior(p, pi_s, iters=200, tol=1e-6):
    """Adjust posteriors p (calibrated at source prior pi_s) to the unlabeled target's prior."""
    pi_t = pi_s
    for _ in range(iters):
        w1, w0 = pi_t / pi_s, (1 - pi_t) / (1 - pi_s)
        q = w1 * p / (w1 * p + w0 * (1 - p))
        new = float(q.mean())
        if abs(new - pi_t) < tol:
            break
        pi_t = new
    return q, pi_t


def fit(variant):
    """Per label: (fitted model or None, source threshold, source prior)."""
    out = {}
    for L in LABELS:
        yy = y(L, "ptb_fit")
        if variant == "B1":
            out[L] = (None, best_thr(yy, np.array([P["ptb"][i][L] for i in SETS["ptb_fit"][1]])), yy.mean())
        else:
            Xf = X(variant, L, "ptb_fit")
            oof = cross_val_predict(LogisticRegression(max_iter=2000), Xf, yy,
                                    cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
            out[L] = (LogisticRegression(max_iter=2000).fit(Xf, yy), best_thr(yy, oof), yy.mean())
    return out


def scores(variant, model, s, em=False):
    dom, ids, _ = SETS[s]
    sc, priors = {}, {}
    for L in LABELS:
        clf, thr, pi_s = model[L]
        p = np.array([P[dom][i][L] for i in ids]) if clf is None else clf.predict_proba(X(variant, L, s))[:, 1]
        if em:
            p, priors[L] = em_prior(p, pi_s)
        sc[L] = (p, thr)
    return sc, priors


def decide(sc, s, at_least_one, fixed_thr=None):
    ids = SETS[s][1]
    pred = {}
    for n, i in enumerate(ids):
        thr = {L: fixed_thr or sc[L][1] for L in LABELS}
        lab = [L for L in LABELS if sc[L][0][n] >= thr[L]]
        if at_least_one and not lab:
            lab = [max(LABELS, key=lambda L: sc[L][0][n] / thr[L])]
        pred[i] = lab
    return pred


def report(pred, s):
    _, ids, gold = SETS[s]
    f1, auc, hl = compute_f1_auc([pred[i] for i in ids], [gold[i] for i in ids])
    per = {L: round(100 * f1_score([L in gold[i] for i in ids], [L in pred[i] for i in ids], zero_division=0), 1)
           for L in LABELS}
    return {"F1": round(100 * f1, 1), "AUC": round(100 * auc, 1), "HL": round(100 * hl, 1), "per_label_F1": per}


res, preds = {}, {}
models = {v: fit(v) for v in ("B1", "S1", "S4", "S3")}
for s in ("ptb_heldout", "cpsc_test", "cpsc_all"):
    alo = s.startswith("cpsc")
    res[s], preds[s] = {}, {}
    if alo:
        preds[s]["B0 (official generation)"] = {i: b0[i] for i in SETS[s][1]}
    sc, _ = scores("B1", models["B1"], s)
    preds[s]["B1@0.5"] = decide(sc, s, alo, fixed_thr=0.5)
    preds[s]["B1@src"] = decide(sc, s, alo)
    for v in ("S1", "S4", "S3"):
        sc, _ = scores(v, models[v], s)
        preds[s][v] = decide(sc, s, alo)
        if alo:
            sc_em, pri = scores(v, models[v], s, em=True)
            preds[s][v + "+EM"] = decide(sc_em, s, alo)
            res.setdefault("em_priors", {})[f"{s}/{v}"] = {L: round(pri[L], 3) for L in LABELS}
    if alo:  # STE sensitivity analysis: STE from raw B1@0.5, others from S3
        sc3, _ = scores("S3", models["S3"], s)
        b1sc, _ = scores("B1", models["B1"], s)
        sc3["STE"] = (b1sc["STE"][0], 0.5)
        preds[s]["S3 (STE raw)"] = decide(sc3, s, alo)
    for k, pr in preds[s].items():
        res[s][k] = report(pr, s)

os.makedirs(os.path.join(ROOT, "outputs/clarity"), exist_ok=True)
json.dump(res, open(os.path.join(ROOT, "outputs/clarity/results_transfer.json"), "w"), indent=1)
json.dump(preds, open(os.path.join(ROOT, "outputs/clarity/preds_transfer.json"), "w"))
for s in ("ptb_heldout", "cpsc_test", "cpsc_all"):
    print(f"\n== {s} (n={len(SETS[s][1])}) ==")
    print(f"{'variant':26s} {'F1':>5s} {'AUC':>5s} {'HL':>5s}  per-label F1")
    for k, r in res[s].items():
        print(f"{k:26s} {r['F1']:5.1f} {r['AUC']:5.1f} {r['HL']:5.1f}  {r['per_label_F1']}")
