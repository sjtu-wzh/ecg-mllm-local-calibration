"""Evaluate B0 (official generation) vs B1 (per-label Yes/No probability) on the CPSC dev/test split.
Thresholds and decision rules are chosen on dev only, then applied unchanged to test.
Also reports how well an uncertainty band (M4 gate) concentrates B1's errors."""
import json, os, runpy
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS = ["NORM", "AF", "I-AVB", "LBBB", "RBBB", "PAC", "PVC", "STD", "STE"]
compute_f1_auc = runpy.run_path(os.path.join(ROOT, "scripts/score_ecgbench.py"), run_name="lib")["compute_f1_auc"]

split = json.load(open(os.path.join(ROOT, "data/splits/cpsc.json")))
gold = {x["id"]: [l for l in LABELS if l in x["conversations"][1]["value"]]
        for x in json.load(open(os.path.join(ROOT, "data/ECGBench/cpsc-test.json")))}
b0 = {json.loads(l)["question_id"]: [t for t in LABELS if t in json.loads(l)["text"]]
      for l in open(os.path.join(ROOT, "outputs/pulse7b/cpsc-test/answers.jsonl"))}


def load_probs(part):
    p = os.path.join(ROOT, f"outputs/clarity/b1_cpsc_{part}.jsonl")
    return {r["id"]: r["probs"] for r in map(json.loads, open(p))} if os.path.exists(p) else {}


def decide(probs, thr, at_least_one):
    pred = [L for L in LABELS if probs[L] >= thr[L]]
    if at_least_one and not pred:  # every CPSC record has >= 1 label
        pred = [max(LABELS, key=lambda L: probs[L] / thr[L])]
    return pred


def metrics(pred, ids):
    f1, auc, hl = compute_f1_auc([pred[i] for i in ids], [gold[i] for i in ids])
    per = {}
    for L in LABELS:
        y = [L in gold[i] for i in ids]; yh = [L in pred[i] for i in ids]
        per[L] = round(100 * f1_score(y, yh, zero_division=0), 1)
    return {"F1": round(100 * f1, 1), "AUC": round(100 * auc, 1), "HL": round(100 * hl, 1), "per_label_F1": per}


def tune_thresholds(P, ids):
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    thr = {}
    for L in LABELS:
        y = np.array([L in gold[i] for i in ids]); s = np.array([P[i][L] for i in ids])
        thr[L] = float(max(grid, key=lambda t: f1_score(y, s >= t, zero_division=0)))
    return thr


Pd, Pt = load_probs("dev"), load_probs("test")
dev = [i for i in split["dev"] if i in Pd]
test = [i for i in split["test"] if i in Pt]
res = {"n_dev": len(dev), "n_test": len(test)}

# --- choose B1 variant on dev ---
variants = {}
flat = {L: 0.5 for L in LABELS}
tuned = tune_thresholds(Pd, dev)
for name, thr in [("thr0.5", flat), ("thr_tuned", tuned)]:
    for alo in (False, True):
        key = f"{name}{'+atleast1' if alo else ''}"
        variants[key] = (thr, alo, metrics({i: decide(Pd[i], thr, alo) for i in dev}, dev)["F1"])
best = max(variants, key=lambda k: variants[k][2])
res["dev"] = {"B0": metrics(b0, dev), "B1_variants_F1": {k: v[2] for k, v in variants.items()}, "B1_selected": best,
              "thresholds": tuned}
thr, alo, _ = variants[best]

# --- report on test (once) ---
if test:
    b1_test = {i: decide(Pt[i], thr, alo) for i in test}
    res["test"] = {"B0": metrics(b0, test), "B1": metrics(b1_test, test)}
    res["test"]["B1"]["prob_macroAUC"] = round(100 * np.mean(
        [roc_auc_score([L in gold[i] for i in test], [Pt[i][L] for i in test]) for L in LABELS]), 1)

# --- M4: does an uncertainty band around the thresholds capture the errors? (dev) ---
gate = {}
b1_dev = {i: decide(Pd[i], thr, alo) for i in dev}
wrong = {i for i in dev if set(b1_dev[i]) != set(gold[i])}
for delta in (0.05, 0.1, 0.2):
    unc = {i for i in dev if any(abs(Pd[i][L] - thr[L]) < delta for L in LABELS)}
    gate[delta] = {"uncertain_frac": round(len(unc) / len(dev), 2),
                   "errors_in_uncertain": round(len(wrong & unc) / max(len(wrong), 1), 2),
                   "error_rate_uncertain": round(len(wrong & unc) / max(len(unc), 1), 2),
                   "error_rate_confident": round(len(wrong - unc) / max(len(set(dev) - unc), 1), 2)}
res["dev"]["M4_gate"] = gate

os.makedirs(os.path.join(ROOT, "outputs/clarity"), exist_ok=True)
json.dump(res, open(os.path.join(ROOT, "outputs/clarity/results_b1.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
