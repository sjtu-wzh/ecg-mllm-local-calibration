"""Confidence intervals for the calibration table and exact counts for the operating-point table (added after Experiment 4).
Re-creates the per-record test probabilities by executing the unchanged probability block of clarity/clinical_eval.py (everything
before `def ece`), checks that every cached point estimate in outputs/clarity/clinical_{data}.json is reproduced exactly, then adds:
  calibration_ci  macro ECE / Brier / AUROC per read-out (RAW, S1, PROBE) and paired differences S1 - RAW and PROBE - S1:
                  percentile bootstrap over test records (B = 2000, default_rng(20261027)); the same record indices for every
                  read-out; per-label metrics re-computed in each replicate and macro-averaged over all labels. A replicate in
                  which any label has no positive or no negative record (AUROC undefined) is redrawn; redraws are counted.
                  Models and thresholds are fixed (test-set uncertainty only). SPH and CPSC test splits have one record per patient.
  operating_counts TP / FP / FN / TN for every operating point of clinical_eval.py (same thresholds), and Clopper-Pearson exact
                  95% intervals for sensitivity, specificity, PPV and NPV.
Output: outputs/clarity/clinical_ci_{data}.json (the cached clinical_{data}.json is not modified)."""
import hashlib, json, os, sys
import numpy as np
from scipy.stats import beta
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "clarity/clinical_eval.py")
src = open(SRC).read(); head = src[:src.index("\n\ndef ece")]
data = sys.argv[sys.argv.index("--data") + 1]
sys.argv = [SRC, "--data", data]; g = {"__file__": SRC, "__name__": "clinical_ci"}
exec(compile(head, SRC, "exec"), g)                      # probability block of clinical_eval.py, unchanged
LABELS, Yt, Yp, prob, oofp, B0m, test = g["LABELS"], g["Yt"], g["Yp"], g["prob"], g["oofp"], g["B0m"], g["test"]
cached = json.load(open(os.path.join(ROOT, f"outputs/clarity/clinical_{data}.json")))
METH = ("RAW", "S1", "PROBE")


def ece(y, p, bins=10):                                  # identical to clinical_eval.ece
    e = np.linspace(0, 1, bins + 1); idx = np.clip(np.digitize(p, e) - 1, 0, bins - 1)
    return float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any()))


def macro(Y, Pm):
    return {"ECE": float(np.mean([ece(Y[:, j], Pm[:, j]) for j in range(Y.shape[1])])),
            "Brier": float(np.mean([np.mean((Pm[:, j] - Y[:, j]) ** 2) for j in range(Y.shape[1])])),
            "AUROC": float(np.mean([roc_auc_score(Y[:, j], Pm[:, j]) for j in range(Y.shape[1])]))}


# ---- 1. reproduce the cached point estimates exactly
point = {m: macro(Yt, prob[m]) for m in METH}
for m in METH:
    for k, v in point[m].items():
        assert abs(v - cached["calibration"][m]["macro"][k]) < 1e-12, (m, k, v, cached["calibration"][m]["macro"][k])

# ---- 2. operating points: same thresholds, exact counts, Clopper-Pearson intervals
cp = lambda x, n: [float(beta.ppf(0.025, x, n - x + 1)) if x > 0 else 0.0, float(beta.ppf(0.975, x + 1, n - x)) if x < n else 1.0]
counts = {}
for L, o in cached["operating"].items():
    j = LABELS.index(L); y = Yt[:, j]; counts[L] = {"n_pos": int(y.sum()), "n": int(len(y))}
    for m, v in o.items():
        d = B0m[:, j] if m == "B0" else prob[m][:, j] >= v["threshold"]
        if m != "B0":                                    # threshold rule of clinical_eval.py, re-derived and checked
            cand = [t for t in np.round(np.arange(0.001, 1.0, 0.001), 3) if (oofp[m][Yp[:, j], j] >= t).mean() >= 0.90]
            assert (max(cand) if cand else 0.0) == v["threshold"], (L, m)
        tp, fp, fn, tn = int((d & y).sum()), int((d & ~y).sum()), int((~d & y).sum()), int((~d & ~y).sum())
        pts = {"sens": (tp, tp + fn), "spec": (tn, tn + fp), "PPV": (tp, tp + fp), "NPV": (tn, tn + fn)}
        for k, (x, n) in pts.items():
            assert round(x / max(n, 1), 3) == v[k][0], (L, m, k)
        counts[L][m] = {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "clopper_pearson": {k: cp(x, n) for k, (x, n) in pts.items()}}

# ---- 3. calibration bootstrap
B, rng, redraws, reps = 2000, np.random.default_rng(20261027), 0, {m: [] for m in METH}
while len(reps["RAW"]) < B:
    idx = rng.integers(0, len(test), len(test)); Yb = Yt[idx]
    if (Yb.sum(0) == 0).any() or (Yb.sum(0) == len(idx)).any(): redraws += 1; continue
    for m in METH: reps[m].append(macro(Yb, prob[m][idx]))
pc = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
ci = {m: {k: {"point": point[m][k], "ci95": pc([r[k] for r in reps[m]])} for k in ("ECE", "Brier", "AUROC")} for m in METH}
diff = {f"{a} - {b}": {k: {"point": point[a][k] - point[b][k], "ci95": pc([ra[k] - rb[k] for ra, rb in zip(reps[a], reps[b])])}
                       for k in ("ECE", "Brier", "AUROC")} for a, b in (("S1", "RAW"), ("PROBE", "S1"))}
sha = lambda p: hashlib.sha256(open(os.path.join(ROOT, p), "rb").read()).hexdigest()
res = {"data": data, "n_test": len(test), "n_labels": len(LABELS), "min_positives_test": int(Yt.sum(0).min()),
       "calibration_ci": {"B": B, "seed": 20261027, "redraws_undefined_auroc": redraws, "per_readout": ci, "paired_differences": diff},
       "operating_counts": counts,
       "inputs_sha256": {p: sha(p) for p in (f"outputs/clarity/clinical_{data}.json", "clarity/clinical_eval.py")}}
json.dump(res, open(os.path.join(ROOT, f"outputs/clarity/clinical_ci_{data}.json"), "w"), indent=1)
print(data, "point estimates reproduced; min positives", res["min_positives_test"], "; redraws", redraws)
for m in METH: print(" ", m, {k: (round(v["point"], 4), [round(x, 4) for x in v["ci95"]]) for k, v in ci[m].items()})
for k, v in diff.items(): print(" ", k, {kk: (round(vv["point"], 4), [round(x, 4) for x in vv["ci95"]]) for kk, vv in v.items()})
