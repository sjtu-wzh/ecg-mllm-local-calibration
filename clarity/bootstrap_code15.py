"""CODE-15% (in-domain for PULSE): paired bootstrap of S1@600 - B0 using the same S1 definition as clarity/code15_curve.py
(fitted on the full 600-record pool: 300 binary + 300 multi-label, seed 0), plus the multi-label probability macro-AUROC.
B = 2000 record resamples of the 800-record test split, default_rng(0). Output: outputs/clarity/code15_bootstrap.json"""
import json, os, runpy, sys
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src = open(os.path.join(ROOT, "clarity/code15_curve.py")).read().split("res = {\"split\"")[0]  # definitions only
ns = {"__file__": os.path.join(ROOT, "clarity/code15_curve.py")}; sys.argv = ["code15_curve.py"]
exec(compile(src, "code15_defs", "exec"), ns)
predict, metrics, test, gold, b0, pool, P, LAB6, test_t, cfa = (ns[k] for k in ("predict", "metrics", "test", "gold", "b0", "pool", "P", "LAB6", "test_t", "compute_f1_auc"))
s1 = predict("S1", pool["bin"], pool["multi"], seed=0); B0 = {i: b0[i] for i in test}
f = lambda pr, ids: 100 * cfa([pr[i] for i in ids], [gold[i] for i in ids])[0]
rng = np.random.default_rng(0); d = []
for _ in range(2000):
    ids = [test[j] for j in rng.integers(0, len(test), len(test))]; d.append(f(s1, ids) - f(B0, ids))
auc = np.mean([roc_auc_score([L in gold[i] for i in test_t["multi"]], [P[i][L] for i in test_t["multi"]]) for L in LAB6])
res = {"point": {"B0": f(B0, test), "S1@600": f(s1, test)}, "S1@600-B0": [round(float(np.mean(d)), 2), round(float(np.percentile(d, 2.5)), 2), round(float(np.percentile(d, 97.5)), 2)],
       "prob_macroAUC_multilabel": round(float(auc), 4)}
json.dump(res, open(os.path.join(ROOT, "outputs/clarity/code15_bootstrap.json"), "w"), indent=1); print(res)
