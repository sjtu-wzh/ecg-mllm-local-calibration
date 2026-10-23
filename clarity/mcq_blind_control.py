"""Image-blind control for the G12EC / CSN multiple-choice subsets (saved version of an interactive check).
Same conditional-logit 'S1' as clarity/mcq_curve.py, but the model's letter log-probabilities are replaced by a
constant, so the score uses only letter-position and diagnosis-code priors learned from n labeled questions.
If this beats PULSE's official accuracy, the multiple-choice items can be solved without the image.
Output: outputs/clarity/mcq_blind_{short}.json"""
import argparse, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(); ap.add_argument("--subset", default="g12-test-no-cot"); a = ap.parse_args()
sys.argv = ["mcq_curve.py", "--subset", a.subset]
src = open(os.path.join(ROOT, "clarity/mcq_curve.py")).read().split("acc = lambda pred")[0]  # definitions only
ns = {"__file__": os.path.join(ROOT, "clarity/mcq_curve.py")}; exec(compile(src, "mcq_curve_defs", "exec"), ns)
LP, fit, predict, pool, test, y, b0, LET = (ns[k] for k in ("LP", "fit", "predict", "pool", "test", "y", "b0", "LET"))
acc = lambda pred: float(np.mean([pred[i] == y[i] for i in test])) * 100
for i in LP: LP[i] = np.zeros(8)  # remove all image information
res = {"subset": a.subset, "B0_official_on_test": acc({i: LET.index(b0[i]) for i in test}), "chance": 12.5, "blind": {}}
for n in (25, 100, 600):
    rng = np.random.default_rng(7); v = []
    for r in range(20 if n < 600 else 1):
        s = list(rng.choice(pool, size=n, replace=False)) if n < 600 else pool
        w, vocab = fit(s, "S1"); v.append(acc(predict(test, "S1", w, vocab)))
    res["blind"][n] = [round(float(np.mean(v)), 2), round(float(np.min(v)), 2), round(float(np.max(v)), 2)]
json.dump(res, open(os.path.join(ROOT, f"outputs/clarity/mcq_blind_{a.subset.split('-')[0]}.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
