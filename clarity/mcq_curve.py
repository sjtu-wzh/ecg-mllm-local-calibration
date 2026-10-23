"""Learning curve for 8-option multiple-choice subsets (G12EC / CSN), out-of-domain for PULSE.
Written before the G12EC letter probabilities were available.

Per question, option k (letter A-H) carries: l_k = log P(letter_k) from clarity/mcq_prob.py, and the set of
diagnosis codes in its text (abbreviation before each '(...)'). Methods (score -> argmax over the 8 options):
  B0   official generation, parsed with evaluate_ecgbench.extract; unparsable -> random letter (seeded per id)
  ZS   argmax l_k (no labels)
  POS  conditional logit on [l_k, letter one-hot]                 (position-bias correction; 'threshold-only' analog)
  S1   conditional logit on [l_k, letter one-hot, code indicators] (adds per-code prior; 'calibration' analog)
Conditional logit = softmax over the 8 options, L2 penalty 0.5*||w||^2 (C = 1), L-BFGS. Code vocabulary = codes
seen in the training sample. Split: 600 pool / rest test, stratified by gold letter, seed 42. 20 draws per n.
Metric: accuracy on the fixed test split; paired bootstrap (B = 2000) for S1@600 and ZS vs B0.
"""
import argparse, json, os, re, runpy, zlib
import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LET = "ABCDEFGH"
SIZES = [25, 50, 100, 200, 300, 400, 600]
REPEATS = 20
ap = argparse.ArgumentParser(); ap.add_argument("--subset", default="g12-test-no-cot"); a = ap.parse_args()
short = a.subset.split("-")[0]
extract = runpy.run_path(os.path.join(ROOT, "scripts/score_ecgbench.py"), run_name="lib")["extract"]

data = json.load(open(os.path.join(ROOT, "data/ECGBench", f"{a.subset}.json")))
ids = [x["id"] for x in data]
gold = {x["id"]: x["conversations"][1]["value"].strip()[0] for x in data}
codes = {}
for x in data:
    opts = dict(re.findall(r"\n([A-H])\. ?([^\n]+)", x["conversations"][0]["value"]))
    assert set(opts) == set(LET), x["id"]
    codes[x["id"]] = [sorted(set(re.findall(r"([A-Za-z0-9]+)\(", opts[L]))) for L in LET]
LP = {r["id"]: np.array([r["logp"][L] for L in LET]) for r in map(json.loads, open(os.path.join(ROOT, f"outputs/clarity/mcq_{short}_all.jsonl")))}
b0, unparsable = {}, 0
for r in map(json.loads, open(os.path.join(ROOT, f"outputs/pulse7b/{a.subset}/answers.jsonl"))):
    p = extract(r["text"])
    if p is None:
        unparsable += 1; p = LET[zlib.crc32(r["question_id"].encode()) % 8]
    b0[r["question_id"]] = p

split_path = os.path.join(ROOT, f"data/splits/{short}.json")
if not os.path.exists(split_path):
    pool, test = train_test_split(sorted(ids), train_size=600, stratify=[gold[i] for i in sorted(ids)], random_state=42)
    json.dump({"pool": sorted(pool), "test": sorted(test)}, open(split_path, "w"), indent=1)
sp = json.load(open(split_path)); pool, test = sp["pool"], sp["test"]
y = {i: LET.index(gold[i]) for i in ids}


def feats(i, method, vocab):
    X = [LP[i][:, None], np.eye(8)[:, 1:]]  # log-prob, letter one-hot (A as reference)
    if method == "S1":
        V = np.zeros((8, len(vocab)))
        for k, cs in enumerate(codes[i]):
            for c in cs:
                if c in vocab: V[k, vocab[c]] = 1
        X.append(V)
    return np.hstack(X)


def fit(sample, method):
    vocab = {}
    if method == "S1":
        for c in sorted({c for i in sample for cs in codes[i] for c in cs}): vocab[c] = len(vocab)
    Xs = np.stack([feats(i, method, vocab) for i in sample]); ys = np.array([y[i] for i in sample])

    def f(w):
        s = Xs @ w; s -= s.max(1, keepdims=True)
        lse = np.log(np.exp(s).sum(1)); nll = (lse - s[np.arange(len(ys)), ys]).sum()
        p = np.exp(s - lse[:, None]); p[np.arange(len(ys)), ys] -= 1
        return nll + 0.5 * w @ w, np.einsum("nk,nkd->d", p, Xs) + w
    w0 = np.zeros(Xs.shape[2]); w0[0] = 1.0
    w = minimize(f, w0, jac=True, method="L-BFGS-B").x
    return w, vocab


def predict(ids_, method, w=None, vocab=None):
    if method == "ZS": return {i: int(np.argmax(LP[i])) for i in ids_}
    return {i: int(np.argmax(feats(i, method, vocab) @ w)) for i in ids_}


acc = lambda pred: float(np.mean([pred[i] == y[i] for i in test])) * 100
res = {"subset": a.subset, "n_pool": len(pool), "n_test": len(test), "b0_unparsable_all": unparsable}
res["B0"] = acc({i: LET.index(b0[i]) for i in test}); res["ZS"] = acc(predict(test, "ZS"))
rng = np.random.default_rng(42); rows = []
for n in SIZES:
    for r in range(REPEATS):
        sample = list(rng.choice(pool, size=n, replace=False))
        for m in ("POS", "S1"):
            w, vocab = fit(sample, m); rows.append({"n": n, "rep": r, "method": m, "acc": acc(predict(test, m, w, vocab))})
    print(f"n={n} done", flush=True)
curve = {}
for m in ("POS", "S1"):
    for n in SIZES:
        v = np.array([x["acc"] for x in rows if x["method"] == m and x["n"] == n])
        curve[f"{m}@{n}"] = {"acc_mean": round(v.mean(), 2), "acc_sd": round(v.std(), 2),
                             "p2.5": round(np.percentile(v, 2.5), 2), "p97.5": round(np.percentile(v, 97.5), 2),
                             "beats_B0": int((v > res["B0"]).sum())}
res["curve"] = curve
w, vocab = fit(pool, "S1"); s1 = predict(test, "S1", w, vocab); zs = predict(test, "ZS")
corr = lambda pr: np.array([pr[i] == y[i] for i in test], float)
cb0, cs1, czs = corr({i: LET.index(b0[i]) for i in test}), corr(s1), corr(zs)
brng = np.random.default_rng(0); bs = {"S1@600-B0": [], "ZS-B0": []}
for _ in range(2000):
    k = brng.integers(0, len(test), len(test))
    bs["S1@600-B0"].append(100 * (cs1[k].mean() - cb0[k].mean())); bs["ZS-B0"].append(100 * (czs[k].mean() - cb0[k].mean()))
res["bootstrap"] = {k: [round(float(np.mean(v)), 2), round(float(np.percentile(v, 2.5)), 2), round(float(np.percentile(v, 97.5)), 2)]
                    for k, v in bs.items()}
top_codes = sorted(vocab, key=lambda c: -w[8 + vocab[c]])
res["S1@600_top_code_priors"] = {"most_positive": [(c, round(float(w[8 + vocab[c]]), 2)) for c in top_codes[:8]],
                                 "most_negative": [(c, round(float(w[8 + vocab[c]]), 2)) for c in top_codes[-8:]],
                                 "logp_coef": round(float(w[0]), 3), "letter_bias": dict(zip(LET[1:], np.round(w[1:8], 2).tolist()))}
json.dump({"results": res, "rows": rows}, open(os.path.join(ROOT, f"outputs/clarity/mcq_curve_{short}.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
