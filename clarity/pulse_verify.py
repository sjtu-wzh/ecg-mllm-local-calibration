"""Spot-check all stored PULSE-7B outputs: recompute K random rows per output file in fresh processes with the SAME
scripts (label_prob.py --ids-file, mcq_prob.py --ids-file, scripts/infer_mps.py on a subset question file) and compare.
Probabilities: max |new - stored|. Generations: exact-text agreement and agreement of the parsed labels.
Output: outputs/clarity/pulse_verify/summary.json
Run: <pulse-llava env python> -I clarity/pulse_verify.py [--k 15]"""
import argparse, json, os, random, re, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs/clarity/pulse_verify")
ap = argparse.ArgumentParser(); ap.add_argument("--k", type=int, default=15); ap.add_argument("--compare-only", action="store_true")
a = ap.parse_args()
PY = sys.executable
ENV = {**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "1", "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.5", "PYTORCH_MPS_LOW_WATERMARK_RATIO": "0.4"}
J = lambda p: json.load(open(os.path.join(ROOT, p)))
rows = lambda p: [json.loads(l) for l in open(os.path.join(ROOT, p)) if l.strip()]

# (name, stored file, id key, value key, command builder)
PROB = [  # label_prob.py: (--data, --set, stored files)
    ("b1_cpsc", "cpsc", "labels", ["outputs/clarity/b1_cpsc_dev.jsonl", "outputs/clarity/b1_cpsc_test.jsonl"]),
    ("ev_cpsc", "cpsc", "evidence", ["outputs/clarity/ev_cpsc_dev.jsonl", "outputs/clarity/ev_cpsc_test.jsonl"]),
    ("b1_ptbxl", "ptbxl", "labels", ["outputs/clarity/b1_ptbxl_all.jsonl"]),
    ("ev_ptbxl", "ptbxl", "evidence", ["outputs/clarity/ev_ptbxl_all.jsonl"]),
    ("b1_code15", "code15", "labels", ["outputs/clarity/b1_code15_all.jsonl"]),
    ("b1_csnml", "csnml", "labels", ["outputs/clarity/b1_csnml_all.jsonl"]),
    ("b1_g12ml", "g12ml", "labels", ["outputs/clarity/b1_g12ml_all.jsonl"]),
]
GEN = [  # scripts/infer_mps.py: (name, question file, stored answers)
    ("b0_cpsc", "data/ECGBench/cpsc-test.json", "outputs/pulse7b/cpsc-test/answers.jsonl"),
    ("b0_code15", "data/ECGBench/code15-test.json", "outputs/pulse7b/code15-test/answers.jsonl"),
    ("b0_mmmu", "data/ECGBench/mmmu-ecg.json", "outputs/pulse7b/mmmu-ecg/answers.jsonl"),
    ("b0_g12mcq", "data/ECGBench/g12-test-no-cot.json", "outputs/pulse7b/g12-test-no-cot/answers.jsonl"),
    ("b0_ptbxl_cpsc9", "data/derived/ptbxl_cpsc9.json", "outputs/pulse7b/ptbxl-cpsc9/answers.jsonl"),
    ("b0_csnml", "data/derived/csn_ml_questions.json", "outputs/pulse7b/csn-ml/answers.jsonl"),
    ("b0_g12ml", "data/derived/g12_ml_questions.json", "outputs/pulse7b/g12-ml/answers.jsonl"),
]
os.makedirs(OUT, exist_ok=True)
pick = lambda ids: sorted(random.Random(0).sample(sorted(ids), a.k))

if not a.compare_only:
    for name, data, qset, files in PROB:
        stored = {r["id"] for f in files for r in rows(f)}
        idf = os.path.join(OUT, f"{name}_ids.json"); json.dump(pick(stored), open(idf, "w"))
        out = os.path.join(OUT, f"{name}.jsonl")
        if os.path.exists(out): os.remove(out)
        subprocess.run([PY, "-I", "clarity/label_prob.py", "--data", data, "--set", qset, "--split", "all", "--ids-file", idf, "--out", out],
                       cwd=ROOT, env=ENV, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); print("done", name, flush=True)
    stored = {r["id"] for r in rows("outputs/clarity/mcq_g12_all.jsonl")}
    idf = os.path.join(OUT, "mcq_g12_ids.json"); json.dump(pick(stored), open(idf, "w")); out = os.path.join(OUT, "mcq_g12.jsonl")
    if os.path.exists(out): os.remove(out)
    subprocess.run([PY, "-I", "clarity/mcq_prob.py", "--subset", "g12-test-no-cot", "--ids-file", idf, "--out", out],
                   cwd=ROOT, env=ENV, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); print("done mcq_g12", flush=True)
    for name, qfile, ans in GEN:
        keep = set(pick({r["question_id"] for r in rows(ans)}))
        sub = [x for x in J(qfile) if x["id"] in keep]; qf = os.path.join(OUT, f"{name}_questions.json"); json.dump(sub, open(qf, "w"))
        out = os.path.join(OUT, f"{name}.jsonl")
        if os.path.exists(out): os.remove(out)
        subprocess.run([PY, "-I", "scripts/infer_mps.py", "--question-file", qf, "--answers-file", out],
                       cwd=ROOT, env=ENV, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); print("done", name, flush=True)

summary = {}
for name, _, _, files in PROB + [("mcq_g12", None, None, ["outputs/clarity/mcq_g12_all.jsonl"])]:
    key = "logp" if name == "mcq_g12" else "probs"
    old = {r["id"]: r[key] for f in files for r in rows(f)}
    new = {r["id"]: r[key] for r in rows(os.path.relpath(os.path.join(OUT, f"{name}.jsonl"), ROOT))}
    if key == "logp":  # compare as probabilities
        import math
        old = {i: {L: math.exp(v) for L, v in d.items()} for i, d in old.items()}; new = {i: {L: math.exp(v) for L, v in d.items()} for i, d in new.items()}
    diffs = [abs(new[i][L] - old[i][L]) for i in new for L in new[i]]
    summary[name] = {"n_rows": len(new), "n_values": len(diffs), "max_abs_diff": round(max(diffs), 5), "mean_abs_diff": round(sum(diffs) / len(diffs), 6)}
codes = lambda t: sorted(set(re.findall(r"[A-Za-z0-9\-]+", t)))
for name, _, ans in GEN:
    old = {r["question_id"]: r["text"] for r in rows(ans)}
    new = {r["question_id"]: r["text"] for r in rows(os.path.relpath(os.path.join(OUT, f"{name}.jsonl"), ROOT))}
    summary[name] = {"n_rows": len(new), "text_identical": sum(new[i] == old[i] for i in new),
                     "token_set_identical": sum(codes(new[i]) == codes(old[i]) for i in new),
                     "differing_examples": [{"id": i, "stored": old[i][:120], "recomputed": new[i][:120]} for i in new if new[i] != old[i]][:5]}
json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
for k, v in summary.items():
    print(k, {kk: vv for kk, vv in v.items() if kk != "differing_examples"})
