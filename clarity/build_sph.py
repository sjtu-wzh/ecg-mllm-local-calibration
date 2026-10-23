"""Experiment 2: SPH (Shandong Provincial Hospital, Liu et al., Sci Data 2022; CC0) as a NEW out-of-domain site.
Fixed before any SPH model output exists:
  * one record per patient (lowest ECG_ID), then a random sample of 3000 records (numpy default_rng(42))
  * labels = base AHA codes (statement before '+'), kept if >= 50 positives in the sample; records with >= 1 label kept
  * prompt = the ECGBench CPSC template with this label list (same as CSN / G12EC multi-label versions)
  * split: 600 pool / rest test, random (seed 42), redrawn until every label has >= 5 positives in pool and test
Outputs: data/derived/sph_ml.json, data/derived/sph_ml_questions.json, data/splits/sph_ml.json
Images are rendered by clarity/render_sph.py into data/sph_images/sph/{ECG_ID}.png"""
import collections, csv, json, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data/sph_raw")
N_SAMPLE, MIN_POS, SEED = 3000, 50, 42
ABBR = {"1": "NORM", "21": "STach", "22": "SB", "23": "SA", "30": "PAC", "50": "AF", "51": "AFL", "60": "PVC",
        "82": "PRL", "101": "LAFB", "104": "LBBB", "105": "IRBBB", "106": "RBBB", "120": "RAD", "121": "LAD",
        "125": "LQRSV", "142": "LVH", "145": "STDEV", "146": "STTC", "147": "TAb"}
TEMPLATE = ("You are provided with an ECG signal image. Please classify the ECG image into the following diagnostic statement classes:\n"
            "{classes}\nDirectly output the class name and separate with a semicolon.")

names = {r["Code"]: r["Description"] for r in csv.DictReader(open(os.path.join(RAW, "code.csv")))}
meta = sorted(csv.DictReader(open(os.path.join(RAW, "metadata.csv"))), key=lambda r: r["ECG_ID"])
first = {}
for r in meta: first.setdefault(r["Patient_ID"], r)
cand = sorted(first.values(), key=lambda r: r["ECG_ID"])
idx = np.random.default_rng(SEED).choice(len(cand), size=N_SAMPLE, replace=False)
sample = [cand[i] for i in sorted(idx)]
base = lambda r: sorted({c.split("+")[0] for c in r["AHA_Code"].split(";")})
cnt = collections.Counter(c for r in sample for c in base(r))
codes = [c for c, n in cnt.most_common() if n >= MIN_POS]
missing = [c for c in codes if c not in ABBR]
assert not missing, f"add abbreviations for {missing}: {[names[c] for c in missing]}"
labels = {ABBR[c]: names[c] for c in codes}
classes = ", ".join(f"{ABBR[c]}({names[c].lower() if ABBR[c] != 'NORM' else 'Normal ECG'})" for c in codes)
recs = []
for r in sample:
    g = [ABBR[c] for c in codes if c in base(r)]
    if g:
        recs.append({"id": f"sph-{r['ECG_ID']}", "image": f"sph/{r['ECG_ID']}.png", "age": r["Age"], "sex": r["Sex"],
                     "conversations": [{"from": "human", "value": TEMPLATE.format(classes=classes)}, {"from": "gpt", "value": ";".join(g)}]})
os.makedirs(os.path.join(ROOT, "data/derived"), exist_ok=True)
json.dump({"labels": labels, "aha_codes": {ABBR[c]: c for c in codes}, "source": "SPH (figshare collection 5779802)",
           "n_sample": N_SAMPLE, "min_positives": MIN_POS, "records": recs}, open(os.path.join(ROOT, "data/derived/sph_ml.json"), "w"), indent=1)
json.dump(recs, open(os.path.join(ROOT, "data/derived/sph_ml_questions.json"), "w"), indent=1)

ids = [x["id"] for x in recs]; gold = {x["id"]: x["conversations"][1]["value"].split(";") for x in recs}
k = 0
while True:
    perm = np.random.default_rng(SEED + k).permutation(ids); pool, test = sorted(perm[:600]), sorted(perm[600:])
    if all(sum(L in gold[i] for i in s) >= 5 for L in labels for s in (pool, test)): break
    k += 1
json.dump({"pool": pool, "test": test, "redraws": k}, open(os.path.join(ROOT, "data/splits/sph_ml.json"), "w"), indent=1)
print(f"SPH: {len(recs)}/{N_SAMPLE} records with >=1 label; {len(labels)} labels:",
      {L: sum(L in gold[i] for i in ids) for L in labels}, f"| pool {len(pool)} / test {len(test)} (redraws {k})")
print("prompt classes:", classes)
