"""Build the derived question files used in this study from ECGBench (and PTB-XL metadata). Deterministic.
  data/derived/{g12,csn}_ml.json + *_ml_questions.json : multi-label versions of the G12EC / CSN multiple-choice subsets.
      Gold = diagnosis codes in the gold option; label set = codes with >= 40 positives; records with >= 1 label kept;
      prompt = the ECGBench CPSC template with this label list.
  data/derived/ptbxl_cpsc9.json : PTB-XL fold-10 images (ECGBench ptb-test) with the EXACT CPSC 9-label prompt;
      gold = CPSC-equivalent labels from SCP codes (needs data/ptbxl_meta/ptbxl_cpsc_labels.json from clarity/ptbxl_labels.py);
      records with >= 1 label kept.
Run after scripts/prepare_ecgbench.py (and clarity/ptbxl_labels.py for the PTB-XL file)."""
import collections, json, os, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
J = lambda p: json.load(open(os.path.join(ROOT, p)))
OUT = os.path.join(ROOT, "data/derived"); os.makedirs(OUT, exist_ok=True)
CPSC_TEMPLATE = ("You are provided with an ECG signal image. Please classify the ECG image into the following diagnostic statement classes:\n"
                 "{classes}\nDirectly output the class name and separate with a semicolon.")

for src, short in (("g12-test-no-cot", "g12"), ("csn-test-no-cot", "csn")):
    d = J(f"data/ECGBench/{src}.json")
    names, gold = {}, {}
    for x in d:
        opts = dict(re.findall(r"\n([A-H])\. ?([^\n]+)", x["conversations"][0]["value"]))
        for v in opts.values():
            for c, nm in re.findall(r"([A-Za-z0-9]+)\(([^)]*)\)", v):
                names.setdefault(c, re.sub(r"\s+", " ", nm).strip())
        g = x["conversations"][1]["value"].strip()[0]
        gold[x["id"]] = sorted(set(re.findall(r"([A-Za-z0-9]+)\(", opts[g])))
    cnt = collections.Counter(c for v in gold.values() for c in v)
    labels = [c for c, n in cnt.most_common() if n >= 40]
    classes = ", ".join(f"{c}({names[c]})" for c in labels)
    recs = []
    for x in d:
        g = [c for c in labels if c in gold[x["id"]]]
        if g:
            recs.append({"id": x["id"], "image": x["image"], "conversations": [
                {"from": "human", "value": CPSC_TEMPLATE.format(classes=classes)}, {"from": "gpt", "value": ";".join(g)}]})
    json.dump({"labels": {c: names[c] for c in labels}, "source": src, "min_positives": 40, "records": recs},
              open(os.path.join(OUT, f"{short}_ml.json"), "w"), indent=1)
    json.dump(recs, open(os.path.join(OUT, f"{short}_ml_questions.json"), "w"), indent=1)
    print(f"{short}: {len(labels)} labels, {len(recs)}/{len(d)} records")

lab_path = os.path.join(ROOT, "data/ptbxl_meta/ptbxl_cpsc_labels.json")
if os.path.exists(lab_path):
    cpsc = J("data/ECGBench/cpsc-test.json"); prompt = cpsc[0]["conversations"][0]["value"]
    assert all(x["conversations"][0]["value"] == prompt for x in cpsc)
    lab = json.load(open(lab_path))["labels"]
    out = [{"id": x["id"], "image": x["image"],
            "conversations": [{"from": "human", "value": prompt}, {"from": "gpt", "value": "; ".join(lab[x["id"]])}]}
           for x in J("data/ECGBench/ptb-test.json") if lab[x["id"]]]
    json.dump(out, open(os.path.join(OUT, "ptbxl_cpsc9.json"), "w"), indent=1)
    print(f"ptbxl_cpsc9: {len(out)} records")
else:
    print("skip ptbxl_cpsc9 (run clarity/ptbxl_labels.py first)")
