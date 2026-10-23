"""Map PTB-XL SCP codes of the ECGBench PTB-XL images (PTB-XL fold 10) to the 9 CPSC labels,
and make a fixed, label-stratified fit / held-out split for the zero-shot transfer calibrator.

STE: PTB-XL has only 3 'STE_' records here, so the primary mapping also counts the injury codes
(INJAS/INJAL/INJIN/INJLA/INJIL, PTB-XL class MI) as ST elevation. This is an approximation and is
reported as such; 'STE_strict' keeps STE_ only for a sensitivity analysis."""
import ast, collections, csv, json, os, re
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS = ["NORM", "AF", "I-AVB", "LBBB", "RBBB", "PAC", "PVC", "STD", "STE"]
SCP_TO_CPSC = {"NORM": ["NORM"], "AF": ["AFIB"], "I-AVB": ["1AVB"], "LBBB": ["CLBBB"], "RBBB": ["CRBBB"],
               "PAC": ["PAC"], "PVC": ["PVC"], "STD": ["STD_"],
               "STE": ["STE_", "INJAS", "INJAL", "INJIN", "INJLA", "INJIL"]}
SEED = 42

db = {int(r["ecg_id"]): ast.literal_eval(r["scp_codes"])
      for r in csv.DictReader(open(os.path.join(ROOT, "data/ptbxl_meta/ptbxl_database.csv")))}
labels, strict_ste = {}, {}
for x in json.load(open(os.path.join(ROOT, "data/ECGBench/ptb-test.json"))):
    codes = db[int(re.match(r"ptb-xl/(\d+)_hr", x["image"]).group(1))]  # any listed code counts (rhythm codes carry likelihood 0)
    labels[x["id"]] = [L for L in LABELS if any(c in codes for c in SCP_TO_CPSC[L])]
    strict_ste[x["id"]] = "STE_" in codes

json.dump({"mapping": SCP_TO_CPSC, "labels": labels, "STE_strict": strict_ste},
          open(os.path.join(ROOT, "data/ptbxl_meta/ptbxl_cpsc_labels.json"), "w"), indent=1)

ids = sorted(labels)
strata = [";".join(labels[i]) or "none" for i in ids]
cnt = collections.Counter(strata)
strata = [s if cnt[s] >= 2 else "rare" for s in strata]
fit, held = train_test_split(ids, train_size=0.7, stratify=strata, random_state=SEED)
json.dump({"fit": sorted(fit), "heldout": sorted(held)}, open(os.path.join(ROOT, "data/splits/ptbxl.json"), "w"), indent=1)

print(f"PTB-XL images: {len(ids)}  fit {len(fit)} / held-out {len(held)}")
for L in LABELS:
    print(f"  {L:6s} fit {sum(L in labels[i] for i in fit):4d}  held-out {sum(L in labels[i] for i in held):4d}")
print("  records with none of the 9 labels:", sum(not labels[i] for i in ids))
