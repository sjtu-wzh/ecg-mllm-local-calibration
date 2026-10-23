"""Step 0: fixed, label-stratified dev/test split of ECGBench subsets.
All tuning (thresholds, prompts, gating) happens on dev; test is reported once."""
import collections, json, os
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(ROOT, "data/ECGBench")
OUT = os.path.join(ROOT, "data/splits")
CPSC_LABELS = ["NORM", "AF", "I-AVB", "LBBB", "RBBB", "PAC", "PVC", "STD", "STE"]
SEED = 42


def split(ids, strata, n_dev):
    # merge rare label combinations so every stratum has >= 2 members
    cnt = collections.Counter(strata)
    strata = [s if cnt[s] >= 2 else "rare" for s in strata]
    dev, test = train_test_split(ids, train_size=n_dev, stratify=strata, random_state=SEED)
    return sorted(dev), sorted(test)


os.makedirs(OUT, exist_ok=True)

d = json.load(open(os.path.join(BENCH, "cpsc-test.json")))
ids = [x["id"] for x in d]
strata = [";".join(l for l in CPSC_LABELS if l in x["conversations"][1]["value"]) for x in d]
dev, test = split(ids, strata, 600)
json.dump({"dev": dev, "test": test}, open(os.path.join(OUT, "cpsc.json"), "w"), indent=1)
print(f"cpsc: dev {len(dev)} / test {len(test)}")

d = json.load(open(os.path.join(BENCH, "mmmu-ecg.json")))
ids = [x["id"] for x in d]
strata = [x["conversations"][1]["value"].strip() for x in d]
dev, test = split(ids, strata, 100)
json.dump({"dev": dev, "test": test}, open(os.path.join(OUT, "mmmu.json"), "w"), indent=1)
print(f"mmmu: dev {len(dev)} / test {len(test)}")
