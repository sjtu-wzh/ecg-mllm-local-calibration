"""Download PULSE-ECG/ECGBench and export images + per-subset JSON (README format)."""
import json, os, sys
from concurrent.futures import ThreadPoolExecutor
from datasets import load_dataset
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_SAVE_DIR = os.path.join(ROOT, "data/ECGBench/images")
JSON_SAVE_DIR = os.path.join(ROOT, "data/ECGBench")
SUBSETS = ['arena', 'code15-test', 'cpsc-test', 'csn-test-no-cot', 'ecgqa-test',
           'g12-test-no-cot', 'mmmu-ecg', 'ptb-test', 'ptb-test-report']

os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)
for name in SUBSETS:
    out = os.path.join(JSON_SAVE_DIR, f"{name}.json")
    if os.path.exists(out):
        print(f"skip {name} (exists)"); continue
    ds = load_dataset("PULSE-ECG/ECGBench", name=name)["test"]

    def work(idx):
        item = ds[idx]
        path = os.path.join(IMAGE_SAVE_DIR, item["image_path"])
        os.makedirs(os.path.dirname(path), exist_ok=True)
        item["image"].save(path)
        return {"id": item["id"], "image": item["image_path"], "conversations": item["conversations"]}

    with ThreadPoolExecutor(max_workers=8) as ex:
        items = list(tqdm(ex.map(work, range(len(ds))), total=len(ds), desc=name))
    with open(out, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=4, ensure_ascii=False)
    print(f"{name}: {len(items)} samples -> {out}", flush=True)
