"""Reference check for MedGemma-ECGInstruct numerics.
Computes P(yes) for a fixed set of (image, label) pairs one question at a time (no batching, no padding) on a chosen
device/dtype, REPEATS times in the same process, and saves them. CPU float32 is the reference; MPS configurations
are compared against it with clarity/mg_reference.py --compare.
Usage: python -I clarity/mg_reference.py --device cpu --dtype float32 --tag cpu_fp32
       python -I clarity/mg_reference.py --device mps --dtype bfloat16 --tag mps_bf16_torchX
       python -I clarity/mg_reference.py --compare cpu_fp32 mps_bf16_torchX"""
import argparse, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs/clarity/mg_reference")
ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cpu"); ap.add_argument("--dtype", default="float32"); ap.add_argument("--tag")
ap.add_argument("--repeats", type=int, default=2); ap.add_argument("--compare", nargs="*")
a = ap.parse_args()

if a.compare:
    ref = json.load(open(os.path.join(OUT, f"{a.compare[0]}.json")))
    for tag in a.compare[1:]:
        t = json.load(open(os.path.join(OUT, f"{tag}.json")))
        d_ref = max(abs(t["p"][k][r] - ref["p"][k][0]) for k in ref["p"] for r in range(len(t["p"][k])))
        d_rep = max(max(v) - min(v) for v in t["p"].values())
        print(f"{tag}: max |vs {a.compare[0]}| = {d_ref:.4f}; max spread across repeats = {d_rep:.4f}; torch {t['torch']}")
    sys.exit(0)

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
MODEL = os.path.join(ROOT, "models/medgemma-4b-ecginstruct-merged")
Q = "You are provided with an ECG signal image. Does this ECG show {name} ({abbr})? Answer Yes or No."
csn = json.load(open(os.path.join(ROOT, "data/derived/csn_ml.json")))
cpsc = json.load(open(os.path.join(ROOT, "data/ECGBench/cpsc-test.json")))
PAIRS = [(x, L, csn["labels"][L]) for x in csn["records"][:3] for L in ("SB", "NSR", "STach")] + \
        [(x, L, n) for x in cpsc[:2] for L, n in (("NORM", "Normal ECG"), ("AF", "Atrial fibrillation"))]
dt = getattr(torch, a.dtype)
proc = AutoProcessor.from_pretrained(MODEL)
model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=dt, attn_implementation="eager").to(a.device).eval()
tok = proc.tokenizer
tid = lambda ws: sorted({i for w in ws for t in (w, "▁" + w) if (i := tok.convert_tokens_to_ids(t)) not in (None, tok.unk_token_id)})
YES, NO = tid(["Yes", "yes", "YES"]), tid(["No", "no", "NO"])
res = {"device": a.device, "dtype": a.dtype, "torch": torch.__version__, "p": {}}
for r in range(a.repeats):
    for x, L, name in PAIRS:
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": Q.format(name=name, abbr=L)}]}]
        v = proc(text=[proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)],
                 images=[[Image.open(os.path.join(ROOT, "data/ECGBench/images", x["image"])).convert("RGB")]], return_tensors="pt").to(a.device)
        v["pixel_values"] = v["pixel_values"].to(dt)
        with torch.inference_mode():
            p = torch.softmax(model(**v).logits[0, -1].float(), -1)
        y, n = p[YES].sum().item(), p[NO].sum().item()
        res["p"].setdefault(f"{x['id']}|{L}", []).append(y / (y + n))
    print(f"repeat {r + 1} done", flush=True)
os.makedirs(OUT, exist_ok=True); json.dump(res, open(os.path.join(OUT, f"{a.tag}.json"), "w"), indent=1)
print({k: [round(v, 4) for v in vs] for k, vs in res["p"].items()})
