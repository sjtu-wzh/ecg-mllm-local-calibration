"""Main experiment on a second base model: Qwen2-VL-7B-Instruct (general VLM, not ECG-trained) on CPSC.
Per image:
  B0 : greedy generation for the official ECGBench CPSC prompt  -> outputs/qwen2vl/cpsc-test/answers.jsonl
  B1 : 9 Yes/No label questions (same wording as clarity/label_prob.py); the shared prefix (system + user header +
       image, ending at <|vision_end|>) is encoded once, its KV cache is reused and cropped back after each question
       -> outputs/clarity/qwen_b1_cpsc_all.jsonl
--check N: compare cached vs full-forward P(yes) on N images and save outputs/clarity/qwen_cache_check.json
Run with the separate env: <qwen-vl env python> -I clarity/qwen_cpsc.py
"""
import argparse, gc, json, os, time
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = "Qwen/Qwen2-VL-7B-Instruct"
CPSC_LABELS = {"NORM": "Normal ECG", "AF": "Atrial fibrillation", "I-AVB": "First-degree atrioventricular block",
               "LBBB": "Left bundle branch block", "RBBB": "Right bundle branch block",
               "PAC": "Premature atrial contraction", "PVC": "Premature ventricular contraction",
               "STD": "ST-segment depression", "STE": "ST-segment elevated"}
QUESTION = "You are provided with an ECG signal image. Does this ECG show {name} ({abbr})? Answer Yes or No."

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--check", type=int, default=0)
ap.add_argument("--skip-b0", action="store_true")
a = ap.parse_args()
dev = "mps"
proc = AutoProcessor.from_pretrained(MODEL, min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28)
model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map=dev).eval()


class F32Head(torch.nn.Module):
    """bf16 logits are quantised coarsely (P(yes) snaps to values like 0.5 / sigmoid(-1)); compute the LM head in fp32."""
    def __init__(self, lin):
        super().__init__(); self.lin = lin.float()

    def forward(self, x):
        return self.lin(x.float())


model.lm_head = F32Head(model.lm_head)
tok = proc.tokenizer
VEND = tok.convert_tokens_to_ids("<|vision_end|>")


def tid(words):
    out = set()
    for w in words:
        for t in (w, "Ġ" + w):
            i = tok.convert_tokens_to_ids(t)
            if i is not None and i != tok.unk_token_id: out.add(i)
    return sorted(out)


YES, NO = tid(["Yes", "yes", "YES"]), tid(["No", "no", "NO"])


def build(img, text):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
    prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return proc(text=[prompt], images=[img], return_tensors="pt").to(dev)


def pyes(logits):
    p = torch.softmax(logits.float(), -1); y, n = p[YES].sum().item(), p[NO].sum().item()
    return y / (y + n), tok.convert_ids_to_tokens(int(logits.argmax()))


@torch.inference_mode()
def b1_cached(img):
    inp = {L: build(img, QUESTION.format(name=n, abbr=L)) for L, n in CPSC_LABELS.items()}
    first = next(iter(inp.values()))
    k = int((first["input_ids"][0] == VEND).nonzero()[0]) + 1
    prefix = first["input_ids"][:, :k]
    assert all(torch.equal(v["input_ids"][:, :k], prefix) for v in inp.values()), "prefix differs"
    model.rope_deltas = None
    out = model(input_ids=prefix, attention_mask=torch.ones_like(prefix), pixel_values=first["pixel_values"],
                image_grid_thw=first["image_grid_thw"], use_cache=True)
    past = out.past_key_values
    probs, top = {}, {}
    for L, v in inp.items():
        suf = v["input_ids"][:, k:]
        o = model(input_ids=suf, attention_mask=torch.ones(1, k + suf.shape[1], dtype=torch.long, device=dev),
                  past_key_values=past, cache_position=torch.arange(k, k + suf.shape[1], device=dev), use_cache=True)
        probs[L], top[L] = pyes(o.logits[0, -1])
        past.crop(k); del o
    del out, past, inp
    return probs, top


@torch.inference_mode()
def b1_full(img, L):
    v = build(img, QUESTION.format(name=CPSC_LABELS[L], abbr=L)); model.rope_deltas = None
    return pyes(model(**v).logits[0, -1])[0]


data = json.load(open(os.path.join(ROOT, "data/ECGBench/cpsc-test.json")))
if a.check:
    rows = []
    for x in data[: a.check]:
        img = Image.open(os.path.join(ROOT, "data/ECGBench/images", x["image"])).convert("RGB")
        cached, _ = b1_cached(img)
        for L in ("NORM", "AF", "STD"):
            rows.append({"id": x["id"], "label": L, "cached": cached[L], "full": b1_full(img, L)})
    diff = max(abs(r["cached"] - r["full"]) for r in rows)
    json.dump({"rows": rows, "max_abs_diff": diff}, open(os.path.join(ROOT, "outputs/clarity/qwen_cache_check.json"), "w"), indent=1)
    print("max |cached - full| =", diff); raise SystemExit

b0_path = os.path.join(ROOT, "outputs/qwen2vl/cpsc-test/answers.jsonl")
b1_path = os.path.join(ROOT, "outputs/clarity/qwen_b1_cpsc_all.jsonl")
os.makedirs(os.path.dirname(b0_path), exist_ok=True)
done = lambda p, k: {json.loads(l)[k] for l in open(p)} if os.path.exists(p) else set()
d0, d1 = done(b0_path, "question_id"), done(b1_path, "id")
f0, f1 = open(b0_path, "a"), open(b1_path, "a")
for x in tqdm(data[: a.limit]):
    if x["id"] in d1 and (a.skip_b0 or x["id"] in d0): continue
    img = Image.open(os.path.join(ROOT, "data/ECGBench/images", x["image"])).convert("RGB")
    t0 = time.time()
    if x["id"] not in d1:
        probs, top = b1_cached(img)
        f1.write(json.dumps({"id": x["id"], "probs": probs, "top": top, "sec": round(time.time() - t0, 2)}) + "\n"); f1.flush()
        gc.collect(); torch.mps.empty_cache()  # free B1 buffers before generation
    if not a.skip_b0 and x["id"] not in d0:
        t1 = time.time(); v = build(img, x["conversations"][0]["value"]); model.rope_deltas = None
        with torch.inference_mode():
            g = model.generate(**v, max_new_tokens=256, do_sample=False)
        text = tok.decode(g[0, v["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        f0.write(json.dumps({"question_id": x["id"], "text": text, "model_id": "Qwen2-VL-7B-Instruct",
                             "sec": round(time.time() - t1, 2)}) + "\n"); f0.flush()
        del g, v
    gc.collect(); torch.mps.empty_cache()
