"""Second ECG-capable base model: MedGemma-4B fine-tuned on ECGInstruct (convaiinnovations/medgemma-4b-ecginstruct-merged).
Same training data as PULSE, different architecture -> CPSC, CSN, G12EC are out-of-domain for it too.
Per image:
  B0 : greedy generation for the dataset prompt (CPSC: ECGBench prompt; csnml/g12ml: CPSC-style prompt with that label list)
       -> outputs/medgemma/{data}/answers.jsonl
  B1 : one Yes/No question per label (same wording as clarity/label_prob.py), all labels of an image in ONE left-padded
       batch; P(yes) = sum P(Yes tokens) / (sum P(Yes) + sum P(No)) at the last position -> outputs/clarity/mg_b1_{data}_all.jsonl
--check N : compare batched P(yes) with one-question-at-a-time forwards on N images -> outputs/clarity/mg_batch_check_{data}.json
Run: <medgemma env python> -I clarity/medgemma_run.py --data cpsc
"""
import argparse, copy, gc, json, os, sys, time
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = "convaiinnovations/medgemma-4b-ecginstruct-merged"
LOCAL = os.path.join(ROOT, "models/medgemma-4b-ecginstruct-merged")  # curl copy (the hub downloader stalled on this machine)
if os.path.exists(os.path.join(LOCAL, "model-00002-of-00002.safetensors")): MODEL = LOCAL
CPSC_LABELS = {"NORM": "Normal ECG", "AF": "Atrial fibrillation", "I-AVB": "First-degree atrioventricular block",
               "LBBB": "Left bundle branch block", "RBBB": "Right bundle branch block",
               "PAC": "Premature atrial contraction", "PVC": "Premature ventricular contraction",
               "STD": "ST-segment depression", "STE": "ST-segment elevated"}
QUESTION = "You are provided with an ECG signal image. Does this ECG show {name} ({abbr})? Answer Yes or No."

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, choices=["cpsc", "csnml", "g12ml", "sphml"])
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--check", type=int, default=0)
ap.add_argument("--verify", type=int, default=0, help="recompute N random already-written rows and compare")
ap.add_argument("--no-cache", action="store_true", help="B1 by one full forward per question (slower reference path)")
a = ap.parse_args()
dev = "mps"

if a.data == "cpsc":
    LABELS = CPSC_LABELS
    data = json.load(open(os.path.join(ROOT, "data/ECGBench/cpsc-test.json")))
else:
    D = json.load(open(os.path.join(ROOT, f"data/derived/{a.data[:-2]}_ml.json"))); LABELS, data = D["labels"], D["records"]

proc = AutoProcessor.from_pretrained(MODEL); proc.tokenizer.padding_side = "right"
# SDPA + RIGHT padding: with left padding the leading pad queries see no valid key -> NaN that leaks through 0-weight
# products; with right padding every query has a valid earlier key. Logits are read at each row's last real token.
# VALIDATED CONFIG (outputs/clarity/mg_reference): eager attention, one question per forward, no padding.
# Batched SDPA with padding on this torch 2.6 / macOS 27 MPS stack produced silently wrong values for some batch shapes.
model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, device_map=dev, attn_implementation="eager").eval()


class F32Head(torch.nn.Module):
    """compute the LM head in fp32 so that P(yes) is not quantised by bf16 logits"""
    def __init__(self, lin):
        super().__init__(); self.lin = lin.float()

    def forward(self, x):
        return self.lin(x.float())


model.lm_head = F32Head(model.lm_head)


_TEXT_CFG = getattr(model.config, "text_config", model.config)
_ORIG_TEXT_ATTN = _TEXT_CFG._attn_implementation


def set_attn(impl=None):
    """Switch ONLY the language model: eager for B0 generation (torch 2.6 MPS SDPA fails on GQA decode steps),
    back to the load-time value (SDPA) for batched B1 prefill. The vision tower is never touched."""
    _TEXT_CFG._attn_implementation = impl or _ORIG_TEXT_ATTN
tok = proc.tokenizer
tid = lambda ws: sorted({i for w in ws for t in (w, "▁" + w) if (i := tok.convert_tokens_to_ids(t)) not in (None, tok.unk_token_id)})
YES, NO = tid(["Yes", "yes", "YES"]), tid(["No", "no", "NO"])
assert YES and NO, (YES, NO)


def prompt(text):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
    return proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


def pyes_rows(logits):
    p = torch.softmax(logits.float(), -1); y, n = p[:, YES].sum(-1), p[:, NO].sum(-1)
    return (y / (y + n)).tolist()


CHUNK = 8  # questions per forward; G12EC's 24-question batch pushed memory to ~48 GB and crashed the process


@torch.inference_mode()
def b1_batched(img):
    res = {}
    names = list(LABELS)
    for k in range(0, len(names), CHUNK):
        part = names[k:k + CHUNK]
        texts = [prompt(QUESTION.format(name=LABELS[L], abbr=L)) for L in part]
        v = proc(text=texts, images=[[img]] * len(texts), return_tensors="pt", padding=True).to(dev)
        v["pixel_values"] = v["pixel_values"].to(torch.bfloat16)
        last = v["attention_mask"].sum(1) - 1                  # index of each row's last real token
        logits = model(**v).logits
        out = logits[torch.arange(logits.shape[0], device=logits.device), last]
        res.update(zip(part, pyes_rows(out))); del out, logits, v
    return res


@torch.inference_mode()
def b1_single(img, L):
    v = proc(text=[prompt(QUESTION.format(name=LABELS[L], abbr=L))], images=[[img]], return_tensors="pt").to(dev)
    v["pixel_values"] = v["pixel_values"].to(torch.bfloat16)
    return pyes_rows(model(**v).logits[:, -1])[0]


def bad_b0(t):
    return (not t.strip()) or sum(ord(c) > 127 for c in t) / max(1, len(t)) > 0.05


def bad_b1(p):
    v = list(p.values()); return any(x != x for x in v) or max(v) - min(v) < 1e-6


IMG_TOK = model.config.image_token_id


@torch.inference_mode()
def b1_cached(img, labels):
    """Encode [system + user header + image] once; reuse a copy of its KV cache for every label question.
    Gemma3 applies the image's bidirectional mask only at prefill, so this equals one full forward per question."""
    inp = {L: proc(text=[prompt(QUESTION.format(name=n, abbr=L))], images=[[img]], return_tensors="pt").to(dev) for L, n in labels.items()}
    first = next(iter(inp.values()))
    k = int((first["input_ids"][0] == IMG_TOK).nonzero()[-1]) + 1
    pre = first["input_ids"][:, :k]
    assert all(torch.equal(v["input_ids"][:, :k], pre) for v in inp.values()), "prefix differs"
    out = model(input_ids=pre, attention_mask=torch.ones_like(pre), token_type_ids=first["token_type_ids"][:, :k],
                pixel_values=first["pixel_values"].to(torch.bfloat16), use_cache=True)
    past, res = out.past_key_values, {}
    for L, v in inp.items():
        suf = v["input_ids"][:, k:]
        o = model(input_ids=suf, attention_mask=torch.ones(1, k + suf.shape[1], dtype=torch.long, device=dev),
                  past_key_values=copy.deepcopy(past), cache_position=torch.arange(k, k + suf.shape[1], device=dev), use_cache=True)
        res[L] = pyes_rows(o.logits[:, -1])[0]; del o
    del out, past, inp
    return res


load = lambda x: Image.open(os.path.join(ROOT, "data/sph_images" if a.data == "sphml" else "data/ECGBench/images", x["image"])).convert("RGB")
if a.check:
    rows = []
    for x in data[: a.check]:
        img = load(x); bat = b1_cached(img, LABELS)
        for L in LABELS:
            rows.append({"id": x["id"], "label": L, "batched": bat[L], "single": b1_single(img, L)})
    diff = max(abs(r["batched"] - r["single"]) for r in rows)  # 'batched' = KV-cached path
    json.dump({"rows": rows, "max_abs_diff": diff}, open(os.path.join(ROOT, f"outputs/clarity/mg_batch_check_{a.data}.json"), "w"), indent=1)
    print("max |batched - single| =", diff); raise SystemExit

if a.verify:
    import random
    b0p, b1p = os.path.join(ROOT, f"outputs/medgemma/{a.data}/answers.jsonl"), os.path.join(ROOT, f"outputs/clarity/mg_b1_{a.data}_all.jsonl")
    old0 = {json.loads(l)["question_id"]: json.loads(l)["text"] for l in open(b0p)}
    old1 = {json.loads(l)["id"]: json.loads(l)["probs"] for l in open(b1p)}
    pick = random.Random(0).sample(sorted(set(old0) & set(old1)), a.verify)
    byid = {x["id"]: x for x in data}; rows = []
    for i in pick:
        img = load(byid[i]); new1 = b1_batched(img)
        v = proc(text=[prompt(byid[i]["conversations"][0]["value"])], images=[[img]], return_tensors="pt").to(dev)
        v["pixel_values"] = v["pixel_values"].to(torch.bfloat16); set_attn("eager")
        with torch.inference_mode(): g = model.generate(**v, max_new_tokens=256, do_sample=False)
        set_attn(); new0 = tok.decode(g[0, v["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        rows.append({"id": i, "b1_max_abs_diff": max(abs(new1[L] - old1[i][L]) for L in new1), "b0_same": new0 == old0[i]})
    out = {"n": len(rows), "b1_max_abs_diff": max(r["b1_max_abs_diff"] for r in rows), "b0_identical": sum(r["b0_same"] for r in rows), "rows": rows}
    json.dump(out, open(os.path.join(ROOT, f"outputs/clarity/mg_verify_{a.data}.json"), "w"), indent=1)
    print(f"verify {a.data}: n={out['n']} B1 max diff={out['b1_max_abs_diff']:.4f} B0 identical={out['b0_identical']}/{out['n']}"); raise SystemExit

b0_path = os.path.join(ROOT, f"outputs/medgemma/{a.data}/answers.jsonl")
b1_path = os.path.join(ROOT, f"outputs/clarity/mg_b1_{a.data}_all.jsonl")
os.makedirs(os.path.dirname(b0_path), exist_ok=True)
done = lambda p, k: {json.loads(l)[k] for l in open(p)} if os.path.exists(p) else set()
d0, d1 = done(b0_path, "question_id"), done(b1_path, "id")
f0, f1 = open(b0_path, "a"), open(b1_path, "a")
REF = json.load(open(os.path.join(ROOT, "outputs/clarity/mg_reference/cpu_fp32.json")))["p"]
CPSC_BY_ID = {x["id"]: x for x in json.load(open(os.path.join(ROOT, "data/ECGBench/cpsc-test.json")))}
CANARY = [(k.split("|")[0], k.split("|")[1]) for k in REF if k.startswith("cpsc")]


def canary_ok():
    by_img = {}
    for i, L in CANARY: by_img.setdefault(i, []).append(L)
    for i, Ls in by_img.items():
        img = Image.open(os.path.join(ROOT, "data/ECGBench/images", CPSC_BY_ID[i]["image"])).convert("RGB")
        got = ({L: b1_single(img, L) for L in Ls} if a.no_cache and set(Ls) <= set(LABELS) else
               b1_cached(img, {L: CPSC_LABELS[L] for L in Ls}))
        for L in Ls:
            if not abs(got[L] - REF[f"{i}|{L}"][0]) < 0.03:
                print(f"CANARY FAILED {i}|{L}: {got[L]:.4f} vs CPU {REF[f'{i}|{L}'][0]:.4f}", flush=True); return False
    return True


if not canary_ok(): sys.exit(4)
buf0, buf1 = [], []


def flush():
    if not canary_ok(): sys.exit(4)  # drop everything since the last passing canary
    f0.writelines(buf0); f1.writelines(buf1); f0.flush(); f1.flush(); buf0.clear(); buf1.clear()


for x in tqdm(data[: a.limit]):
    if x["id"] in d0 and x["id"] in d1: continue
    img = load(x)
    if x["id"] not in d1:
        t0 = time.time(); probs = {L: b1_single(img, L) for L in LABELS} if a.no_cache else b1_cached(img, LABELS)
        if bad_b1(probs):  # corrupted MPS state (NaN / all-equal): stop; the outer loop restarts a fresh process
            print(f"CORRUPT B1 for {x['id']} - exiting", flush=True); sys.exit(3)
        buf1.append(json.dumps({"id": x["id"], "probs": probs, "sec": round(time.time() - t0, 2)}) + "\n")
    if x["id"] not in d0:
        t0 = time.time()
        v = proc(text=[prompt(x["conversations"][0]["value"])], images=[[img]], return_tensors="pt").to(dev)
        v["pixel_values"] = v["pixel_values"].to(torch.bfloat16)
        with torch.inference_mode():
            g = model.generate(**v, max_new_tokens=256, do_sample=False)
        text = tok.decode(g[0, v["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        if bad_b0(text):  # empty or garbled text (e.g. a run of 260 non-English token strings on CSN): fresh process
            print(f"CORRUPT B0 for {x['id']}: {text[:60]!r} - exiting", flush=True); sys.exit(3)
        buf0.append(json.dumps({"question_id": x["id"], "text": text, "model_id": "medgemma-4b-ecginstruct",
                                "sec": round(time.time() - t0, 2)}) + "\n")
        del g, v
    gc.collect(); torch.mps.empty_cache()
    if len(buf1) >= 50: flush()
flush()
