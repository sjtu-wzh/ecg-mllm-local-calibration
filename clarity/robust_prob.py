"""Experiment 3 (robustness): per-label Yes/No probabilities from frozen PULSE-7B for perturbed SPH images or
paraphrased label questions. Same computation as clarity/label_prob.py (frozen; not modified): the image prefix is
encoded once, its KV cache is reused for every label question, P(yes) = mass(Yes tokens) / (mass(Yes) + mass(No)).
  --image-root  folder holding sph/{ECG_ID}.png (data/sph_images = clean, data/sph_images_robust/<COND> = perturbed)
  --template    ORIG (label_prob.py wording) or a key of data/derived/sph_robust_prompts.json (QA, QB)
  --split       pool | test | both  (subsets in data/splits/sph_robust.json)
Output jsonl (resumable): {"id", "probs", "top", "sec"}
"""
import argparse, gc, json, os, time
import torch
from PIL import Image
from tqdm import tqdm
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORIG = "You are provided with an ECG signal image. Does this ECG show {name} ({abbr})? Answer Yes or No."
ap = argparse.ArgumentParser()
ap.add_argument("--model-path", default="PULSE-ECG/PULSE-7B")
ap.add_argument("--image-root", required=True)
ap.add_argument("--template", default="ORIG")
ap.add_argument("--split", default="both", choices=["pool", "test", "both"])
ap.add_argument("--out", required=True)
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--device", default="mps")
a = ap.parse_args()
TEMPLATE = ORIG if a.template == "ORIG" else json.load(open(os.path.join(ROOT, "data/derived/sph_robust_prompts.json")))[a.template]

disable_torch_init()
tok, model, proc, _ = load_pretrained_model(a.model_path, None, get_model_name_from_path(a.model_path), device_map=a.device, device=a.device)
model.eval()


def token_ids(words):
    ids = set()
    for w in words:
        for t in (w, "▁" + w):
            i = tok.convert_tokens_to_ids(t)
            if i is not None and i != tok.unk_token_id:
                ids.add(i)
    return sorted(ids)


YES, NO = token_ids(["Yes", "yes", "YES"]), token_ids(["No", "no", "NO"])


def split_prompt(question):
    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question); conv.append_message(conv.roles[1], None)
    ids = tokenizer_image_token(conv.get_prompt(), tok, IMAGE_TOKEN_INDEX, return_tensors="pt")
    k = (ids == IMAGE_TOKEN_INDEX).nonzero()[0].item()
    return ids[: k + 1], ids[k + 1:]


D = json.load(open(os.path.join(ROOT, "data/derived/sph_ml.json"))); data = {x["id"]: x for x in D["records"]}
parts = {L: split_prompt(TEMPLATE.format(name=n, abbr=L)) for L, n in D["labels"].items()}
prefix_ids = next(iter(parts.values()))[0]
assert all(torch.equal(p, prefix_ids) for p, _ in parts.values()), "prefix must be label-independent"
sp = json.load(open(os.path.join(ROOT, "data/splits/sph_robust.json")))
ids = (sp["pool"] + sp["test"]) if a.split == "both" else sp[a.split]
ids = ids[: a.limit]
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
done = {json.loads(l)["id"] for l in open(a.out)} if os.path.exists(a.out) else set()
out = open(a.out, "a"); embed = model.get_model().embed_tokens

for i in tqdm([i for i in ids if i not in done]):
    t0 = time.time()
    img = Image.open(os.path.join(ROOT, a.image_root, data[i]["image"])).convert("RGB")
    pix = process_images([img], proc, model.config)[0].unsqueeze(0).to(a.device, dtype=torch.float16)
    probs, top = {}, {}
    with torch.inference_mode():
        _, _, _, _, pre_emb, _ = model.prepare_inputs_labels_for_multimodal(
            prefix_ids.unsqueeze(0).to(a.device), None, None, None, None, pix, image_sizes=[img.size])
        past = model(inputs_embeds=pre_emb, use_cache=True).past_key_values
        n_pre = pre_emb.shape[1]
        for L, (_, suf) in parts.items():
            suf = suf.unsqueeze(0).to(a.device)
            mask = torch.ones(1, n_pre + suf.shape[1], dtype=torch.long, device=a.device)
            logits = model(inputs_embeds=embed(suf), past_key_values=past, attention_mask=mask, use_cache=True).logits[0, -1].float()
            p = torch.softmax(logits, -1); y, n = p[YES].sum().item(), p[NO].sum().item()
            probs[L] = y / (y + n); top[L] = tok.convert_ids_to_tokens(int(logits.argmax()))
    out.write(json.dumps({"id": i, "probs": probs, "top": top, "sec": round(time.time() - t0, 2)}) + "\n"); out.flush()
    del past, pre_emb, pix
    gc.collect()
    if a.device == "mps":
        torch.mps.empty_cache()
