"""Experiment 1 (baselines): frozen PULSE-7B features for linear probes.
One forward per image with a NEUTRAL prompt ("Describe the ECG.") so no diagnosis name enters the features.
Saved per image (float16):
  llm_L{8,16,24,32}_img : LLM hidden state at layer L, mean over the image-token span
  llm_L{8,16,24,32}_last: LLM hidden state at layer L, last prompt token
  clip_cls              : CLIP ViT-L/14-336 (PULSE's vision tower) pooled CLS on the whole image resized to 336
  clip_patch            : CLIP penultimate-layer patch tokens, mean (the layer LLaVA feeds to the projector)
Output: outputs/features/pulse_{data}.npz  (ids + one array per feature). --check N compares a cached-free re-run.
Run: <pulse-llava env python> -I clarity/extract_features.py --data cpsc"""
import argparse, json, os, time
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token, process_images
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAYERS = [8, 16, 24, 32]
PROMPT = "Describe the ECG."
ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, choices=["cpsc", "csnml", "g12ml", "sphml"])
ap.add_argument("--limit", type=int, default=None)
a = ap.parse_args()
dev = "mps"

if a.data == "cpsc":
    recs = json.load(open(os.path.join(ROOT, "data/ECGBench/cpsc-test.json")))
else:
    recs = json.load(open(os.path.join(ROOT, f"data/derived/{a.data[:-2]}_ml.json")))["records"]
recs = recs[: a.limit]
IMG_ROOT = os.path.join(ROOT, "data/sph_images" if a.data == "sphml" else "data/ECGBench/images")

disable_torch_init()
tok, model, proc, _ = load_pretrained_model("PULSE-ECG/PULSE-7B", None, "PULSE-7B", device_map=dev, device=dev)
model.eval()
clip = model.get_model().get_vision_tower().vision_tower  # HF CLIPVisionModel (frozen during PULSE training)
conv = conv_templates["llava_v1"].copy()
conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + PROMPT); conv.append_message(conv.roles[1], None)
ids = tokenizer_image_token(conv.get_prompt(), tok, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(dev)
k = int((ids[0] == IMAGE_TOKEN_INDEX).nonzero()[0])  # image span starts here after expansion

feats = {f"llm_L{L}_{p}": [] for L in LAYERS for p in ("img", "last")}
feats.update(clip_cls=[], clip_patch=[])
t0 = time.time()
for x in tqdm(recs):
    img = Image.open(os.path.join(IMG_ROOT, x["image"])).convert("RGB")
    pix = process_images([img], proc, model.config)[0].unsqueeze(0).to(dev, dtype=torch.float16)  # anyres stack, [1, P, 3, 336, 336]
    with torch.inference_mode():
        _, _, _, _, emb, _ = model.prepare_inputs_labels_for_multimodal(ids, None, None, None, None, pix, image_sizes=[img.size])
        n_img = emb.shape[1] - (ids.shape[1] - 1)
        hs = model(inputs_embeds=emb, output_hidden_states=True).hidden_states  # tuple: embeddings + 32 layers
        for L in LAYERS:
            feats[f"llm_L{L}_img"].append(hs[L][0, k:k + n_img].float().mean(0).cpu().numpy())
            feats[f"llm_L{L}_last"].append(hs[L][0, -1].float().cpu().numpy())
        base = pix[0, :1]  # anyres patch 0 = whole image resized to 336x336
        co = clip(base, output_hidden_states=True)
        feats["clip_cls"].append(co.pooler_output[0].float().cpu().numpy())
        feats["clip_patch"].append(co.hidden_states[-2][0, 1:].float().mean(0).cpu().numpy())
    del hs, emb, co
    torch.mps.empty_cache()
os.makedirs(os.path.join(ROOT, "outputs/features"), exist_ok=True)
out = os.path.join(ROOT, f"outputs/features/pulse_{a.data}{'' if a.limit is None else '_smoke'}.npz")
np.savez_compressed(out, ids=np.array([x["id"] for x in recs]), **{f: np.stack(v).astype(np.float16) for f, v in feats.items()})
print(f"saved {out}: {len(recs)} images, {time.time() - t0:.0f}s, dims", {f: len(v[0]) for f, v in feats.items()})
