"""B1 for multiple-choice subsets (G12EC, CSN): option-letter probabilities from frozen PULSE-7B.
The official question is kept verbatim; the assistant turn is primed with "Answer:" and the next-token
distribution is renormalised over the 8 option letters A-H (both "A" and "▁A" token forms).
Output jsonl (resumable): {"id", "logp": {letter: log P(letter | A-H)}, "top": argmax token over the vocabulary, "sec"}
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
LETTERS = "ABCDEFGH"

ap = argparse.ArgumentParser()
ap.add_argument("--model-path", default="PULSE-ECG/PULSE-7B")
ap.add_argument("--subset", required=True, choices=["g12-test-no-cot", "csn-test-no-cot"])
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--ids-file", default=None, help="JSON list of record ids to compute (verification); requires --out")
ap.add_argument("--device", default="mps")
ap.add_argument("--out", default=None)
a = ap.parse_args()
out_path = a.out or os.path.join(ROOT, f"outputs/clarity/mcq_{a.subset.split('-')[0]}_all.jsonl")

disable_torch_init()
tok, model, proc, _ = load_pretrained_model(a.model_path, None, get_model_name_from_path(a.model_path),
                                            device_map=a.device, device=a.device)
model.eval()
letter_ids = {L: sorted({i for t in (L, "▁" + L) if (i := tok.convert_tokens_to_ids(t)) not in (None, tok.unk_token_id)})
              for L in LETTERS}
assert all(letter_ids.values()), letter_ids

data = json.load(open(os.path.join(ROOT, "data/ECGBench", f"{a.subset}.json")))[: a.limit]
if a.ids_file:
    keep = set(json.load(open(a.ids_file))); data = [x for x in json.load(open(os.path.join(ROOT, "data/ECGBench", f"{a.subset}.json"))) if x["id"] in keep]
done = {json.loads(l)["id"] for l in open(out_path)} if os.path.exists(out_path) else set()
out = open(out_path, "a")
for x in tqdm([x for x in data if x["id"] not in done]):
    t0 = time.time()
    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + x["conversations"][0]["value"])
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt() + " Answer:"
    ids = tokenizer_image_token(prompt, tok, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(a.device)
    img = Image.open(os.path.join(ROOT, "data/ECGBench/images", x["image"])).convert("RGB")
    pix = process_images([img], proc, model.config)[0].unsqueeze(0).to(a.device, dtype=torch.float16)
    with torch.inference_mode():
        logits = model(input_ids=ids, images=pix, image_sizes=[img.size]).logits[0, -1].float()
    p = torch.softmax(logits, -1)
    mass = {L: p[letter_ids[L]].sum().item() for L in LETTERS}
    z = sum(mass.values())
    logp = {L: float(torch.log(torch.tensor(max(mass[L], 1e-30) / z))) for L in LETTERS}
    out.write(json.dumps({"id": x["id"], "logp": logp, "letter_mass": round(z, 4),
                          "top": tok.convert_ids_to_tokens(int(logits.argmax())), "sec": round(time.time() - t0, 2)}) + "\n")
    out.flush()
    del logits, p, pix, ids
    gc.collect()
    if a.device == "mps":
        torch.mps.empty_cache()
