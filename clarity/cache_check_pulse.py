"""Check that the KV-cached Yes/No probabilities of clarity/label_prob.py equal a plain full forward pass (PULSE-7B).
Compares outputs/clarity/b1_cpsc_dev.jsonl (cached path) with a full forward of the same prompt on N dev images.
Output: outputs/clarity/pulse_cache_check.json"""
import argparse, json, os
import torch
from PIL import Image
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token, process_images
from llava.model.builder import load_pretrained_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAMES = {"NORM": "Normal ECG", "AF": "Atrial fibrillation", "I-AVB": "First-degree atrioventricular block",
         "STD": "ST-segment depression", "STE": "ST-segment elevated"}
QUESTION = "You are provided with an ECG signal image. Does this ECG show {name} ({abbr})? Answer Yes or No."
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=5); a = ap.parse_args()
tok, model, proc, _ = load_pretrained_model("PULSE-ECG/PULSE-7B", None, "PULSE-7B", device_map="mps", device="mps")
ids = lambda ws: sorted({i for w in ws for t in (w, "▁" + w) if (i := tok.convert_tokens_to_ids(t)) not in (None, tok.unk_token_id)})
YES, NO = ids(["Yes", "yes", "YES"]), ids(["No", "no", "NO"])
cached = {r["id"]: r["probs"] for r in map(json.loads, open(os.path.join(ROOT, "outputs/clarity/b1_cpsc_dev.jsonl")))}
data = {x["id"]: x for x in json.load(open(os.path.join(ROOT, "data/ECGBench/cpsc-test.json")))}
rows = []
for i in list(cached)[: a.n]:
    img = Image.open(os.path.join(ROOT, "data/ECGBench/images", data[i]["image"])).convert("RGB")
    pix = process_images([img], proc, model.config)[0].unsqueeze(0).to("mps", dtype=torch.float16)
    for L, nm in NAMES.items():
        conv = conv_templates["llava_v1"].copy()
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + QUESTION.format(name=nm, abbr=L)); conv.append_message(conv.roles[1], None)
        x = tokenizer_image_token(conv.get_prompt(), tok, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to("mps")
        with torch.inference_mode():
            p = torch.softmax(model(input_ids=x, images=pix, image_sizes=[img.size]).logits[0, -1].float(), -1)
        y, n = p[YES].sum().item(), p[NO].sum().item()
        rows.append({"id": i, "label": L, "cached": cached[i][L], "full": y / (y + n)})
diff = max(abs(r["cached"] - r["full"]) for r in rows)
json.dump({"rows": rows, "max_abs_diff": diff, "n_images": a.n}, open(os.path.join(ROOT, "outputs/clarity/pulse_cache_check.json"), "w"), indent=1)
print("max |cached - full| =", diff)
