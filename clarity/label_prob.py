"""B1 (+ M4 signal): per-label Yes/No probabilities from frozen PULSE-7B (CPSC, PTB-XL or CODE-15% images).

The image prefix (system prompt + image tokens) is encoded once per ECG and its KV cache is
reused for every label question, so 9 labels cost ~1 full forward + 9 short ones.
Output jsonl (resumable): {"id", "probs": {label: P(yes)}, "top": {label: argmax token}, "sec"}
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
CPSC_LABELS = {  # abbreviations and names exactly as in the ECGBench CPSC prompt
    "NORM": "Normal ECG", "AF": "Atrial fibrillation", "I-AVB": "First-degree atrioventricular block",
    "LBBB": "Left bundle branch block", "RBBB": "Right bundle branch block",
    "PAC": "Premature atrial contraction", "PVC": "Premature ventricular contraction",
    "STD": "ST-segment depression", "STE": "ST-segment elevated",
}
CODE15_LABELS = {  # names as in the ECGBench CODE-15% prompt; "NORM" serves the normal/abnormal task
    "NORM": "Normal ECG", "1dAVb": "1st degree av block", "RBBB": "right bundle branch block",
    "LBBB": "left bundle branch block", "SB": "sinus bradycardia", "ST": "sinus tachycardia", "AF": "atrial fibrillation",
}
QUESTION = "You are provided with an ECG signal image. Does this ECG show {name} ({abbr})? Answer Yes or No."
# M2: clinical evidence questions; M3 links each diagnosis only to its supporting evidence
EVIDENCE = {
    "irregular": "Is the rhythm irregularly irregular?",
    "no_p": "Are P waves absent and replaced by fibrillatory waves?",
    "long_pr": "Is the PR interval prolonged (longer than 200 ms)?",
    "wide_qrs": "Is the QRS duration prolonged (120 ms or longer)?",
    "rsr_v1": "Is there an rSR' pattern in lead V1?",
    "notched_r_lat": "Is there a broad notched R wave in leads I, V5 or V6?",
    "prem_atrial": "Are there premature beats with a narrow QRS complex and an abnormal P wave?",
    "prem_vent": "Are there premature beats with a wide and bizarre QRS complex?",
    "st_elev": "Is there ST-segment elevation in two or more contiguous leads?",
    "st_dep": "Is there ST-segment depression in two or more contiguous leads?",
    "t_inv": "Are there T wave inversions?",
}
EV_QUESTION = "You are provided with an ECG signal image. {q} Answer Yes or No."

ap = argparse.ArgumentParser()
ap.add_argument("--model-path", default="PULSE-ECG/PULSE-7B")
ap.add_argument("--data", default="cpsc", choices=["cpsc", "ptbxl", "code15", "g12ml", "csnml", "sphml"],
                help="ptbxl = ECGBench PTB-XL images (fold 10); code15 uses its own 7 label questions")
ap.add_argument("--split", default="dev", choices=["dev", "test", "all"])
ap.add_argument("--set", default="labels", choices=["labels", "evidence"])
ap.add_argument("--out", default=None)
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--ids-file", default=None, help="JSON list of record ids to compute (verification); requires --out")
ap.add_argument("--device", default="mps")
a = ap.parse_args()
out_path = a.out or os.path.join(ROOT, f"outputs/clarity/{'b1' if a.set == 'labels' else 'ev'}_{a.data}_{a.split}.jsonl")

disable_torch_init()
tok, model, proc, _ = load_pretrained_model(a.model_path, None, get_model_name_from_path(a.model_path),
                                            device_map=a.device, device=a.device)
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
assert YES and NO, (YES, NO)


def split_prompt(question):
    """Token ids for the shared prefix (ending at the image token) and the label-specific suffix."""
    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question)
    conv.append_message(conv.roles[1], None)
    ids = tokenizer_image_token(conv.get_prompt(), tok, IMAGE_TOKEN_INDEX, return_tensors="pt")
    k = (ids == IMAGE_TOKEN_INDEX).nonzero()[0].item()
    return ids[: k + 1], ids[k + 1:]


if a.set == "labels":
    if a.data in ("g12ml", "csnml", "sphml"):  # multi-label versions of G12EC / CSN (data/derived/*_ml.json)
        label_names = json.load(open(os.path.join(ROOT, f"data/derived/{a.data[:-2]}_ml.json")))["labels"]
    else:
        label_names = CODE15_LABELS if a.data == "code15" else CPSC_LABELS
    parts = {L: split_prompt(QUESTION.format(name=n, abbr=L)) for L, n in label_names.items()}
else:
    parts = {k: split_prompt(EV_QUESTION.format(q=q)) for k, q in EVIDENCE.items()}
prefix_ids = next(iter(parts.values()))[0]
assert all(torch.equal(p, prefix_ids) for p, _ in parts.values()), "prefix must be label-independent"

if a.data in ("g12ml", "csnml", "sphml"):
    data = {x["id"]: x for x in json.load(open(os.path.join(ROOT, f"data/derived/{a.data[:-2]}_ml.json")))["records"]}
else:
    bench = {"cpsc": "cpsc-test.json", "ptbxl": "ptb-test.json", "code15": "code15-test.json"}[a.data]
    data = {x["id"]: x for x in json.load(open(os.path.join(ROOT, "data/ECGBench", bench)))}
ids = sorted(data) if a.split == "all" else json.load(open(os.path.join(ROOT, f"data/splits/{a.data}.json")))[a.split]
ids = ids[: a.limit]
if a.ids_file:
    ids = json.load(open(a.ids_file))
os.makedirs(os.path.dirname(out_path), exist_ok=True)
done = {json.loads(l)["id"] for l in open(out_path)} if os.path.exists(out_path) else set()
out = open(out_path, "a")
embed = model.get_model().embed_tokens

for i in tqdm([i for i in ids if i not in done]):
    t0 = time.time()
    img = Image.open(os.path.join(ROOT, "data/sph_images" if a.data == "sphml" else "data/ECGBench/images", data[i]["image"])).convert("RGB")
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
            logits = model(inputs_embeds=embed(suf), past_key_values=past, attention_mask=mask,
                           use_cache=True).logits[0, -1].float()
            p = torch.softmax(logits, -1)
            y, n = p[YES].sum().item(), p[NO].sum().item()
            probs[L] = y / (y + n)
            top[L] = tok.convert_ids_to_tokens(int(logits.argmax()))
    out.write(json.dumps({"id": i, "probs": probs, "top": top, "sec": round(time.time() - t0, 2)}) + "\n")
    out.flush()
    del past, pre_emb, pix
    gc.collect()
    if a.device == "mps":
        torch.mps.empty_cache()
