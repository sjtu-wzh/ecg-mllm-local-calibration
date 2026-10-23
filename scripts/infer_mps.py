"""PULSE-7B inference on Apple GPU (MPS). Mirrors LLaVA/llava/eval/model_ecg.py, no CUDA.
Output: jsonl with {question_id, prompt, text, ...} (resumable)."""
import argparse, gc, json, os, time
import torch
from PIL import Image
from tqdm import tqdm
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

ap = argparse.ArgumentParser()
ap.add_argument("--model-path", default="PULSE-ECG/PULSE-7B")
ap.add_argument("--image-folder", default="data/ECGBench/images")
ap.add_argument("--question-file", required=True)
ap.add_argument("--answers-file", required=True)
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--max-new-tokens", type=int, default=1024)
ap.add_argument("--device", default="mps")
a = ap.parse_args()

disable_torch_init()
name = get_model_name_from_path(a.model_path)
tok, model, proc, _ = load_pretrained_model(a.model_path, None, name, device_map=a.device, device=a.device)
model.eval()

qs_all = json.load(open(a.question_file))[: a.limit]
os.makedirs(os.path.dirname(a.answers_file) or ".", exist_ok=True)
done = set()
if os.path.exists(a.answers_file):
    done = {json.loads(l)["question_id"] for l in open(a.answers_file) if l.strip()}
out = open(a.answers_file, "a")
n_done = 0
for q in tqdm(qs_all):
    if q["id"] in done:
        continue
    text = q["conversations"][0]["value"]
    text = (DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN if model.config.mm_use_im_start_end
            else DEFAULT_IMAGE_TOKEN) + "\n" + text
    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], text); conv.append_message(conv.roles[1], None)
    ids = tokenizer_image_token(conv.get_prompt(), tok, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(a.device)
    img = Image.open(os.path.join(a.image_folder, q["image"])).convert("RGB")
    t = process_images([img], proc, model.config)[0].unsqueeze(0).to(a.device, dtype=torch.float16)
    t0 = time.time()
    with torch.inference_mode():
        o = model.generate(ids, images=t, image_sizes=[img.size], do_sample=False, num_beams=1,
                           max_new_tokens=a.max_new_tokens, use_cache=True)
    ans = tok.batch_decode(o, skip_special_tokens=True)[0].strip()
    out.write(json.dumps({"question_id": q["id"], "prompt": q["conversations"][0]["value"], "text": ans,
                          "model_id": name, "sec": round(time.time() - t0, 1)}, ensure_ascii=False) + "\n")
    out.flush()
    del o, t, ids
    n_done += 1
    if a.device == "mps" and n_done % 1 == 0:  # MPS allocator keeps cache; release periodically
        gc.collect(); torch.mps.empty_cache()
