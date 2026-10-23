"""Experiment 3 (robustness), step 1: fixed record subsets, perturbed SPH images and paraphrased prompts.
Frozen with docs/prereg_robustness.md before any model output on perturbed images or paraphrased prompts.

Subsets (from data/splits/sph_ml.json): 300 pool + 800 test records, numpy default_rng(42 + k), k = first redraw with
>= 5 positives per label in the pool subset and >= 10 in the test subset.  -> data/splits/sph_robust.json

Image conditions (applied to the clean SPH renderings; per-record seed = crc32("<condition>|<id>")):
  ROT    rotation by +-U(4, 10) degrees about the centre, white fill (clean renderings have <= 3 degrees on half)
  PHOTO  phone photo of a printout: perspective warp (each corner moved inwards by U(0, 6)% of width / height, grey
         background), uneven lighting (multiplicative linear gradient 0.65-1.0 in a random direction), warm colour cast,
         Gaussian blur radius 1.0, JPEG quality 70
  SCAN   low-quality scan / forwarded image: greyscale, downsample to 40% and back (bilinear), Gaussian noise sd 8, JPEG quality 25
  -> data/sph_images_robust/{ROT,PHOTO,SCAN}/sph/{ECG_ID}.png

Prompts (data/derived/sph_robust_prompts.json):
  QA, QB  paraphrases of the Yes/No label question
  P1      paraphrase of the generated-answer prompt (same class list, same order)
Question files for the generated answer (800 test records): sph_robust_test_questions.json (original prompt),
sph_robust_P1_questions.json (P1).
Run: python -I clarity/robust_setup.py
"""
import io, json, os, zlib
import numpy as np
from PIL import Image, ImageFilter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = json.load(open(os.path.join(ROOT, "data/derived/sph_ml.json"))); S = json.load(open(os.path.join(ROOT, "data/splits/sph_ml.json")))
Q = {q["id"]: q for q in json.load(open(os.path.join(ROOT, "data/derived/sph_ml_questions.json")))}
gold = {r["id"]: set(r["conversations"][-1]["value"].split(";")) for r in D["records"]}
LABELS = list(D["labels"])

for k in range(100):
    rng = np.random.default_rng(42 + k)
    pool = sorted(rng.choice(S["pool"], 300, replace=False).tolist()); test = sorted(rng.choice(S["test"], 800, replace=False).tolist())
    if all(sum(L in gold[i] for i in pool) >= 5 and sum(L in gold[i] for i in test) >= 10 for L in LABELS):
        break
json.dump({"pool": pool, "test": test, "redraws": k}, open(os.path.join(ROOT, "data/splits/sph_robust.json"), "w"), indent=1)


def jpeg(img, q):
    b = io.BytesIO(); img.save(b, "JPEG", quality=q); b.seek(0); return Image.open(b).convert("RGB")


def perspective_coeffs(src, dst):
    """PIL PERSPECTIVE coefficients mapping output (dst) corners to input (src) corners."""
    A, B = [], []
    for (x, y), (u, v) in zip(dst, src):
        A += [[x, y, 1, 0, 0, 0, -u * x, -u * y], [0, 0, 0, x, y, 1, -v * x, -v * y]]; B += [u, v]
    return np.linalg.solve(np.array(A, float), np.array(B, float)).tolist()


def perturb(img, cond, r):
    W, H = img.size
    if cond == "ROT":
        return img.rotate(r.uniform(4, 10) * r.choice([-1, 1]), resample=Image.BICUBIC, expand=False, fillcolor=(255, 255, 255))
    if cond == "PHOTO":
        src = [(0, 0), (W, 0), (W, H), (0, H)]
        dst = [(x + (1 if x == 0 else -1) * r.uniform(0, 0.06) * W, y + (1 if y == 0 else -1) * r.uniform(0, 0.06) * H) for x, y in src]
        img = img.transform((W, H), Image.PERSPECTIVE, perspective_coeffs(src, dst), Image.BICUBIC, fillcolor=(90, 90, 90))
        th = r.uniform(0, 2 * np.pi); yy, xx = np.mgrid[0:H, 0:W]
        g = np.cos(th) * xx / W + np.sin(th) * yy / H; g = (g - g.min()) / (g.max() - g.min() + 1e-9)
        a = np.asarray(img).astype(float) * (0.65 + 0.35 * g)[..., None] * np.array([1.0, 0.96, 0.88])
        return jpeg(Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.0)), 70)
    if cond == "SCAN":
        img = img.convert("L").resize((int(W * 0.4), int(H * 0.4)), Image.BILINEAR).resize((W, H), Image.BILINEAR)
        a = np.asarray(img).astype(float) + r.normal(0, 8, (H, W))
        return jpeg(Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).convert("RGB"), 25)
    raise ValueError(cond)


for cond in ("ROT", "PHOTO", "SCAN"):
    out = os.path.join(ROOT, "data/sph_images_robust", cond, "sph"); os.makedirs(out, exist_ok=True)
    for i in pool + test:
        dst = os.path.join(out, os.path.basename(Q[i]["image"]))
        if not os.path.exists(dst):
            r = np.random.default_rng(zlib.crc32(f"{cond}|{i}".encode()))
            perturb(Image.open(os.path.join(ROOT, "data/sph_images", Q[i]["image"])).convert("RGB"), cond, r).save(dst)

PROMPTS = {
    "QA": "Based on this 12-lead ECG image, is {name} ({abbr}) present? Answer Yes or No.",
    "QB": "Look at the ECG image. Would you diagnose {name} ({abbr})? Please answer with Yes or No.",
    "P1": "This is a 12-lead ECG image. Which of the following diagnoses are present?\n{classes}\n"
          "List every class that applies, using the abbreviations only, separated by semicolons.",
}
json.dump(PROMPTS, open(os.path.join(ROOT, "data/derived/sph_robust_prompts.json"), "w"), indent=1)
orig = Q[test[0]]["conversations"][0]["value"]
classes = orig.split("classes:\n")[1].split("\nDirectly output")[0]
json.dump([Q[i] for i in test], open(os.path.join(ROOT, "data/derived/sph_robust_test_questions.json"), "w"), indent=1)
json.dump([{**Q[i], "conversations": [{"from": "human", "value": PROMPTS["P1"].format(classes=classes)}, Q[i]["conversations"][1]]} for i in test],
          open(os.path.join(ROOT, "data/derived/sph_robust_P1_questions.json"), "w"), indent=1)
print(f"subsets: pool {len(pool)}, test {len(test)} (redraws {k}); perturbed images: 3 x {len(pool) + len(test)}")
