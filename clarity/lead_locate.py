"""M1 pilot: locate the 12 printed lead labels on an ECG image with Apple Vision OCR.

ocr_leads(path) -> {lead: (x0, y0, x1, y1)} in pixel coords (top-left origin).
Run as a script to measure detection rate on a sample and save box visualizations."""
import argparse, json, os, random, re
import Quartz, Vision
from Foundation import NSURL
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
# OCR confusions for the short limb-lead labels
ALIASES = {"I": {"I", "l", "1", "|"}, "II": {"II", "Il", "lI", "ll", "11"}, "III": {"III", "IIl", "lll", "111"}}


def normalize(text):
    t = re.sub(r"[\s\.\:\-_,]", "", text)
    for lead, al in ALIASES.items():
        if t in al:
            return lead
    u = t.upper().replace("0", "O")
    for lead in ("aVR", "aVL", "aVF"):
        if u == lead.upper():
            return lead
    m = re.fullmatch(r"V([1-6])", u)
    return f"V{m.group(1)}" if m else None


def ocr(path):
    """All text observations: list of (string, (x0, y0, x1, y1)) in pixels, top-left origin."""
    w, h = Image.open(path).size
    req = Vision.VNRecognizeTextRequest.alloc().init()
    req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    req.setUsesLanguageCorrection_(False)
    req.setMinimumTextHeight_(0.0)
    handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(NSURL.fileURLWithPath_(path), None)
    ok, err = handler.performRequests_error_([req], None)
    if not ok:
        raise RuntimeError(err)
    res = []
    for obs in req.results() or []:
        cand = obs.topCandidates_(1)
        if not cand:
            continue
        b = obs.boundingBox()  # normalized, bottom-left origin
        x0, y1 = b.origin.x * w, (1 - b.origin.y) * h
        x1, y0 = x0 + b.size.width * w, y1 - b.size.height * h
        res.append((str(cand[0].string()), (x0, y0, x1, y1)))
    return res


def ocr_leads(path):
    found = {}
    for text, box in ocr(path):
        # a single observation may hold several labels, e.g. "aVR V1"
        for tok in text.split():
            lead = normalize(tok)
            if lead and lead not in found:  # first occurrence (rhythm strips repeat II / V1)
                found[lead] = box
    return found


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", default="cpsc-test,mmmu-ecg,ptb-test,g12-test-no-cot,csn-test-no-cot")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--vis", type=int, default=6, help="visualizations saved per subset")
    a = ap.parse_args()
    random.seed(0)
    vis_dir = os.path.join(ROOT, "outputs/clarity/m1_vis"); os.makedirs(vis_dir, exist_ok=True)
    summary = {}
    for s in a.subsets.split(","):
        data = json.load(open(os.path.join(ROOT, f"data/ECGBench/{s}.json")))
        if s == "cpsc-test":  # dev only, never peek at the test half
            dev = set(json.load(open(os.path.join(ROOT, "data/splits/cpsc.json")))["dev"])
            data = [x for x in data if x["id"] in dev]
        sample = random.sample(data, min(a.n, len(data)))
        counts = []
        for k, x in enumerate(sample):
            p = os.path.join(ROOT, "data/ECGBench/images", x["image"])
            leads = ocr_leads(p)
            counts.append(len(leads))
            if k < a.vis:
                im = Image.open(p).convert("RGB"); d = ImageDraw.Draw(im)
                for L, (x0, y0, x1, y1) in leads.items():
                    d.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=3); d.text((x1 + 3, y0), L, fill=(255, 0, 0))
                im.save(os.path.join(vis_dir, f"{s}_{k}.png"))
        summary[s] = {"n": len(counts), "mean_leads_found": round(sum(counts) / len(counts), 2),
                      "all12_rate": round(sum(c == 12 for c in counts) / len(counts), 2),
                      ">=10_rate": round(sum(c >= 10 for c in counts) / len(counts), 2)}
        print(s, summary[s], flush=True)
    json.dump(summary, open(os.path.join(ROOT, "outputs/clarity/m1_ocr_pilot.json"), "w"), indent=1)
