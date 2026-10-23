"""Render SPH 12-lead signals (500 Hz, mV) as ECG images in the visual conventions of ECGBench (fixed before any model output):
  * first 10 s; 25 mm/s, 10 mm/mV; 1 mm minor / 5 mm major grid; 1 mV x 0.2 s calibration pulse per row; lead names
  * layout per record (seeded by ECG_ID): 3x4 + lead-II rhythm strip (50%), 6x2 (25%), 12x1 (25%)
  * grid colour per record: red/pink (60%), blue (25%), grey (15%); footer '25mm/s  10mm/mV'
  * mild acquisition variation on 50% of records: rotation U(-3,3) deg, background tint, light blur (ECGBench mixes clean and
    augmented images ~1:1); no wrinkles / perspective (those are reserved for the robustness experiment)
Output: data/sph_images/sph/{ECG_ID}.png (~120 dpi; width ~1350 px)
Run: python -I clarity/render_sph.py [--limit N] [--jobs 8]"""
import argparse, json, os, zlib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
FS, SECS = 500, 10.0
GRIDS = [("#f19c9c", "#fbdada"), ("#7fa7d8", "#d6e4f5"), ("#9a9a9a", "#e2e2e2")]


def render(ecg_id, out_path):
    import h5py, matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image, ImageFilter
    rng = np.random.default_rng(zlib.crc32(ecg_id.encode()))
    x = h5py.File(os.path.join(ROOT, "data/sph_raw/records", f"{ecg_id}.h5"), "r")["ecg"][()].astype(np.float32)[:, : int(FS * SECS)]
    layout = rng.choice(["3x4+II", "6x2", "12x1"], p=[0.5, 0.25, 0.25])
    major, minor = GRIDS[rng.choice(3, p=[0.6, 0.25, 0.15])]
    if layout == "3x4+II":
        rows = [[(LEADS[r + 3 * c], c * 2.5, 2.5) for c in range(4)] for r in range(3)] + [[("II", 0.0, 10.0)]]
    elif layout == "6x2":
        rows = [[(LEADS[r], 0.0, 5.0), (LEADS[r + 6], 5.0, 5.0)] for r in range(6)]
    else:
        rows = [[(L, 0.0, 10.0)] for L in LEADS]
    row_h = 25.0 if len(rows) <= 4 else (18.0 if len(rows) == 6 else 12.5)          # mm per row
    W, H = 10 + 250 + 10, 15 + row_h * len(rows) + 15                              # mm
    fig = plt.figure(figsize=(W / 25.4, H / 25.4), dpi=127)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    for v in np.arange(0, W + 0.01, 1.0): ax.axvline(v, color=minor if v % 5 else major, lw=0.25 if v % 5 else 0.55, zorder=0)
    for v in np.arange(0, H + 0.01, 1.0): ax.axhline(v, color=minor if v % 5 else major, lw=0.25 if v % 5 else 0.55, zorder=0)
    for r, segs in enumerate(rows):
        y0 = H - 15 - row_h * (r + 0.5)
        ax.plot([2, 3, 3, 8, 8, 9], [y0, y0, y0 + 10, y0 + 10, y0, y0], color="black", lw=0.7)   # calibration pulse
        for lead, t0, dur in segs:
            sig = x[LEADS.index(lead), int(t0 * FS): int((t0 + dur) * FS)]
            t = 10 + t0 * 25 + np.arange(len(sig)) / FS * 25
            ax.plot(t, y0 + sig * 10, color="black", lw=0.6)
            ax.text(10 + t0 * 25 + 1, y0 + 6, lead, fontsize=6.5)
            if t0 > 0: ax.plot([10 + t0 * 25] * 2, [y0 - 3, y0 + 3], color="black", lw=0.6)
    ax.text(15, 5, "25mm/s", fontsize=6); ax.text(60, 5, "10mm/mV", fontsize=6)
    tmp = out_path + ".tmp.png"; fig.savefig(tmp, dpi=127, facecolor="white"); plt.close(fig)
    im = Image.open(tmp).convert("RGB"); os.remove(tmp)
    if rng.random() < 0.5:  # mild acquisition variation
        tint = np.array([rng.uniform(0.85, 1.0), rng.uniform(0.8, 1.0), rng.uniform(0.75, 1.0)])
        im = Image.fromarray((np.asarray(im, dtype=np.float32) * tint).clip(0, 255).astype(np.uint8))
        im = im.rotate(rng.uniform(-3, 3), resample=Image.BICUBIC, expand=True, fillcolor=(0, 0, 0))
        if rng.random() < 0.5: im = im.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 0.8)))
    im.save(out_path)
    return layout


if __name__ == "__main__":
    from joblib import Parallel, delayed
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=None); ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    recs = json.load(open(os.path.join(ROOT, "data/derived/sph_ml.json")))["records"][: a.limit]
    out_dir = os.path.join(ROOT, "data/sph_images/sph"); os.makedirs(out_dir, exist_ok=True)
    todo = [x for x in recs if not os.path.exists(os.path.join(ROOT, "data/sph_images", x["image"]))]
    lays = Parallel(n_jobs=a.jobs)(delayed(render)(x["image"].split("/")[1][:-4], os.path.join(ROOT, "data/sph_images", x["image"])) for x in todo)
    print(f"rendered {len(lays)} images; layouts:", {k: lays.count(k) for k in set(lays)})
