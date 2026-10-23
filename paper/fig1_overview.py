"""Figure 1 — Overview. One frozen ECG multimodal LLM, three read-outs, chosen by the local label budget.
The ECG thumbnail is an SPH record rendered by clarity/render_sph.py (SPH is CC0)."""
import json, os, runpy
import numpy as np
from PIL import Image
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
S = runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "figstyle.py"))
plt, C, MM, ROOT = S["plt"], S["C"], S["MM"], S["ROOT"]
fig = plt.figure(figsize=(183 * MM, 82 * MM)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 183); ax.set_ylim(0, 82); ax.axis("off")


def box(x, y, w, h, text, fc, ec, fs=7, bold=False, tc="black"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.5", fc=fc, ec=ec, lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", color=tc, linespacing=1.25)


def arrow(x0, y0, x1, y1, col="#555555"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8, lw=0.9, color=col, shrinkA=0, shrinkB=0))


# input ECG
rec = json.load(open(os.path.join(ROOT, "data/derived/sph_ml.json")))["records"][1]
im = np.asarray(Image.open(os.path.join(ROOT, "data/sph_images", rec["image"])).convert("RGB"))
ax.imshow(im, extent=(3, 33, 40, 62), aspect="auto", zorder=1)
ax.text(18, 64, "12-lead ECG image\n(new site)", ha="center", fontsize=7)
arrow(34, 51, 41, 51)
box(42, 40, 26, 22, "Frozen ECG\nmultimodal LLM\n\nPULSE-7B\nMedGemma-4B\n(no retraining)", "#EEF2F8", "#0F4D92", fs=7, bold=False)
ax.text(55, 64.5, "unchanged weights", ha="center", fontsize=6.3, color="#555555")
# three read-outs
ys = [70, 51, 30]
for y in ys: arrow(69, 51, 80, y + 4)
box(81, 66, 40, 8, "(1) Generate an answer\n“AF; RBBB”", "#F0F0F0", C["B0"], fs=6.8)
box(81, 47, 40, 8, "(2) Ask Yes/No per diagnosis\nread P(yes) from next-token logits", "#EAF0F8", C["S1"], fs=6.8)
box(81, 26, 40, 8, "(3) Read hidden state\n(LLM layer 24–32, last token)", "#F8ECEC", C["PROBE"], fs=6.8)
arrow(122, 70, 130, 70); arrow(122, 51, 130, 51); arrow(122, 30, 130, 30)
box(131, 66, 49, 8, "Standard use (B0)\nunder-uses what the model knows", "white", C["B0"], fs=6.5)
box(131, 45, 49, 12, "Calibrate with n local labels\n• threshold per label (very small n)\n• S1: logistic regression on all\n  label logits (n ≈ 100–300)", "white", C["S1"], fs=6.3)
box(131, 26, 49, 8, "Linear probe on hidden state\n(n ≥ 200–300)", "white", C["PROBE"], fs=6.5)
# label-budget ruler
y0 = 11; ax.plot([20, 175], [y0, y0], color="#333333", lw=0.9)
LX = lambda n: 20 + (np.log(n) - np.log(25)) / (np.log(600) - np.log(25)) * 155  # log scale, as in Fig. 3
ticks = {n: LX(n) for n in (25, 50, 100, 200, 300, 600)}
for n, x in ticks.items():
    ax.plot([x, x], [y0 - 1, y0 + 1], color="#333333", lw=0.8); ax.text(x, y0 - 4, str(n), ha="center", fontsize=6.5)
ax.text(97, y0 - 8.5, "Local labelled ECGs available (n)", ha="center", fontsize=7)
for x0, x1, col, lab in ((LX(25), LX(100), C["THR"], "probability + threshold"), (LX(100), LX(250), C["S1"], "S1"), (LX(250), LX(600), C["PROBE"], "hidden-state probe")):
    ax.add_patch(FancyBboxPatch((x0 + 0.6, y0 + 2), x1 - x0 - 1.2, 3.2, boxstyle="round,pad=0,rounding_size=1", fc=col, ec="none", alpha=0.85))
    ax.text((x0 + x1) / 2, y0 + 3.6, lab, ha="center", va="center", fontsize=6.2, color="white", fontweight="bold")
ax.text(3, y0 + 3.6, "Recommended\nread-out", fontsize=6.5, va="center")
S["save"](fig, "Fig1_overview")
