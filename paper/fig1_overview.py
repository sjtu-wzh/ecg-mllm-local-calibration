"""Figure 1 — Study overview (v3).
a One frozen ECG multimodal LLM (PULSE-7B) receives the same ECG image with three different prompts: a classification prompt
  (generated answer, B0), one Yes/No question per diagnosis (per-label P(yes): ZS, THR, S1) and a neutral prompt (hidden state:
  linear probe). Dashed borders mark read-outs that require local labeled data.
b Design of the main analysis (CPSC, CSN, G12EC, SPH): n records are sampled from the 600-record development pool, read-outs are
  fitted on them and applied unchanged to the held-out test ECGs; test reference labels enter only the evaluation. In the original
  analysis the probe's layer and token were selected once on the full development pool.
The example ECG is SPH record A00024 rendered for this study (reference label NORM; PULSE-7B generated answer "NORM")."""
import json, os, runpy
import numpy as np
from PIL import Image
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
S = runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "figstyle.py"))
plt, C, MM, ROOT = S["plt"], S["C"], S["MM"], S["ROOT"]
W, H = 183, 146
fig = plt.figure(figsize=(W * MM, H * MM)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
GREY, DASH = "#444444", (0, (3, 1.6))


def box(x, y, w, h, text, ec, fc="white", fs=7.4, bold_first=False, dashed=False, lw=0.9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=fc, ec=ec, lw=lw, ls=DASH if dashed else "-"))
    lines = text.split("\n")
    if bold_first and len(lines) > 1:
        step = fs * 0.42
        top = y + h / 2 + (len(lines) - 1) * step / 2
        ax.text(x + w / 2, top, lines[0], ha="center", va="center", fontsize=fs, fontweight="bold")
        for k, l in enumerate(lines[1:], 1):
            ax.text(x + w / 2, top - k * step, l, ha="center", va="center", fontsize=fs - 0.4, color="#222222")
    else:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.2,
                fontweight="bold" if bold_first else "normal")


def arrow(x0, y0, x1, y1, dashed=False, lw=0.9, col=GREY):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, lw=lw, color=col,
                                 ls=DASH if dashed else "-", shrinkA=0, shrinkB=0))


def line(xs, ys, dashed=False):
    ax.plot(xs, ys, color=GREY, lw=0.9, ls=DASH if dashed else "-", solid_capstyle="butt")


# ================================================================ panel a: read-outs
from matplotlib.patches import Rectangle
plt.rcParams["mathtext.fontset"] = "stixsans"
ax.text(2, H - 2.5, "a", fontsize=11, fontweight="bold", va="top")
ax.text(8, H - 2.5, "Read-outs of one frozen ECG multimodal LLM", fontsize=9, fontweight="bold", va="top")
R1, R2, R3 = 127, 108, 76
rec = json.load(open(os.path.join(ROOT, "data/derived/sph_ml.json")))["records"][1]
assert rec["id"] == "sph-A00024"
im = np.asarray(Image.open(os.path.join(ROOT, "data/sph_images", rec["image"])).convert("RGB"))
ax.imshow(im, extent=(3, 29, R2 - 8.5, R2 + 8.5), aspect="auto", zorder=1)
ax.text(16, R2 - 10.5, "12-lead ECG image", ha="center", va="top", fontsize=7.4)
ax.text(48.5, 136.5, "Model input: image + prompt", ha="center", fontsize=7.2, color=GREY, fontstyle="italic")
INPUTS = [(R1, "ECG image +\nclassification prompt"), (R2, "ECG image +\nYes/No question\nper diagnosis"),
          (R3, "ECG image +\nneutral prompt\n“Describe the ECG.”")]
for y, txt in INPUTS:
    arrow(29.5, R2, 33.5, y, lw=0.8)
    box(34, y - 6.5, 29, 13, txt, GREY, fc="#F4F4F4", fs=7.0)
    arrow(63.5, y, 67.5, y)
box(68, R3 - 6.5, 21, R1 - R3 + 13, "Frozen\nPULSE-7B\n\nweights fixed;\nno fine-tuning\n\nshared by all\nthree branches",
    GREY, fc="#F4F4F4", fs=7.6)
OUTPUTS = [(R1, "Generated text\n(e.g., “NORM”)"), (R2, "Per-label P(yes)\nfrom next-token\nlogits"),
           (R3, "Hidden state\n(selected layer,\nlast prompt token)")]
for y, txt in OUTPUTS:
    arrow(89.5, y, 93.5, y)
    box(94, y - 6, 29, 12, txt, GREY, fs=7.2)
# read-outs (dashed border = requires local labeled data; colored tab = read-out identity)
arrow(123.5, R1, 127.5, R1)
box(128, R1 - 5, 53, 10, "B0: generated-answer baseline\nlabels parsed from the text", GREY, fs=7.4, bold_first=True)
box(128, 115.5, 53, 5.2, "ZS: P(yes) ≥ 0.5 (no fitting)", C["ZS"], fs=7.2)
box(128, 109, 53, 5.2, "THR: threshold per label", GREY, fs=7.2, dashed=True, lw=1.0)
for yt in (118.1, 111.6, 101.5):
    arrow(123.5, R2, 127.6, yt, lw=0.8)


def grid_icon(x0, y0, rows, cols, cell, col):
    for r in range(rows):
        for c in range(cols):
            ax.add_patch(Rectangle((x0 + c * cell, y0 + r * cell), cell, cell, fc=col, ec="white", lw=0.35))


def readout_card(y0, h, tag, title, tagcol, fill, rows, cols, dims, formula):
    box(128, y0, 53, h, "", GREY, fc=fill, dashed=True, lw=1.0)
    ax.add_patch(Rectangle((129.6, y0 + h - 5.4), 0.8, 3.6, fc=tagcol, ec="none"))
    ax.text(131.6, y0 + h - 3.6, tag, fontsize=7.8, fontweight="bold", color=tagcol, va="center")
    ax.text(131.6 + (4.2 if tag == "S1" else 9.0), y0 + h - 3.6, title, fontsize=7.4, fontweight="bold", va="center")
    cell = 1.15
    grid_icon(131.5, y0 + 3.6, rows, cols, cell, tagcol + "40")
    ax.text(131.5 + cols * cell / 2, y0 + 2.0, dims, fontsize=6.8, ha="center", va="center", fontstyle="italic", color="#333333")
    ax.text(131.5 + cols * cell + (181 - 131.5 - cols * cell) / 2, y0 + 3.6 + rows * cell / 2, formula, fontsize=9.2, ha="center", va="center")


readout_card(89, 18, "S1", "Cross-label logistic read-out", C["S1"], "#F3F6FB", 7, 7, r"$K\times K$",
             r"$\mathbf{q}=\sigma\,(\mathbf{W}\,\mathrm{logit}(\mathbf{p})+\mathbf{b})$")
arrow(123.5, R3, 127.5, R3)
readout_card(67, 18, "Probe", "Linear read-out (PULSE only)", C["PROBE"], "#FBF4F4", 7, 11, r"$K\times d$",
             r"$\mathbf{q}=\sigma\,(\mathbf{V}\,\mathbf{z}+\mathbf{a})$")
ax.text(3, 63.5, "Replication: MedGemma-4B-ECGInstruct (probability read-outs)", fontsize=7, color=GREY, va="center")
ax.text(3, 60.3, "General-VLM control: Qwen2-VL-7B-Instruct (probability read-outs)", fontsize=7, color=GREY, va="center")
ax.text(181, 63.5, "Same model; different prompts.", fontsize=7, color=GREY, va="center", ha="right")

# ================================================================ panel b: design
ax.plot([2, W - 2], [57.5, 57.5], color="#DDDDDD", lw=0.6)
ax.text(2, 56.5, "b", fontsize=11, fontweight="bold", va="top")
ax.text(8, 56.5, "Local fitting and held-out evaluation (main analysis: CPSC, CSN, G12EC, SPH)", fontsize=9, fontweight="bold", va="top")
yb, hb = 23, 14                                              # main row: bottom, height
box(3, yb, 33, hb, "Development pool\n600 labeled ECG records", GREY, fc="#F4F4F4", fs=7.4, bold_first=True)
arrow(36.5, yb + hb / 2, 40.5, yb + hb / 2)
box(41, yb, 34, hb, "Sample n records\nn = 25, 50, 100, 200, 300,\n400 or 600; 20 draws each", GREY, fs=7.4, bold_first=True)
arrow(75.5, yb + hb / 2, 79.5, yb + hb / 2, dashed=True)
box(80, yb, 32, hb, "Fit read-outs\non the n records\n(THR, S1, probe)", GREY, fs=7.4, bold_first=True, dashed=True, lw=1.0)
arrow(112.5, yb + hb / 2, 116.5, yb + hb / 2)
box(117, yb, 27, hb, "Apply fitted\nread-outs\nunchanged", GREY, fs=7.4)
arrow(144.5, yb + hb / 2, 148.5, yb + hb / 2)
box(149, yb, 32, hb, "Evaluate on the\nheld-out test split\nB0, ZS, THR, S1, probe", GREY, fc="#F4F4F4", fs=7.4, bold_first=True)
# probe feature selection on the full pool (original analysis)
box(40, 40, 36.5, 11, "Probe feature selection\n(layer, token): 5-fold CV\non the full development pool", GREY, fs=6.9, dashed=True, lw=1.0)
from matplotlib.patches import Rectangle as _R; ax.add_patch(_R((41.0, 43.7), 0.8, 3.6, fc=C["PROBE"], ec="none"))
line([19.5, 19.5], [yb + hb, 45.5], dashed=True); arrow(19.5, 45.5, 39.5, 45.5, dashed=True)
line([77, 96], [45.5, 45.5], dashed=True); arrow(96, 45.5, 96, yb + hb + 0.5, dashed=True)
ax.text(97.5, 41.5, "Selected\nlayer / token\n(probe only)", fontsize=6.6, color=GREY, ha="left", va="center", linespacing=1.15)
# test inputs and predictions that need no fitting
box(117, 40, 27, 11, "Held-out test ECGs:\nP(yes) / hidden state", GREY, fs=7.0)
arrow(130.5, 39.5, 130.5, yb + hb + 0.5)
box(149, 40, 32, 11, "B0 and ZS predictions\n(no fitting)", GREY, fs=7.0)
arrow(165, 39.5, 165, yb + hb + 0.5)
box(149, 3, 32, 11, "Test reference labels", GREY, fs=7.2)
arrow(165, 14.5, 165, yb - 0.5)
# legend
ax.plot([3, 10], [16, 16], color=GREY, lw=0.9); ax.text(11.5, 16, "Prediction / data flow", fontsize=7, color=GREY, va="center")
ax.plot([3, 10], [10.5, 10.5], color=GREY, lw=0.9, ls=DASH); ax.text(11.5, 10.5, "Development-stage fitting and selection", fontsize=7, color=GREY, va="center")
ax.add_patch(FancyBboxPatch((3, 3.3), 7, 3.4, boxstyle="round,pad=0,rounding_size=0.8", fc="white", ec=GREY, lw=1, ls=DASH))
ax.text(11.5, 5, "Requires local labeled data", fontsize=7, color=GREY, va="center")
S["save"](fig, "Fig1_overview")
