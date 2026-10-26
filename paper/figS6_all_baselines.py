"""Supplementary Figure S6 — All read-outs with the same number of fitting records (frozen PULSE-7B) and hidden-state feature selection.
a-d Learning curves of every read-out (mean of 20 draws). e Pool-only stage-1 selection: 5-fold OOF macro-F1 of linear probes on each
saved feature (LLM layer 8/16/24/32, image-token mean or last prompt token; CLIP CLS / patch mean). Data: baselines_*.json"""
import os, runpy
import numpy as np
S = runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "figstyle.py"))
plt, C, MM, SIZES = S["plt"], S["C"], S["MM"], S["SIZES"]
M = [("THR", C["THR"], "-", "Threshold only"), ("S1-uni", "#76B5BD", "--", "Per-label Platt (S1-uni)"), ("S1", C["S1"], "-", "S1"),
     ("PROBE-LLM", C["PROBE"], "-", "Hidden-state probe"), ("COMBO", C["COMBO"], "--", "Probe + label logits"), ("PROBE-CLIP", C["CLIP"], "-", "CLIP probe")]
fig = plt.figure(figsize=(183 * MM, 112 * MM)); gs = fig.add_gridspec(2, 4, height_ratios=[1, 0.9], hspace=0.75, wspace=0.35)
for k, (d, title) in enumerate(S["DATASETS"]):
    ax = fig.add_subplot(gs[0, k]); r = S["baselines"](d)
    for key, col, ls, lab in M:
        ax.plot(SIZES, [r["curve"][f"{key}@{n}"]["F1"][0] for n in SIZES], color=col, ls=ls, marker="o", ms=2.2, lw=1.1, label=lab)
    ax.axhline(r["reference"]["B0"]["F1"], color=C["B0"], ls=(0, (4, 2)), lw=0.9, label="Generated (B0)")
    ax.set_xscale("log"); ax.set_xticks([25, 100, 600]); ax.set_xticklabels([25, 100, 600]); ax.minorticks_off(); ax.set_xlabel("Labelled ECGs (n)")
    ax.set_title(title, loc="left", fontsize=8.2); ax.grid(axis="y", color=C["grid"], lw=0.5); S["panel_label"](ax, "abcd"[k], x=-0.27)
    if k == 0: ax.set_ylabel("Macro-F1")
h, l = fig.axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="center", ncol=7, bbox_to_anchor=(0.5, 0.455), fontsize=6.4)
ax = fig.add_subplot(gs[1, 1:3]); feats = [f"llm_L{L}_{p}" for p in ("img", "last") for L in (8, 16, 24, 32)] + ["clip_cls", "clip_patch"]
mat = np.array([[S["baselines"](d)["stage1_pool_oof_F1"][f] for f in feats] for d, _ in S["DATASETS"]])
im = ax.imshow(mat, cmap="Blues", aspect="auto")
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        sel = S["baselines"](S["DATASETS"][i][0])["selected"]["llm"] == feats[j]
        ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center", fontsize=5.6, color="white" if mat[i, j] > mat.max() * 0.8 else "black", fontweight="bold" if sel else "normal")
ax.set_xticks(range(len(feats))); ax.set_xticklabels(["L8", "L16", "L24", "L32"] * 2 + ["CLS", "patch"], fontsize=6.2)
for x0, x1, lab in ((0, 3, "LLM, image-token mean"), (4, 7, "LLM, last prompt token"), (8, 9, "CLIP")):
    ax.annotate("", xy=(x0 - 0.4, -0.2), xytext=(x1 + 0.4, -0.2), xycoords=("data", "axes fraction"), arrowprops=dict(arrowstyle="-", color="#555555", lw=0.7))
    ax.text((x0 + x1) / 2, -0.27, lab, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6.4)
ax.set_yticks(range(4)); ax.set_yticklabels([t for _, t in S["DATASETS"]], fontsize=6.6)
ax.set_title("Stage-1 feature selection (pool CV macro-F1; bold = selected)", loc="left", fontsize=8.2); S["panel_label"](ax, "e", x=-0.18)
plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
S["save"](fig, "FigS6_all_baselines", supp=True)
