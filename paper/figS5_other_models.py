"""Supplementary Figure S5 — Learning curves for the other base models (threshold-only and S1; mean and 2.5-97.5th percentile of 20 draws).
MedGemma-4B-ECGInstruct (same ECGInstruct training data as PULSE, different architecture) on CPSC, CSN and SPH; Qwen2-VL-7B (general VLM) on CPSC.
Data: mg_curve_cpsc.json, mg_ml_curve_{csn,sph}.json, qwen_curve_cpsc.json"""
import os, runpy
S = runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "figstyle.py"))
plt, C, MM, J, SIZES = S["plt"], S["C"], S["MM"], S["J"], S["SIZES"]
P = [("MedGemma · CPSC", J("outputs/clarity/mg_curve_cpsc.json")["results"], "B1@0.5"), ("MedGemma · CSN", J("outputs/clarity/mg_ml_curve_csn.json")["results"], "ZS"),
     ("MedGemma · SPH*", J("outputs/clarity/mg_ml_curve_sph.json")["results"], "ZS"), ("Qwen2-VL · CPSC", J("outputs/clarity/qwen_curve_cpsc.json")["results"], "B1@0.5")]
fig, axes = plt.subplots(1, 4, figsize=(183 * MM, 55 * MM))
for k, (ax, (title, r, zs)) in enumerate(zip(axes, P)):
    for key, col, lab in (("THR", C["THR"], "Threshold only"), ("S1", C["S1"], "S1")):
        cu = r["curve"]
        ax.fill_between(SIZES, [cu[f"{key}@{n}"]["p2.5"] for n in SIZES], [cu[f"{key}@{n}"]["p97.5"] for n in SIZES], color=col, alpha=0.14, lw=0)
        ax.plot(SIZES, [cu[f"{key}@{n}"]["F1_mean"] for n in SIZES], color=col, marker="o", ms=2.6, label=lab)
    ax.axhline(r["B0"][0], color=C["B0"], ls=(0, (4, 2)), lw=1.0, label="Generated (B0)"); ax.axhline(r[zs][0], color=C["ZS"], ls=(0, (1, 1.5)), lw=1.0, label="Zero-shot P≥0.5")
    ax.set_xscale("log"); ax.set_xticks([25, 100, 600]); ax.set_xticklabels([25, 100, 600]); ax.minorticks_off(); ax.set_xlabel("Labeled ECGs (n)")
    ax.set_title(title, loc="left", fontsize=8.3); ax.grid(axis="y", color=C["grid"], lw=0.5); S["panel_label"](ax, "abcd"[k], x=-0.25)
    ax.text(0.04, 0.93, f"prob. AUROC {r['prob_macroAUC']:.2f}", transform=ax.transAxes, ha="left", fontsize=6.2, color="#555555")
axes[0].set_ylabel("Macro-F1"); h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.1), fontsize=6.8); fig.tight_layout(w_pad=1.0)
S["save"](fig, "FigS5_other_models", supp=True)
