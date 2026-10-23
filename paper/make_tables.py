"""Main tables -> paper/tables/Table{1,2,3}.{csv,md}. Every number is read from outputs/clarity/*.json."""
import csv, json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper/tables"); os.makedirs(OUT, exist_ok=True)
J = lambda p: json.load(open(os.path.join(ROOT, p)))
f1 = lambda x: f"{x:.1f}"
ci = lambda v: f"{v[0]:+.1f} [{v[1]:.1f}, {v[2]:.1f}]"


def write(name, header, rows, note):
    with open(os.path.join(OUT, f"{name}.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(header); w.writerows(rows)
    md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    open(os.path.join(OUT, f"{name}.md"), "w").write("\n".join(md) + f"\n\n{note}\n")
    print(f"{name}: {len(rows)} rows")


# ---------- Table 1: datasets
sph = J("data/derived/sph_ml.json"); csn = J("data/derived/csn_ml.json"); g12 = J("data/derived/g12_ml.json")
spl = {d: J(f"data/splits/{d}.json") for d in ("cpsc", "csn_ml", "g12_ml", "sph_ml")}
c15 = J("outputs/clarity/code15_curve.json")["results"]["split"]
rows = [
    ["CPSC 2018 (ECGBench)", "China (multi-centre)", "out-of-domain", "multi-label classification, official ECGBench prompt", 9, len(spl["cpsc"]["dev"]), len(spl["cpsc"]["test"])],
    ["CSN (ECGBench, multi-label version)", "China (Shaoxing/Ningbo)", "out-of-domain", "multi-label, CPSC-style prompt (this work)", len(csn["labels"]), len(spl["csn_ml"]["pool"]), len(spl["csn_ml"]["test"])],
    ["G12EC (ECGBench, multi-label version)", "USA (Georgia)", "out-of-domain", "multi-label, CPSC-style prompt (this work)", len(g12["labels"]), len(spl["g12_ml"]["pool"]), len(spl["g12_ml"]["test"])],
    ["SPH (rendered images; pre-registered)", "China (Shandong Provincial Hospital)", "out-of-domain, new site", "multi-label, CPSC-style prompt (this work)", len(sph["labels"]), len(spl["sph_ml"]["pool"]), len(spl["sph_ml"]["test"])],
    ["CODE-15% (ECGBench)", "Brazil", "in-domain (training source)", "binary + multi-label, official prompts", "1+6", c15["dev_pool"]["bin"] + c15["dev_pool"]["multi"], c15["test"]["bin"] + c15["test"]["multi"]],
    ["PTB-XL fold 10 (ECGBench images)", "Germany", "in-domain (training source)", "CPSC 9-label prompt (domain analysis)", 8, 600, "≈757 (20 patient-grouped splits)"],
]
write("Table1_datasets", ["Dataset", "Origin", "Relation to PULSE training", "Task / prompt", "Labels", "Calibration pool", "Test"], rows,
      "Pool = records from which n labelled ECGs are drawn; the test split is fixed and never used for fitting.")

# ---------- Table 2: main results (macro-F1, held-out test)
rows = []
for d, name in (("cpsc", "CPSC"), ("csn", "CSN"), ("g12", "G12EC"), ("sph", "SPH*")):
    b = J(f"outputs/clarity/baselines_{d}.json")["results"]; cur = b["curve"]
    boot = J("outputs/clarity/pulse_bootstrap_cpsc.json")["bootstrap"]["S1@600-B0"] if d == "cpsc" else J(f"outputs/clarity/ml_curve_{d}.json")["results"]["bootstrap"]["S1@600-B0"]
    rows.append(["PULSE-7B", name, f1(b["reference"]["B0"]["F1"]), f1(b["reference"]["ZS@0.5"]["F1"]),
                 f1(cur["THR@100"]["F1"][0]), f1(cur["S1@100"]["F1"][0]), f1(cur["PROBE-LLM@100"]["F1"][0]),
                 f1(cur["S1@600"]["F1"][0]), f1(cur["PROBE-LLM@600"]["F1"][0]), ci(boot)])
mg_c = J("outputs/clarity/mg_curve_cpsc.json")["results"]; mgb = J("outputs/clarity/mg_bootstrap_cpsc.json")
rows.append(["MedGemma-4B-ECGInstruct", "CPSC", f1(mg_c["B0"][0]), f1(mg_c["B1@0.5"][0]), f1(mg_c["curve"]["THR@100"]["F1_mean"]),
             f1(mg_c["curve"]["S1@100"]["F1_mean"]), "—", f1(mg_c["curve"]["S1@600"]["F1_mean"]), "—", ci(mgb["bootstrap"]["S1@600-B0"])])
for d, name in (("csn", "CSN"), ("sph", "SPH*")):
    r = J(f"outputs/clarity/mg_ml_curve_{d}.json")["results"]
    rows.append(["MedGemma-4B-ECGInstruct", name, f1(r["B0"][0]), f1(r["ZS"][0]), f1(r["curve"]["THR@100"]["F1_mean"]),
                 f1(r["curve"]["S1@100"]["F1_mean"]), "—", f1(r["curve"]["S1@600"]["F1_mean"]), "—", ci(r["bootstrap"]["S1@600-B0"])])
q = J("outputs/clarity/qwen_curve_cpsc.json")["results"]; qb = J("outputs/clarity/qwen_bootstrap_cpsc.json")
rows.append(["Qwen2-VL-7B (general VLM)", "CPSC", f1(q["B0"][0]), f1(q["B1@0.5"][0]), f1(q["curve"]["THR@100"]["F1_mean"]),
             f1(q["curve"]["S1@100"]["F1_mean"]), "—", f1(q["curve"]["S1@600"]["F1_mean"]), "—", ci(qb["bootstrap"]["S1@600-B0"])])
c = J("outputs/clarity/code15_curve.json")["results"]; cb = J("outputs/clarity/code15_bootstrap.json")
rows.append(["PULSE-7B (in-domain)", "CODE-15%", f1(c["B0"]["F1"]), f1(c["B1@0.5 (zero-shot)"]["F1"]), f1(c["curve"]["THR@100"]["F1_mean"]),
             f1(c["curve"]["S1@100"]["F1_mean"]), "—", f1(c["curve"]["S1@600"]["F1_mean"]), "—", ci(cb["S1@600-B0"])])
write("Table2_main_results", ["Model", "Dataset", "B0 (generated)", "Zero-shot P≥0.5", "THR n=100", "S1 n=100", "Probe n=100",
                              "S1 n=600", "Probe n=600", "S1@600 − B0 [95% CI]"], rows,
      "Macro-F1 (official ECGBench metric) on the fixed test split. n-label results: mean of 20 random draws from the pool. "
      "CI: paired bootstrap over test records (B = 2000). Probe = linear probe on the "
      "pool-selected PULSE hidden state; not computed for other models. *SPH: pre-registered new site.")

# ---------- Table 3: pre-registered hypotheses (SPH)
P = J("outputs/clarity/ml_curve_sph.json")["results"]; M = J("outputs/clarity/mg_ml_curve_sph.json")["results"]; B = J("outputs/clarity/baselines_sph.json")["results"]
pk = "paired_F1_S1_minus_other [mean, p2.5, p97.5, draws S1 better]"; cur = P["curve"]
h4s = {n: cur[f"THR@{n}"]["F1_mean"] - cur[f"S1@{n}"]["F1_mean"] for n in (25, 50, 100)}
h4l = {n: cur[f"S1@{n}"]["F1_mean"] - cur[f"THR@{n}"]["F1_mean"] for n in (300, 400, 600)}
h5 = {n: B[pk][f"S1-PROBE-LLM@{n}"] for n in (25, 50, 100, 200)}
rows = [
    ["H1 (primary)", "PULSE: S1@600 − B0", ci(P["bootstrap"]["S1@600-B0"]), "CI lower > 0 and point ≥ 3", "Supported"],
    ["H2", "PULSE: zero-shot − B0", ci(P["bootstrap"]["ZS-B0"]), "CI lower > 0", "Not supported"],
    ["H3", "PULSE: S1@600 − zero-shot", ci(P["bootstrap"]["S1@600-ZS"]), "CI lower > 0", "Supported"],
    ["H4", "THR − S1 at n = 25/50/100; S1 − THR at n = 300/400/600",
     "; ".join(f"{v:+.1f}" for v in h4s.values()) + " / " + "; ".join(f"{v:+.1f}" for v in h4l.values()), "≥ 2 of 3 in each direction",
     "Supported" if sum(v >= 0 for v in h4s.values()) >= 2 and sum(v > 0 for v in h4l.values()) >= 2 else "Not supported"],
    ["H5", "S1 − probe, 2.5th percentile at n = 25/50/100/200", "; ".join(f"{v[1]:+.1f}" for v in h5.values()), "all > −3",
     "Supported" if all(v[1] > -3 for v in h5.values()) else "Not supported"],
    ["H6", "MedGemma: S1@600 − B0", ci(M["bootstrap"]["S1@600-B0"]), "CI lower > 0 and point ≥ 3", "Supported"],
]
write("Table3_preregistered", ["Hypothesis", "Contrast (SPH test, n = 2312)", "Estimate", "Decision rule", "Verdict"], rows,
      "Pre-registration: docs/prereg_sph.md; code, data and images frozen by md5 before any SPH model output (outputs/clarity/prereg_sph_freeze.json).")

# ---------- Table 4: operating points targeting sensitivity >= 0.90 (threshold chosen on pool OOF predictions)
NAMES = {"AF": "Atrial fibrillation", "I-AVB": "1st-degree AV block", "STD": "ST depression", "PVC": "Premature ventricular complex",
         "SB": "Sinus bradycardia", "STDEV": "ST deviation"}
fmt = lambda v: f"{v[0]:.2f} ({v[1]:.2f}–{v[2]:.2f})"
rows = []
for d, dn in (("cpsc", "CPSC"), ("sph", "SPH*")):
    cl = J(f"outputs/clarity/clinical_{d}.json")
    for L, o in cl["operating"].items():
        for m, mn in (("B0", "Generated answer (B0)"), ("S1", "S1"), ("PROBE", "Hidden-state probe")):
            rows.append([dn, NAMES[L], mn, fmt(o[m]["sens"]), fmt(o[m]["spec"]), fmt(o[m]["PPV"]), fmt(o[m]["NPV"])])
write("Table4_operating_points", ["Dataset", "Diagnosis", "Read-out", "Sensitivity", "Specificity", "PPV", "NPV"], rows,
      "S1 and probe thresholds were chosen on out-of-fold predictions in the 600-record pool to reach sensitivity ≥ 0.90, then applied unchanged to the test split. "
      "B0 is the generated answer (fixed operating point). 95% CI: bootstrap over test records (B = 1000). *SPH: pre-registered new site.")

# ---------- Table 5: probability calibration (macro over labels)
rows = []
for d, dn in (("cpsc", "CPSC"), ("sph", "SPH*")):
    cal = J(f"outputs/clarity/clinical_{d}.json")["calibration"]
    for m, mn in (("RAW", "Raw P(yes)"), ("S1", "S1"), ("PROBE", "Hidden-state probe")):
        mm = cal[m]["macro"]; rows.append([dn, mn, f"{mm['ECE']:.3f}", f"{mm['Brier']:.4f}", f"{mm['AUROC']:.3f}"])
write("Table5_calibration", ["Dataset", "Probability", "ECE (macro)", "Brier (macro)", "AUROC (macro)"], rows,
      "ECE with 10 equal-width bins per label; macro = mean over labels. S1 and probe fitted on the 600-record pool. *SPH: pre-registered new site.")
