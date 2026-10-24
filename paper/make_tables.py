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


# ---------- Table 1: datasets, study roles and analysis splits (every count computed from the split / cohort files)
import csv as _csv, re as _re
import numpy as np
sph = J("data/derived/sph_ml.json"); csn = J("data/derived/csn_ml.json"); g12 = J("data/derived/g12_ml.json")
spl = {d: J(f"data/splits/{d}.json") for d in ("cpsc", "csn_ml", "g12_ml", "sph_ml")}
c15 = J("outputs/clarity/code15_curve.json")["results"]["split"]
cpsc_all = J("data/ECGBench/cpsc-test.json")
# PTB-XL domain-analysis cohort: replay the 20 patient-grouped outer partitions of clarity/domain_analysis.py (no redraws occurred)
assert all(k == 0 for k in J("outputs/clarity/domain_results.json")["redraws"]["PTB"])
ptb_dom = J("data/derived/ptbxl_cpsc9.json")
db = {int(r["ecg_id"]): r for r in _csv.DictReader(open(os.path.join(ROOT, "data/ptbxl_meta/ptbxl_database.csv")))}
grp = np.array([str(db[int(_re.match(r"ptb-xl-(\d+)", x["id"]).group(1))]["patient_id"]) for x in ptb_dom])
ug = np.unique(grp); size = {g: int((grp == g).sum()) for g in ug}; pools = []
for r in range(20):
    cnt = 0
    for g in np.random.default_rng(r * 1000).permutation(ug):
        if cnt >= 600: break
        cnt += size[g]
    pools.append(cnt)
dev_rng = f"{min(pools)}–{max(pools)}" if min(pools) != max(pools) else str(pools[0])
test_rng = f"{len(ptb_dom) - max(pools)}–{len(ptb_dom) - min(pools)}" if min(pools) != max(pools) else str(len(ptb_dom) - pools[0])
# PTB-XL transfer-source cohort (clarity/ptbxl_labels.py): all ECGBench PTB-XL images, record-level 70/30 split
tr = J("data/splits/ptbxl.json"); img = {x["id"]: x["image"] for x in J("data/ECGBench/ptb-test.json")}
pid = lambda i: db[int(_re.match(r"ptb-xl/(\d+)_hr", img[i]).group(1))]["patient_id"]
shared_pat = len({pid(i) for i in tr["fit"]} & {pid(i) for i in tr["heldout"]})
n_unlab = sum(not v for v in J("data/ptbxl_meta/ptbxl_cpsc_labels.json")["labels"].values())
src_csn, src_g12 = len(J("data/ECGBench/csn-test-no-cot.json")), len(J("data/ECGBench/g12-test-no-cot.json"))
f = lambda n: f"{n:,}" if isinstance(n, int) else n
rows = [
    ["**A. Main analyses**", "", "", "", "", "", "", ""],
    ["CPSC 2018 (China)", "Development and evaluation", "No", "Multi-label (9)", f(len(cpsc_all)), f(len(spl["cpsc"]["dev"])), f(len(spl["cpsc"]["test"])), "Fixed, record-level^a^"],
    ["CSN (China)", "Out-of-domain evaluation", "No", f"Multi-label ({len(csn['labels'])})^b^", f(len(csn["records"])), f(len(spl["csn_ml"]["pool"])), f(len(spl["csn_ml"]["test"])), "Fixed, record-level"],
    ["G12EC (USA)", "Out-of-domain evaluation", "No", f"Multi-label ({len(g12['labels'])})^b^", f(len(g12["records"])), f(len(spl["g12_ml"]["pool"])), f(len(spl["g12_ml"]["test"])), "Fixed, record-level"],
    ["SPH (China)", "Pre-registered validation", "No", f"Multi-label ({len(sph['labels'])})^c^", f(len(sph["records"])), f(len(spl["sph_ml"]["pool"])), f(len(spl["sph_ml"]["test"])), "Fixed; one ECG per patient"],
    ["**B. Supporting analyses**", "", "", "", "", "", "", ""],
    ["CODE-15%: binary (Brazil)", "In-domain boundary condition", "Yes", "Binary (2)^d^", f(c15["dev_pool"]["bin"] + c15["test"]["bin"]), f(c15["dev_pool"]["bin"]), f(c15["test"]["bin"]), "Fixed, record-level"],
    ["CODE-15%: multi-label (Brazil)", "In-domain boundary condition", "Yes", "Multi-label (6)^d^", f(c15["dev_pool"]["multi"] + c15["test"]["multi"]), f(c15["dev_pool"]["multi"]), f(c15["test"]["multi"]), "Fixed, record-level"],
    ["PTB-XL: domain analysis (Germany)", "Matched domain comparison", "Yes", "Multi-label (8)^e^", f(len(ptb_dom)), dev_rng, test_rng, "20 patient-grouped partitions"],
    ["PTB-XL: transfer source (Germany)", "Transfer source", "Yes", "Multi-label (9)^f^", f(len(tr["fit"]) + len(tr["heldout"])), f(len(tr["fit"])), f(len(tr["heldout"])) + "^f^", "Fixed, record-level^f^"],
]
write("Table1_datasets", ["Dataset (country)", "Study role", "PULSE training source", "Task (labels)", "Included<br>(n)", "Development<br>(n)", "Held-out<br>(n)", "Split"], rows,
      "All counts are ECG records, not patients. Task (labels) gives the number of evaluated classes: labels for multi-label tasks and two mutually exclusive "
      "classes for the binary task. PULSE training source indicates whether the source dataset is among the training sources disclosed for PULSE "
      "(ECGBench classification); it does not establish record-level overlap. Development records were used only for fitting and threshold selection. "
      "Images come from ECGBench, except SPH, which was rendered from the released signals. Labels and prompts are listed in Tables S1 and S2. "
      f"^a^ CPSC test results were inspected repeatedly during method development. In the domain analysis, all {len(cpsc_all):,} CPSC records were "
      "re-partitioned 20 times at record level (600 development records each) and evaluated on 8 labels. "
      f"^b^ Reference labels derived from the correct ECGBench multiple-choice options; diagnoses absent from the options count as negative, so labels may be "
      f"incomplete. Labels with at least 40 positives were kept: {src_csn:,} → {len(csn['records']):,} (CSN) and {src_g12:,} → {len(g12['records']):,} (G12EC) records. "
      f"^c^ Retained primary AHA statement codes after removal of secondary modifiers; one ECG per patient from a random sample of 3,000 patients "
      f"(→ {len(sph['records']):,} records with at least one retained label). "
      f"^d^ Binary: normal versus abnormal; multi-label: six diagnoses. The two tasks use disjoint records; a label budget n is split as floor(n/2) binary "
      f"and the remainder multi-label records, and the reported macro-F1 is computed over all {c15['test']['bin'] + c15['test']['multi']} test records "
      "(NORM, ABNORMAL and the six diagnoses). "
      f"^e^ PTB-XL fold 10, records with at least one CPSC-mapped label; pre-specified analysis (Section 2.10). Nine-label prompt with ST elevation excluded "
      f"from the evaluation. Whole patients were added to the development set until it held at least 600 records ({dev_rng}); at most 600 were used for fitting. "
      f"^f^ PTB-XL fold 10, all ECGBench images ({n_unlab} without a CPSC-mapped label); a different cohort from the domain analysis. Read-outs were fitted on "
      f"the {len(tr['fit']):,} fit records and evaluated on the CPSC test split; the {len(tr['heldout'])} records are a source-domain hold-out set not used for any "
      f"reported result. The split is record-level and not patient-disjoint ({shared_pat} patients appear in both parts).")

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
