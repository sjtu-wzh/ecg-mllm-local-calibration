"""Supplementary tables -> paper/tables_supp/TableS{1..5}.{csv,md}. Every number is read from data/ or outputs/clarity/.
S1 label sets and prevalence, S2 prompts, S3 numerical validation, S4 exploratory / negative results, S5 pre-specified domain analysis."""
import collections, csv, json, os, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper/tables_supp"); os.makedirs(OUT, exist_ok=True)
J = lambda p: json.load(open(os.path.join(ROOT, p)))
ci = lambda v, d=1: f"{v[0]:+.{d}f} [{v[1]:.{d}f}, {v[2]:.{d}f}]"


def write(name, header, rows, note):
    with open(os.path.join(OUT, f"{name}.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(header); w.writerows(rows)
    cell = lambda x: str(x).replace("|", "\\|").replace("\n", " ")
    md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(map(cell, r)) + " |" for r in rows]
    md = [re.sub(r"(?<![\w.])-(?=\d)", "\u2212", l) for l in md]   # typographic minus for negative numbers (csv keeps ASCII)
    open(os.path.join(OUT, f"{name}.md"), "w").write("\n".join(md) + f"\n\n{note}\n")
    print(f"{name}: {len(rows)} rows")


gold = lambda rec: {s.strip() for s in rec["conversations"][-1]["value"].split(";") if s.strip()}

# ---------- Table S1: label sets, source codes and prevalence (pool / test)
CPSC = {"NORM": "Normal ECG", "AF": "Atrial fibrillation", "I-AVB": "First-degree atrioventricular block", "LBBB": "Left bundle branch block",
        "RBBB": "Right bundle branch block", "PAC": "Premature atrial contraction", "PVC": "Premature ventricular contraction",
        "STD": "ST-segment depression", "STE": "ST-segment elevated"}
SCP = {"NORM": "NORM", "AF": "AFIB", "I-AVB": "1AVB", "LBBB": "CLBBB", "RBBB": "CRBBB", "PAC": "PAC", "PVC": "PVC", "STD": "STD_",
       "STE": "STE_ + INJAS/INJAL/INJIN/INJLA/INJIL (not in primary 8-label domain analysis)"}
rows = []
cp = {x["id"]: x for x in J("data/ECGBench/cpsc-test.json")}; sp = J("data/splits/cpsc.json")
for L, nm in CPSC.items():
    npos = lambda ids: sum(L in gold(cp[i]) for i in ids)
    rows.append(["CPSC 2018", L, nm, "ECGBench CPSC label", npos(sp["dev"]), npos(sp["test"]), f"PTB-XL SCP: {SCP[L]}"])
AHA = {r["Code"]: r["Description"] for r in csv.DictReader(open(os.path.join(ROOT, "data/sph_raw/code.csv")))}
SPH_RAW = {r["ECG_ID"]: r["AHA_Code"].split(";") for r in csv.DictReader(open(os.path.join(ROOT, "data/sph_raw/metadata.csv")))}


def sph_note(code, recs):
    """Which raw AHA statements (base code + modifiers) were collapsed into this label, with counts in the sampled records."""
    c = collections.Counter(st for r in recs for st in SPH_RAW[r["id"][4:]] if st.split("+")[0] == code)
    parts = [f"{st} {' + '.join(AHA[m].lower() for m in st.split('+')[1:])} ({n})" if "+" in st else f"{st} unmodified ({n})" for st, n in c.most_common()]
    return "no modifiers" if len(parts) == 1 else "merged: " + "; ".join(parts)


for d, name, f in (("csn_ml", "CSN", "csn_ml"), ("g12_ml", "G12EC", "g12_ml"), ("sph_ml", "SPH", "sph_ml")):
    D = J(f"data/derived/{f}.json"); by = {r["id"]: r for r in D["records"]}; s = J(f"data/splits/{d}.json")
    for L, nm in D["labels"].items():
        npos = lambda ids: sum(L in gold(by[i]) for i in ids)
        src = f"AHA statement code {D['aha_codes'][L]}" if "aha_codes" in D else "ECGBench MCQ gold option"
        rows.append([name, L, nm, src, npos(s["pool"]), npos(s["test"]), sph_note(D["aha_codes"][L], D["records"]) if "aha_codes" in D else ""])
write("TableS1_labels", ["Dataset", "Abbreviation", "Label (as in prompt)", "Source of gold label", "Positives in pool", "Positives in test", "Mapping note"], rows,
      "Pool = CPSC dev split (600) or the 600-record calibration pool; test = fixed held-out split. CSN/G12EC multi-label versions: gold = union of the "
      "correct MCQ options of each record, labels kept if they have at least 40 positive records, and records with none of them removed (CSN 1611 → 1579; G12EC 2026 → 1996; data/derived/*_ml.json). "
      "SPH: labels are the SPH primary statements of the AHA/ACC/HRS standard (Liu et al., Sci Data 2022), defined before any model output (docs/prereg_sph.md). "
      "Rules applied (clarity/build_sph.py): (1) one record per patient (lowest ECG_ID), random sample of 3000 (seed 42); (2) each statement is reduced to its "
      "primary code by dropping secondary modifiers after '+' (e.g. 145+362 'ST deviation, depression' and 145+363 'ST deviation, elevation' both count as 145); "
      "the merged statements are listed in the last column; (3) primary codes with ≥ 50 positives are kept; (4) records with none of them are excluded (2912 remain). "
      "No other merging was done, and no SPH label was mapped to a CPSC, CSN or G12EC label. Abbreviations are ours; STDEV (code 145, ST deviation, either "
      "direction), STTC (code 146, ST deviation with T-wave change) and TAb (code 147) keep their AHA meaning and are not equivalent to CPSC STD/STE or to the "
      "CSN/G12EC T-wave labels despite similar names. Results are therefore compared across datasets only as within-dataset contrasts (e.g. S1 − B0 on the same "
      "test split), never label-by-label. PTB-XL (fold 10) SCP codes were mapped to CPSC labels only for the domain analysis and the transfer experiment, by the "
      "rules in the last column (clarity/ptbxl_labels.py).")

# ---------- Table S2: prompts
lp = open(os.path.join(ROOT, "clarity/label_prob.py")).read()
ev = dict(re.findall(r'"(\w+)": "([^"]+\?)"', lp.split("EVIDENCE = {")[1].split("}")[0]))
rows = [["B0 (generated answer)", "All models; CPSC, CSN, G12EC, SPH, PTB-XL",
         "You are provided with an ECG signal image. Please classify the ECG image into the following diagnostic statement classes:\\n"
         "{ABBR1}({name1}), {ABBR2}({name2}), …\\nDirectly output the class name and separate with a semicolon.",
         "ECGBench CPSC template; label list replaced per dataset. Output parsed by the official scorer."],
        ["B0, CODE-15%", "PULSE-7B", "Official ECGBench CODE-15% prompts (binary normal/abnormal and 6-label multi-label)", "Native (training) prompt"],
        ["Yes/No label question (ZS, THR, S1)", "All models",
         "You are provided with an ECG signal image. Does this ECG show {name} ({abbr})? Answer Yes or No.",
         "P(yes) = softmax over the 'Yes'/'No' next-token logits; image prefix KV-cached and reused across labels."],
        ["Neutral prompt (hidden-state probe)", "PULSE-7B", "Describe the ECG.",
         "One forward pass per image; no diagnosis name in the prompt. Features: hidden state of the selected layer at the last prompt token (or mean over image tokens; Figure S6e)."]]
rows += [[f"Evidence question: {k}", "PULSE-7B, CPSC and PTB-XL (exploratory)", f"You are provided with an ECG signal image. {q} Answer Yes or No.", "Table S4 (variants E1–E3)"] for k, q in ev.items()]
RP = J("data/derived/sph_robust_prompts.json")
rows += [["Robustness: Yes/No paraphrase QA", "PULSE-7B, SPH (Experiment 3)", RP["QA"], "Table S8, R3"],
         ["Robustness: Yes/No paraphrase QB", "PULSE-7B, SPH (Experiment 3)", RP["QB"], "Table S8, R3"],
         ["Robustness: generated-answer paraphrase P1", "PULSE-7B, SPH (Experiment 3)", RP["P1"].replace("\n", "\\n"), "Table S8, R4"]]
write("TableS2_prompts", ["Use", "Model / data", "Prompt (verbatim)", "Note"], rows,
      "Chat templates: PULSE-7B uses the LLaVA-v1.6 Vicuna template; Qwen2-VL and MedGemma use their own processor chat templates with the same user text.")

# ---------- Table S3: numerical validation
pv = open(os.path.join(ROOT, "outputs/clarity/pulse_verify.log")).read()
b0v = re.findall(r"^(b0_\w+) \{'n_rows': (\d+), 'text_identical': (\d+)", pv, re.M)
b1v = re.findall(r"^(\w+) \{'n_rows': (\d+), 'n_values': (\d+), 'max_abs_diff': ([\d.e-]+)", pv, re.M)
pc, qc = J("outputs/clarity/pulse_cache_check.json"), J("outputs/clarity/qwen_cache_check.json")
mb = {d: J(f"outputs/clarity/mg_batch_check_{d}.json") for d in ("cpsc", "csnml", "g12ml")}
cpu, mps = J("outputs/clarity/mg_reference/cpu_fp32.json")["p"], J("outputs/clarity/mg_reference/mps_bf16_t26.json")["p"]
diffs = [abs(cpu[k][0] - x) for k in cpu if k in mps for x in mps[k]]
mv = {d: J(f"outputs/clarity/mg_verify_{d}.json") for d in ("cpsc", "csnml")}
rows = [
    ["PULSE-7B", "Re-run of generated answers (B0), fresh process",
     f"{len(b0v)} sets × {b0v[0][1]} records", f"text identical {sum(int(t) for _, _, t in b0v)}/{sum(int(n) for _, n, _ in b0v)}", "Pass"],
    ["PULSE-7B", "Re-run of P(yes) (label, evidence, MCQ files)", f"{len(b1v)} sets, {sum(int(v) for _, _, v, _ in b1v)} values",
     f"max |Δ| = {max(float(m) for *_, m in b1v):.4f}", "Pass"],
    ["PULSE-7B", "KV-cached image prefix vs full forward pass", f"{pc['n_images']} images, {len(pc['rows'])} values", f"max |Δ| = {pc['max_abs_diff']:.4f}", "Pass"],
    ["Qwen2-VL-7B", "KV-cached prefix vs full forward pass (fp32 LM head)", f"{len(qc['rows'])} values", f"max |Δ| = {qc['max_abs_diff']:.4f}", "Pass"],
    ["MedGemma-4B", "Label-batched vs single question", "; ".join(f"{d}: {len(v['rows'])}" for d, v in mb.items()),
     "; ".join(f"{d} {v['max_abs_diff']:.1e}" for d, v in mb.items()), "Pass"],
    ["MedGemma-4B", "Final MPS bf16 configuration vs CPU fp32 reference", f"{len(diffs)} values (3 repeats)", f"max |Δ| = {max(diffs):.3f}; mean {sum(diffs)/len(diffs):.4f}; repeats identical", "Pass (< 0.03)"],
    ["MedGemma-4B", "In-run canary: CPSC reference probabilities re-computed at start and every 50 images", "all final runs (CPSC, CSN, SPH)",
     "tolerance 0.03 vs CPU fp32; failure discards rows since the last pass", "No failure logged"],
    ["MedGemma-4B", "Diagnostic on the first (SDPA, padded-batch) runs, before the fix",
     "; ".join(f"{d}: n = {v['n']}" for d, v in mv.items()), "; ".join(f"{d} max |Δ| {v['b1_max_abs_diff']:.2f}, B0 identical {v['b0_identical']}/{v['n']}" for d, v in mv.items()),
     "Failed → all MedGemma outputs discarded and recomputed"],
]
write("TableS3_numerical_validation", ["Model", "Check", "Sample", "Result", "Outcome"], rows,
      "Apple-silicon (MPS) inference. PULSE: torch 2.1.2 / transformers 4.37.2; Qwen2-VL: torch 2.5.1 / transformers 4.47.1; MedGemma: torch 2.6 / transformers 4.57.3, "
      "eager attention, one sequence per forward pass, deep-copied KV cache, fp32 LM head. Every reported MedGemma number comes from the recomputed runs.")

# ---------- Table S4: exploratory and negative results (CPSC test, PULSE-7B, fitted on 600 dev records)
m3 = J("outputs/clarity/results_m3.json")["test"]; st = J("outputs/clarity/stats_cpsc.json"); bt = st["bootstrap_fewshot"]
ocr = J("outputs/clarity/m1_ocr_pilot.json"); tr = st["bootstrap_transfer"]["cpsc_test"]
pct = lambda v: (f"{v[0]:+.1f} [{v[1]:.1f}, {v[2]:.1f}], " + ("p < 0.001" if v[3] < 0.001 else f"p = {v[3]:.3f}")).replace("+0.0 ", "0.0 ").replace("-0.0 ", "0.0 ")
rows = [
    ["H: per-label P(yes) + generated answer (no cross-label terms)", f"{m3['H']['F1']:.1f}", f"vs S1 {m3['S1']['F1']:.1f}", "Cross-label terms, not the generated answer, carry the gain", "Not adopted"],
    ["E1: label + clinically linked evidence", f"{m3['S2']['F1']:.1f}", f"vs S1 {m3['S1']['F1']:.1f}", "Clinically linked evidence alone is weaker than all label logits", "Not adopted"],
    ["E3: S1 + clinically linked evidence", f"{m3['S4']['F1']:.1f}", f"E3 − S1 {pct(bt['S4 - S1'])}", "Not significant", "Not adopted"],
    ["E2: all labels + all 11 evidence answers", f"{m3['S3']['F1']:.1f}", f"E2 − S1 {pct(bt['S3 - S1'])}", "Small gain for 2.2× questions per image", "Reported, not main method"],
    ["E2 + uncertainty gating (M4)", f"{m3['S3+M4(d=0.1)']['F1']:.1f}", f"dev-selected δ routes {m3['S3+M4(d=0.1)']['uncertain_frac']:.0%} of records", "Gate degenerates to E2", "Not adopted"],
    ["Control: S1 protocol on B0 0/1 indicators", f"{st['control_b0_regression']['macroF1']:.1f}", f"B0 {st['control_b0_regression']['B0']:.1f}", "Gain needs the probabilities, not re-weighting of answers", "Control"],
    ["M1 lead localisation by OCR (pilot)", "—", "; ".join(f"{k.split('-')[0]} {v['mean_leads_found']:.1f}/12" for k, v in ocr.items()), "Lead labels not found reliably on ECGBench images", "Abandoned"],
] + [[f"Transfer PTB-XL → CPSC: {k.split(' (')[0].replace('S3', 'E2').replace(' - ', ' − ')}", "—", pct(v), "No transferred variant beats B0 on the new site", "Negative (Fig. S4b)"] for k, v in tr.items()]
write("TableS4_exploratory", ["Variant", "CPSC test macro-F1", "Comparison [95% CI]", "Interpretation", "Status"], rows,
      "All variants fitted on the 600-record CPSC dev split and evaluated once on the 1461-record test split (clarity/verify_stack.py, clarity/stats_cpsc.py). "
      "Paired bootstrap over test records (B = 2000). These analyses were exploratory and are not used for any main claim. "
      "Variant codes E1–E3 replace the internal codes S2/S3/S4 (Table S6).")

# ---------- Table S5: pre-specified domain analysis (PTB-XL fold 10 vs CPSC, same 9-label prompt, 8 primary labels)
D = J("outputs/clarity/domain_results.json"); P1, P2, P3, P4 = D["P1"], D["P2"], D["P3"], D["P4"]
mv = P4["mapping_variants"]
rows = [
    ["P1", "Gain G = S1@600 − B0", f"CPSC {ci(P1['G_CPSC_600'])}; PTB raw {ci(P1['G_PTBraw_600'])}; DiD {ci(P1['DiD_600'])}; n* CPSC {P1['n_star']['CPSC']}, PTB none",
     P1["verdict"]],
    ["P2", "Case-mix standardisation (PTB reweighted to CPSC label-set mix)",
     f"G_PTB,std {ci(P2['G_PTBstd_600'])}; R_std {ci(P2['R_std'], 2)}; CPSC→PTB mix {ci(P2['G_CPSC_to_PTBmix_600'])}; test-fold ESS ≥ {P2['ESS_test_min']:.0f}",
     "Inconclusive (R_std CI width > 0.8)"],
    ["P3", "Mechanism decomposition (read-out a / re-ranking c / operating point b / ≥1-label rule e)",
     "; ".join(f"{arm}: " + ", ".join(f"{c} {P3[arm][c]:+.1f}" for c in "acbe") for arm in ("CPSC", "PTB_raw", "PTB_std")) + f"; Spearman r_L {P3['spearman_rL_CPSC_PTB']:.2f}",
     "Rules met: " + ", ".join(P3["rules_met"]) + " → no single cause"],
    ["P4", "Label concordance and mapping robustness",
     f"Spearman G_CPSC vs G_PTB,std {P4['spearman_G_CPSC_vs_PTBstd']:+.2f}; top-3 CPSC {'/'.join(P4['top3']['CPSC'])}, PTB {'/'.join(P4['top3']['PTB_std'])}; "
     "R_std across mapping variants " + ", ".join(f"{k} {v['R_std_point']:.2f}" for k, v in mv.items()),
     f"H_label not supported; mapping-sensitive = {P4['mapping_sensitive']}"],
]
write("TableS5_domain_analysis", ["Analysis", "Question", "Estimates (macro-F1 points, 95% CI)", "Pre-specified verdict"], rows,
      "Plan, decision rules and code frozen by md5 during PTB-XL inference (487 of 1357 generated answers existed; about 150–200 interim answers had been inspected for error patterns during plan design; no PTB-XL metric was computed before the freeze; outputs/clarity/preregistration_domain.json). 20 patient-grouped pool/test partitions; "
      "bootstrap B = 200 per partition. Pre-registered secondary analyses not implemented: "
      + "; ".join(re.sub(r"^S\d+_", "", x).replace("_", " ") for x in D["not_implemented"]) + ".")

# ---------- Table S6: key from paper variant codes to the internal codes used in scripts and result files
rows = [
    ["E1", "S2", "Per-label stack: label P(yes) + its clinically linked evidence P(yes)", "clarity/verify_stack.py; results_m3.json"],
    ["E1 + B0", "S2H", "E1 + generated answer for that label", "clarity/verify_stack.py; results_m3.json"],
    ["E2", "S3", "Stack over all label P(yes) + all 11 evidence P(yes) (no clinical constraint)", "verify_stack.py, transfer.py, stats_cpsc.py; results_m3.json, results_transfer.json, stats_cpsc.json"],
    ["E2 + M4", "S3+M4(d=δ)", "E2 only for records whose S1 score lies within δ of its threshold, S1 elsewhere", "clarity/verify_stack.py; results_m3.json"],
    ["E2 + EM", "S3+EM", "E2 fitted on PTB-XL; label priors re-estimated by EM on the unlabelled CPSC images (Saerens et al. 2002)", "clarity/transfer.py; results_transfer.json"],
    ["E3", "S4", "S1 + only the clinically linked evidence P(yes)", "verify_stack.py, stats_cpsc.py; results_m3.json, stats_cpsc.json"],
    ["H (unchanged)", "H", "Per-label P(yes) + generated answer, no cross-label terms", "clarity/verify_stack.py; results_m3.json"],
    ["S1 (unchanged)", "S1", "Main read-out: logistic regression per label on all label logits", "all scripts"],
]
write("TableS6_variant_codes", ["Code in paper", "Internal code (scripts, result files)", "Definition", "Where"], rows,
      "Renamed so that exploratory variants cannot be confused with Supplementary Figure/Table numbers. Result files keep the internal codes so that "
      "published numbers can be traced to the frozen outputs; Supplementary Figures and Tables keep their own numbering (Figure S2, Table S4, ...).")

# ---------- Table S7: pre-registered robustness experiment (Experiment 3; written before its results existed)
if os.path.exists(os.path.join(ROOT, "outputs/clarity/robust_results.json")):
    RR = J("outputs/clarity/robust_results.json"); F, A = RR["F1"], RR["prob_macroAUC"]
    NAME = {"CLEAN": "Clean images", "ROT": "Rotated 4–10°", "PHOTO": "Phone photo of printout", "SCAN": "Low-quality scan",
            "QA": "Yes/No question paraphrase QA", "QB": "Yes/No question paraphrase QB", "P1": "Generated-answer prompt paraphrase P1"}
    g = lambda k: f"{F[k]:.1f}" if k in F else "—"
    rows = [[NAME[a], g(f"B0_{a}") if a not in ("QA", "QB") else f"({F['B0_CLEAN']:.1f})", g(f"ZS_{a}"), g(f"S1_{a}"), g(f"S1_clean->{a}"),
             f"{A[a]:.3f}" if a in A else "—"] for a in NAME]
    write("TableS7_robustness_arms", ["Arm", "B0", "Zero-shot P≥0.5", "S1 (own pool)", "S1 (clean pool)", "P(yes) macro AUROC"], rows,
          "Frozen PULSE-7B on SPH; 300-record pool subset and 800-record test subset (data/splits/sph_robust.json). S1 fitted on all 300 pool records "
          "of the arm (own pool) or on the 300 clean pool records (clean pool). For QA/QB the generated answer is the clean one (in brackets). "
          "Pre-registration: docs/prereg_robustness.md (frozen by md5 before any perturbed or paraphrased output).")
    rows = [[k.split(":")[0], k.split(":", 1)[1].strip().replace("S1_clean->", "S1 clean→").replace("_", " ").replace(" - ", " − "),
             ci(v["estimate"]), v["rule"].replace("lo>0", "CI lower > 0").replace("pt>=3", "point ≥ 3").replace("&", "and"), v["verdict"].capitalize()]
            for k, v in RR["hypotheses"].items()]
    rows += [["Descriptive", k.replace("S1_clean->", "S1 clean→").replace("_", " ").replace(" - ", " − "), ci(v), "—", "—"] for k, v in RR["descriptive"].items()]
    write("TableS8_robustness_contrasts", ["ID", "Contrast (800 test ECGs)", "Δ macro-F1 [95% CI]", "Decision rule", "Verdict"], rows,
          "Paired bootstrap over test records (B = 2000). No multiplicity correction (pre-specified).")
