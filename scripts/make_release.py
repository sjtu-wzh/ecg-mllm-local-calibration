"""Assemble a clean, GitHub-ready copy of this project in release/ecg-mllm-local-calibration/.
Copies code, docs, splits and the final model outputs; strips ECGBench prompt text from generated-answer files;
leaves out logs, superseded/faulty runs, images, model weights, ECGBench data and the official PULSE code."""
import glob, json, os, shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DST = os.path.join(ROOT, "release/ecg-mllm-local-calibration")
os.makedirs(DST, exist_ok=True)
for e in os.listdir(DST):  # rebuild from scratch but keep the git history of the release repository
    if e != ".git":
        (shutil.rmtree if os.path.isdir(os.path.join(DST, e)) else os.remove)(os.path.join(DST, e))
cp = lambda rel: (os.makedirs(os.path.dirname(os.path.join(DST, rel)), exist_ok=True), shutil.copy2(os.path.join(ROOT, rel), os.path.join(DST, rel)))

CODE_EXCLUDE = {"clarity/drop_nan_rows.py", "clarity/mg_purge_bad.py"}
files = ["README.md", "LICENSE", "verify_freeze.py", "docs/REPRODUCE.md", "docs/prereg_sph.md", "docs/prereg_robustness.md", "environment/README.md", *glob.glob("environment/*.txt", root_dir=ROOT),
         "scripts/infer_mps.py", "scripts/prepare_ecgbench.py", "scripts/make_scorer.py", "scripts/build_derived_datasets.py",
         "scripts/run_all.sh", "scripts/download_ptbxl_meta.sh", "scripts/download_models.sh", "scripts/make_release.py",
         *[f for f in glob.glob("clarity/*.py", root_dir=ROOT) if f not in CODE_EXCLUDE],
         *glob.glob("data/splits/*.json", root_dir=ROOT),
         # SPH (CC0) label set, prompts and question files; frozen in the pre-registrations
         "data/derived/sph_ml.json", "data/derived/sph_ml_questions.json", "data/derived/sph_robust_prompts.json",
         "data/derived/sph_robust_test_questions.json", "data/derived/sph_robust_P1_questions.json",
         # figure/table scripts and their outputs (manuscript and submission files are not released)
         *glob.glob("paper/*.py", root_dir=ROOT), *glob.glob("paper/tables/*", root_dir=ROOT), *glob.glob("paper/tables_supp/*", root_dir=ROOT),
         *glob.glob("paper/figures/*", root_dir=ROOT), *glob.glob("paper/figures_supp/*", root_dir=ROOT)]

# final outputs only (no logs, prevrun/eager_partial/backup copies, faulty-config checks, image visualisations)
OUT_KEEP = ["b1_cpsc_dev.jsonl", "b1_cpsc_test.jsonl", "b1_cpsc_all.jsonl", "ev_cpsc_dev.jsonl", "ev_cpsc_test.jsonl",
            "b1_ptbxl_all.jsonl", "ev_ptbxl_all.jsonl", "b1_code15_all.jsonl", "b1_csnml_all.jsonl", "b1_g12ml_all.jsonl",
            "mcq_g12_all.jsonl", "qwen_b1_cpsc_all.jsonl", "mg_b1_cpsc_all.jsonl", "mg_b1_csnml_all.jsonl",
            "results_b1.json", "results_m3.json", "preds_test.json", "results_transfer.json", "preds_transfer.json",
            "bootstrap.txt", "bootstrap_transfer.txt", "stats_cpsc.json", "clinical_s1.txt",
            "learning_curve.json", "code15_curve.json", "ml_curve_csn.json", "ml_curve_g12.json", "mcq_curve_g12.json",
            "mcq_blind_g12.json", "qwen_curve_cpsc.json", "mg_curve_cpsc.json", "mg_ml_curve_csn.json",
            "pulse_bootstrap_cpsc.json", "mg_bootstrap_cpsc.json", "preregistration_domain.json", "domain_results.json",
            "m1_ocr_pilot.json", "pulse_cache_check.json", "qwen_cache_check.json", "mg_batch_check_cpsc.json",
            "mg_batch_check_csnml.json", "mg_batch_check_g12ml.json", "mg_verify_cpsc.json", "mg_verify_csnml.json",
            "baselines_cpsc.json", "baselines_csn.json", "baselines_g12.json", "baselines_sph.json", "ml_curve_sph.json",
            "mg_ml_curve_sph.json", "b1_sphml_all.jsonl", "mg_b1_sphml_all.jsonl", "clinical_cpsc.json", "clinical_sph.json",
            "code15_bootstrap.json", "qwen_bootstrap_cpsc.json", "prereg_sph_freeze.json", "prereg_sph_verdicts.txt",
            "prereg_robust_freeze.json", "robust_results.json", "domain_rerun_check.json", "learning_curve.png", "learning_curves_cpsc_vs_code15.png", "learning_curves_3ood.png"]
files += [f"outputs/clarity/{f}" for f in OUT_KEEP]
files += ["outputs/clarity/mg_reference/cpu_fp32.json", "outputs/clarity/mg_reference/mps_bf16_t26.json",
          *glob.glob("outputs/clarity/pulse_verify/*", root_dir=ROOT), "outputs/pulse7b/cpsc_score.txt"]
missing = [f for f in files if not os.path.exists(os.path.join(ROOT, f))]
if missing:
    raise SystemExit(f"missing files: {missing}")
for f in files:
    if f.endswith("_questions.json") and "pulse_verify" in f:
        continue  # subsets of ECGBench questions: not redistributed
    if f.endswith(".jsonl"):  # strip ECGBench prompt text wherever it appears
        os.makedirs(os.path.dirname(os.path.join(DST, f)), exist_ok=True)
        with open(os.path.join(ROOT, f)) as fi, open(os.path.join(DST, f), "w") as fo:
            for line in fi:
                r = json.loads(line); r.pop("prompt", None); fo.write(json.dumps(r, ensure_ascii=False) + "\n")
        continue
    cp(f)

# generated answers: keep id, text, model, timing; drop the ECGBench prompt
ANSWERS = [*glob.glob("outputs/pulse7b/*/answers.jsonl", root_dir=ROOT), "outputs/qwen2vl/cpsc-test/answers.jsonl",
           "outputs/medgemma/cpsc/answers.jsonl", "outputs/medgemma/csnml/answers.jsonl", "outputs/medgemma/sphml/answers.jsonl",
           *glob.glob("outputs/robust/*.jsonl", root_dir=ROOT)]
for f in ANSWERS:
    os.makedirs(os.path.dirname(os.path.join(DST, f)), exist_ok=True)
    with open(os.path.join(ROOT, f)) as fi, open(os.path.join(DST, f), "w") as fo:
        for line in fi:
            r = json.loads(line); r.pop("prompt", None); fo.write(json.dumps(r, ensure_ascii=False) + "\n")

# experiment 3 queue: replace the absolute project path with the repository root
q = open(os.path.join(ROOT, "outputs/robust/queue_robust.sh")).read().replace(f"cd {ROOT}", 'cd "$(dirname "$0")/../.."')
os.makedirs(os.path.join(DST, "outputs/robust"), exist_ok=True); open(os.path.join(DST, "outputs/robust/queue_robust.sh"), "w").write(q)
for d, _, fs in os.walk(DST):  # no personal paths in the release
    for f in fs:
        if f.endswith((".py", ".sh", ".md", ".json", ".jsonl", ".txt", ".csv")):
            assert os.path.expanduser("~") not in open(os.path.join(d, f), errors="ignore").read(), os.path.join(d, f)
open(os.path.join(DST, ".gitignore"), "w").write(
    "PULSE/\ndata/ECGBench/\ndata/derived/*\n!data/derived/sph_*\ndata/ptbxl_meta/\ndata/sph_raw/\ndata/sph_images/\ndata/sph_images_robust/\noutputs/features/\nmodels/\nscripts/score_ecgbench.py\nrelease/\n"
    "__pycache__/\n*.pyc\n*.log\n.DS_Store\n")
n = sum(len(fs) for _, _, fs in os.walk(DST)); size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(DST) for f in fs)
print(f"release: {DST}  ({n} files, {size / 1e6:.1f} MB)")
