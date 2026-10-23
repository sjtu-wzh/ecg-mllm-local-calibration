# Reproducing every reported number

Two levels:

* **Analysis only (CPU, minutes):** the repository ships every model output (`outputs/`). After downloading ECGBench
  (needed for the gold labels) you can re-run all analyses and figures without a GPU — steps 0, 1 and 5–7.
* **Full re-run (Apple-silicon GPU, ~1–2 days):** additionally re-generate the model outputs (steps 2–4).

All commands are run from the repository root. `$PY` = the `pulse-llava` interpreter, `$QW` = `qwen-vl`,
`$MG` = `medgemma` (see `environment/README.md`). Every script uses paths relative to the repository root.

## 0. Data and scorer

```bash
git clone https://github.com/AIMedLab/PULSE.git          # official code (scorer + LLaVA)
$PY -I scripts/make_scorer.py                            # -> scripts/score_ecgbench.py (official metric, local paths)
$PY -I scripts/prepare_ecgbench.py                       # ECGBench -> data/ECGBench/{*.json, images/}
bash scripts/download_ptbxl_meta.sh                      # PTB-XL metadata (CC BY 4.0)
$PY -I clarity/ptbxl_labels.py                           # SCP -> CPSC labels for ECGBench PTB-XL images; data/splits/ptbxl.json
$PY -I scripts/build_derived_datasets.py                 # data/derived/{csn,g12}_ml*.json, ptbxl_cpsc9.json
```
`data/splits/*.json` (fixed dev/test splits) are shipped; `clarity/split_dev_test.py` regenerates the CPSC / MMMU splits.

## 1. Official baseline (B0) for PULSE-7B

```bash
PY=$PY bash scripts/run_all.sh                           # outputs/pulse7b/{mmmu-ecg,cpsc-test,code15-test,g12-test-no-cot}
$PY -I -c "import runpy;ns=runpy.run_path('scripts/score_ecgbench.py',run_name='lib');ns['eval_cpsc_test']('outputs/pulse7b/cpsc-test')"
```

## 2. PULSE-7B Yes/No probabilities (B1) and other prompts

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5
$PY -I clarity/label_prob.py --data cpsc --split dev ; $PY -I clarity/label_prob.py --data cpsc --split test
$PY -I clarity/label_prob.py --data cpsc --split dev --set evidence ; $PY -I clarity/label_prob.py --data cpsc --split test --set evidence
$PY -I clarity/label_prob.py --data ptbxl --split all ; $PY -I clarity/label_prob.py --data ptbxl --split all --set evidence
$PY -I clarity/label_prob.py --data code15 --split all
$PY -I clarity/label_prob.py --data csnml --split all ; $PY -I clarity/label_prob.py --data g12ml --split all
$PY -I clarity/mcq_prob.py --subset g12-test-no-cot                                   # option-letter probabilities
$PY -I scripts/infer_mps.py --question-file data/derived/ptbxl_cpsc9.json --answers-file outputs/pulse7b/ptbxl-cpsc9/answers.jsonl
$PY -I scripts/infer_mps.py --question-file data/derived/csn_ml_questions.json --answers-file outputs/pulse7b/csn-ml/answers.jsonl
$PY -I scripts/infer_mps.py --question-file data/derived/g12_ml_questions.json --answers-file outputs/pulse7b/g12-ml/answers.jsonl
```

## 3. Qwen2-VL-7B-Instruct (general VLM, CPSC)

```bash
$QW -I clarity/qwen_cpsc.py --check 5     # KV-cache vs full forward -> outputs/clarity/qwen_cache_check.json
$QW -I clarity/qwen_cpsc.py               # B0 + B1; resumable — re-run until 2061 rows (MPS memory restarts)
```

## 4. MedGemma-4B-ECGInstruct (second ECG-trained model, CPSC + CSN)

```bash
bash scripts/download_models.sh
$MG -I clarity/mg_reference.py --device cpu --dtype float32 --tag cpu_fp32 --repeats 1   # CPU reference (canary)
$MG -I clarity/medgemma_run.py --data cpsc --check 5      # cached vs full forward -> mg_batch_check_cpsc.json
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.85
$MG -I clarity/medgemma_run.py --data cpsc ; $MG -I clarity/medgemma_run.py --data csnml   # re-run until complete
```

## 5. Analyses

| Reported result | Command | Output |
|---|---|---|
| CPSC: B0 vs B1 thresholds | `$PY -I clarity/evaluate.py` | `results_b1.json` |
| CPSC: S1 vs E1–E3 evidence variants (internal codes S2/S3/S4; 600 dev → 1461 test) | `$PY -I clarity/verify_stack.py` | `results_m3.json`, `preds_test.json` |
| Bootstraps, B0-regression control, S1 coefficients, clinical translation | `$PY -I clarity/stats_cpsc.py` | `bootstrap.txt`, `bootstrap_transfer.txt`, `stats_cpsc.json`, `clinical_s1.txt` |
| Zero-shot transfer PTB-XL → CPSC | `$PY -I clarity/transfer.py` | `results_transfer.json`, `preds_transfer.json` |
| CPSC learning curve (PULSE) | `$PY -I clarity/learning_curve.py` | `learning_curve.json` |
| CODE-15% learning curve (in-domain) | `$PY -I clarity/code15_curve.py` | `code15_curve.json` |
| CSN / G12EC multi-label curves (PULSE) | `$PY -I clarity/ml_curve.py --data csn` (and `g12`) | `ml_curve_{csn,g12}.json` |
| G12EC multiple-choice curve + image-blind control | `$PY -I clarity/mcq_curve.py --subset g12-test-no-cot`; `$PY -I clarity/mcq_blind_control.py` | `mcq_curve_g12.json`, `mcq_blind_g12.json` |
| Qwen2-VL CPSC curve | `$PY -I clarity/qwen_curve.py` | `qwen_curve_cpsc.json` |
| MedGemma CPSC curve | `$PY -I clarity/qwen_curve.py --b1 outputs/clarity/mg_b1_cpsc_all.jsonl --b0 outputs/medgemma/cpsc/answers.jsonl --out outputs/clarity/mg_curve_cpsc.json` | `mg_curve_cpsc.json` |
| MedGemma CSN curve | `$PY -I clarity/ml_curve.py --data csn --model medgemma` | `mg_ml_curve_csn.json` |
| CPSC bootstraps for any model (S1@600 vs B0 / ZS) | `$PY -I clarity/bootstrap_cpsc_model.py --b1 … --b0 … --out …` | `pulse_bootstrap_cpsc.json`, `mg_bootstrap_cpsc.json` |
| Pre-registered domain analysis (PTB-XL vs CPSC, same prompt) | `$PY -I clarity/domain_analysis.py` | `domain_results.json` (plan: `preregistration_domain.json`) |
| Experiment 3 robustness (pre-registered: `docs/prereg_robustness.md`, freeze `prereg_robust_freeze.json`) | `python -I clarity/robust_setup.py`; then `bash outputs/robust/queue_robust.sh` (B0 via `scripts/infer_mps.py`, P(yes) via `clarity/robust_prob.py`, analysis `clarity/robust_eval.py`) | `outputs/robust/*.jsonl`, `robust_results.json` |
| OCR lead-localisation pilot (negative) | `$PY -I clarity/lead_locate.py` | `m1_ocr_pilot.json` |
| PULSE cache check / random re-computation | `$PY -I clarity/cache_check_pulse.py`; `$PY -I clarity/pulse_verify.py` | `pulse_cache_check.json`, `pulse_verify/summary.json` |

`outputs/clarity/b1_cpsc_all.jsonl` is simply `b1_cpsc_dev.jsonl` + `b1_cpsc_test.jsonl` concatenated.

## 6. Figures

```bash
$PY -I clarity/plot_learning_curve.py        # learning_curve.png (CPSC, PULSE)
$PY -I clarity/plot_curves_2panel.py         # learning_curves_cpsc_vs_code15.png
$PY -I clarity/plot_curves_3ood.py           # learning_curves_3ood.png (CPSC, CSN, G12EC)
```

## 7. Notes on protocol

* Naming: the paper calls the exploratory evidence variants E1/E2/E3; scripts and result files keep the internal
  codes S2/S3/S4 (key: `paper/tables_supp/TableS6_variant_codes.md`).

* All thresholds / coefficients are fitted on the dev pool only; learning curves use 20 random draws per n.
  Bands in figures are draw percentiles on a fixed test split, not confidence intervals.
* Variants added after seeing results are labelled *exploratory* in code and figures (evidence variants E2/E3 = internal S3/S4, S1+B0,
  the threshold-vs-S1 rule). The CPSC test split was evaluated many times; only large effects (S1 vs B0) should be read
  as robust. The domain analysis was pre-registered and its code frozen by md5 before PTB-XL B0 was read
  (`preregistration_domain.json`, `_freeze`).
