# Frozen ECG multimodal LLMs know more than they say: label-budget-aware local calibration

Code, model outputs, pre-registrations and analyses for the study

> Wang Z, Fan X, Chen L, Shi Y, Xiang S, Jiang L, Shao Q. *Frozen ECG Multimodal Large Language Models Know More Than They Say:
> Label-Budget-Aware Local Calibration for 12-Lead ECG Image Diagnosis.* (submitted)

The study builds on **PULSE-7B** and the **ECGBench** benchmark (Liu et al., *npj Digital Medicine* 2026;
[AIMedLab/PULSE](https://github.com/AIMedLab/PULSE)). No model weights are changed.

## What the study does

ECG multimodal LLMs are evaluated by the text they generate. We compared that text (**B0**) with read-outs of the same frozen model obtained with different prompts:

| Read-out | What it uses | Labels needed |
|---|---|---|
| Zero-shot (ZS) | P(yes) for one Yes/No question per diagnosis, threshold 0.5 | none |
| Threshold only (THR) | one threshold per label on P(yes) | tens |
| **S1** | logistic regression per label on the logits of *all* label probabilities | tens to hundreds |
| Hidden-state probe | L2 logistic regression on a language-model hidden state | ≈ 200–300 or more |

Each read-out is fitted on n = 25–600 labelled ECGs from the target site and evaluated on a fixed test split.

## Main results (macro-F1, official ECGBench metric, held-out test splits)

| Model / dataset (not used for training) | B0 | S1, 600 labels | S1@600 − B0 [95% CI] |
|---|---|---|---|
| PULSE-7B / CPSC 2018 | 57.6 | 68.8 | +10.7 [8.2, 13.3] |
| PULSE-7B / CSN (multi-label) | 39.3 | 59.2 | +20.5 [17.5, 23.5] |
| PULSE-7B / G12EC (multi-label) | 32.4 | 49.0 | +17.2 [15.4, 19.1] |
| PULSE-7B / SPH (pre-registered new site) | 47.4 | 61.8 | +13.5 [11.2, 15.8] |
| MedGemma-4B-ECGInstruct / CPSC | 23.5 | 45.8 | +22.3 [20.1, 24.5] |
| MedGemma-4B-ECGInstruct / CSN | 16.5 | 44.2 | +27.7 [25.2, 30.2] |
| MedGemma-4B-ECGInstruct / SPH (pre-registered) | 21.2 | 41.8 | +20.7 [18.5, 22.9] |

**Other findings:**

* **The best read-out depends on the label budget.** Probability-based read-outs (THR or S1, depending on the dataset) are best with up to about 100 labels and the hidden-state probe from about 200–300; with 600 labels the probe adds a further 2.4–4.1 points. The probe's layer and token were selected on the full 600-record pool, so its advantage below 600 labels may be overstated.
* **SPH validation.** It was pre-registered (`docs/prereg_sph.md`). Five of six hypotheses were supported.
* **Robustness (pre-registered, `docs/prereg_robustness.md`).** The gain persists under simulated phone photographs, low-quality scans, rotation and prompt paraphrases, at +11.5 to +13.7.
* **Calibration.** Local calibration reduces the macro ECE from 0.09–0.12 to 0.014.
* **No gain in three cases:**
  * a general VLM (Qwen2-VL-7B: P(yes) AUROC 0.53, gain +3.4);
  * PULSE in its training domain with its training prompt (CODE-15%: −2.4);
  * calibrators transferred between sites without local labels (PTB-XL → CPSC).
* **Domain analysis.** The pre-registered in-domain versus out-of-domain comparison was inconclusive (`outputs/clarity/domain_results.json`).

## Repository layout

```
clarity/        experiment and analysis scripts (the name comes from the original project plan)
scripts/        data preparation, PULSE inference on Apple MPS, scorer generator, release builder
data/splits/    fixed pool/test splits (record ids only)
data/derived/   SPH label set, prompts and question files (SPH is CC0); other derived files are rebuilt by scripts
outputs/        model outputs (probabilities, generated answers without prompt text), results, pre-registration freezes
paper/          figure and table scripts, figures (PDF/SVG/PNG) and tables (CSV/MD)
docs/           REPRODUCE.md (command → output for every number), pre-registrations
environment/    pip freezes and notes on the three Python environments (including MPS numerical caveats)
verify_freeze.py  checks the md5 hashes recorded in the pre-registration freezes
```

**Not included.** Download or regenerate these with the scripts: ECGBench images and questions, the official PULSE code and scorer, PTB-XL metadata, the SPH signals, rendered SPH images, model weights, and hidden-state feature files.

## Pre-registrations

| Experiment | Plan | Freeze record (md5 of code, data, prompts) |
|---|---|---|
| SPH new-site validation (H1–H6) | `docs/prereg_sph.md` | `outputs/clarity/prereg_sph_freeze.json` |
| In-domain vs out-of-domain analysis (P0–P4) | inside the freeze record | `outputs/clarity/preregistration_domain.json` |
| Robustness to image degradation and prompt wording (R1–R4) | `docs/prereg_robustness.md` | `outputs/clarity/prereg_robust_freeze.json` |

The pre-registrations are time-stamped documents in this repository. They were not deposited in an external registry. Run `python verify_freeze.py` to recompute the md5 hashes.

Two files of the domain-analysis freeze were edited after its analysis had run:

* `clarity/label_prob.py` gained new dataset options;
* `scripts/score_ecgbench.py` was regenerated by `scripts/make_scorer.py`.

Re-running the domain analysis with the current files reproduces `domain_results.json` exactly (`outputs/clarity/domain_rerun_check.json`).

## Reproducing

See `docs/REPRODUCE.md` for the full pipeline and `environment/README.md` for the environments. Analyses run on CPU from the shipped outputs. Model inference needs the model weights; we used one Apple-silicon (MPS) machine.

## Validation of the inference pipeline

* **PULSE-7B on MPS** reproduces the official ECGBench scores within 0.1 points.
  * KV-cached probabilities match a full forward pass (max |Δ| 0.0019).
  * Re-running stored outputs in fresh processes gives identical probabilities and generated text (`outputs/clarity/pulse_verify/`).
* **MedGemma** needed eager attention, single-sequence KV-cached inference and a CPU-fp32 canary every 50 images. All outputs of an earlier faulty configuration were discarded and recomputed (`environment/README.md`; manuscript Table S3).

## Limitations

* All images are digital renderings or simulated degradations. There are no real photographs or scans of paper ECGs.
* The CSN and G12EC multi-label versions were built from ECGBench multiple-choice gold options, so their labels may be incomplete.
* The CPSC test split was used during development, so confirmatory evidence comes from SPH.
* There was no prospective or clinical validation.

## Licences and attribution

The code in this repository is released under the MIT License (see `LICENSE`). It covers our code and analysis outputs only. Data and models keep their own terms:

* ECGBench and PULSE-7B;
* PTB-XL (CC BY 4.0, PhysioNet);
* CODE-15% (Zenodo);
* SPH (CC0, figshare);
* CPSC 2018, Chapman–Shaoxing/Ningbo and Georgia (PhysioNet/CinC 2020);
* Qwen2-VL-7B-Instruct;
* MedGemma-4B-ECGInstruct (convaiinnovations).

Please cite PULSE/ECGBench and the dataset publications when using this code.
