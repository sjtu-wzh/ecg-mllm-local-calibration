# Pre-registration — Experiment 3: robustness to image degradation and prompt wording

Written before any model output on perturbed images or paraphrased prompts exists. The code, the subsets, the perturbed
images and the prompts are frozen by md5 in `outputs/clarity/prereg_robust_freeze.json` before inference starts.
Each arm is evaluated once. Any later analysis is labelled exploratory.

## Question

Does the gain of local calibration over the generated answer persist in two situations?

* ECG images are degraded in ways that are common in practice (rotation, phone photographs of printouts, low-quality scans).
* The question wording changes.

## Data and model

* **Model:** frozen PULSE-7B, the same weights and inference code as in the main analysis.
* **Data:** SPH, the pre-registered new site.
  * Fixed subsets were drawn from the SPH pool and test splits (`clarity/robust_setup.py`; `data/splits/sph_robust.json`):
    300 pool records (n is at most 300 here, instead of 600) and 800 test records.
  * Subset sizes were set by the compute budget (about 10 GPU-hours).
* **Image conditions** (`clarity/robust_setup.py`) are applied to both the pool and the test images of an arm:
  * **ROT:** rotation of 4–10°.
  * **PHOTO:** phone photo of a printout: perspective warp up to 6%, uneven lighting, colour cast, blur, JPEG quality 70.
  * **SCAN:** low-quality scan or forwarded image: greyscale, 40% downsampling, noise, JPEG quality 25.

  The parameters were fixed after looking at a single preview record, before any model output existed.
* **Prompt variants** (clean images):
  * **QA and QB:** two paraphrases of the Yes/No label question.
  * **P1:** a paraphrase of the generated-answer prompt, with the same label list.
* **Not included:**
  * the hidden-state probe, because of compute;
  * MedGemma;
  * combinations of conditions;
  * other severities.

## Outputs

| Arm | Generated answer (B0), 800 test records | P(yes), 1100 records |
|---|---|---|
| ROT, PHOTO, SCAN | `scripts/infer_mps.py` with the original prompt | `clarity/robust_prob.py` with the original question |
| QA, QB | — | `clarity/robust_prob.py --template QA/QB` |
| P1 | `scripts/infer_mps.py` with the P1 prompt | — |

The CLEAN arm reuses the existing SPH outputs, restricted to the same subsets.

## Read-outs and analysis (`clarity/robust_eval.py`)

The read-out rules are those of the main analysis:

* S1 is one logistic regression per label on all 12 label logits, with C = 1.
* Decision thresholds come from 5-fold out-of-fold predictions.
* Every record receives at least one label.
* The generated answer (B0) is parsed by exact code tokens.
* The metric is the official macro-F1.

Each arm uses two versions of S1:

* **S1_a:** fitted on the arm's own 300 pool records (local calibration on images like the deployment images).
* **S1_clean→a:** fitted on the clean pool and applied unchanged to arm a.

The paired bootstrap runs over the 800 test records (B = 2000).

| ID | Contrast | Supported if |
|---|---|---|
| R1 (per image condition) | S1_c − B0_c | 95% CI lower bound > 0 **and** point estimate ≥ 3 |
| R2 (per image condition) | S1_clean→c − B0_c | CI lower bound > 0 |
| R3 (per Yes/No paraphrase) | S1_q − B0_CLEAN | CI lower bound > 0 **and** point estimate ≥ 3 |
| R4 | S1_CLEAN − B0_P1 | CI lower bound > 0 **and** point estimate ≥ 3 |

The following are reported without decision rules:

* B0_c − B0_CLEAN and S1_c − S1_CLEAN (degradation of each read-out);
* S1_clean→a − S1_a (cost of not recalibrating);
* B0_P1 − B0_CLEAN;
* macro AUROC of P(yes) in each arm;
* learning curves for THR and S1, with n ∈ {25, 50, 100, 200, 300} and 20 draws per arm.

We apply no correction for multiplicity. The 9 contrasts are reported together, and each one is interpreted against its own rule.

## Deviations

Crashes, restarts and any numerical-validation failure are logged in the freeze file. Failed rows are recomputed, never excluded.
