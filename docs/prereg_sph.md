# Pre-registration — Experiment 2: SPH as a new, untouched out-of-domain site

Written before any model output on SPH exists. Code and data are frozen by md5 in `outputs/clarity/prereg_sph_freeze.json`
before inference starts. The SPH test split is evaluated **once**; any later analysis is labelled exploratory.

## Data
* SPH (Shandong Provincial Hospital; Liu et al., *Sci Data* 2022; CC0). Not a PULSE / ECGInstruct source.
* `clarity/build_sph.py`: one record per patient, random 3000 records (seed 42), base AHA codes with ≥ 50 positives
  → 12 labels, 2912 records; split 600 pool / 2312 test (`data/splits/sph_ml.json`).
* `clarity/render_sph.py`: ECGBench-style images (10 s, 25 mm/s, 10 mm/mV; 3×4+II / 6×2 / 12×1; mild acquisition
  variation on 50 %).

## Models and outputs
* PULSE-7B (primary) and MedGemma-4B-ECGInstruct (replication).
* B0: generation for the CPSC-style prompt with the 12-label list (`scripts/infer_mps.py`, `clarity/medgemma_run.py`),
  parsed by exact code tokens. B1: one Yes/No question per label (`clarity/label_prob.py --data sphml`).
* Features for Experiment-1 baselines: `clarity/extract_features.py --data sphml`.

## Hypotheses and decision rules (macro-F1, official metric, SPH test split)
| ID | Hypothesis | Analysis | Supported if |
|---|---|---|---|
| H1 (primary) | Local calibration improves over the generated answer | `ml_curve.py`: S1@600 − B0, paired bootstrap (B = 2000) | 95 % CI lower bound > 0 **and** point estimate ≥ 3 |
| H2 | Reading probabilities already helps without labels | ZS(P ≥ 0.5) − B0 | CI lower bound > 0 |
| H3 | Calibration adds beyond the probability read-out | S1@600 − ZS | CI lower bound > 0 |
| H4 | Label-budget rule (found post hoc on CSN/G12EC): threshold-only is preferable at small n, S1 at large n | mean over 20 draws: THR − S1 at n ∈ {25, 50, 100}; S1 − THR at n ∈ {300, 400, 600} | THR ≥ S1 at ≥ 2 of the 3 small n **and** S1 > THR at ≥ 2 of the 3 large n |
| H5 | S1 is non-inferior to a linear probe on PULSE's internal features at small label budgets | `baselines.py --data sph`: paired S1 − PROBE-LLM per draw at n ∈ {25, 50, 100, 200} | 2.5th percentile of the paired difference > −3 at each n (non-inferiority margin 3 F1) |
| H6 | H1 replicates on a second ECG-trained model | MedGemma: S1@600 − B0 | as H1 |

PROBE-LLM uses the feature chosen in the pool-only stage 1 of `baselines.py`. All thresholds and model choices are
fitted on pool records only. No other variants will be reported as confirmatory.

## Deviations
Any deviation (crashes, corrupted rows, re-runs) is logged in the freeze file with the reason; numerical-validation
failures lead to re-computation, never to excluding records.
