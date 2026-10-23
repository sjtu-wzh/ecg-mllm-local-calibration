| Arm | B0 | Zero-shot P≥0.5 | S1 (own pool) | S1 (clean pool) | P(yes) macro AUROC |
|---|---|---|---|---|---|
| Clean images | 46.9 | 43.8 | 60.1 | — | 0.898 |
| Rotated 4–10° | 43.9 | 40.4 | 57.5 | 56.2 | 0.887 |
| Phone photo of printout | 49.4 | 45.1 | 60.9 | 61.1 | 0.901 |
| Low-quality scan | 48.0 | 45.4 | 60.1 | 60.7 | 0.903 |
| Yes/No question paraphrase QA | (46.9) | 42.8 | 59.3 | 55.5 | 0.893 |
| Yes/No question paraphrase QB | (46.9) | 43.3 | 59.0 | 58.3 | 0.888 |
| Generated-answer prompt paraphrase P1 | 46.5 | — | — | — | — |

Frozen PULSE-7B on SPH; 300-record pool subset and 800-record test subset (data/splits/sph_robust.json). S1 fitted on all 300 pool records of the arm (own pool) or on the 300 clean pool records (clean pool). For QA/QB the generated answer is the clean one (in brackets). Pre-registration: docs/prereg_robustness.md (frozen by md5 before any perturbed or paraphrased output).
