| Variant | CPSC test macro-F1 | Comparison [95% CI] | Interpretation | Status |
|---|---|---|---|---|
| H: per-label P(yes) + generated answer (no cross-label terms) | 61.3 | vs S1 68.3 | Cross-label terms, not the generated answer, carry the gain | Not adopted |
| E1: label + clinically linked evidence | 66.6 | vs S1 68.3 | Clinically linked evidence alone is weaker than all label logits | Not adopted |
| E3: S1 + clinically linked evidence | 69.6 | E3 − S1 +1.3 [-0.3, 3.0], p = 0.112 | Not significant | Not adopted |
| E2: all labels + all 11 evidence answers | 70.2 | E2 − S1 +1.9 [0.1, 3.5], p = 0.032 | Small gain for 2.2× questions per image | Reported, not main method |
| E2 + uncertainty gating (M4) | 70.2 | dev-selected δ routes 100% of records | Gate degenerates to E2 | Not adopted |
| Control: S1 protocol on B0 0/1 indicators | 56.9 | B0 57.6 | Gain needs the probabilities, not re-weighting of answers | Control |
| M1 lead localisation by OCR (pilot) | — | cpsc 0.6/12; mmmu 1.4/12; ptb 0.6/12; g12 0.5/12; csn 0.4/12 | Lead labels not found reliably on ECGBench images | Abandoned |
| Transfer PTB-XL → CPSC: B1@src − B0 | — | +1.0 [-1.5, 3.4], p = 0.374 | No transferred variant beats B0 on the new site | Negative (Fig. S4b) |
| Transfer PTB-XL → CPSC: B1@0.5 − B0 | — | 0.0 [-2.7, 2.6], p = 0.984 | No transferred variant beats B0 on the new site | Negative (Fig. S4b) |
| Transfer PTB-XL → CPSC: S1 − B0 | — | -2.1 [-4.9, 0.5], p = 0.114 | No transferred variant beats B0 on the new site | Negative (Fig. S4b) |
| Transfer PTB-XL → CPSC: E2 − B0 | — | -4.8 [-7.5, -2.3], p < 0.001 | No transferred variant beats B0 on the new site | Negative (Fig. S4b) |

All variants fitted on the 600-record CPSC dev split and evaluated once on the 1461-record test split (clarity/verify_stack.py, clarity/stats_cpsc.py). Paired bootstrap over test records (B = 2000). These analyses were exploratory and are not used for any main claim. Variant codes E1–E3 replace the internal codes S2/S3/S4 (Table S6).
