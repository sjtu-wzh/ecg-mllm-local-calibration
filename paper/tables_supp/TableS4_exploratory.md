| Variant | CPSC test macro-F1 | Comparison [95% CI] | Interpretation | Status |
|---|---|---|---|---|
| H: per-label P(yes) + generated answer (no cross-label terms) | 61.3 | vs S1 68.3 | Cross-label terms, not the generated answer, carry the gain | Not adopted |
| E1: label + clinically linked evidence | 66.6 | vs S1 68.3 | Clinically linked evidence alone is weaker than all label logits | Not adopted |
| E3: S1 + clinically linked evidence | 69.6 | E3 − S1 +1.3 [−0.3, 3.0], p = 0.112 | CI includes 0 | Not adopted |
| E2: all labels + all 11 evidence answers | 70.2 | E2 − S1 +1.9 [0.1, 3.5], p = 0.032 | Small gain for 2.2× questions per image | Reported, not main method |
| E2 + uncertainty gating (M4) | 70.2 | development-selected δ routes 100% of records | Gate degenerates to E2 | Not adopted |
| Control: S1 protocol on B0 0/1 indicators | 56.9 | B0 57.6 | Gain needs the probabilities, not re-weighting of answers | Control |
| Lead localisation by OCR (pilot) | — | lead names found per image (of 12): CPSC 0.6, MMMU-ECG 1.4, PTB-XL 0.6, G12EC 0.5, CSN 0.4 | Lead labels not found reliably on ECGBench images | Abandoned |
| Transfer PTB-XL → CPSC: ZS, P(yes) ≥ 0.5 (no fitting; reference) | 57.6 | ZS − B0 0.0 [−2.7, 2.6], p = 0.984 | Uses no labels; equals B0 on average | Reference (Fig. S4b) |
| Transfer PTB-XL → CPSC: THR with PTB-XL thresholds | 58.6 | THR − B0 +1.0 [−1.5, 3.4], p = 0.374 | CI includes 0 | Negative (Fig. S4b) |
| Transfer PTB-XL → CPSC: S1 fitted on PTB-XL | 55.5 | S1 − B0 −2.1 [−4.9, 0.5], p = 0.114 | CI includes 0 | Negative (Fig. S4b) |
| Transfer PTB-XL → CPSC: S1 fitted on PTB-XL + EM prior shift | 51.7 | S1 + EM − B0 −5.9 (observed; no bootstrap) | EM prior correction lowers macro-F1 | Negative (Fig. S4b) |
| Transfer PTB-XL → CPSC: E3 fitted on PTB-XL | 54.0 | E3 − B0 −3.6 (observed; no bootstrap) | Below B0 | Negative (Fig. S4b) |
| Transfer PTB-XL → CPSC: E3 fitted on PTB-XL + EM prior shift | 53.4 | E3 + EM − B0 −4.2 (observed; no bootstrap) | EM prior correction lowers macro-F1 | Negative (Fig. S4b) |
| Transfer PTB-XL → CPSC: E2 fitted on PTB-XL | 52.8 | E2 − B0 −4.8 [−7.5, −2.3], p < 0.001 | Below B0 | Negative (Fig. S4b) |
| Transfer PTB-XL → CPSC: E2 fitted on PTB-XL + EM prior shift | 49.6 | E2 + EM − B0 −8.0 (observed; no bootstrap) | EM prior correction lowers macro-F1 | Negative (Fig. S4b) |

Rows above the transfer block: fitted on the 600-record CPSC development pool and evaluated once on the 1461-record test split (clarity/verify_stack.py, clarity/stats_cpsc.py). Transfer rows: fitted on the 1457 PTB-XL fit records (main text, Section 2.8) and applied unchanged to the CPSC test split, without CPSC labels; EM re-estimates the label priors on the unlabelled CPSC test images (clarity/transfer.py). All transfer variants were specified in advance and all are shown. B0 = 57.6. Paired bootstrap over test records (B = 1000, clarity/stats_cpsc.py; point estimate = mean of the bootstrap differences) was computed for the contrasts shown with a CI; the other transfer rows give the observed difference only. These analyses were exploratory and are not used for any main claim. Variant codes E1–E3 replace the internal codes S2/S3/S4, and ZS/THR replace B1@0.5/B1@src (Table S6). OCR pilot: MMMU-ECG is an ECGBench image set used only in this pilot.
