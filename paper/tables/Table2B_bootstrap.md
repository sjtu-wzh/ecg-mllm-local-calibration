| Model | Dataset | Bootstrap mean Δ macro-F1 (S1 − B0) | 95% percentile interval |
|---|---|---|---|
| PULSE-7B | CPSC | +10.7 | [8.2, 13.3] |
| PULSE-7B | CSN | +20.5 | [17.5, 23.5] |
| PULSE-7B | G12EC | +17.2 | [15.4, 19.1] |
| PULSE-7B | SPH (pre-registered) | +13.5 | [11.2, 15.8] |
| MedGemma-4B-ECGInstruct | CPSC | +22.3 | [20.1, 24.5] |
| MedGemma-4B-ECGInstruct | CSN | +27.7 | [25.2, 30.2] |
| MedGemma-4B-ECGInstruct | SPH (pre-registered) | +20.7 | [18.5, 22.9] |
| Qwen2-VL-7B-Instruct | CPSC | +3.4 | [2.2, 4.7] |
| PULSE-7B (in-domain) | CODE-15%^a^ | −2.4 | [−4.4, −0.4] |

S1 was fitted once on all 600 development records (thresholds from out-of-fold predictions) and compared with B0 on the same test records by paired bootstrap over test records (B = 2000). The point estimate is the mean of the bootstrap differences, as implemented in the pre-registered analysis; the interval is the 2.5th–97.5th percentile. These values are not differences between entries in Table 2 (A), which are means of repeated fits. ^a^ As in Table 2 (A).
