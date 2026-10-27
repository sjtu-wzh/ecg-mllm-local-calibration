| Model | Check | Sample | Result | Outcome |
|---|---|---|---|---|
| PULSE-7B | Re-run of generated answers (B0), fresh process | 7 sets × 15 records | text identical 105/105 | Pass |
| PULSE-7B | Re-run of P(yes) (label, evidence, MCQ files) | 8 sets, 1350 values | max \|Δ\| = 0.0000 | Pass |
| PULSE-7B | KV-cached image prefix vs full forward pass | 5 images, 25 values | max \|Δ\| = 0.0019 | Pass |
| Qwen2-VL-7B | KV-cached prefix vs full forward pass (fp32 LM head) | 15 values | max \|Δ\| = 0.0082 | Pass |
| MedGemma-4B | KV-cached image prefix vs one question per forward pass (final configuration) | CPSC: 45 values; CSN: 33 values | CPSC max \|Δ\| = 6.7e-06; CSN max \|Δ\| = 9.5e-06 | Pass |
| MedGemma-4B | Final MPS bf16 configuration vs CPU fp32 reference | 39 values (3 repeats) | max \|Δ\| = 0.021; mean 0.0095; repeats identical | Pass (< 0.03) |
| MedGemma-4B | In-run canary: CPSC reference probabilities re-computed at start and every 50 images | all final runs (CPSC, CSN, SPH) | tolerance 0.03 vs CPU fp32; failure discards rows since the last pass | No failure logged |
| MedGemma-4B | Diagnostic on the first (SDPA, padded-batch) runs, before the fix | CPSC: n = 20; CSN: n = 20 | CPSC max \|Δ\| 0.22, B0 identical 18/20; CSN max \|Δ\| 0.93, B0 identical 2/20 | Failed → all MedGemma outputs discarded and recomputed |

Apple-silicon (MPS) inference. PULSE: torch 2.1.2 / transformers 4.37.2; Qwen2-VL: torch 2.5.1 / transformers 4.47.1; MedGemma: torch 2.6 / transformers 4.57.3, eager attention, one sequence per forward pass, deep-copied KV cache, fp32 LM head. Every reported MedGemma number comes from the recomputed runs. An earlier cache check on 6 G12EC values was run before the fix and is not shown; MedGemma was not evaluated on G12EC.
