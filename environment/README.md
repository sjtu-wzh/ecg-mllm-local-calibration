# Environments

All runs used Apple silicon (M5 Max, 128 GB, macOS 27) with the PyTorch **MPS** backend. Three separate conda
environments (Python 3.10) are needed because the three base models require incompatible `transformers` versions.

| Environment | Used for | Key pins |
|---|---|---|
| `pulse-llava` | PULSE-7B inference, every analysis script, figures | torch 2.1.2, transformers 4.37.2 (pinned by PULSE/LLaVA) |
| `qwen-vl` | Qwen2-VL-7B-Instruct inference (`clarity/qwen_cpsc.py`) | torch 2.5.1, transformers 4.47.1 |
| `medgemma` | MedGemma-4B-ECGInstruct inference (`clarity/medgemma_run.py`, `clarity/mg_reference.py`) | torch 2.6.0, transformers 4.57.3 |

`*.txt` in this folder are `pip freeze` outputs. scipy 1.15.2 and scikit-learn 1.7.2 were installed from
**conda-forge** in all three environments (the pip wheel of scipy failed to load on macOS 27), so they are not listed.

```bash
conda create -n pulse-llava python=3.10 -y
conda install -n pulse-llava -c conda-forge scipy=1.15.2 scikit-learn=1.7.2 -y
conda run -n pulse-llava pip install -r environment/pulse-llava.txt
git clone https://github.com/AIMedLab/PULSE.git            # official code; LLaVA/ is Apache-2.0
conda run -n pulse-llava pip install --no-deps -e PULSE/LLaVA
# same pattern for qwen-vl and medgemma with their requirement files
```

PULSE/LLaVA is installed with `--no-deps`; deepspeed, bitsandbytes, flash-attn and gradio are not needed for inference.
`pyobjc-framework-Vision` (in `pulse-llava`) is only used by the OCR pilot `clarity/lead_locate.py`.

## MPS numerics (read before re-running on a Mac)

* PULSE-7B (torch 2.1.2, fp16): reproduces the official ECGBench scores to within 0.1 on four subsets; the KV-cached
  Yes/No path matches a full forward to 0.0019 (`outputs/clarity/pulse_cache_check.json`); random re-computation of
  stored rows is summarised in `outputs/clarity/pulse_verify/summary.json`.
* Qwen2-VL (bf16): bf16 logits quantise P(yes) coarsely; the LM head is computed in fp32. The process leaks MPS memory
  across differently sized images and is restarted automatically (outputs are resumable).
* MedGemma-ECGInstruct (bf16, torch 2.6, macOS 27): batched SDPA attention with padding produced NaN, garbled text and
  **silently wrong probabilities** for some batch shapes. The validated configuration is eager attention, one sequence at
  a time, a KV-cached image prefix, and a canary check against a CPU float32 reference every 50 images
  (`clarity/mg_reference.py`, `outputs/clarity/mg_reference/`).
