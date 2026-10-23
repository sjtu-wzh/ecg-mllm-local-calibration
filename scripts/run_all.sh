#!/bin/bash
# Official PULSE-7B generation (B0) on the ECGBench subsets used here, on Apple-silicon MPS. Resumable.
# Usage: PY=/path/to/pulse-llava/bin/python bash scripts/run_all.sh
set -e
cd "$(dirname "$0")/.."
PY=${PY:-python}
export PYTORCH_ENABLE_MPS_FALLBACK=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.4
for s in mmmu-ecg cpsc-test code15-test g12-test-no-cot; do
  "$PY" -I scripts/infer_mps.py --question-file data/ECGBench/$s.json --answers-file outputs/pulse7b/$s/answers.jsonl
done
