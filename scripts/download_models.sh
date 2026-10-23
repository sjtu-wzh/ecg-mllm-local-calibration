#!/bin/bash
# Model weights. PULSE-7B and Qwen2-VL download into the Hugging Face cache on first use; MedGemma-ECGInstruct is fetched
# with curl into ./models because the hub downloader stalled on our machine (clarity/medgemma_run.py prefers this copy).
set -e
cd "$(dirname "$0")/.."
M=convaiinnovations/medgemma-4b-ecginstruct-merged; D=models/medgemma-4b-ecginstruct-merged; mkdir -p $D
for f in $(curl -s https://huggingface.co/api/models/$M | python3 -c "import sys,json;print(' '.join(s['rfilename'] for s in json.load(sys.stdin)['siblings']))"); do
  curl -sL -C - --retry 5 -o "$D/$f" "https://huggingface.co/$M/resolve/main/$f"
done
echo "MedGemma-ECGInstruct -> $D"
