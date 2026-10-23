#!/bin/bash
# Experiment 3 queue (docs/prereg_robustness.md). One GPU process at a time; resumable; each step retried until complete.
cd "$(dirname "$0")/../.."
caffeinate -i -s -w $$ &
export PYTORCH_ENABLE_MPS_FALLBACK=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.4
P=$HOME/miniforge3/envs/pulse-llava/bin/python; L=outputs/robust/logs; mkdir -p $L
n() { [ -f "$1" ] && grep -c . "$1" || echo 0; }
step() {  # name, expected rows, output, command...
  local name=$1 want=$2 out=$3; shift 3; local k=0
  while [ "$(n $out)" -lt "$want" ]; do
    k=$((k+1)); [ $k -gt 20 ] && { echo "[$(date +%m-%d\ %H:%M)] $name GAVE UP at $(n $out)"; return; }
    "$@" >> $L/$name.log 2>&1; echo "[$(date +%m-%d\ %H:%M)] $name try $k rc=$? rows=$(n $out)"
  done
}
for C in ROT PHOTO SCAN; do
  step b0_$C 800 outputs/robust/b0_$C.jsonl $P -I scripts/infer_mps.py --question-file data/derived/sph_robust_test_questions.json \
       --image-folder data/sph_images_robust/$C --answers-file outputs/robust/b0_$C.jsonl
  step b1_$C 1100 outputs/robust/b1_$C.jsonl $P -I clarity/robust_prob.py --image-root data/sph_images_robust/$C --out outputs/robust/b1_$C.jsonl
done
for Q in QA QB; do
  step b1_$Q 1100 outputs/robust/b1_$Q.jsonl $P -I clarity/robust_prob.py --image-root data/sph_images --template $Q --out outputs/robust/b1_$Q.jsonl
done
step b0_P1 800 outputs/robust/b0_P1.jsonl $P -I scripts/infer_mps.py --question-file data/derived/sph_robust_P1_questions.json \
     --image-folder data/sph_images --answers-file outputs/robust/b0_P1.jsonl
$P -I clarity/robust_eval.py > $L/robust_eval.log 2>&1; echo "[$(date +%m-%d\ %H:%M)] robust_eval rc=$?"
echo ROBUST_QUEUE_DONE
