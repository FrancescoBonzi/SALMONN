#!/usr/bin/env bash
# Eval pretrained SALMONN (no finetuning) on MMAU + MMAR.
# Usage: bash eval/eval_benchmarks_salmonn_.sh
# Override: SEED=1 PROMPT_TYPE=official_reasoning bash eval/eval_benchmarks_salmonn_.sh

cd "$(dirname "$0")/.."

MODEL_TYPE="${MODEL_TYPE:-salmonn}"
SEED="${SEED:-42}"
NUM_WORKERS="${NUM_WORKERS:-4}"
BATCH_SIZE="${BATCH_SIZE:-4}"
DEVICE="${DEVICE:-cuda:0}"
PROMPT_TYPE="${PROMPT_TYPE:-official_reasoning}"
CKPT="${CKPT:-pretrained/salmonn_v1.pth}"

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

EVAL_NAME="${MODEL_TYPE}_pretrained_seed${SEED}.json"
echo "Checkpoint: ${CKPT}"

echo "Evaluating MMAU..."
python eval/evaluate_mmau.py \
  --cfg-path "recipes/afthink/${MODEL_TYPE}.yaml" \
  --ckpt "${CKPT}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --device "${DEVICE}" \
  --prompt-type "${PROMPT_TYPE}" \
  --output-file "outputs/mmau/${EVAL_NAME}" \
  --options \
    datasets.test_ann_path=data/MMAU/annotations/test_mmau.json \
    run.seed="${SEED}"

echo "Evaluating MMAR..."
python eval/evaluate_mmar.py \
  --cfg-path "recipes/afthink/${MODEL_TYPE}.yaml" \
  --ckpt "${CKPT}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --device "${DEVICE}" \
  --prompt-type "${PROMPT_TYPE}" \
  --output-file "outputs/mmar/${EVAL_NAME}" \
  --options \
    datasets.test_ann_path=data/MMAR/annotations/test_mmar.json \
    run.seed="${SEED}"

echo "Done."
