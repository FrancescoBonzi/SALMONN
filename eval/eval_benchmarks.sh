#!/usr/bin/env bash
# Eval finetuned checkpoint on MMAU + MMAR.
# Usage: bash eval/eval_benchmarks.sh
# Override: MODEL_TYPE=bert_conclusion_spare SEED=1 CKPT=path/to/ckpt.pth bash eval/eval_benchmarks.sh

cd "$(dirname "$0")/.."

MODEL_TYPE="${MODEL_TYPE:-salmonn}"
SEED="${SEED:-42}"
NUM_WORKERS="${NUM_WORKERS:-4}"
BATCH_SIZE="${BATCH_SIZE:-4}"
DEVICE="${DEVICE:-cuda:0}"
PROMPT_TYPE="${PROMPT_TYPE:-afthink}"

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

if [[ -n "${CKPT:-}" ]]; then
  BEST_CKPT="${CKPT}"
else
  OUTPUT_DIR=$(ls -dt "outputs/afthink_youtube8m/${MODEL_TYPE}/${SEED}"/* | head -n 1)
  BEST_CKPT="${OUTPUT_DIR}/checkpoint_best.pth"
fi

EVAL_NAME="${MODEL_TYPE}_seed${SEED}.json"
echo "Checkpoint: ${BEST_CKPT}"

echo "Evaluating MMAU..."
python eval/evaluate_mmau.py \
  --cfg-path "recipes/afthink/${MODEL_TYPE}.yaml" \
  --ckpt "${BEST_CKPT}" \
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
  --ckpt "${BEST_CKPT}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --device "${DEVICE}" \
  --prompt-type "${PROMPT_TYPE}" \
  --output-file "outputs/mmar/${EVAL_NAME}" \
  --options \
    datasets.test_ann_path=data/MMAR/annotations/test_mmar.json \
    run.seed="${SEED}"

echo "Done."
