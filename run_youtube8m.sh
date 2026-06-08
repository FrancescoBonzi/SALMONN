#!/usr/bin/env bash
# Train on YouTube8M, then eval on MMAU + MMAR.
# Usage: bash run_youtube8m.sh
# Override: MODEL_TYPE=bert_conclusion_spare SEED=1 NPROC=2 bash run_youtube8m.sh

cd "$(dirname "$0")"

MODEL_TYPE="${MODEL_TYPE:-salmonn}"
SEED="${SEED:-42}"
NPROC="${NPROC:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
BATCH_SIZE="${BATCH_SIZE:-4}"
DEVICE="${DEVICE:-cuda:0}"
CKPT_PATH="${CKPT_PATH:-pretrained/salmonn_v1.pth}"
PROMPT_TYPE="${PROMPT_TYPE:-afthink}"

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

echo "Training ${MODEL_TYPE} (seed=${SEED}, gpus=${NPROC})..."

torchrun --nproc_per_node="${NPROC}" train.py \
  --cfg-path "recipes/afthink/${MODEL_TYPE}.yaml" \
  --options \
    model.ckpt="${CKPT_PATH}" \
    datasets.train_ann_path=data/YouTube8M/annotations/train_youtube8m.json \
    datasets.valid_ann_path=data/YouTube8M/annotations/test_youtube8m.json \
    datasets.test_ann_path=data/YouTube8M/annotations/test_youtube8m.json \
    run.seed="${SEED}" \
    run.output_dir="outputs/afthink_youtube8m/${MODEL_TYPE}/${SEED}" \
    run.num_workers="${NUM_WORKERS}"

OUTPUT_DIR=$(ls -dt "outputs/afthink_youtube8m/${MODEL_TYPE}/${SEED}"/* | head -n 1)
BEST_CKPT="${OUTPUT_DIR}/checkpoint_best.pth"
EVAL_NAME="${MODEL_TYPE}_seed${SEED}.json"

echo "Evaluating MMAU (ckpt=${BEST_CKPT})..."
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
