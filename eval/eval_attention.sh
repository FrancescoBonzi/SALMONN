#!/usr/bin/env bash
# Attention visualization on YouTube8M test set.
# Usage: bash eval/eval_attention.sh
# Override: MODEL_TYPE=bert_conclusion_spare SEED=1 CKPT=path/to/ckpt.pth bash eval/eval_attention.sh

cd "$(dirname "$0")/.."

MODEL_TYPE="${MODEL_TYPE:-salmonn}"
SEED="${SEED:-42}"
NUM_WORKERS="${NUM_WORKERS:-4}"

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

if [[ -n "${CKPT:-}" ]]; then
  BEST_CKPT="${CKPT}"
  CKPT_DIR="${CKPT_DIR:-$(dirname "${CKPT}")}"
else
  CKPT_DIR=$(ls -dt "outputs/afthink_youtube8m/${MODEL_TYPE}/${SEED}"/* | head -n 1)
  BEST_CKPT="${CKPT_DIR}/checkpoint_best.pth"
fi

echo "Checkpoint: ${BEST_CKPT}"

python eval/visualize_attention.py \
  --cfg-path "recipes/afthink/${MODEL_TYPE}.yaml" \
  --options \
    model.ckpt="${BEST_CKPT}" \
    datasets.valid_ann_path=data/YouTube8M/annotations/test_youtube8m.json \
    run.output_dir="${CKPT_DIR}" \
    run.seed="${SEED}" \
    run.num_workers="${NUM_WORKERS}"

echo "Done. Metrics saved to ${CKPT_DIR}/attention_metrics.json"
