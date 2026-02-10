#!/bin/bash
#SBATCH --time=3:0:0
#SBATCH --account=aip-csubakan
#SBATCH --cpus-per-task=48
#SBATCH --mem=488G
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --nodes=1
#SBATCH --array=0-0

# Define seeds array
seeds=(42)

# Get the seed for this job array index
seed=${seeds[$SLURM_ARRAY_TASK_ID]}

model_type="salmonn"

# Copy data to SLURM_TMPDIR for fast I/O
echo "Copying data to SLURM_TMPDIR..."

# Copy LibriSpeech audio data
mkdir -p "$SLURM_TMPDIR/data"
cp -r "data/LibriSpeech" "$SLURM_TMPDIR/data/" &
COPY_AUDIO_PID=$!

# Copy annotation files
cp -r "data/LibriSpeech/annotations" "$SLURM_TMPDIR/data/" &
COPY_ANN_PID=$!

# Copy pretrained models
mkdir -p "$SLURM_TMPDIR/pretrained"
cp -r "pretrained/whisper-large-v2" "$SLURM_TMPDIR/pretrained/" &
COPY_WHISPER_PID=$!

cp -r "pretrained/vicuna-13b-v1.1" "$SLURM_TMPDIR/pretrained/" &
COPY_VICUNA_PID=$!

cp "pretrained/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt" "$SLURM_TMPDIR/pretrained/" &
COPY_BEATS_PID=$!

# Wait for all copies to finish
wait $COPY_AUDIO_PID $COPY_ANN_PID $COPY_WHISPER_PID $COPY_VICUNA_PID $COPY_BEATS_PID
echo "Data copy complete!"

# Update annotation paths to point to SLURM_TMPDIR
# Replace any absolute path ending with /data/LibriSpeech
echo "Updating annotation paths..."
sed -i "s|[^\"]*data/LibriSpeech|$SLURM_TMPDIR/data/LibriSpeech|g" "$SLURM_TMPDIR/data/librispeech/"*.json

# Activate environment
module load StdEnv/2023 cuda/12.2
module load httpproxy
source .venv/bin/activate

# Run evaluation on the best checkpoint
echo "Starting evaluation..."

# Find the output directory
OUTPUT_DIR=$(ls -dt outputs/librispeech_asr/$model_type/$seed/* | head -n 1)
BEST_CKPT="${OUTPUT_DIR}/checkpoint_best.pth"
EVAL_OUTPUT="${OUTPUT_DIR}/eval_results"

echo "Using checkpoint: $BEST_CKPT"
echo "Saving evaluation results to: $EVAL_OUTPUT"

mkdir -p "$EVAL_OUTPUT"

python evaluate.py \
    --cfg-path recipes/librispeech/salmonn.yaml \
    --ckpt "$BEST_CKPT" \
    --split test \
    --batch-size 8 \
    --num-workers "$SLURM_CPUS_PER_TASK" \
    --device cuda:0 \
    --output-dir "$EVAL_OUTPUT" \
    --options \
    model.llama_path="$SLURM_TMPDIR/pretrained/vicuna-13b-v1.1" \
    model.whisper_path="$SLURM_TMPDIR/pretrained/whisper-large-v2" \
    model.beats_path="$SLURM_TMPDIR/pretrained/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt" \
    datasets.test_ann_path="$SLURM_TMPDIR/data/librispeech/test_librispeech.json" \
    datasets.whisper_path="$SLURM_TMPDIR/pretrained/whisper-large-v2"

echo "Evaluation finished at $(date)"
echo "Job finished at $(date)"