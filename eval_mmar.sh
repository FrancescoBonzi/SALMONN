#!/bin/bash
#SBATCH --time=1:0:0
#SBATCH --account=def-ravanelm
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

# Copy MMAR audio data
mkdir -p "$SLURM_TMPDIR/data"
cp -r "data/MMAR" "$SLURM_TMPDIR/data/" &
COPY_AUDIO_PID=$!

# Copy annotation files
cp -r "data/MMAR/annotations" "$SLURM_TMPDIR/data/" &
COPY_ANN_PID=$!

# Copy pretrained models
mkdir -p "$SLURM_TMPDIR/pretrained"
cp -r "pretrained/whisper-large-v2" "$SLURM_TMPDIR/pretrained/" &
COPY_WHISPER_PID=$!

cp -r "pretrained/vicuna-13b-v1.1" "$SLURM_TMPDIR/pretrained/" &
COPY_VICUNA_PID=$!

cp "pretrained/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt" "$SLURM_TMPDIR/pretrained/" &
COPY_BEATS_PID=$!

cp "pretrained/salmonn_v1.pth" "$SLURM_TMPDIR/pretrained/" &
COPY_SALMONN_PID=$!

# Wait for all copies to finish
wait $COPY_AUDIO_PID $COPY_ANN_PID $COPY_WHISPER_PID $COPY_VICUNA_PID $COPY_BEATS_PID $COPY_SALMONN_PID
echo "Data copy complete!"

# Update annotation paths to point to SLURM_TMPDIR
echo "Updating annotation paths..."
sed -i "s|[^\"]*data/MMAR|$SLURM_TMPDIR/data/MMAR|g" "$SLURM_TMPDIR/data/MMAR/annotations/"*.json

# Activate environment
module load StdEnv/2023 cuda/12.2
module load httpproxy
source .venv/bin/activate

# Run evaluation on the best checkpoint
echo "Starting evaluation..."

EVAL_DIR="outputs/mmar/salmonn/$seed/eval_results"
mkdir -p "$EVAL_DIR"

python evaluate_mmar.py \
    --cfg-path recipes/mmar/salmonn.yaml \
    --ckpt "$SLURM_TMPDIR/pretrained/salmonn_v1.pth" \
    --batch-size 4 \
    --num-workers "$SLURM_CPUS_PER_TASK" \
    --device cuda:0 \
    --output-dir "$EVAL_DIR" \
    --use-cot \
    --options \
    model.llama_path="$SLURM_TMPDIR/pretrained/vicuna-13b-v1.1" \
    model.whisper_path="$SLURM_TMPDIR/pretrained/whisper-large-v2" \
    model.beats_path="$SLURM_TMPDIR/pretrained/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt" \
    datasets.test_ann_path="$SLURM_TMPDIR/data/MMAR/annotations/test_mmar.json" \
    datasets.whisper_path="$SLURM_TMPDIR/pretrained/whisper-large-v2"

echo "MMAR evaluation finished at $(date)"
echo "Job finished at $(date)"
