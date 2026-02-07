#!/bin/bash
#SBATCH --time=12:0:0
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
cp -r "data/Clotho-AQA" "$SLURM_TMPDIR/data/" &
COPY_DATA_PID=$!

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
wait $COPY_DATA_PID $COPY_WHISPER_PID $COPY_VICUNA_PID $COPY_BEATS_PID $COPY_SALMONN_PID
echo "Data and pretrained models copy complete!"

# Update annotation paths to point to SLURM_TMPDIR
# Replace any absolute path ending with /data/Clotho-AQA
echo "Updating annotation paths..."
sed -i "s|[^\"]*data/Clotho-AQA|$SLURM_TMPDIR/data/Clotho-AQA|g" "$SLURM_TMPDIR/data/Clotho-AQA/annotations/"*.json

# Activate environment
module load StdEnv/2023 cuda/12.2
module load httpproxy
source .venv/bin/activate

# Run training
echo "Starting training..."

torchrun --nproc_per_node=4 train.py --cfg-path recipes/afthink/$model_type.yaml \
    --options \
    model.llama_path="$SLURM_TMPDIR/pretrained/vicuna-13b-v1.1" \
    model.whisper_path="$SLURM_TMPDIR/pretrained/whisper-large-v2" \
    model.beats_path="$SLURM_TMPDIR/pretrained/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt" \
    model.ckpt="$SLURM_TMPDIR/pretrained/salmonn_v1.pth" \
    datasets.train_ann_path="$SLURM_TMPDIR/data/Clotho-AQA/annotations/train_clotho_aqa.json" \
    datasets.valid_ann_path="$SLURM_TMPDIR/data/Clotho-AQA/annotations/train_clotho_aqa.json" \
    datasets.test_ann_path="$SLURM_TMPDIR/data/Clotho-AQA/annotations/train_clotho_aqa.json" \
    datasets.whisper_path="$SLURM_TMPDIR/pretrained/whisper-large-v2" \
    run.seed="$seed" \
    run.output_dir="outputs/afthink_clotho_aqa/$model_type/$seed" \
    run.num_workers="$SLURM_CPUS_PER_TASK"

echo "Training finished at $(date)"

# Run evaluation on the best checkpoint
echo "Starting evaluation..."

# Find the output directory
OUTPUT_DIR=$(ls -dt outputs/afthink_clotho_aqa/$model_type/$seed/* | head -n 1)
BEST_CKPT="${OUTPUT_DIR}/checkpoint_best.pth"
EVAL_OUTPUT="${OUTPUT_DIR}/eval_results"

echo "Using checkpoint: $BEST_CKPT"
echo "Saving evaluation results to: $EVAL_OUTPUT"

mkdir -p "$EVAL_OUTPUT"

echo "Evaluating MMAU at $(date)"
python evaluate_mmau.py \
    --cfg-path recipes/afthink/$model_type.yaml \
    --ckpt "$BEST_CKPT" \
    --batch-size 4 \
    --num-workers "$SLURM_CPUS_PER_TASK" \
    --device cuda:0 \
    --output-dir "$EVAL_OUTPUT" \
    --options \
    model.llama_path="$SLURM_TMPDIR/pretrained/vicuna-13b-v1.1" \
    model.whisper_path="$SLURM_TMPDIR/pretrained/whisper-large-v2" \
    model.beats_path="$SLURM_TMPDIR/pretrained/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt" \
    datasets.test_ann_path="$SLURM_TMPDIR/data/MMAU/annotations/test_mmau.json" \
    datasets.whisper_path="$SLURM_TMPDIR/pretrained/whisper-large-v2"

echo "Evaluating MMAR at $(date)"
python evaluate_mmar.py \
    --cfg-path recipes/afthink/$model_type.yaml \
    --ckpt "$BEST_CKPT" \
    --batch-size 4 \
    --num-workers "$SLURM_CPUS_PER_TASK" \
    --device cuda:0 \
    --output-dir "$EVAL_OUTPUT" \
    --options \
    model.llama_path="$SLURM_TMPDIR/pretrained/vicuna-13b-v1.1" \
    model.whisper_path="$SLURM_TMPDIR/pretrained/whisper-large-v2" \
    model.beats_path="$SLURM_TMPDIR/pretrained/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt" \
    datasets.test_ann_path="$SLURM_TMPDIR/data/MMAR/annotations/test_mmar.json" \
    datasets.whisper_path="$SLURM_TMPDIR/pretrained/whisper-large-v2"

echo "Evaluation finished at $(date)"