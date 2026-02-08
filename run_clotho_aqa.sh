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

# Define config variables
model_type="salmonn"
ckpt_path="pretrained/salmonn_v1.pth"
ckpt_type="pretrained"
prompt_type="official"
eval_filename="${model_type}_13B_${ckpt_type}_${prompt_type}prompt_seed${seed}.json"

# Copy data to SLURM_TMPDIR for fast I/O
echo "Copying data to SLURM_TMPDIR..."

# Copy Clotho-AQA dataset
mkdir -p "$SLURM_TMPDIR/data/Clotho-AQA/audio_files"
cp -r "data/Clotho-AQA/audio_files" "$SLURM_TMPDIR/data/Clotho-AQA/audio_files/" &
COPY_CLOTH_AUDIO_DATA_PID=$!

mkdir -p "$SLURM_TMPDIR/data/Clotho-AQA/annotations"
cp -r "data/Clotho-AQA/annotations" "$SLURM_TMPDIR/data/Clotho-AQA/annotations/" &
COPY_CLOTH_ANN_DATA_PID=$!

# Copy MMAR and MMAU benchmarks
mkdir -p "$SLURM_TMPDIR/data/MMAR/audio_files"
cp -r "data/MMAR/audio_files" "$SLURM_TMPDIR/data/MMAR/audio_files/" &
COPY_MMAR_AUDIO_DATA_PID=$!

mkdir -p "$SLURM_TMPDIR/data/MMAR/annotations"
cp -r "data/MMAR/annotations" "$SLURM_TMPDIR/data/MMAR/annotations/" &
COPY_MMAR_ANN_DATA_PID=$!

mkdir -p "$SLURM_TMPDIR/data/MMAU/audio_files"
cp -r "data/MMAU/audio_files" "$SLURM_TMPDIR/data/MMAU/audio_files/" &
COPY_MMAU_AUDIO_DATA_PID=$!

mkdir -p "$SLURM_TMPDIR/data/MMAU/annotations"
cp -r "data/MMAU/annotations" "$SLURM_TMPDIR/data/MMAU/annotations/" &
COPY_MMAU_ANN_DATA_PID=$!

# Wait for all copies to finish
wait $COPY_CLOTH_AUDIO_DATA_PID $COPY_CLOTH_ANN_DATA_PID $COPY_MMAR_AUDIO_DATA_PID $COPY_MMAR_ANN_DATA_PID $COPY_MMAU_AUDIO_DATA_PID $COPY_MMAU_ANN_DATA_PID
echo "Data copy complete!"

# Copy pretrained models
mkdir -p "$SLURM_TMPDIR/pretrained"
cp -r "pretrained/whisper-large-v2" "$SLURM_TMPDIR/pretrained/" &
COPY_WHISPER_PID=$!

cp -r "pretrained/vicuna-13b-v1.1" "$SLURM_TMPDIR/pretrained/" &
COPY_VICUNA_PID=$!

cp "pretrained/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt" "$SLURM_TMPDIR/pretrained/" &
COPY_BEATS_PID=$!

cp "$ckpt_path" "$SLURM_TMPDIR/pretrained/ckpt.pth" &
COPY_SALMONN_PID=$!

# Wait for all copies to finish
wait $COPY_WHISPER_PID $COPY_VICUNA_PID $COPY_BEATS_PID $COPY_SALMONN_PID
echo "Pretrained models copy complete!"

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
    model.ckpt="$SLURM_TMPDIR/pretrained/ckpt.pth" \
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

echo "Using checkpoint: $BEST_CKPT"

# Run evaluation on the best checkpoint
echo "Evaluating MMAU (cuda:0) and MMAR (cuda:1) in parallel..."

python evaluate_mmau.py \
    --cfg-path recipes/afthink/$model_type.yaml \
    --ckpt "$BEST_CKPT" \
    --batch-size 4 \
    --num-workers "$SLURM_CPUS_PER_TASK" \
    --device cuda:0 \
    --output-file "outputs/mmau/$eval_filename" \
    --prompt-type "$prompt_type" \
    --options \
    model.llama_path="$SLURM_TMPDIR/pretrained/vicuna-13b-v1.1" \
    model.whisper_path="$SLURM_TMPDIR/pretrained/whisper-large-v2" \
    model.beats_path="$SLURM_TMPDIR/pretrained/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt" \
    datasets.test_ann_path="$SLURM_TMPDIR/data/MMAU/annotations/test_mmau.json" \
    datasets.whisper_path="$SLURM_TMPDIR/pretrained/whisper-large-v2" \
    run.seed="$seed" \
    run.num_workers="$SLURM_CPUS_PER_TASK" \
    &
MMAU_PID=$!

echo "Evaluating MMAR..."

python evaluate_mmar.py \
    --cfg-path recipes/afthink/$model_type.yaml \
    --ckpt "$BEST_CKPT" \
    --batch-size 4 \
    --num-workers "$SLURM_CPUS_PER_TASK" \
    --device cuda:1 \
    --output-file "outputs/mmar/$eval_filename" \
    --prompt-type "$prompt_type" \
    --options \
    model.llama_path="$SLURM_TMPDIR/pretrained/vicuna-13b-v1.1" \
    model.whisper_path="$SLURM_TMPDIR/pretrained/whisper-large-v2" \
    model.beats_path="$SLURM_TMPDIR/pretrained/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt" \
    datasets.test_ann_path="$SLURM_TMPDIR/data/MMAR/annotations/test_mmar.json" \
    datasets.whisper_path="$SLURM_TMPDIR/pretrained/whisper-large-v2" \
    run.seed="$seed" \
    run.num_workers="$SLURM_CPUS_PER_TASK" \
    &

MMAR_PID=$!

wait $MMAU_PID $MMAR_PID
echo "MMAU and MMAR evaluation finished at $(date)"
