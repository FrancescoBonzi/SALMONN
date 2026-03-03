#!/bin/bash
#SBATCH --time=12:0:0
#SBATCH --account=aip-csubakan
#SBATCH --cpus-per-task=48
#SBATCH --mem=488G
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --nodes=1
#SBATCH --array=0-5

# Qwen2.5-Omni base model training on YouTube8M

seeds=(1 2 3 4 5 6)
seed=${seeds[$SLURM_ARRAY_TASK_ID]}

model_type="qwen"
qwen_path="Qwen/Qwen2.5-Omni-7B"
prompt_type="afthink"
eval_filename="${model_type}_7B_${prompt_type}prompt/seed${seed}.json"

# Copy data to SLURM_TMPDIR for fast I/O
echo "Copying data to SLURM_TMPDIR..."

mkdir -p "$SLURM_TMPDIR/data/YouTube8M/"
cp -r "data/YouTube8M/" "$SLURM_TMPDIR/data/" &
COPY_YOUTUBE8M_DATA_PID=$!

mkdir -p "$SLURM_TMPDIR/data/MMAR/"
cp -r "data/MMAR/" "$SLURM_TMPDIR/data/" &
COPY_MMAR_DATA_PID=$!

mkdir -p "$SLURM_TMPDIR/data/MMAU/"
cp -r "data/MMAU/" "$SLURM_TMPDIR/data/" &
COPY_MMAU_DATA_PID=$!

wait $COPY_YOUTUBE8M_DATA_PID $COPY_MMAR_DATA_PID $COPY_MMAU_DATA_PID
echo "Data copy complete!"

# Extract YouTube8M audio archive if present
YT8M_ARCHIVE="$SLURM_TMPDIR/data/YouTube8M/youtube8m_audio_files.tar.gz"
[ -f "$YT8M_ARCHIVE" ] && tar -xzf "$YT8M_ARCHIVE" -C "$SLURM_TMPDIR/data/YouTube8M/" && rm "$YT8M_ARCHIVE"

echo "Extracted youtube8m_audio_files.tar.gz"

# Copy Qwen2.5-Omni if available locally (optional - otherwise downloads from HuggingFace)
mkdir -p "$SLURM_TMPDIR/pretrained"
if [ -d "pretrained/Qwen2.5-Omni-7B" ]; then
    echo "Copying Qwen2.5-Omni from pretrained/..."
    cp -r "pretrained/Qwen2.5-Omni-7B" "$SLURM_TMPDIR/pretrained/"
    qwen_path="$SLURM_TMPDIR/pretrained/Qwen2.5-Omni-7B"
else
    echo "Using HuggingFace path for Qwen2.5-Omni (will download if not cached)"
fi

# Update annotation paths
echo "Updating annotation paths..."
sed -i "s|[^\"]*data/YouTube8M|$SLURM_TMPDIR/data/YouTube8M|g" "$SLURM_TMPDIR/data/YouTube8M/annotations/"*.json
sed -i "s|[^\"]*data/MMAR|$SLURM_TMPDIR/data/MMAR|g" "$SLURM_TMPDIR/data/MMAR/annotations/"*.json
sed -i "s|[^\"]*data/MMAU|$SLURM_TMPDIR/data/MMAU|g" "$SLURM_TMPDIR/data/MMAU/annotations/"*.json

# Activate environment
module load StdEnv/2023 cuda/12.2
module load httpproxy
source .venv/bin/activate

# Run training
echo "Starting training..."

torchrun --nproc_per_node=4 train_qwen.py --cfg-path recipes/afthink/$model_type.yaml \
    --options \
    model.qwen25_omni_path="$qwen_path" \
    datasets.train_ann_path="$SLURM_TMPDIR/data/YouTube8M/annotations/train_youtube8m.json" \
    datasets.valid_ann_path="$SLURM_TMPDIR/data/YouTube8M/annotations/test_youtube8m.json" \
    datasets.test_ann_path="$SLURM_TMPDIR/data/YouTube8M/annotations/test_youtube8m.json" \
    datasets.whisper_path="$qwen_path" \
    run.seed="$seed" \
    run.output_dir="outputs/afthink_youtube8m/$model_type/$seed" \
    run.num_workers=8

echo "Training finished at $(date)"

# Run evaluation
echo "Starting evaluation..."
OUTPUT_DIR=$(ls -dt outputs/afthink_youtube8m/$model_type/$seed/* 2>/dev/null | head -n 1)
BEST_CKPT="${OUTPUT_DIR}/checkpoint_best.pth"
echo "Using checkpoint: $BEST_CKPT"

echo "Evaluating MMAU..."
python evaluate_qwen_mmau.py \
    --cfg-path recipes/afthink/$model_type.yaml \
    --ckpt "$BEST_CKPT" \
    --batch-size 4 \
    --num-workers "$SLURM_CPUS_PER_TASK" \
    --device cuda:0 \
    --output-file "outputs/mmau/$eval_filename" \
    --prompt-type "$prompt_type" \
    --options \
    model.qwen25_omni_path="$qwen_path" \
    datasets.test_ann_path="$SLURM_TMPDIR/data/MMAU/annotations/test_mmau.json" \
    datasets.whisper_path="$qwen_path" \
    run.seed="$seed" \
    run.num_workers="$SLURM_CPUS_PER_TASK"

echo "MMAU evaluation finished at $(date)"

echo "Evaluating MMAR..."
python evaluate_qwen_mmar.py \
    --cfg-path recipes/afthink/$model_type.yaml \
    --ckpt "$BEST_CKPT" \
    --batch-size 4 \
    --num-workers "$SLURM_CPUS_PER_TASK" \
    --device cuda:0 \
    --output-file "outputs/mmar/$eval_filename" \
    --prompt-type "$prompt_type" \
    --options \
    model.qwen25_omni_path="$qwen_path" \
    datasets.test_ann_path="$SLURM_TMPDIR/data/MMAR/annotations/test_mmar.json" \
    datasets.whisper_path="$qwen_path" \
    run.seed="$seed" \
    run.num_workers="$SLURM_CPUS_PER_TASK"

echo "MMAR evaluation finished at $(date)"
echo "Job finished at $(date)"
