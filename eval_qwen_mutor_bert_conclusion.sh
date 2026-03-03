#!/bin/bash
#SBATCH --time=12:0:0
#SBATCH --account=aip-csubakan
#SBATCH --cpus-per-task=48
#SBATCH --mem=488G
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --nodes=1
#SBATCH --array=0-0

# Qwen2.5-Omni + MuToR BERT Conclusion training on YouTube8M

seeds=(1)
seed=${seeds[$SLURM_ARRAY_TASK_ID]}

model_type="qwen_mutor_bert_conclusion"
qwen_path="Qwen/Qwen2.5-Omni-7B"
prompt_type="afthink"
eval_filename="${model_type}_7B_${prompt_type}prompt/seed${seed}.json"

# Copy data to SLURM_TMPDIR for fast I/O
echo "Copying data to SLURM_TMPDIR..."

mkdir -p "$SLURM_TMPDIR/data/MMAR/"
cp -r "data/MMAR/" "$SLURM_TMPDIR/data/" &
COPY_MMAR_DATA_PID=$!

mkdir -p "$SLURM_TMPDIR/data/MMAU/"
cp -r "data/MMAU/" "$SLURM_TMPDIR/data/" &
COPY_MMAU_DATA_PID=$!

wait $COPY_YOUTUBE8M_DATA_PID $COPY_MMAR_DATA_PID $COPY_MMAU_DATA_PID
echo "Data copy complete!"

# Extract YouTube8M audio archive if present
# YT8M_ARCHIVE="$SLURM_TMPDIR/data/YouTube8M/youtube8m_audio_files.tar.gz"
# [ -f "$YT8M_ARCHIVE" ] && tar -xzf "$YT8M_ARCHIVE" -C "$SLURM_TMPDIR/data/YouTube8M/" && rm "$YT8M_ARCHIVE"

# echo "Extracted youtube8m_audio_files.tar.gz"

# Copy Qwen2.5-Omni if available locally (optional - otherwise downloads from HuggingFace)
mkdir -p "$SLURM_TMPDIR/pretrained"
if [ -d "pretrained/Qwen2.5-Omni-7B" ]; then
    echo "Copying Qwen2.5-Omni from pretrained/..."
    cp -r "pretrained/Qwen2.5-Omni-7B" "$SLURM_TMPDIR/pretrained/"
    qwen_path="$SLURM_TMPDIR/pretrained/Qwen2.5-Omni-7B"
else
    echo "Using HuggingFace path for Qwen2.5-Omni (will download if not cached)"
fi

# Copy BERT conclusion model if available (required when cluster proxy blocks HuggingFace)
# Pre-download: huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 --local-dir pretrained/all-MiniLM-L6-v2
bert_path=""
if [ -d "pretrained/all-MiniLM-L6-v2" ]; then
    echo "Copying all-MiniLM-L6-v2 from pretrained/..."
    cp -r "pretrained/all-MiniLM-L6-v2" "$SLURM_TMPDIR/pretrained/"
    bert_path="$SLURM_TMPDIR/pretrained/all-MiniLM-L6-v2"
else
    echo "WARNING: pretrained/all-MiniLM-L6-v2 not found. HuggingFace download may fail if proxy blocks it."
fi

# Update annotation paths
echo "Updating annotation paths..."
sed -i "s|[^\"]*data/MMAR|$SLURM_TMPDIR/data/MMAR|g" "$SLURM_TMPDIR/data/MMAR/annotations/"*.json
sed -i "s|[^\"]*data/MMAU|$SLURM_TMPDIR/data/MMAU|g" "$SLURM_TMPDIR/data/MMAU/annotations/"*.json

# Activate environment
module load StdEnv/2023 cuda/12.2
module load httpproxy
source .venv/bin/activate

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
    --num-workers 8 \
    --device cuda:0 \
    --output-file "outputs/mmau/$eval_filename" \
    --prompt-type "$prompt_type" \
    --options \
    model.qwen25_omni_path="$qwen_path" \
    model.bert_conclusion_path="$bert_path" \
    datasets.test_ann_path="$SLURM_TMPDIR/data/MMAU/annotations/test_mmau.json" \
    datasets.whisper_path="$qwen_path" \
    run.seed="$seed" \
    run.num_workers=8

echo "MMAU evaluation finished at $(date)"

echo "Evaluating MMAR..."
python evaluate_qwen_mmar.py \
    --cfg-path recipes/afthink/$model_type.yaml \
    --ckpt "$BEST_CKPT" \
    --batch-size 4 \
    --num-workers 8 \
    --device cuda:0 \
    --output-file "outputs/mmar/$eval_filename" \
    --prompt-type "$prompt_type" \
    --options \
    model.qwen25_omni_path="$qwen_path" \
    model.bert_conclusion_path="$bert_path" \
    datasets.test_ann_path="$SLURM_TMPDIR/data/MMAR/annotations/test_mmar.json" \
    datasets.whisper_path="$qwen_path" \
    run.seed="$seed" \
    run.num_workers=8

echo "MMAR evaluation finished at $(date)"
echo "Job finished at $(date)"