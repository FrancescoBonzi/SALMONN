#!/bin/bash
#SBATCH --time=1:0:0
#SBATCH --account=aip-csubakan
#SBATCH --cpus-per-task=48
#SBATCH --mem=488G
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --nodes=1

# Qwen2.5-Omni evaluation on MMAU and MMAR
# Run with: sbatch evaluate_qwen_benchmarks.sh
# For local runs: set SLURM_TMPDIR=$PWD and SLURM_CPUS_PER_TASK to desired workers

: ${SLURM_TMPDIR:=$PWD}
: ${SLURM_CPUS_PER_TASK:=4}

model_type="qwen25_omni"
prompt_type="official"
eval_filename="qwen_7B_pretrained_zeroshot.json"

# Copy data to SLURM_TMPDIR for fast I/O
echo "Copying data to SLURM_TMPDIR..."

mkdir -p "$SLURM_TMPDIR/data/MMAR/"
cp -r "data/MMAR/" "$SLURM_TMPDIR/data/" &
COPY_MMAR_DATA_PID=$!

mkdir -p "$SLURM_TMPDIR/data/MMAU/"
cp -r "data/MMAU/" "$SLURM_TMPDIR/data/" &
COPY_MMAU_DATA_PID=$!

wait $COPY_MMAR_DATA_PID $COPY_MMAU_DATA_PID
echo "Data copy complete!"

# Update annotation paths to point to SLURM_TMPDIR
echo "Updating annotation paths..."
sed -i "s|[^\"]*data/MMAR|$SLURM_TMPDIR/data/MMAR|g" "$SLURM_TMPDIR/data/MMAR/annotations/"*.json
sed -i "s|[^\"]*data/MMAU|$SLURM_TMPDIR/data/MMAU|g" "$SLURM_TMPDIR/data/MMAU/annotations/"*.json

# Activate environment
module load StdEnv/2023 cuda/12.2
module load httpproxy
source .venv/bin/activate

echo "Evaluating Qwen2.5-Omni on MMAU and MMAR..."

echo "Evaluating MMAU..."
python evaluate_qwen_mmau.py \
    --cfg-path recipes/mmau/qwen.yaml \
    --batch-size 4 \
    --num-workers $SLURM_CPUS_PER_TASK \
    --device cuda:0 \
    --output-file "outputs/mmau/$eval_filename" \
    --prompt-type "$prompt_type" \
    --options \
    datasets.test_ann_path="$SLURM_TMPDIR/data/MMAU/annotations/test_mmau.json" \
    run.num_workers="$SLURM_CPUS_PER_TASK"

echo "MMAU evaluation finished at $(date)"

echo "Evaluating MMAR..."
python evaluate_qwen_mmar.py \
    --cfg-path recipes/mmar/qwen.yaml \
    --batch-size 4 \
    --num-workers $SLURM_CPUS_PER_TASK \
    --device cuda:0 \
    --output-file "outputs/mmar/$eval_filename" \
    --prompt-type "$prompt_type" \
    --options \
    datasets.test_ann_path="$SLURM_TMPDIR/data/MMAR/annotations/test_mmar.json" \
    run.num_workers="$SLURM_CPUS_PER_TASK"

echo "MMAR evaluation finished at $(date)"
echo "Job finished at $(date)"
