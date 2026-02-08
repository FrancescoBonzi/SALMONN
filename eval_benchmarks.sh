#!/bin/bash
#SBATCH --time=1:0:0
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

# Copy MMAR
mkdir -p "$SLURM_TMPDIR/data/MMAR/audio_files"
cp -r "data/MMAR/audio_files" "$SLURM_TMPDIR/data/MMAR/audio_files/" &
COPY_MMAR_AUDIO_PID=$!

mkdir -p "$SLURM_TMPDIR/data/MMAR/annotations"
cp -r "data/MMAR/annotations" "$SLURM_TMPDIR/data/MMAR/annotations/" &
COPY_MMAR_ANN_PID=$!

# Copy MMAU
mkdir -p "$SLURM_TMPDIR/data/MMAU/audio_files"
cp -r "data/MMAU/audio_files" "$SLURM_TMPDIR/data/MMAU/audio_files/" &
COPY_MMAU_AUDIO_PID=$!

mkdir -p "$SLURM_TMPDIR/data/MMAU/annotations"
cp -r "data/MMAU/annotations" "$SLURM_TMPDIR/data/MMAU/annotations/" &
COPY_MMAU_ANN_PID=$!

# Wait for all copies to finish
wait $COPY_MMAR_AUDIO_PID $COPY_MMAR_ANN_PID $COPY_MMAU_AUDIO_PID $COPY_MMAU_ANN_PID
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
echo "Updating annotation paths..."
sed -i "s|[^\"]*data/MMAR|$SLURM_TMPDIR/data/MMAR|g" "$SLURM_TMPDIR/data/MMAR/annotations/"*.json
sed -i "s|[^\"]*data/MMAU|$SLURM_TMPDIR/data/MMAU|g" "$SLURM_TMPDIR/data/MMAU/annotations/"*.json

# Activate environment
module load StdEnv/2023 cuda/12.2
module load httpproxy
source .venv/bin/activate

echo "Evaluating MMAU (cuda:0) and MMAR (cuda:1) in parallel..."

python evaluate_mmau.py \
    --cfg-path recipes/afthink/$model_type.yaml \
    --ckpt "$SLURM_TMPDIR/pretrained/ckpt.pth" \
    --batch-size 4 \
    --num-workers $SLURM_CPUS_PER_TASK \
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
    generate.max_new_tokens=20 \
    generate.num_beams=1 \
    &
MMAU_PID=$!

python evaluate_mmar.py \
    --cfg-path recipes/afthink/$model_type.yaml \
    --ckpt "$SLURM_TMPDIR/pretrained/ckpt.pth" \
    --batch-size 4 \
    --num-workers $SLURM_CPUS_PER_TASK \
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
    generate.max_new_tokens=20 \
    generate.num_beams=1 \
    &
MMAR_PID=$!

wait $MMAU_PID $MMAR_PID
echo "MMAU and MMAR evaluation finished at $(date)"
