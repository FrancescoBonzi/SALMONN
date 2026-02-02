#!/bin/bash
# Download pretrained models for SALMONN

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
mkdir -p "$PROJECT_ROOT/pretrained" && cd "$PROJECT_ROOT/pretrained"

# 1. Whisper Large V2
echo "Downloading Whisper Large V2..."
hf download openai/whisper-large-v2 --local-dir whisper-large-v2

# 2. BEATs iter3+ AS2M (from HuggingFace mirror)
echo "Downloading BEATs..."
wget -O BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt "https://huggingface.co/THUdyh/Ola_speech_encoders/resolve/main/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt?download=true"

# 3. Vicuna 13B v1.1
echo "Downloading Vicuna 13B v1.1..."
hf download lmsys/vicuna-13b-v1.1 --local-dir vicuna-13b-v1.1

# 4. SALMONN pretrained checkpoint
echo "Downloading SALMONN pretrained checkpoint..."
wget -O salmonn_v1.pth "https://huggingface.co/tsinghua-ee/SALMONN/resolve/main/salmonn_v1.pth?download=true"
echo "Done!"

