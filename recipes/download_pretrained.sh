#!/bin/bash
# Download pretrained models for SALMONN

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
mkdir -p "$PROJECT_ROOT/pretrained" && cd "$PROJECT_ROOT/pretrained"

# 1. Whisper Large V2
echo "Downloading Whisper Large V2..."
huggingface-cli download openai/whisper-large-v2 --local-dir whisper-large-v2

# 2. BEATs iter3+ AS2M
echo "Downloading BEATs..."
wget -O BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt "https://valle.blob.core.windows.net/share/BEATs/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt?sv=2020-08-04&st=2023-03-01T07%3A51%3A05Z&se=2033-03-02T07%3A51%3A00Z&sr=c&sp=rl&sig=QJXmSJG9DbMKf48UDIU1MfzIro8HQOf3sqlNXiflY1I%3D"

# 3. Vicuna 13B v1.1
echo "Downloading Vicuna 13B v1.1..."
huggingface-cli download lmsys/vicuna-13b-v1.1 --local-dir vicuna-13b-v1.1

echo "Done!"

