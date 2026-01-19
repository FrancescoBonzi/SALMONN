# AF-Think Dataset Preparation

Prepares the [NVIDIA AF-Think](https://huggingface.co/datasets/nvidia/AF-Think) dataset for SALMONN training.

> **Note**: AF-Think provides audio reasoning annotations (chain-of-thought) but does **not** include audio files. Audio must be downloaded from the original source datasets.

## Quick Start

```bash
# Show dataset info and source requirements
python recipes/afthink/prepare_afthink.py --info

# Download available datasets (HuggingFace sources only)
python recipes/afthink/prepare_afthink.py --download-audio

# Download specific splits
python recipes/afthink/prepare_afthink.py --splits urbansound8k esc50 --download-audio

# Debug mode (limited samples)
python recipes/afthink/prepare_afthink.py --splits urbansound8k --max-samples 100 --download-audio
```

## Source Datasets

### Auto-Downloadable (HuggingFace)

| Split | Source |
|-------|--------|
| `urbansound8k` | danavery/urbansound8K |
| `esc50` | ashraq/esc50 |
| `meld` | declare-lab/MELD |
| `af_cot_train_clotho_v2` | d0rj/clotho-dataset |
| `af_cot_train_gtzan` | marsyas/gtzan |
| `af_cot_train_fsd50k` | Fhrozen/FSD50k |

### Manual Download Required

| Split | Source |
|-------|--------|
| `audioset` | [Google AudioSet](https://research.google.com/audioset/download.html) |
| `musiccaps` | [MusicCaps](https://www.kaggle.com/datasets/googleai/musiccaps) (YouTube download) |
| `vggsound` | [VGGSound](https://www.robots.ox.ac.uk/~vgg/data/vggsound/) |
| `af_cot_train_fma` | [Free Music Archive](https://github.com/mdeff/fma) |

For manual datasets, download audio files and use:

```bash
python recipes/afthink/prepare_afthink.py --existing-audio-dir /path/to/audio
```

## Output

Annotations are saved to `data/afthink/`:
- `afthink_all.json` — All processed samples
- `train_afthink.json` — Training split
- `eval_afthink.json` — Evaluation split

Audio files are saved to `data/afthink/audio/<split>/`.

## Options

| Flag | Description |
|------|-------------|
| `--info` | Print dataset info and exit |
| `--download-audio` | Download audio from HuggingFace sources |
| `--splits` | Specific splits to process |
| `--max-samples` | Limit samples per split (debugging) |
| `--output-dir` | Custom annotation output directory |
| `--audio-dir` | Custom audio output directory |
| `--existing-audio-dir` | Use pre-downloaded audio files |
