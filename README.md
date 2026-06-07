# SPARE

![SPARE training framework](resource/spare_method.svg)

Reference code for **SPARE** (Semantic Prediction for Audio REasoning), from the paper *Enhancing Audio Reasoning via Semantic Summary Prediction* (Interspeech 2026).

Built on [SALMONN 13B](https://github.com/bytedance/SALMONN).

## What is this

Large audio-language models often have a **reasoning gap**: explicit chain-of-thought (CoT) hurts accuracy because attention drifts from the audio to the text the model already generated.

SPARE fixes this at training time. You inject a register token `[REG]` right after the audio/prompt and before the reasoning chain:

```
[Question, Audio, Choices, [REG], <SUMMARY>..., <CAPTION>..., <REASONING>..., <CONCLUSION>...]
```

During training, the hidden state of `[REG]` is aligned to a frozen [Sentence-BERT](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) embedding of the conclusion (cosine similarity loss). Total loss:

```
L = L_CE + λ · L_align     (λ = 2.0 in the paper)
```

At inference, `[REG]` and the projection head are dropped. Same decoding as standard SFT, zero extra cost.

**Results** (zero-shot, SALMONN 13B backbone, MMAU / MMAR):

| Method | MMAU | MMAR |
|---|---|---|
| Zero-shot | 36.0% | 33.3% |
| Zero-shot (CoT) | 18.0% | 12.3% |
| SFT | 54.7% | 38.2% |
| Audio MuToR | 53.0% | 38.4% |
| **SPARE** | **58.0%** | **40.3%** |

## Install

```bash
git clone <this-repo>
cd SALMONN
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt huggingface_hub
```

GPU recommended. Full pretrained weights ~80GB. Debug runs with tiny models.

## Quick Start (Debug)

Debug data is bundled in `resource/afthink_debug/` (2 wav files, a few annotations). No prep needed.

```bash
wget -O pretrained/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt \
  "https://huggingface.co/THUdyh/Ola_speech_encoders/resolve/main/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt?download=true"

hf download openai/whisper-tiny --local-dir pretrained/whisper-tiny
python recipes/create_mock_llm.py

python train.py --cfg-path recipes/afthink/debug.yaml
```

Optional attention analysis after training:

```bash
python eval/visualize_attention.py \
  --cfg-path recipes/afthink/debug.yaml \
  --options model.ckpt=outputs/afthink_debug/<run_id>/checkpoint_best.pth
```

## Full Experiments

### Pretrained Models

```bash
bash recipes/download_pretrained.sh
```

Downloads Whisper-large-v2, BEATs, Vicuna-13B, and `salmonn_v1.pth` into `pretrained/`.

### Training Data

Fine-tuning uses [AF-Think](https://huggingface.co/datasets/nvidia/AF-Think) labels on a YouTube8M subset (160k train / 40k val in the paper). Each sample has four chapters: `<SUMMARY>`, `<CAPTION>`, `<REASONING>`, `<CONCLUSION>`.

1. Put `AFThink.tar.gz` at `data/YouTube8M/AFThink.tar.gz`
2. `python recipes/afthink/prepare_cot_youtube8m.py`
3. Resample mp3 → 16 kHz wav:

```bash
cd data/YouTube8M/audio_files
find . -maxdepth 1 -name "*.mp3" -print0 | \
  parallel -0 -j 48 ffmpeg -i {} -ar 16000 -ac 1 -c:a pcm_s16le {.}.wav
```

**MMAU** eval: `python recipes/mmau/prepare_mmau.py` (downloads [MMAU-test-mini](https://huggingface.co/datasets/gamma-lab-umd/MMAU-test-mini) from HuggingFace and writes `data/MMAU/audio_files/` + `data/MMAU/annotations/test_mmau.json`).

**MMAR** eval: `python recipes/mmar/prepare_mmar.py`.

### Train

Recipes in `recipes/afthink/`. Set `datasets.*_ann_path` in the yaml (shipped empty). Paper configs:

| Recipe | What it reproduces |
|---|---|
| `salmonn.yaml` | SFT baseline |
| `mutor.yaml` | Audio MuToR baseline |
| `mutor_with_alpha_decay.yaml` | Audio MuToR with λ decay |
| `bert_conclusion_spare.yaml` | **SPARE** (single `[REG]`, set `mutor_alpha: 2.0`) |
| `bert_multi_conclusion_spare.yaml` | Multi-conclusion ablation (register per chapter) |
| `cot_salmonn.yaml` | CoT-focused training variant |

```bash
torchrun --nproc_per_node=4 train.py \
  --cfg-path recipes/afthink/bert_conclusion_spare.yaml \
  --options \
    model.mutor_alpha=2.0 \
    datasets.train_ann_path=data/YouTube8M/annotations/train_youtube8m.json \
    datasets.valid_ann_path=data/YouTube8M/annotations/test_youtube8m.json \
    datasets.test_ann_path=data/YouTube8M/annotations/test_youtube8m.json
```

Or the full train + eval pipeline:

```bash
MODEL_TYPE=bert_conclusion_spare bash run_youtube8m.sh
```

## Eval

Benchmarks: [MMAU](https://arxiv.org/abs/2410.19168) and [MMAR](https://arxiv.org/abs/2505.03054). Evaluations are zero-shot with `--prompt-type afthink` (CoT output, conclusion extracted for scoring).

```bash
python eval/evaluate_mmau.py \
  --cfg-path recipes/afthink/bert_conclusion_spare.yaml \
  --ckpt outputs/bert_conclusion_spare/<run_id>/checkpoint_best.pth \
  --prompt-type afthink \
  --output-file outputs/mmau/spare_seed1.json \
  --options datasets.test_ann_path=data/MMAU/annotations/test_mmau.json
```

MMAR: same with `eval/evaluate_mmar.py`. Shell wrappers: `bash eval/eval_benchmarks.sh`, `bash eval/eval_benchmarks_salmonn_.sh` (pretrained baseline), `bash eval/eval_attention.sh`.

## Qualitative Analysis

- `eval/visualize_attention.py` — attention maps (mechanistic analysis from the paper)
- `eval/analyze_mmau_caption_words.py` — caption length vs correctness
- `eval/compare_model_outputs.py` — overlap of correct samples between two runs

## Structure

```
train.py                        training
models/salmonn.py               SALMONN, MuToR, SPARE classes
recipes/afthink/                configs + data prep
eval/                           eval + analysis
resource/afthink_debug/         bundled debug data
```

## Cite

```bibtex
@inproceedings{spare2026,
  title={Enhancing Audio Reasoning via Semantic Summary Prediction},
  author={Anonymous},
  booktitle={Interspeech},
  year={2026}
}
```
