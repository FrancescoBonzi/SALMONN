# AF-Think debug dataset

Bundled mini dataset for `recipes/afthink/debug.yaml`. No preparation needed — clone the repo and run training.

## Contents

| File | Samples | Description |
|---|---|---|
| `train_afthink.json` | 4 | Training annotations (2 unique audio clips, duplicated) |
| `valid_afthink.json` | 2 | Validation annotations |
| `test_afthink.json` | 2 | Test annotations |
| `samples/` | 2 WAV files | 16 kHz mono audio (`Pas sur le sol.wav`, `Bear Last Audio.wav`) |

Each annotation uses `task: "afthink_reasoning"` with `<SUMMARY>`, `<CAPTION>`, `<REASONING>`, and `<CONCLUSION>` tags in the answer field.

## Quick start

```bash
python recipes/create_mock_llm.py
hf download openai/whisper-tiny --local-dir pretrained/whisper-tiny
# BEATs: see recipes/download_pretrained.sh

python train.py --cfg-path recipes/afthink/debug.yaml
```
