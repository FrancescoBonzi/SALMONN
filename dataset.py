# Copyright (2024) Tsinghua University, Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import soundfile as sf
import numpy as np
from transformers import WhisperFeatureExtractor


class SALMONNDataset(Dataset):
    def __init__(self, ann_path, whisper_path):
        super().__init__()

        self.annotation = json.load(open(ann_path, "r"))["annotation"]

        self.wav_processor = WhisperFeatureExtractor.from_pretrained(whisper_path)

    def __len__(self):
        return len(self.annotation)

    def collater(self, samples):
        samples_spectrogram = [s["spectrogram"] for s in samples]
        cat_spectrogram = torch.stack(samples_spectrogram, dim=0)

        raw_wav = [torch.from_numpy(s["raw_wav"]) for s in samples]
        raw_wav_length = torch.tensor([len(s["raw_wav"]) for s in samples])
        raw_wav = pad_sequence(raw_wav, batch_first=True, padding_value=0)
        paddding_mask = torch.arange(raw_wav.size(1)).unsqueeze(0) >= raw_wav_length.unsqueeze(1)

        text = [s["text"] for s in samples]
        task = [s["task"] for s in samples]
        Q = [s["Q"] for s in samples]
        id = [s["id"] for s in samples]

        result = {
            "spectrogram": cat_spectrogram,
            "raw_wav": raw_wav,
            "padding_mask": paddding_mask,
            "text": text,
            "task": task,
            "Q": Q,
            "id": id,
        }
        
        # For reasoning tasks: add question and answer only if present
        if any("question" in s for s in samples):
            result["question"] = [s.get("question", "") for s in samples]
            result["answer"] = [s.get("answer", "") for s in samples]

        return result

    def __getitem__(self, index):
        ann = self.annotation[index]

        audio, sr = sf.read(ann["path"])
        if len(audio.shape) == 2: # stereo to mono
            audio = audio[:, 0]
        if "expand_wav" in ann:
            for p in ann["expand_wav"]:
                expand_audio, _ = sf.read(p)
                if len(expand_audio.shape) == 2:
                    expand_audio = expand_audio[:, 0]
                sil = np.zeros(1600, dtype=float)
                audio = np.concatenate((audio, sil, expand_audio), axis=0)
        if len(audio) < sr: # pad audio to at least 1s
            sil = np.zeros(sr - len(audio), dtype=float)
            audio = np.concatenate((audio, sil), axis=0)
        audio = audio[: sr * 30] # truncate audio to at most 30s

        spectrogram = self.wav_processor(audio, sampling_rate=sr, return_tensors="pt")["input_features"].squeeze()
        text = ann.get("text", "")
        task = ann.get("task", "asr")
        Q = ann.get("Q", "")

        result = {
            "spectrogram": spectrogram,
            "raw_wav": audio,
            "text": text,
            "task": task,
            "Q": Q,
            "id": ann["path"],
        }
        
        # For reasoning tasks: add question and answer only if present
        if "question" in ann:
            result["question"] = ann["question"]
            result["answer"] = ann.get("answer", "")

        return result


class Qwen2AudioDataset(Dataset):
    """
    Dataset for Qwen2.5-Omni / Qwen2-Audio training.
    Uses the same JSON annotation format as SALMONNDataset so existing
    annotations can be reused across both model families.

    Annotation JSON format:
    {
        "annotation": [
            {
                "path": "/path/to/audio.wav",
                "text": "transcription or answer",
                "task": "asr",
                "Q": "",
                "question": "optional question",
                "answer": "optional answer with <CONCLUSION>...</CONCLUSION>"
            },
            ...
        ]
    }
    """

    def __init__(self, ann_path, whisper_path="Qwen/Qwen2.5-Omni-7B"):
        super().__init__()
        self.annotation = json.load(open(ann_path, "r"))["annotation"]
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_path)

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]

        audio, sr = sf.read(ann["path"])
        if len(audio.shape) == 2:
            audio = audio[:, 0]

        if "expand_wav" in ann:
            for p in ann["expand_wav"]:
                expand_audio, _ = sf.read(p)
                if len(expand_audio.shape) == 2:
                    expand_audio = expand_audio[:, 0]
                sil = np.zeros(1600, dtype=float)
                audio = np.concatenate((audio, sil, expand_audio), axis=0)

        if len(audio) < sr:
            sil = np.zeros(sr - len(audio), dtype=float)
            audio = np.concatenate((audio, sil), axis=0)

        audio = audio[:sr * 30]

        features = self.feature_extractor(
            audio,
            sampling_rate=sr,
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_features = features["input_features"].squeeze(0)
        feature_attention_mask = features["attention_mask"].squeeze(0)

        result = {
            "input_features": input_features,
            "feature_attention_mask": feature_attention_mask,
            "text": ann.get("text", ""),
            "task": ann.get("task", "asr"),
            "Q": ann.get("Q", ""),
            "id": ann["path"],
        }

        if "question" in ann:
            result["question"] = ann["question"]
            result["answer"] = ann.get("answer", "")

        return result

    def collater(self, samples):
        input_features = torch.stack([s["input_features"] for s in samples], dim=0)
        feature_attention_mask = torch.stack(
            [s["feature_attention_mask"] for s in samples], dim=0
        )

        result = {
            "input_features": input_features,
            "feature_attention_mask": feature_attention_mask,
            "text": [s["text"] for s in samples],
            "task": [s["task"] for s in samples],
            "Q": [s["Q"] for s in samples],
            "id": [s["id"] for s in samples],
        }

        if any("question" in s for s in samples):
            result["question"] = [s.get("question", "") for s in samples]
            result["answer"] = [s.get("answer", "") for s in samples]

        return result
