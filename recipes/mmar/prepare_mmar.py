# prepare_mmar.py
import os
import json
from pathlib import Path

import tarfile
import urllib.request
import numpy as np
import soundfile as sf
import librosa
from datasets import load_dataset

TARGET_SAMPLE_RATE = 16000  # Whisper expects 16 kHz
AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg"}


def resample_audio_dir(root_dir: Path, target_sr: int = TARGET_SAMPLE_RATE):
    """Resample all audio files under root_dir to target_sr (16 kHz for Whisper)."""
    root_dir = Path(root_dir)
    resampled = 0
    for path in root_dir.rglob("*"):
        if path.suffix.lower() not in AUDIO_EXTENSIONS or not path.is_file():
            continue
        try:
            audio, sr = sf.read(path)
            if sr == target_sr:
                continue
            if len(audio.shape) == 2:
                audio = audio.mean(axis=1)
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

            # Normalize audio peak
            audio = audio / (np.abs(audio).max() + 1e-9)

            sf.write(path, audio, target_sr)
            resampled += 1
        except Exception as e:
            print(f"Warning: failed to resample {path}: {e}")
    if resampled:
        print(f"Resampled {resampled} audio file(s) to {target_sr} Hz.")


def prepare_mmar_annotations(output_dir: str):
    """Load MMAR test split from Hugging Face and write test_mmar.json and test_mmar_metadata.json."""
    os.makedirs(output_dir, exist_ok=True)
    output_dir = Path(output_dir)

    ds = load_dataset("BoJack/MMAR")
    annotations = []
    for row in ds["test"]:
        audio_path = row["audio_path"]
        if not os.path.isabs(audio_path):
            audio_path = str(output_dir / audio_path)

        annotations.append({
            "path": audio_path,
            "text": row["answer"],
            "task": "mmar",
            **{key: value for key, value in row.items() if key != "audio_path" and key != "answer"}
        })

    (output_dir / "annotations").mkdir(parents=True, exist_ok=True)
    ann_path = output_dir / "annotations" / "test_mmar.json"
    with open(ann_path, "w") as f:
        json.dump({"annotation": annotations}, f, indent=2)
    print(f"Saved {len(annotations)} samples to {ann_path}")


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent.parent / "data" / "MMAR"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Download and extract mmar-audio.tar.gz
    url = "https://huggingface.co/datasets/BoJack/MMAR/resolve/main/mmar-audio.tar.gz?download=true"
    archive_path = data_dir / "mmar-audio.tar.gz"
    if not archive_path.exists():
        print("Downloading mmar-audio.tar.gz...")
        urllib.request.urlretrieve(url, archive_path)
    print("Extracting...")
    with tarfile.open(archive_path, "r:gz") as tf:
        tf.extractall(data_dir)

    print("Resampling audio to 16 kHz for Whisper...")
    resample_audio_dir(data_dir)

    prepare_mmar_annotations(str(data_dir))
    print("Done. Dataset in", data_dir)
