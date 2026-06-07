"""Prepare MMAU test-mini from the official HuggingFace parquet release.

Dataset: https://huggingface.co/datasets/gamma-lab-umd/MMAU-test-mini

Audio is embedded in the parquet `context` column. This script extracts it,
resamples to 16 kHz WAV files, and writes SALMONN-compatible annotations.
"""

import argparse
import io
import json
from pathlib import Path

import librosa
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import hf_hub_download
from tqdm import tqdm

TARGET_SAMPLE_RATE = 16000
HF_DATASET = "gamma-lab-umd/MMAU-test-mini"
HF_PARQUET = "test_mini.parquet"


def decode_audio_bytes(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(io.BytesIO(audio_bytes))
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32), int(sr)


def resample_audio(audio: np.ndarray, sr: int, target_sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    if sr == target_sr:
        return audio
    return librosa.resample(audio, orig_sr=sr, target_sr=target_sr).astype(np.float32)


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / (peak + 1e-9)
    return audio


def parse_other_attributes(raw_attrs) -> dict:
    if isinstance(raw_attrs, str):
        return json.loads(raw_attrs)
    return raw_attrs


def prepare_mmau(
    output_dir: Path,
    dataset_name: str = HF_DATASET,
    parquet_file: str = HF_PARQUET,
    skip_existing_audio: bool = True,
):
    output_dir = Path(output_dir)
    audio_dir = output_dir / "audio_files"
    ann_dir = output_dir / "annotations"
    audio_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {dataset_name}/{parquet_file} from HuggingFace...")
    parquet_path = hf_hub_download(dataset_name, parquet_file, repo_type="dataset")
    table = pq.read_table(
        parquet_path,
        columns=["context", "instruction", "choices", "answer", "other_attributes"],
    )
    rows = table.to_pylist()
    print(f"Loaded {len(rows)} samples.")

    annotations = []
    for row in tqdm(rows, desc="Extracting audio"):
        attrs = parse_other_attributes(row["other_attributes"])
        sample_id = attrs["id"]
        audio_path = audio_dir / f"{sample_id}.wav"

        if not (skip_existing_audio and audio_path.exists()):
            audio_bytes = row["context"]["bytes"]
            if not audio_bytes:
                raise ValueError(f"Missing audio bytes for sample {sample_id}")

            audio, sr = decode_audio_bytes(audio_bytes)
            audio = resample_audio(audio, sr)
            audio = normalize_audio(audio)
            sf.write(audio_path, audio, TARGET_SAMPLE_RATE)

        annotations.append({
            "path": str(audio_path.resolve()),
            "text": row["answer"],
            "id": sample_id,
            "question": row["instruction"],
            "choices": row["choices"],
            "answer": row["answer"],
            "task": attrs["task"],
            "difficulty": attrs["difficulty"],
            "sub-category": attrs.get("sub-category"),
            "dataset": attrs.get("dataset"),
            "category": attrs.get("category"),
            "split": attrs.get("split"),
        })

    ann_path = ann_dir / "test_mmau.json"
    with open(ann_path, "w") as f:
        json.dump({"annotation": annotations}, f, indent=2)

    print(f"Saved {len(annotations)} annotations to {ann_path}")
    print(f"Audio files in {audio_dir}")
    return ann_path


def main():
    parser = argparse.ArgumentParser(description="Prepare MMAU test-mini from HuggingFace parquet.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <repo>/data/MMAU)",
    )
    parser.add_argument("--dataset", default=HF_DATASET, help="HuggingFace dataset id")
    parser.add_argument("--parquet-file", default=HF_PARQUET, help="Parquet filename in the dataset repo")
    parser.add_argument(
        "--force-audio",
        action="store_true",
        help="Re-extract audio even if WAV files already exist",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    output_dir = args.output_dir or (script_dir.parent.parent / "data" / "MMAU")

    prepare_mmau(
        output_dir=output_dir,
        dataset_name=args.dataset,
        parquet_file=args.parquet_file,
        skip_existing_audio=not args.force_audio,
    )
    print("Done.")


if __name__ == "__main__":
    main()
