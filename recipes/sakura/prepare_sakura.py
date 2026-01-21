import os
import csv
import json
from pathlib import Path

import soundfile as sf
from datasets import load_dataset

TASKS = {
    "Gender": "gender_QA",
    "Language": "QA",
    "Emotion": "emotion_recognition",
    "Animal": "QA",
}

HF_DATASETS = {
    "Animal": "SLLM-multi-hop/AnimalQA",
    "Gender": "SLLM-multi-hop/GenderQA",
    "Emotion": "SLLM-multi-hop/EmotionQA",
    "Language": "SLLM-multi-hop/LanguageQA",
}


def download_sakura(download_dir: Path):
    """
    Download SAKURA from Hugging Face and materialize it
    in the official SAKURA directory structure.

    This version intentionally relies on TorchCodec
    to decode Arrow-embedded audio (Linux / CC friendly).
    """

    download_dir.mkdir(parents=True, exist_ok=True)

    for track, hf_name in HF_DATASETS.items():
        print(f"Downloading {track}...")

        track_dir = download_dir / track
        audio_dir = track_dir / "audio"
        metadata_path = track_dir / "metadata.csv"

        # Load dataset from HF
        dataset = load_dataset(
            hf_name,
            split="test",
            cache_dir=str(download_dir),
        )

        # Skip only if data is actually materialized
        if metadata_path.exists() and audio_dir.exists() and any(audio_dir.iterdir()):
            print(f"{track} already exists, skipping...")
            continue

        audio_dir.mkdir(parents=True, exist_ok=True)

        with open(metadata_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "file",
                    "attribute_label",
                    "single_instruction",
                    "single_answer",
                    "multi_instruction",
                    "multi_answer",
                ]
            )

            for idx, sample in enumerate(dataset):
                audio = sample["audio"]  # decoded via TorchCodec

                file_name = f"{idx:06d}.wav"
                audio_path = audio_dir / file_name

                # Write decoded audio
                sf.write(
                    audio_path,
                    audio["array"],
                    audio["sampling_rate"],
                )

                writer.writerow(
                    [
                        file_name,
                        sample.get("attribute_label", ""),
                        sample["single_instruction"],
                        sample["single_answer"],
                        sample["multi_instruction"],
                        sample["multi_answer"],
                    ]
                )

        print(f"{track} downloaded")
        print(f"Audio files: {len(dataset)}")
        print(f"Metadata: {metadata_path}")


def get_annotations(download_dir: Path, track_name: str):
    track_dir = download_dir / track_name
    audio_dir = track_dir / "audio"
    metadata_file = track_dir / "metadata.csv"

    annotations = []

    with open(metadata_file, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            audio_file = audio_dir / row["file"]
            if not audio_file.exists():
                print(f"Warning: missing {audio_file}, skipping")
                continue

            # Single-hop
            annotations.append(
                {
                    "path": str(audio_file),
                    "task": TASKS[track_name],
                    "question": row["single_instruction"],
                    "text": row["single_answer"],
                }
            )

            # Multi-hop
            annotations.append(
                {
                    "path": str(audio_file),
                    "task": TASKS[track_name],
                    "question": row["multi_instruction"],
                    "text": row["multi_answer"],
                }
            )

    return annotations


def main():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent

    data_dir = project_root / "data"
    output_dir = data_dir / "sakura"
    output_dir.mkdir(parents=True, exist_ok=True)

    download_sakura(data_dir)

    all_annotations = []

    for track_name in TASKS:
        print(f"Processing track: {track_name}")
        track_ann = get_annotations(data_dir, track_name)
        all_annotations.extend(track_ann)
        print(f"Added {len(track_ann)} samples from {track_name}")

    output_file = output_dir / "test_sakura.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"annotation": all_annotations}, f, indent=2)

    print(f"Done! Total samples: {len(all_annotations)}")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()
