import os
import json
import tarfile
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import hf_hub_download


def prepare_mmar_annotations(output_dir: str, download_dir: str = "./data"):
    """Download MMAR dataset and create SALMONN-compatible annotations."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(download_dir, exist_ok=True)

    mmar_audio_dir = os.path.join(download_dir, "MMAR", "audio")
    os.makedirs(mmar_audio_dir, exist_ok=True)

    print("Loading MMAR dataset from HuggingFace...")
    dataset = load_dataset("BoJack/MMAR", cache_dir=download_dir)

    if "test" not in dataset:
        print("Error: 'test' split not found")
        return

    print("\nDownloading audio archive (~698MB)...")
    audio_tar_path = hf_hub_download(
        repo_id="BoJack/MMAR",
        filename="mmar-audio.tar.gz",
        repo_type="dataset",
        cache_dir=download_dir
    )

    print(f"Extracting audio files to {mmar_audio_dir}...")
    with tarfile.open(audio_tar_path, 'r:gz') as tar:
        tar.extractall(path=os.path.join(download_dir, "MMAR"))
    print("Audio extraction complete!")

    test_data = dataset["test"]
    all_annotations = []
    choice_letters = ["A", "B", "C", "D", "E", "F", "G", "H"]

    print(f"\nProcessing {len(test_data)} samples...")

    for i, sample in enumerate(test_data):
        audio_rel_path = sample.get("audio_path", "").replace("./", "")
        if audio_rel_path.startswith("audio/"):
            audio_rel_path = audio_rel_path[6:]

        local_audio_path = os.path.join(mmar_audio_dir, audio_rel_path)

        choices = sample.get("choices", [])
        formatted_choices = ", ".join([f"{choice_letters[j]}. {c}" for j, c in enumerate(choices)])

        answer_text = sample.get("answer", "")
        answer_letter = next((choice_letters[j] for j, c in enumerate(choices)
                             if c.strip().lower() == answer_text.strip().lower()), answer_text)

        annotation = {
            "path": local_audio_path,
            "text": f"{sample.get('question', '')} {formatted_choices}",
            "answer": str(answer_letter),
            "task": "mmar",
            "id": sample.get("id", ""),
            "modality": sample.get("modality", ""),
            "category": sample.get("category", ""),
        }
        all_annotations.append(annotation)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(test_data)} samples...")

    print(f"\nTotal samples processed: {len(all_annotations)}")

    # Create train/val/test split (60/20/20)
    import random
    random.seed(42)
    indices = list(range(len(all_annotations)))
    random.shuffle(indices)

    n_train, n_val = int(0.6 * len(indices)), int(0.2 * len(indices))
    splits = {
        "train_mmar.json": [all_annotations[i] for i in indices[:n_train]],
        "valid_mmar.json": [all_annotations[i] for i in indices[n_train:n_train + n_val]],
        "test_mmar.json": [all_annotations[i] for i in indices[n_train + n_val:]]
    }

    for filename, annotations in splits.items():
        output_path = os.path.join(output_dir, filename)
        with open(output_path, "w") as f:
            json.dump({"annotation": annotations}, f, indent=2)
        print(f"Saved {len(annotations)} samples to {output_path}")

    print("\nMMAR dataset preparation complete!")
    print(f"Audio files: {mmar_audio_dir}/")


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    data_dir = project_root / "data"

    prepare_mmar_annotations(
        output_dir=str(data_dir / "mmar"),
        download_dir=str(data_dir)
    )