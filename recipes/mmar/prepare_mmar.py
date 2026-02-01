import os
import json
import tarfile
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm


def resample_audio_file(input_path: str, output_path: str, target_sr: int = 16000):
    """Resample audio file to target sampling rate."""
    try:
        audio, sr = torchaudio.load(input_path)

        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        # Resample if necessary
        if sr != target_sr:
            resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
            audio = resampler(audio)

        # Save resampled audio
        torchaudio.save(output_path, audio, target_sr)
        return True
    except Exception as e:
        print(f"Error resampling {input_path}: {e}")
        return False


def prepare_mmar_annotations(output_dir: str, download_dir: str = "./data"):
    """Download MMAR dataset and create SALMONN-compatible annotations."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(download_dir, exist_ok=True)

    mmar_audio_dir = os.path.join(download_dir, "MMAR", "audio")
    mmar_audio_16k_dir = os.path.join(download_dir, "MMAR", "audio_16k")
    os.makedirs(mmar_audio_dir, exist_ok=True)
    os.makedirs(mmar_audio_16k_dir, exist_ok=True)

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

    # Resample all audio files to 16kHz
    print("\nResampling audio files to 16kHz...")
    audio_files = list(Path(mmar_audio_dir).rglob("*.wav")) + \
                  list(Path(mmar_audio_dir).rglob("*.mp3")) + \
                  list(Path(mmar_audio_dir).rglob("*.flac"))

    if not audio_files:
        print(f"Warning: No audio files found in {mmar_audio_dir}")
        # Try to find files in subdirectories
        audio_files = list(Path(mmar_audio_dir).rglob("*.*"))
        print(f"Found {len(audio_files)} files total")

    resampled_count = 0
    failed_files = []

    for audio_file in tqdm(audio_files, desc="Resampling audio"):
        rel_path = audio_file.relative_to(mmar_audio_dir)
        output_path = Path(mmar_audio_16k_dir) / rel_path

        # Create subdirectories if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Change extension to .wav for consistency
        output_path = output_path.with_suffix('.wav')

        if resample_audio_file(str(audio_file), str(output_path)):
            resampled_count += 1
        else:
            failed_files.append(str(audio_file))

    print(f"Successfully resampled {resampled_count}/{len(audio_files)} files")
    if failed_files:
        print(f"Failed to resample {len(failed_files)} files:")
        for f in failed_files[:5]:  # Show first 5 failures
            print(f"  - {f}")

    # Process dataset annotations
    test_data = dataset["test"]
    all_annotations = []
    choice_letters = ["A", "B", "C", "D", "E", "F", "G", "H"]
    skipped_samples = []

    print(f"\nProcessing {len(test_data)} samples...")

    for i, sample in enumerate(test_data):
        audio_rel_path = sample.get("audio_path", "").replace("./", "")
        if audio_rel_path.startswith("audio/"):
            audio_rel_path = audio_rel_path[6:]

        # Point to resampled audio file
        local_audio_path = os.path.join(mmar_audio_16k_dir, audio_rel_path)

        # Change extension to .wav
        local_audio_path = str(Path(local_audio_path).with_suffix('.wav'))

        # Check if resampled file exists
        if not os.path.exists(local_audio_path):
            skipped_samples.append((i, sample.get("id", ""), audio_rel_path))
            continue

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
    if skipped_samples:
        print(f"Skipped {len(skipped_samples)} samples due to missing audio files:")
        for idx, sample_id, path in skipped_samples[:5]:
            print(f"  - Index {idx}, ID: {sample_id}, Path: {path}")

    # Create train/val/test split (60/20/20)
    import random
    random.seed(42)
    indices = list(range(len(all_annotations)))
    random.shuffle(indices)

    n_train = int(0.6 * len(indices))
    n_val = int(0.2 * len(indices))

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

    print("\n" + "=" * 60)
    print("MMAR dataset preparation complete!")
    print("=" * 60)
    print(f"Resampled audio files (16kHz): {mmar_audio_16k_dir}/")
    print(f"Annotation files: {output_dir}/")
    print(f"Total samples: {len(all_annotations)}")
    print(f"  - Train: {len(splits['train_mmar.json'])}")
    print(f"  - Valid: {len(splits['valid_mmar.json'])}")
    print(f"  - Test: {len(splits['test_mmar.json'])}")


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    data_dir = project_root / "data"

    prepare_mmar_annotations(
        output_dir=str(data_dir / "mmar"),
        download_dir=str(data_dir)
    )