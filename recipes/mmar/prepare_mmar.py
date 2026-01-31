import os
import json
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import shutil


def prepare_mmar_annotations(output_dir: str, download_dir: str = "./data"):
    """
    Download MMAR dataset from HuggingFace and create SALMONN-compatible annotations.

    MMAR (Music Multiple-choice Question Answering) is a dataset for music understanding
    with multiple-choice questions about music audio clips.

    Args:
        output_dir: Directory to save annotation JSON files
        download_dir: Directory where MMAR will be downloaded/cached
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(download_dir, exist_ok=True)

    mmar_audio_dir = os.path.join(download_dir, "MMAR")
    os.makedirs(mmar_audio_dir, exist_ok=True)

    print("Loading MMAR dataset from HuggingFace...")

    # Load the metadata
    try:
        dataset = load_dataset("BoJack/MMAR", cache_dir=download_dir)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Make sure you have the 'datasets' library installed: pip install datasets")
        return

    # MMAR only has a test split
    if "test" not in dataset:
        print("Error: 'test' split not found in dataset")
        return

    print("\nDownloading MMAR audio files...")
    print("Note: This may take a while as audio files are ~698MB")

    # Download the audio zip file from the HuggingFace repo
    try:
        # The audio files are stored in a separate repository structure
        # We'll download them using hf_hub_download
        audio_zip_path = hf_hub_download(
            repo_id="BoJack/MMAR",
            filename="audio.zip",
            repo_type="dataset",
            cache_dir=download_dir
        )

        # Extract audio files
        import zipfile
        print(f"Extracting audio files to {mmar_audio_dir}...")
        with zipfile.ZipFile(audio_zip_path, 'r') as zip_ref:
            zip_ref.extractall(mmar_audio_dir)
        print("Audio extraction complete!")

    except Exception as e:
        print(f"Warning: Could not download audio.zip: {e}")
        print("You may need to manually download audio files from:")
        print("https://huggingface.co/datasets/BoJack/MMAR")
        print("Continuing with metadata processing...")

    # Process the test split
    test_data = dataset["test"]
    all_annotations = []

    print(f"\nProcessing test split ({len(test_data)} samples)...")

    for i, sample in enumerate(test_data):
        # MMAR dataset structure:
        # - id: unique identifier
        # - audio_path: relative path to audio file (e.g., "./audio/xxx.wav")
        # - question: the question text
        # - choices: list of answer choices
        # - answer: the correct answer text
        # - modality, category, sub-category, language, source, url, timestamp

        # Get the audio path and convert it to absolute path
        audio_rel_path = sample.get("audio_path", "")
        # Remove the "./" prefix if present
        audio_rel_path = audio_rel_path.replace("./", "")
        audio_path = os.path.join(mmar_audio_dir, audio_rel_path)

        # Format choices as a string
        choices = sample.get("choices", [])
        choice_letters = ["A", "B", "C", "D", "E", "F", "G", "H"]

        if isinstance(choices, list):
            # Format as "A. choice1, B. choice2, C. choice3, D. choice4"
            formatted_choices = ", ".join([
                f"{choice_letters[j]}. {choice}"
                for j, choice in enumerate(choices)
            ])
        else:
            formatted_choices = str(choices)

        # Get the answer text
        answer_text = sample.get("answer", "")

        # Try to convert answer to letter format if it matches a choice
        answer_letter = None
        for j, choice in enumerate(choices):
            if choice.strip().lower() == answer_text.strip().lower():
                answer_letter = choice_letters[j]
                break

        # If no match found, keep the original answer
        if answer_letter is None:
            answer_letter = answer_text

        # Get question text
        question_text = sample.get("question", "")

        # Create the complete text field that combines question and choices
        # This will be formatted into the prompt template during training
        text = f"{question_text} {formatted_choices}"

        # Annotation structure matching LibriSpeech format
        annotation = {
            "path": audio_path,
            "text": text,  # Combined question and choices
            "answer": str(answer_letter),  # The correct answer
            "task": "mmar",
            # Optional: store additional metadata
            "id": sample.get("id", ""),
            "modality": sample.get("modality", ""),
            "category": sample.get("category", ""),
        }

        all_annotations.append(annotation)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(test_data)} samples...")

    print(f"\nTotal samples processed: {len(all_annotations)}")

    # Since MMAR only has test split, we'll create a suggested train/val/test split
    # 60% train, 20% val, 20% test
    import random
    random.seed(42)  # For reproducibility

    indices = list(range(len(all_annotations)))
    random.shuffle(indices)

    n_train = int(0.6 * len(indices))
    n_val = int(0.2 * len(indices))

    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    train_annotations = [all_annotations[i] for i in train_indices]
    val_annotations = [all_annotations[i] for i in val_indices]
    test_annotations = [all_annotations[i] for i in test_indices]

    # Save splits
    splits = {
        "train_mmar.json": train_annotations,
        "valid_mmar.json": val_annotations,
        "test_mmar.json": test_annotations
    }

    for filename, annotations in splits.items():
        output_path = os.path.join(output_dir, filename)
        with open(output_path, "w") as f:
            json.dump({"annotation": annotations}, f, indent=2)
        print(f"Saved {len(annotations)} samples to {output_path}")

    print("\nMMAR dataset preparation complete!")
    print("\nNote on usage:")
    print("- MMAR originally has only a test split")
    print("- We've split it into train (60%), val (20%), test (20%) with seed=42")
    print("- Annotations contain 'text' field with question and choices")
    print("- Prompts are loaded from prompts/train_prompt.json during training")
    print("- Make sure to update prompts/train_prompt.json with MMAR prompts")
    print("\nAudio file location:")
    print(f"- Audio files should be in: {mmar_audio_dir}/audio/")


if __name__ == "__main__":
    # Get the SALMONN project root directory (two levels up from this script)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    data_dir = project_root / "data"

    prepare_mmar_annotations(
        output_dir=str(data_dir / "mmar"),
        download_dir=str(data_dir)  # MMAR will be cached in SALMONN/data/
    )