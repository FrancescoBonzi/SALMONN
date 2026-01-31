import os
import json
from pathlib import Path
from datasets import load_dataset
import torch
import torchaudio


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

    print("Loading MMAR dataset from HuggingFace...")

    # Load the dataset - MMAR has train, validation, and test splits
    try:
        dataset = load_dataset("BoJack/MMAR", cache_dir=download_dir)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Make sure you have the 'datasets' library installed: pip install datasets")
        return

    # Process each split
    for split_name in ["train", "validation", "test"]:
        if split_name not in dataset:
            print(f"Warning: {split_name} split not found in dataset")
            continue

        split_data = dataset[split_name]
        annotations = []

        print(f"\nProcessing {split_name} split...")

        for i, sample in enumerate(split_data):
            # MMAR dataset structure:
            # - audio: audio file
            # - question: the question text
            # - choices: list of answer choices (typically A, B, C, D)
            # - answer: the correct answer (index or letter)

            # Save audio file to disk using torchaudio
            audio_filename = f"{split_name}_{i:06d}.wav"
            audio_path = os.path.join(download_dir, "MMAR", split_name, audio_filename)
            os.makedirs(os.path.dirname(audio_path), exist_ok=True)

            # Save the audio file using torchaudio
            audio_data = sample["audio"]
            if "array" in audio_data and "sampling_rate" in audio_data:
                # Convert numpy array to torch tensor
                waveform = torch.from_numpy(audio_data["array"]).float()
                # Ensure waveform is 2D (channels, samples)
                if waveform.dim() == 1:
                    waveform = waveform.unsqueeze(0)
                # Save using torchaudio
                torchaudio.save(audio_path, waveform, audio_data["sampling_rate"])
            elif "path" in audio_data and audio_data["path"]:
                # If audio is already a file, load and save it
                waveform, sr = torchaudio.load(audio_data["path"])
                torchaudio.save(audio_path, waveform, sr)

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

            # Get the answer
            answer = sample.get("answer", "")
            # Convert answer index to letter if it's a number
            if isinstance(answer, int):
                answer = choice_letters[answer] if answer < len(choice_letters) else str(answer)

            # Get question text
            question_text = sample.get("question", "")

            # Create the complete text field that combines question and choices
            # This will be formatted into the prompt template during training
            text = f"{question_text} {formatted_choices}"

            # Annotation structure matching LibriSpeech format
            annotation = {
                "path": audio_path,
                "text": text,  # Combined question and choices
                "answer": str(answer),  # The correct answer (A, B, C, or D)
                "task": "mmar"
            }

            annotations.append(annotation)

            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(split_data)} samples...")

        # Save annotations for this split
        if split_name == "train":
            output_filename = "train_mmar.json"
        elif split_name == "validation":
            output_filename = "valid_mmar.json"
        else:  # test
            output_filename = "test_mmar.json"

        output_path = os.path.join(output_dir, output_filename)

        with open(output_path, "w") as f:
            json.dump({"annotation": annotations}, f, indent=2)

        print(f"Saved {len(annotations)} {split_name} samples to {output_path}")

    print("\nMMAR dataset preparation complete!")
    print("\nNote on usage:")
    print("- Annotations contain 'text' field with question and choices")
    print("- Prompts are loaded from prompts/train_prompt.json during training")
    print("- The 'text' field gets inserted into the prompt template via .format()")
    print("- Make sure to update prompts/train_prompt.json with MMAR prompts")


if __name__ == "__main__":
    # Get the SALMONN project root directory (two levels up from this script)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    data_dir = project_root / "data"

    prepare_mmar_annotations(
        output_dir=str(data_dir / "mmar"),
        download_dir=str(data_dir)  # MMAR will be cached in SALMONN/data/
    )