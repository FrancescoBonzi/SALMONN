# prepare_librispeech.py
import os
import json
from pathlib import Path

from torchaudio.datasets import LIBRISPEECH


def prepare_librispeech_annotations(output_dir: str, download_dir: str = "./data"):
    """
    Download LibriSpeech using torchaudio and create SALMONN-compatible annotations.
    
    Args:
        output_dir: Directory to save annotation JSON files
        download_dir: Directory where LibriSpeech will be downloaded
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(download_dir, exist_ok=True)
    
    # Training splits
    train_splits = ["train-clean-100", "train-clean-360", "train-other-500"]
    
    # Validation and test splits
    val_split = "dev-clean"
    test_split = "test-clean"
    
    # Process training data
    train_annotations = []
    for split in train_splits:
        print(f"Loading {split}...")
        dataset = LIBRISPEECH(root=download_dir, url=split, download=True)
        
        for i in range(len(dataset)):
            waveform, sample_rate, transcript, speaker_id, chapter_id, utterance_id = dataset[i]
            
            # Construct the path to the audio file (torchaudio downloads to LibriSpeech subdir)
            audio_path = os.path.join(
                download_dir, 
                "LibriSpeech", 
                split, 
                str(speaker_id), 
                str(chapter_id), 
                f"{speaker_id}-{chapter_id}-{utterance_id:04d}.flac"
            )
            
            train_annotations.append({
                "path": str(Path(audio_path).absolute()),
                "text": transcript.lower(),  # LibriSpeech text is uppercase
                "task": "asr"
            })
        
        print(f"  Processed {len(dataset)} samples from {split}")
    
    # Save training annotations
    with open(os.path.join(output_dir, "train_librispeech.json"), "w") as f:
        json.dump({"annotation": train_annotations}, f, indent=2)
    print(f"Saved {len(train_annotations)} training samples")
    
    # Process validation data (dev-clean)
    print(f"Loading {val_split}...")
    val_dataset = LIBRISPEECH(root=download_dir, url=val_split, download=True)
    val_annotations = []
    
    for i in range(len(val_dataset)):
        waveform, sample_rate, transcript, speaker_id, chapter_id, utterance_id = val_dataset[i]
        
        audio_path = os.path.join(
            download_dir,
            "LibriSpeech",
            val_split,
            str(speaker_id),
            str(chapter_id),
            f"{speaker_id}-{chapter_id}-{utterance_id:04d}.flac"
        )
        
        val_annotations.append({
            "path": str(Path(audio_path).absolute()),
            "text": transcript.lower(),
            "task": "asr"
        })
    
    with open(os.path.join(output_dir, "valid_librispeech.json"), "w") as f:
        json.dump({"annotation": val_annotations}, f, indent=2)
    print(f"Saved {len(val_annotations)} validation samples")
    
    # Process test data (test-clean)
    print(f"Loading {test_split}...")
    test_dataset = LIBRISPEECH(root=download_dir, url=test_split, download=True)
    test_annotations = []
    
    for i in range(len(test_dataset)):
        waveform, sample_rate, transcript, speaker_id, chapter_id, utterance_id = test_dataset[i]
        
        audio_path = os.path.join(
            download_dir,
            "LibriSpeech",
            test_split,
            str(speaker_id),
            str(chapter_id),
            f"{speaker_id}-{chapter_id}-{utterance_id:04d}.flac"
        )
        
        test_annotations.append({
            "path": str(Path(audio_path).absolute()),
            "text": transcript.lower(),
            "task": "asr"
        })
    
    with open(os.path.join(output_dir, "test_librispeech.json"), "w") as f:
        json.dump({"annotation": test_annotations}, f, indent=2)
    print(f"Saved {len(test_annotations)} test samples")


if __name__ == "__main__":
    # Get the SALMONN project root directory (two levels up from this script)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    data_dir = project_root / "data"
    
    prepare_librispeech_annotations(
        output_dir=str(data_dir / "librispeech"),
        download_dir=str(data_dir)  # LibriSpeech will be downloaded to SALMONN/data/LibriSpeech/
    )
