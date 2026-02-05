import os
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile
import tarfile

import soundfile as sf
import librosa
from tqdm import tqdm

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
            print(f"Resampled {path} to {target_sr} Hz.")
            sf.write(path, audio, target_sr)
            resampled += 1
        except Exception as e:
            print(f"Warning: failed to resample {path}: {e}")
    if resampled:
        print(f"Resampled {resampled} audio file(s) to {target_sr} Hz.")


def prepare_mmau_annotations(output_dir: str):
    """Load MMAU from Hugging Face and write train_mmau.json."""
    os.makedirs(output_dir, exist_ok=True)
    output_dir = Path(output_dir)

    url = "https://github.com/Sakshi113/MMAU/archive/refs/heads/main.zip"
    archive_path = output_dir / "main.zip"
    if not archive_path.exists():
        print("Downloading main.zip...")
        urllib.request.urlretrieve(url, archive_path)
    print("Extracting...")
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(output_dir)

    test_mini_json = output_dir / "MMAU-main" / "mmau-test-mini.json"
    with open(test_mini_json, "r") as f:
        test_mini_data = json.load(f)

    annotations = []
    for item in tqdm(test_mini_data):
        audio_path = str(output_dir / "audio_files" / Path(item["id"] + ".wav"))
        annotations.append({
            "path": audio_path,
            "text": item["answer"],
            "task": "mmau",
            **{key: value for key, value in item.items() if key != "audio_id" and key != "answer" and key != "task"}
        })

    ann_path = output_dir / "annotations" / "test_mmau.json"
    os.makedirs(ann_path.parent, exist_ok=True)
    with open(ann_path, "w") as f:
        json.dump({"annotation": annotations}, f, indent=2)
    print(f"Saved {len(annotations)} samples to {ann_path}")

    if (output_dir / "MMAU-main").exists():
        shutil.rmtree(output_dir / "MMAU-main")
        print("Removed", output_dir / "MMAU-main")

    if (output_dir / "main.zip").exists():
        os.remove(output_dir / "main.zip")
        print("Removed", output_dir / "main.zip")


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent.parent / "data" / "MMAU"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Extract MMAU test-mini-audios.tar
    archive_path = data_dir / "test-mini-audios.tar"
    if not archive_path.exists():
        raise FileNotFoundError(f"MMAU test-mini-audios.tar not found at {archive_path}, \
            downlad it from https://drive.usercontent.google.com/download?id=1fERNIyTa0HWry6iIG1X-1ACPlUlhlRWA&export=download&authuser=0")

    audio_files_dir = data_dir / "audio_files"
    with tarfile.open(archive_path, "r") as tf:
        tf.extractall(data_dir)

    # Rename test-mini-audios/ to audio_files/
    shutil.move(data_dir / "test-mini-audios", audio_files_dir)

    print("Resampling audio to 16 kHz for Whisper...")
    resample_audio_dir(audio_files_dir)

    prepare_mmau_annotations(str(data_dir))

    #if archive_path.exists():
    #    archive_path.unlink()
    #    print("Removed", archive_path.name)

    print("Done. Dataset in", data_dir)