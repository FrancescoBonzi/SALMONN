import os
import json
import shutil
from pathlib import Path
import zipfile
import urllib.request

import soundfile as sf
import librosa
from tqdm import tqdm
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
            sf.write(path, audio, target_sr)
            resampled += 1
        except Exception as e:
            print(f"Warning: failed to resample {path}: {e}")
    if resampled:
        print(f"Resampled {resampled} audio file(s) to {target_sr} Hz.")


def prepare_cot_gtzan_annotations(output_dir: str):
    """Load COT-GTZAN from AF-Think and write SALMONN-compatible train_gtzan.json."""
    os.makedirs(output_dir, exist_ok=True)
    output_dir = Path(output_dir)

    ds_afthink_stream = load_dataset("nvidia/AF-Think", "afthink", streaming=True)
    train_annotations = []
    for item in tqdm(ds_afthink_stream["af_cot_train_gtzan"]):
        sound_name = Path(item["sound"])
        audio_path = str(output_dir / "audio_files" / sound_name)
        train_annotations.append({
            "path": audio_path,
            "task": "reasoning",
            "question": item["conversations"][0]["value"].replace("<sound>", "").replace("Output the answer with <SUMMARY>, <CAPTION>, <REASONING>, and <CONCLUSION> tags.", "").strip(),
            "answer": item["conversations"][1]["value"].strip(),
        })

    ann_path = output_dir / "annotations" / "train_gtzan.json"
    os.makedirs(ann_path.parent, exist_ok=True)
    with open(ann_path, "w") as f:
        json.dump({"annotation": train_annotations}, f, indent=2)
    print(f"Saved {len(train_annotations)} samples to {ann_path}")


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent.parent / "data" / "GTZAN"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Download and extract master.zip (GTZAN, GitHub)
    url = "https://github.com/EsratMaria/MusicGenreRecogniton/archive/refs/heads/master.zip"
    archive_path = data_dir / "master.zip"
    if not archive_path.exists():
        print("Downloading master.zip...")
        urllib.request.urlretrieve(url, archive_path)

    print("Extracting...")
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(data_dir)

    audio_files_dir = data_dir / "audio_files"
    audio_files_dir.mkdir(parents=True, exist_ok=True)

    # Move genres to audio_files/
    genres_src = data_dir / "MusicGenreRecogniton-master" / "genres"
    if genres_src.exists():
        shutil.move(str(genres_src), str(audio_files_dir))

    # Remove extracted directory
    extracted_dir = data_dir / "MusicGenreRecogniton-master"
    if extracted_dir.exists():
        shutil.rmtree(extracted_dir)
        print("Removed MusicGenreRecogniton-master")

    print("Resampling audio to 16 kHz for Whisper...")
    resample_audio_dir(audio_files_dir)

    prepare_cot_gtzan_annotations(str(data_dir))

    if archive_path.exists():
        archive_path.unlink()
        print("Removed", archive_path.name)

    print("Done. Dataset in", data_dir)
