import os

# Limit BLAS/OMP threads to avoid "pthread_create failed" when using ProcessPoolExecutor.
# Each worker already runs in parallel; multithreading inside workers causes thread explosion.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import sys
from pathlib import Path
import shutil
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Optional, Tuple
import urllib.request
import random
import soundfile as sf
import librosa
from tqdm import tqdm

TARGET_SAMPLE_RATE = 16000  # Whisper expects 16 kHz


def _resample_one(wav_path: str, target_sr: int) -> Tuple[int, Optional[str]]:
    """
    Resample a single file. 
    Returns (1 if processed, error message or None).
    """
    try:
        # Use librosa.load to handle MP3s, resampling, and mono conversion in one step.
        mp3_path = wav_path.replace(".wav", ".mp3")
        audio, _ = librosa.load(mp3_path, sr=target_sr, mono=True)

        # Write to WAV. 
        sf.write(wav_path, audio, target_sr, subtype='PCM_16')
        
        # Remove original mp3
        os.remove(mp3_path) 
        
        return (1, None)

    except Exception as e:
        return (0, f"Warning: failed to resample {wav_path}: {e}")


def resample_audio_files(
    paths: List[str],
    target_sr: int = TARGET_SAMPLE_RATE,
    n_workers: Optional[int] = None,
):
    """
    Resample audio files to target_sr (16 kHz for Whisper).
    Converts to mono and saves as WAV.
    """
    n_workers = n_workers or min(32, (os.cpu_count() or 8))
    
    print(f"Resampling {len(paths)} files with {n_workers} workers...", flush=True)
    
    resampled_count = 0
    
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        # Map futures to paths
        fut_to_path = {pool.submit(_resample_one, p, target_sr): p for p in paths}
        
        pbar = tqdm(
            as_completed(fut_to_path),
            total=len(paths),
            desc="Resampling",
            unit="file",
            file=sys.stderr,
        )
        
        for f in pbar:
            count, err = f.result()
            resampled_count += count
            
            if err:
                tqdm.write(err, file=sys.stderr)
            else:
                pbar.set_postfix_str(f"cnt={resampled_count}")

    print(f"\nProcessing complete. {resampled_count}/{len(paths)} files converted.", flush=True)


def prepare_cot_youtube8m_annotations(output_dir: str):
    """Load COT-YouTube8M from AF-Think and write SALMONN-compatible train_youtube8m.json and test_youtube8m.json."""
    os.makedirs(output_dir, exist_ok=True)
    output_dir = Path(output_dir)

    # Download labels with CoT
    url = "https://huggingface.co/datasets/nvidia/AF-Think/resolve/main/afcot/YouTube8M.json"
    json_path = output_dir / "YouTube8M.json"
    if not json_path.exists():
        print("Downloading YouTube8M.json...")
        urllib.request.urlretrieve(url, json_path)

    with open(json_path, "r") as f:
        data = json.load(f)

    print("Processing YouTube8M annotations...")

    # Build set of existing audio filenames once (O(1) lookup vs os.path.exists per item)
    audio_dir = output_dir / "audio_files"
    existing = set(p.name for p in audio_dir.iterdir() if p.is_file()) if audio_dir.exists() else set()

    annotations = []
    suffix_remove = "Output the answer with <SUMMARY>, <CAPTION>, <REASONING>, and <CONCLUSION> tags."
    for item in tqdm(data):
        filename = item["sound"].split("/")[-1]
        if filename not in existing:
            continue
        question = item["conversations"][0]["value"].replace("<sound>", "").replace(suffix_remove, "").strip()
        annotations.append({
            "path": str(audio_dir / filename.replace(".mp3", ".wav")),
            "task": "reasoning",
            "answer": item["conversations"][1]["value"].strip(),
            "question": question,
        })
    
    # Split in train and test
    random.seed(42)
    random.shuffle(annotations)
    train_annotations = annotations[:int(len(annotations) * 0.8)]
    test_annotations = annotations[int(len(annotations) * 0.8):]

    # Save annotations
    train_ann_path = output_dir / "annotations" / "train_youtube8m.json"
    test_ann_path = output_dir / "annotations" / "test_youtube8m.json"
    os.makedirs(train_ann_path.parent, exist_ok=True)
    os.makedirs(test_ann_path.parent, exist_ok=True)
    with open(train_ann_path, "w") as f:
        json.dump({"annotation": train_annotations}, f, indent=2)
    print(f"Saved {len(train_annotations)} samples to {train_ann_path}")
    with open(test_ann_path, "w") as f:
        json.dump({"annotation": test_annotations}, f, indent=2)
    print(f"Saved {len(test_annotations)} samples to {test_ann_path}")

    # Remove YouTube8M.json
    #if json_path.exists():
    #    json_path.unlink()
    #    print("Removed", json_path.name)

    return train_annotations, test_annotations


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent.parent / "data" / "YouTube8M"
    data_dir.mkdir(parents=True, exist_ok=True)
    archive_path = data_dir / "AFThink.tar.gz"

    # Extract AFThink.tar.gz
    print("Extracting AFThink.tar.gz...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(data_dir)

    # Rename audio_files to audio_files/
    audio_files_dir = data_dir / "AFThink"
    if audio_files_dir.exists():
        shutil.move(str(audio_files_dir), str(data_dir / "audio_files"))

    print("Generating annotations...")
    train_annotations, test_annotations = prepare_cot_youtube8m_annotations(str(data_dir))

    print("Resampling train audio files to 16 kHz for Whisper...")
    resample_audio_files([ann["path"] for ann in train_annotations])
    print("Resampling test audio files to 16 kHz for Whisper...")
    resample_audio_files([ann["path"] for ann in test_annotations])

    #if archive_path.exists():
    #    archive_path.unlink()
    #    print("Removed", archive_path.name)

    print("Done. Dataset in", data_dir)
