"""
BLAB (Brutally Long Audio Bench) dataset preparation script.

Steps:
  1. download_blab_audio   – download audio from YouTube URLs via yt-dlp
  2. prepare_blab_annotations – load each task from HuggingFace, write JSON

Usage:
    # Download audio only
    python prepare_blab.py --step download

    # Build annotation JSONs only (audio already downloaded)
    python prepare_blab.py --step annotate

    # Both (default)
    python prepare_blab.py

Requirements:
    pip install datasets yt-dlp soundfile librosa tqdm numpy
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TARGET_SAMPLE_RATE = 16_000  # Whisper / most audio-LMs expect 16 kHz

ALL_TASKS = [
    "word_localization",
    "named_entity_localization",
    "advertisement_localization",
    "speaker_number_estimation",
    "entire_duration",
    "event_duration",
    "emotion_reasoning",
    "emotion_ranking",
]

AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".webm", ".opus"}


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def resample_audio_dir(root_dir: Path, target_sr: int = TARGET_SAMPLE_RATE):
    """Resample all audio files under root_dir to target_sr in-place."""
    root_dir = Path(root_dir)
    resampled = 0
    for path in root_dir.rglob("*"):
        if path.suffix.lower() not in AUDIO_EXTENSIONS or not path.is_file():
            continue
        try:
            audio, sr = sf.read(path)
            if sr == target_sr:
                continue
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            audio = audio / (np.abs(audio).max() + 1e-9)
            # Write back as wav (guarantees soundfile can re-read it)
            wav_path = path.with_suffix(".wav")
            sf.write(wav_path, audio, target_sr)
            if wav_path != path:
                path.unlink()
            resampled += 1
        except Exception as e:
            print(f"  Warning: failed to resample {path}: {e}")
    if resampled:
        print(f"Resampled {resampled} audio file(s) to {target_sr} Hz.")


# ---------------------------------------------------------------------------
# Step 1 – Download audio from YouTube
# ---------------------------------------------------------------------------

def _check_ytdlp():
    """Verify yt-dlp is available."""
    try:
        subprocess.run(
            ["yt-dlp", "--version"],
            check=True,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ERROR: yt-dlp not found. Install it with:  pip install yt-dlp")
        sys.exit(1)


def download_blab_audio(audio_dir: Path, tasks: list[str] | None = None, resample: bool = True):
    """
    For every task in *tasks*, load the HuggingFace dataset, collect all
    unique YouTube URLs, and download each audio file with yt-dlp.

    Files are saved as  <audio_dir>/<video_id>.wav
    """
    _check_ytdlp()
    audio_dir.mkdir(parents=True, exist_ok=True)
    tasks = tasks or ALL_TASKS

    # Collect unique URLs across all tasks
    url_set: set[str] = set()
    print("Loading dataset metadata from HuggingFace …")
    for task in tasks:
        try:
            ds = load_dataset("oreva/blab_long_audio", task, trust_remote_code=True)
        except Exception as e:
            print(f"  Warning: could not load task '{task}': {e}")
            continue
        for split in ds.values():
            for row in split:
                url = row.get("audio_url") or row.get("url") or row.get("youtube_url")
                if url:
                    url_set.add(url)

    print(f"Found {len(url_set)} unique audio URLs to download.")

    failed: list[str] = []
    for url in tqdm(sorted(url_set), desc="Downloading audio"):
        # Derive a stable filename from the video ID
        video_id = url.split("v=")[-1].split("&")[0] if "v=" in url else url.split("/")[-1]
        out_path = audio_dir / f"{video_id}.wav"

        if out_path.exists():
            continue  # already downloaded

        cmd = [
            "yt-dlp",
            "--no-playlist",
            "--extract-audio",
            "--audio-format", "wav",
            "--audio-quality", "0",         # best quality
            "--postprocessor-args", f"-ar {TARGET_SAMPLE_RATE}",   # resample on the fly
            "-o", str(audio_dir / f"{video_id}.%(ext)s"),
            url,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"\n  FAILED: {url}\n  {result.stderr[-400:]}")
            failed.append(url)

    if failed:
        fail_log = audio_dir / "failed_downloads.txt"
        fail_log.write_text("\n".join(failed))
        print(f"\n{len(failed)} download(s) failed. URLs saved to {fail_log}")

    if resample:
        print("Resampling any non-16 kHz files …")
        resample_audio_dir(audio_dir)

    print(f"Audio download complete. Files in: {audio_dir}")


# ---------------------------------------------------------------------------
# Step 2 – Build annotation JSONs
# ---------------------------------------------------------------------------

def prepare_blab_annotations(data_dir: Path, audio_dir: Path, tasks: list[str] | None = None):
    """
    For every task, load the HuggingFace dataset and write
      <data_dir>/annotations/<task>.json
    with the schema:
      { "annotation": [ { "path": ..., "answer": ..., <other fields> }, ... ] }
    """
    ann_dir = data_dir / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    tasks = tasks or ALL_TASKS

    for task in tasks:
        print(f"\nProcessing task: {task}")
        try:
            ds = load_dataset("oreva/blab_long_audio", task, trust_remote_code=True)
        except Exception as e:
            print(f"  Warning: could not load task '{task}': {e}")
            continue

        for split_name, split in ds.items():
            annotations = []
            for row in tqdm(split, desc=f"  {split_name}"):
                # Resolve audio path from URL
                url = row.get("audio_url") or row.get("url") or row.get("youtube_url") or ""
                if url:
                    video_id = url.split("v=")[-1].split("&")[0] if "v=" in url else url.split("/")[-1]
                    audio_path = str(audio_dir / f"{video_id}.wav")
                else:
                    audio_path = ""

                # Build annotation entry – keep all original fields
                entry = {"path": audio_path}
                for k, v in row.items():
                    entry[k] = v

                # Normalise answer field name
                if "answer" not in entry:
                    for candidate in ("answers", "label", "labels", "ground_truth"):
                        if candidate in entry:
                            entry["answer"] = entry[candidate]
                            break

                annotations.append(entry)

            out_path = ann_dir / f"{split_name}_{task}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"annotation": annotations}, f, indent=2, ensure_ascii=False)
            print(f"  Saved {len(annotations)} samples → {out_path}")

    print(f"\nAll annotation files written to: {ann_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Prepare BLAB dataset for evaluation.")
    p.add_argument(
        "--step",
        choices=["download", "annotate", "all"],
        default="all",
        help="Which step(s) to run (default: all)",
    )
    p.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        metavar="TASK",
        help=f"Subset of tasks to process. Defaults to all: {ALL_TASKS}",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Root data directory (default: ../../data/BLAB relative to this script)",
    )
    p.add_argument(
        "--no-resample",
        action="store_true",
        help="Skip resampling to 16 kHz after download",
    )
    return p.parse_args()


def main():
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    data_dir: Path = args.data_dir or (script_dir.parent.parent / "data" / "BLAB")
    audio_dir = data_dir / "audio_files"
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data directory : {data_dir}")
    print(f"Audio directory: {audio_dir}")
    print(f"Tasks          : {args.tasks or 'ALL'}")
    print()

    if args.step in ("download", "all"):
        download_blab_audio(
            audio_dir=audio_dir,
            tasks=args.tasks,
            resample=not args.no_resample,
        )

    if args.step in ("annotate", "all"):
        prepare_blab_annotations(
            data_dir=data_dir,
            audio_dir=audio_dir,
            tasks=args.tasks,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()