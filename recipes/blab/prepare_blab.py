"""
BLAB (Brutally Long Audio Bench) dataset preparation script.

The dataset uses a custom loading script that is no longer supported by
newer versions of `datasets`. This script bypasses it entirely by:
  1. Using huggingface_hub to snapshot the raw repo files.
  2. Auto-detecting the data file format (JSON / JSONL / CSV / Parquet)
     from the downloaded files and reading them with pandas/pyarrow.
  3. Downloading YouTube audio via yt-dlp.
  4. Writing per-task annotation JSONs.

Steps (can be run independently):
  --step download   Download audio from YouTube URLs
  --step annotate   Build annotation JSONs from raw data files
  --step all        Both (default)

Usage:
    pip install huggingface_hub datasets pandas pyarrow yt-dlp soundfile librosa tqdm numpy

    python prepare_blab.py                          # full pipeline
    python prepare_blab.py --step download          # audio only
    python prepare_blab.py --step annotate          # annotations only
    python prepare_blab.py --tasks word_localization entire_duration
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
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TARGET_SAMPLE_RATE = 16_000

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

HF_REPO_ID = "oreva/blab_long_audio"


# ---------------------------------------------------------------------------
# Repo snapshot (bypasses the broken loading script)
# ---------------------------------------------------------------------------

def snapshot_repo(cache_dir: Path) -> Path:
    """
    Download the raw HuggingFace repo files (everything except model weights)
    using huggingface_hub.snapshot_download.  Returns the local repo root.
    """
    from huggingface_hub import snapshot_download

    print(f"Snapshotting HF repo '{HF_REPO_ID}' into {cache_dir} …")
    repo_dir = snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        local_dir=str(cache_dir / "hf_repo"),
        # Exclude large binary blobs that aren't data tables
        ignore_patterns=["*.bin", "*.pt", "*.safetensors"],
    )
    print(f"Repo files available at: {repo_dir}")
    return Path(repo_dir)


# ---------------------------------------------------------------------------
# Data file discovery & parsing
# ---------------------------------------------------------------------------

def _read_data_file(path: Path) -> list[dict]:
    """Read a single data file (jsonl / json / csv / parquet) into a list of dicts."""
    suf = path.suffix.lower()
    if suf in (".jsonl", ".ndjson"):
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if suf == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        # May be a list or a dict with a key containing the list
        if isinstance(data, list):
            return data
        for v in data.values():
            if isinstance(v, list):
                return v
        return [data]
    if suf == ".csv":
        import csv
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    if suf == ".parquet":
        import pyarrow.parquet as pq
        tbl = pq.read_table(path)
        return tbl.to_pydict()  # returns dict-of-lists; convert below
    raise ValueError(f"Unsupported format: {path}")


def _parquet_dict_to_rows(d: dict) -> list[dict]:
    """Convert pyarrow dict-of-lists to list-of-dicts."""
    keys = list(d.keys())
    n = len(d[keys[0]])
    return [{k: d[k][i] for k in keys} for i in range(n)]


def load_task_data(repo_dir: Path, task: str) -> dict[str, list[dict]]:
    """
    Find and load all data files for *task* under the repo directory.
    Returns a dict  { split_name -> [row, ...] }.
    """
    repo_dir = Path(repo_dir)
    result: dict[str, list[dict]] = {}

    # Strategy 1: look for files/dirs named after the task
    candidates: list[tuple[str, Path]] = []

    for path in sorted(repo_dir.rglob("*")):
        if not path.is_file():
            continue
        name_lower = path.stem.lower()
        suf = path.suffix.lower()
        if suf not in (".json", ".jsonl", ".ndjson", ".csv", ".parquet"):
            continue
        if task.lower() in name_lower or name_lower in task.lower():
            # Infer split from filename or parent dir name
            split = "test"
            for s in ("train", "validation", "val", "dev", "test"):
                if s in name_lower or s in path.parent.name.lower():
                    split = s
                    break
            candidates.append((split, path))

    # Strategy 2: look in a subdirectory whose name matches the task
    task_dir = repo_dir / task
    if task_dir.is_dir():
        for path in sorted(task_dir.rglob("*")):
            if not path.is_file():
                continue
            suf = path.suffix.lower()
            if suf not in (".json", ".jsonl", ".ndjson", ".csv", ".parquet"):
                continue
            split = path.stem.lower()
            if split not in ("train", "validation", "val", "dev", "test"):
                split = "test"
            candidates.append((split, path))

    if not candidates:
        print(f"  Warning: no data files found for task '{task}' in {repo_dir}")
        return result

    for split, path in candidates:
        try:
            raw = _read_data_file(path)
            if isinstance(raw, dict):          # parquet dict-of-lists
                rows = _parquet_dict_to_rows(raw)
            else:
                rows = raw
            result.setdefault(split, []).extend(rows)
            print(f"  Loaded {len(rows)} rows from {path.relative_to(repo_dir)}")
        except Exception as e:
            print(f"  Warning: could not read {path}: {e}")

    return result


def collect_all_urls(repo_dir: Path, tasks: list[str]) -> set[str]:
    """Scan all task data files and return every unique YouTube URL."""
    url_set: set[str] = set()
    url_keys = ("audio_url", "url", "youtube_url", "yt_url", "link")

    for task in tasks:
        splits = load_task_data(repo_dir, task)
        for rows in splits.values():
            for row in rows:
                for k in url_keys:
                    v = row.get(k)
                    if v and isinstance(v, str) and ("youtube" in v or "youtu.be" in v):
                        url_set.add(v)
    return url_set


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def resample_audio_dir(root_dir: Path, target_sr: int = TARGET_SAMPLE_RATE):
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
            wav_path = path.with_suffix(".wav")
            sf.write(wav_path, audio, target_sr)
            if wav_path != path:
                path.unlink()
            resampled += 1
        except Exception as e:
            print(f"  Warning: failed to resample {path}: {e}")
    if resampled:
        print(f"Resampled {resampled} file(s) to {target_sr} Hz.")


# ---------------------------------------------------------------------------
# Step 1 – Download audio
# ---------------------------------------------------------------------------

def _check_ytdlp():
    try:
        subprocess.run(["yt-dlp", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ERROR: yt-dlp not found.  Install with:  pip install yt-dlp")
        sys.exit(1)


def _video_id_from_url(url: str) -> str:
    if "v=" in url:
        return url.split("v=")[-1].split("&")[0]
    if "youtu.be/" in url:
        return url.split("youtu.be/")[-1].split("?")[0]
    return url.split("/")[-1].split("?")[0]


def download_blab_audio(
    repo_dir: Path,
    audio_dir: Path,
    tasks: list[str],
    resample: bool = True,
):
    _check_ytdlp()
    audio_dir.mkdir(parents=True, exist_ok=True)

    print("Collecting YouTube URLs from repo data files …")
    url_set = collect_all_urls(repo_dir, tasks)
    print(f"Found {len(url_set)} unique URL(s).")

    if not url_set:
        print("No URLs found – check repo structure above.")
        return

    failed: list[str] = []
    for url in tqdm(sorted(url_set), desc="Downloading audio"):
        vid = _video_id_from_url(url)
        out_wav = audio_dir / f"{vid}.wav"
        if out_wav.exists():
            continue

        cmd = [
            "yt-dlp",
            "--no-playlist",
            "--extract-audio",
            "--audio-format", "wav",
            "--audio-quality", "0",
            "--postprocessor-args", f"-ar {TARGET_SAMPLE_RATE}",
            "-o", str(audio_dir / f"{vid}.%(ext)s"),
            url,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"\n  FAILED: {url}\n  {res.stderr[-400:]}")
            failed.append(url)

    if failed:
        fail_log = audio_dir / "failed_downloads.txt"
        fail_log.write_text("\n".join(failed))
        print(f"\n{len(failed)} failure(s). Saved to {fail_log}")

    if resample:
        print("Resampling audio to 16 kHz …")
        resample_audio_dir(audio_dir)

    print(f"Audio complete. Files in: {audio_dir}")


# ---------------------------------------------------------------------------
# Step 2 – Build annotation JSONs
# ---------------------------------------------------------------------------

def prepare_blab_annotations(
    repo_dir: Path,
    data_dir: Path,
    audio_dir: Path,
    tasks: list[str],
):
    ann_dir = data_dir / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    url_keys = ("audio_url", "url", "youtube_url", "yt_url", "link")

    for task in tasks:
        print(f"\nProcessing task: {task}")
        splits = load_task_data(repo_dir, task)
        if not splits:
            continue

        for split_name, rows in splits.items():
            annotations = []
            for row in tqdm(rows, desc=f"  {split_name}"):
                # Resolve audio path
                url = ""
                for k in url_keys:
                    v = row.get(k)
                    if v and isinstance(v, str):
                        url = v
                        break
                if url:
                    vid = _video_id_from_url(url)
                    audio_path = str(audio_dir / f"{vid}.wav")
                else:
                    audio_path = ""

                entry = {"path": audio_path}
                entry.update(row)

                # Normalise answer field
                if "answer" not in entry:
                    for candidate in ("answers", "label", "labels", "ground_truth", "response"):
                        if candidate in entry:
                            entry["answer"] = entry[candidate]
                            break

                annotations.append(entry)

            out_path = ann_dir / f"{split_name}_{task}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"annotation": annotations}, f, indent=2, ensure_ascii=False)
            print(f"  Saved {len(annotations)} samples → {out_path}")

    print(f"\nAnnotations written to: {ann_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Prepare BLAB dataset for evaluation.")
    p.add_argument(
        "--step", choices=["download", "annotate", "all"], default="all",
        help="Which step(s) to run (default: all)",
    )
    p.add_argument(
        "--tasks", nargs="+", default=None, metavar="TASK",
        help=f"Tasks to process (default: all). Choices: {ALL_TASKS}",
    )
    p.add_argument(
        "--data-dir", type=Path, default=None,
        help="Root data directory (default: ../../data/BLAB relative to this script)",
    )
    p.add_argument(
        "--no-resample", action="store_true",
        help="Skip resampling to 16 kHz after download",
    )
    return p.parse_args()


def main():
    args = parse_args()
    tasks = args.tasks or ALL_TASKS

    script_dir = Path(__file__).resolve().parent
    data_dir: Path = args.data_dir or (script_dir.parent.parent / "data" / "BLAB")
    audio_dir = data_dir / "audio_files"
    cache_dir = data_dir / "_cache"
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data directory : {data_dir}")
    print(f"Audio directory: {audio_dir}")
    print(f"Tasks          : {tasks}\n")

    # Always need the repo snapshot for both steps
    repo_dir = snapshot_repo(cache_dir)

    # Print repo contents so we can debug if data discovery fails
    print("\nRepo files discovered:")
    for p in sorted(Path(repo_dir).rglob("*")):
        if p.is_file() and p.suffix.lower() in (".json", ".jsonl", ".csv", ".parquet", ".py", ".md"):
            print(f"  {p.relative_to(repo_dir)}")
    print()

    if args.step in ("download", "all"):
        download_blab_audio(
            repo_dir=repo_dir,
            audio_dir=audio_dir,
            tasks=tasks,
            resample=not args.no_resample,
        )

    if args.step in ("annotate", "all"):
        prepare_blab_annotations(
            repo_dir=repo_dir,
            data_dir=data_dir,
            audio_dir=audio_dir,
            tasks=tasks,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()