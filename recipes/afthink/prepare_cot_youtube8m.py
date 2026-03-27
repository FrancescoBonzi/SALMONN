import os
import json
from pathlib import Path
import shutil
import tarfile
import urllib.request
import random
from tqdm import tqdm


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
            "task": "afthink_reasoning",
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


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent.parent / "data" / "YouTube8M"
    data_dir.mkdir(parents=True, exist_ok=True)
    archive_path = data_dir / "AFThink.tar.gz"

    # Extract AFThink.tar.gz
    if not (data_dir / "audio_files").exists():
        print("Extracting AFThink.tar.gz...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(data_dir)

        # Rename audio_files to audio_files/
        audio_files_dir = data_dir / "AFThink"
        if audio_files_dir.exists():
            shutil.move(str(audio_files_dir), str(data_dir / "audio_files"))
    else:
        print("Audio files already extracted")

    print("Generating annotations...")
    prepare_cot_youtube8m_annotations(str(data_dir))

    print("TO-DO: Resample audio from 48kHz (.mp3) to 16 kHz (.wav) for Whisper")
    print("Recommended to use the following command:")
    print("cd data/YouTube8M/audio_files")
    print("find . -maxdepth 1 -name \"*.mp3\" -print0 | parallel -0 -j $SLURM_CPUS_PER_TASK ffmpeg -i {} -ar 16000 -ac 1 -c:a pcm_s16le {.}.wav")
    print("Optional: Remove original mp3 files with the following command:")
    print("find . -name \"*.mp3\" | xargs -I \{\} rm -f \"\{\}\"")
    print("WARNING: Resample the audio files on a COMPUTE NODE!")
    print("Note: Estimated time to resample ~300k audio files is 4-5 hours on a compute node with 48 CPUs.")
    print("Note: Estimated disk space savings is 100GB.")
    print("Note: Keep in mind that you can't have more than 1000K audio files in your cluster allocation.")

    # Remove YouTube8M.json
    #if json_path.exists():
    #    json_path.unlink()
    #    print("Removed", json_path.name)

    # Remove AFThink.tar.gz
    #if archive_path.exists():
    #    archive_path.unlink()
    #    print("Removed", archive_path.name)

    print("Done. Dataset in", data_dir)
