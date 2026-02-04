import os
import json
from pathlib import Path
import shutil
import urllib.request
import tarfile

from tqdm import tqdm
from datasets import load_dataset


def prepare_cot_chime_home_annotations(output_dir: str):
    """Load COT-Chime-Home from AF-Think and write SALMONN-compatible train_chime_home.json."""
    os.makedirs(output_dir, exist_ok=True)
    output_dir = Path(output_dir)

    ds_afthink_stream = load_dataset("nvidia/AF-Think", "afthink", streaming=True)
    train_annotations = []
    for item in tqdm(ds_afthink_stream["af_cot_train_chime_home"]):
        sound_name = Path("audio_files/" + item["sound"][len("chunks/"):-len(".48kHz.wav")] + ".16kHz.wav")
        audio_path = str(output_dir / sound_name)
        train_annotations.append({
            "path": audio_path,
            "task": "reasoning",
            "prompt": item["conversations"][0]["value"].replace("<sound>", "").strip(),
            "text": item["conversations"][1]["value"].strip(),
        })

    ann_path = output_dir / "annotations" / "train_chime_home.json"
    os.makedirs(ann_path.parent, exist_ok=True)
    with open(ann_path, "w") as f:
        json.dump({"annotation": train_annotations}, f, indent=2)
    print(f"Saved {len(train_annotations)} samples to {ann_path}")


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent.parent / "data" / "Chime-Home"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Download and extract chime_home.tar.gz (Chime-Home, Archive.org)
    url = "https://archive.org/download/chime-home/chime_home.tar.gz"
    archive_path = data_dir / "chime_home.tar.gz"
    if not archive_path.exists():
        print("Downloading chime_home.tar.gz...")
        urllib.request.urlretrieve(url, archive_path)

    print("Extracting...")
    with tarfile.open(archive_path, "r") as tar:
        tar.extractall(path=data_dir)

    # Rename chunks to audio_files
    chunks_src = data_dir / "chime_home" / "chunks"
    if chunks_src.exists():
        shutil.move(str(chunks_src), str(data_dir / "audio_files"))

    # Remove extracted directory
    extracted_dir = data_dir / "chime_home"
    if extracted_dir.exists():
        shutil.rmtree(extracted_dir)
        print("Removed chime_home")

    prepare_cot_chime_home_annotations(str(data_dir))

    if archive_path.exists():
        archive_path.unlink()
        print("Removed", archive_path.name)

    print("Done. Dataset in", data_dir)
