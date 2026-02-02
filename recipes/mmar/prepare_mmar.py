# prepare_mmar.py
import os
import json
from pathlib import Path

import tarfile
import urllib.request

from datasets import load_dataset


def prepare_mmar_annotations(output_dir: str):
    """Load MMAR test split from Hugging Face and write test_mmar.json and test_mmar_metadata.json."""
    os.makedirs(output_dir, exist_ok=True)
    output_dir = Path(output_dir)

    ds = load_dataset("BoJack/MMAR")
    annotations = []
    for row in ds["test"]:
        audio_path = row["audio_path"]
        if not os.path.isabs(audio_path):
            audio_path = str(output_dir / audio_path)

        annotations.append({
            "path": audio_path,
            "text": row["answer"],
            "task": "mmar",
            **{key: value for key, value in row.items() if key != "audio_path" and key != "answer"}
        })

    (output_dir / "annotations").mkdir(parents=True, exist_ok=True)
    ann_path = output_dir / "annotations" / "test_mmar.json"
    with open(ann_path, "w") as f:
        json.dump({"annotation": annotations}, f, indent=2)
    print(f"Saved {len(annotations)} samples to {ann_path}")


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent.parent / "data" / "MMAR"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Download and extract mmar-audio.tar.gz
    url = "https://huggingface.co/datasets/BoJack/MMAR/resolve/main/mmar-audio.tar.gz?download=true"
    archive_path = data_dir / "mmar-audio.tar.gz"
    if not archive_path.exists():
        print("Downloading mmar-audio.tar.gz...")
        urllib.request.urlretrieve(url, archive_path)
    print("Extracting...")
    with tarfile.open(archive_path, "r:gz") as tf:
        tf.extractall(data_dir)

    prepare_mmar_annotations(str(data_dir))
    print("Done. Dataset in", data_dir)
