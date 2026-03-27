import os
import json
import re
from pathlib import Path


def prepare_af_youtube8m_annotations(source_annotations_dir: str, target_annotations_dir: str):
    """Load AF-YouTube8M from AF-Think and write SALMONN-compatible train_youtube8m.json and test_youtube8m.json."""
    target_annotations_dir = Path(target_annotations_dir)

    # Read the .json files in the source annotations directory
    for json_file in source_annotations_dir.glob("*.json"):
        target_annotation_file = target_annotations_dir / json_file.name
        print(f"Processing {json_file.name}...")
        with open(json_file, "r") as f:
            source_annotations = json.load(f)
            source_annotations = source_annotations["annotation"]
            print(f"Found {len(source_annotations)} annotations in {json_file.name}")

        # Remove the reasoning part from the question
        for annotation in source_annotations:
            match = re.search(r"<CONCLUSION>(.*?)</CONCLUSION>", annotation["answer"], re.DOTALL)
            answer = match.group(1).strip()
            annotation["answer"] = answer

        # Save the target annotations
        with open(target_annotation_file, "w") as f:
            json.dump({"annotation": source_annotations}, f, indent=2)
            print(f"Saved {len(source_annotations)} annotations to {target_annotation_file}")


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    source_annotations_dir = script_dir.parent.parent / "data" / "YouTube8M" / "annotations"
    target_annotations_dir = script_dir.parent.parent / "data" / "YouTube8M" / "annotations_no_reasoning"
    os.makedirs(target_annotations_dir, exist_ok=True)

    print("Generating annotations...")
    prepare_af_youtube8m_annotations(source_annotations_dir, target_annotations_dir)

    print("Done. Dataset in", target_annotations_dir)
