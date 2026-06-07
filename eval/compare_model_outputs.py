#!/usr/bin/env python3
"""
Compare two model output JSON files using the string_match evaluation from eval/evaluate_mmau.py.
Shows correct samples per model and overlap/unique correct samples between the two.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json

from eval.evaluate_mmau import string_match


def load_and_evaluate(json_path):
    """Load JSON file and return set of sample IDs that are correctly classified."""
    with open(json_path, "r") as f:
        data = json.load(f)

    correct_ids = set()
    sample_by_id = {}

    for sample in data:
        sample_id = sample.get("id")
        if sample_id is None:
            continue
        sample_by_id[sample_id] = sample

        if "model_output" not in sample:
            continue

        answer = sample["answer"]
        prediction = sample["model_output"]
        choices = sample.get("choices", [])

        if string_match(answer, prediction, choices):
            correct_ids.add(sample_id)

    return correct_ids, sample_by_id, len(data)


def main():
    parser = argparse.ArgumentParser(
        description="Compare correct classifications between two model output JSON files"
    )
    parser.add_argument("file1", type=str, help="Path to first model output JSON")
    parser.add_argument("file2", type=str, help="Path to second model output JSON")
    parser.add_argument("--show-ids", action="store_true", help="Print sample IDs for each category")
    args = parser.parse_args()

    # Evaluate both files
    correct1, samples1, total1 = load_and_evaluate(args.file1)
    correct2, samples2, total2 = load_and_evaluate(args.file2)

    # Ensure we're comparing the same sample space (by ID)
    common_ids = set(samples1.keys()) & set(samples2.keys())

    # Overlap: correct in BOTH models
    overlap = correct1 & correct2

    # Unique to model 1: correct only in model 1
    unique_to_1 = correct1 - correct2

    # Unique to model 2: correct only in model 2
    unique_to_2 = correct2 - correct1

    # Print results
    print("=" * 60)
    print("Model Comparison Report")
    print("=" * 60)
    print(f"\nFile 1: {args.file1}")
    print(f"  Total samples: {total1}")
    print(f"  Correct: {len(correct1)} ({100 * len(correct1) / total1:.2f}%)" if total1 > 0 else "  Correct: 0")

    print(f"\nFile 2: {args.file2}")
    print(f"  Total samples: {total2}")
    print(f"  Correct: {len(correct2)} ({100 * len(correct2) / total2:.2f}%)" if total2 > 0 else "  Correct: 0")

    print("\n" + "-" * 60)
    print("Overlap & Unique Correct Samples")
    print("-" * 60)
    print(f"  Correct in BOTH models:     {len(overlap)}")
    print(f"  Correct ONLY in model 1:    {len(unique_to_1)}")
    print(f"  Correct ONLY in model 2:    {len(unique_to_2)}")
    print("-" * 60)

    if args.show_ids:
        print("\n--- Correct in BOTH (overlap) ---")
        for sid in sorted(overlap):
            print(f"  {sid}")

        print("\n--- Correct ONLY in model 1 ---")
        for sid in sorted(unique_to_1):
            print(f"  {sid}")

        print("\n--- Correct ONLY in model 2 ---")
        for sid in sorted(unique_to_2):
            print(f"  {sid}")

    if len(common_ids) != total1 or len(common_ids) != total2:
        print(f"\nNote: File 1 has {total1} samples, File 2 has {total2} samples.")
        print(f"      {len(common_ids)} sample IDs appear in both files.")


if __name__ == "__main__":
    main()
