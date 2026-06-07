import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import os
import re
import string
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from jiwer import wer, cer
from omegaconf import OmegaConf

from config import Config
from models.salmonn import SALMONN, MutorSALMONN
from dataset import SALMONNDataset
from utils import get_prompts


def normalize_text(text: str) -> str:
    """
    Normalize text for WER/CER computation.
    - Lowercase
    - Remove punctuation
    - Collapse multiple spaces
    - Strip whitespace
    """
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SALMONN ASR with WER/CER")
    parser.add_argument("--cfg-path", type=str, required=True, help="Path to config file")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--split", type=str, default="test", choices=["valid", "test"],
                        help="Which split to evaluate on")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for inference")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--output-file", type=str, default="eval_results.json", help="Output file")
    parser.add_argument(
        "--options",
        nargs="+",
        help="Override config options in key=value format",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config(args)

    print("=" * 60)
    print("SALMONN AF-Think Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {args.ckpt}")
    print(f"Split: {args.split}")
    print(f"Device: {args.device}")
    print("=" * 60)

    # Set checkpoint path in config so from_config loads it automatically
    cfg.config.model.ckpt = args.ckpt
    
    # Use appropriate model class based on model_type
    model_type = cfg.config.model.get("model_type", "salmonn")
    if model_type == "mutor":
        print("Loading MutorSALMONN model...")
        model = MutorSALMONN.from_config(cfg.config.model)
    else:
        print("Loading SALMONN model...")
        model = SALMONN.from_config(cfg.config.model)
    model.to(args.device)
    model.eval()
    
    # Load dataset
    data_config = cfg.config.datasets
    if args.split == "test":
        ann_path = data_config.test_ann_path
    else:
        ann_path = data_config.valid_ann_path
    
    print(f"Loading dataset from: {ann_path}")
    dataset = SALMONNDataset(ann_path, data_config.whisper_path)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=dataset.collater,
        pin_memory=True,
    )

    metadata_list = json.load(open(ann_path, "r"))["annotation"]

    # Extract choices from question
    for item in metadata_list:
        item["choices"] = [m[1].strip() for m in re.findall(r"\(([A-Z])\) (.*?)(?:\.(?=\s|$)|$)", item["question"], re.MULTILINE)]

    metadata = {
        item["path"]: {k: v for k, v in item.items() if k != "path"}
        for item in metadata_list
    }
    
    # Run inference
    print("Running inference...")
    all_results = []
    all_refs = []
    all_hyps = []
    all_conclusions_hyps = []
    all_conclusions_refs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move to device
            batch["spectrogram"] = batch["spectrogram"].to(args.device)
            batch["raw_wav"] = batch["raw_wav"].to(args.device)
            batch["padding_mask"] = batch["padding_mask"].to(args.device)
            
            # Create prompts for batch
            prompts = get_prompts(batch, metadata, cfg, prompt_type="afthink")
            
            # Generate transcriptions
            with torch.amp.autocast('cuda', dtype=torch.float16):
                hypotheses = model.generate(batch, cfg.config.generate, prompts=prompts)
            
            references = batch["text"]
            ids = batch["id"]
            for ref, hyp, uid in zip(references, hypotheses, ids):
                # Clean up hypothesis (remove special tokens, extra whitespace)
                hyp_clean = hyp.replace("</s>", "").replace("<s>", "").replace("<unk>", "").strip()

                # Extract the CONCLUSION tag from the hypothesis
                conclusion_hypothesis_tags = re.search(r"<CONCLUSION>(.*?)</CONCLUSION>", hyp_clean)
                if conclusion_hypothesis_tags is not None:
                    conclusion_hypothesis = conclusion_hypothesis_tags.group(1).strip()
                else:
                    conclusion_hypothesis = hyp_clean
                # Remove letter choice prefix from conclusion (e.g. "(A) ", "(B) ") - only at the beginning
                conclusion_hypothesis = re.sub(r"^\([A-Z]\)\s*", "", conclusion_hypothesis).strip()

                # Extract the CONCLUSION tag from the reference
                conclusion_reference_tags = re.search(r"<CONCLUSION>(.*?)</CONCLUSION>", ref)
                if conclusion_reference_tags is not None:
                    conclusion_reference = conclusion_reference_tags.group(1).strip()
                else:
                    conclusion_reference = ref
                # Remove letter choice prefix from conclusion (e.g. "(A) ", "(B) ") - only at the beginning
                conclusion_reference = re.sub(r"^\([A-Z]\)\s*", "", conclusion_reference).strip()
                
                all_conclusions_hyps.append(conclusion_hypothesis)
                all_conclusions_refs.append(conclusion_reference)
                all_refs.append(ref)
                all_hyps.append(hyp_clean)
                all_results.append({
                    "id": uid,
                    "conclusion_hypothesis": conclusion_hypothesis.lower(),
                    "conclusion_reference": conclusion_reference.lower(),
                    "reference": ref,
                    "hypothesis": hyp_clean,
                    "reference_normalized": normalize_text(ref),
                    "hypothesis_normalized": normalize_text(hyp_clean),
                })
    
    # Normalize for WER/CER computation
    refs_normalized = [normalize_text(r) for r in all_refs]
    hyps_normalized = [normalize_text(h) for h in all_hyps]

    # Compute string match accuracy
    corr, total = 0, 0
    for hyp, ref in zip(all_conclusions_hyps, all_conclusions_refs):
        if hyp == ref:
            corr += 1
        total += 1
    string_match_accuracy = corr / total * 100
    print(f"String Match Accuracy: {string_match_accuracy:.2f}%")
    
    # Compute metrics
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    # Overall WER/CER
    overall_wer = wer(refs_normalized, hyps_normalized) * 100
    overall_cer = cer(refs_normalized, hyps_normalized) * 100
    
    print(f"Word Error Rate (WER):      {overall_wer:.2f}%")
    print(f"Character Error Rate (CER): {overall_cer:.2f}%")
    print(f"Number of samples:          {len(all_refs)}")
    print("=" * 60)
    
    # Save results
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    results_summary = {
        "checkpoint": args.ckpt,
        "split": args.split,
        "num_samples": len(all_refs),
        "string_match_accuracy": string_match_accuracy,
        "wer": overall_wer,
        "cer": overall_cer,
        "generate_config": OmegaConf.to_container(cfg.config.generate),
    }
    
    summary_path = args.output_file.replace(".json", "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")
    
    # Save detailed results
    details_path = args.output_file.replace(".json", "_details.json")
    with open(details_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Detailed results saved to: {details_path}")
    
    # Save a few examples
    print("\n" + "=" * 60)
    print("SAMPLE OUTPUTS (first 5)")
    print("=" * 60)
    for i, res in enumerate(all_results[:5]):
        print(f"\n[{i+1}] ID: {res['id']}")
        print(f"    CONCLUSION HYP: {res['conclusion_hypothesis']}")
        print(f"    CONCLUSION REF: {res['conclusion_reference']}")
        print(f"    REF: {res['reference']}")
        print(f"    HYP: {res['hypothesis']}")


if __name__ == "__main__":
    """
    Evaluate SALMONN AF-Think performance with WER/CER metrics and string match accuracy.

    Usage:
        python eval/evaluate_afthink.py \
            --cfg-path recipes/afthink/salmonn.yaml \
            --ckpt <checkpoint_path> \
            --split test \
            --batch-size <batch_size> \
            --num-workers <num_workers> \
            --device <device> \
            --output-file <output_file> \
            --options <options>
    """
    main()