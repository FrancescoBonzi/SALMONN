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
from models.salmonn import SALMONN
from dataset import SALMONNDataset


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
    parser.add_argument("--output-dir", type=str, default="eval_results", help="Output directory")
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
    print("SALMONN ASR Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {args.ckpt}")
    print(f"Split: {args.split}")
    print(f"Device: {args.device}")
    print("=" * 60)

    # Set checkpoint path in config so from_config loads it automatically
    cfg.config.model.ckpt = args.ckpt
    
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
    
    # ASR prompt (from test_prompt.json)
    prompt_template = cfg.config.model.prompt_template
    asr_prompt = "<Speech><SpeechHere></Speech> Recognize the speech and give me the transcription."
    
    # Run inference
    print("Running inference...")
    all_results = []
    all_refs = []
    all_hyps = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move to device
            batch["spectrogram"] = batch["spectrogram"].to(args.device)
            batch["raw_wav"] = batch["raw_wav"].to(args.device)
            batch["padding_mask"] = batch["padding_mask"].to(args.device)
            
            # Create prompts for batch
            prompts = [prompt_template.format(asr_prompt)] * len(batch["text"])
            
            # Generate transcriptions
            with torch.amp.autocast('cuda', dtype=torch.float16):
                hypotheses = model.generate(batch, cfg.config.generate, prompts=prompts)[0]
            
            references = batch["text"]
            ids = batch["id"]
            
            # Store results
            for ref, hyp, uid in zip(references, hypotheses, ids):
                # Clean up hypothesis (remove special tokens, extra whitespace)
                hyp_clean = hyp.replace("</s>", "").replace("<s>", "").strip()
                
                all_refs.append(ref)
                all_hyps.append(hyp_clean)
                all_results.append({
                    "id": uid,
                    "reference": ref,
                    "hypothesis": hyp_clean,
                    "reference_normalized": normalize_text(ref),
                    "hypothesis_normalized": normalize_text(hyp_clean),
                })
    
    # Normalize for WER/CER computation
    refs_normalized = [normalize_text(r) for r in all_refs]
    hyps_normalized = [normalize_text(h) for h in all_hyps]
    
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
    
    # Comparison with paper
    print("\nComparison with SALMONN Paper (Tang et al., 2024):")
    print("-" * 60)
    print(f"  Paper (test-clean):  WER = 2.1%")
    print(f"  Paper (test-other):  WER = 4.9%")
    print(f"  Current model:       WER = {overall_wer:.2f}%")
    print("-" * 60)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    results_summary = {
        "checkpoint": args.ckpt,
        "split": args.split,
        "num_samples": len(all_refs),
        "wer": overall_wer,
        "cer": overall_cer,
        "generate_config": OmegaConf.to_container(cfg.config.generate),
        "paper_comparison": {
            "salmonn_test_clean_wer": 2.1,
            "salmonn_test_other_wer": 4.9,
        }
    }
    
    summary_path = os.path.join(args.output_dir, f"summary_{args.split}.json")
    with open(summary_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")
    
    # Save detailed results
    details_path = os.path.join(args.output_dir, f"details_{args.split}.json")
    with open(details_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Detailed results saved to: {details_path}")
    
    # Save a few examples
    print("\n" + "=" * 60)
    print("SAMPLE OUTPUTS (first 5)")
    print("=" * 60)
    for i, res in enumerate(all_results[:5]):
        print(f"\n[{i+1}] ID: {res['id']}")
        print(f"    REF: {res['reference']}")
        print(f"    HYP: {res['hypothesis']}")


if __name__ == "__main__":
    """
    Evaluate SALMONN ASR performance with WER/CER metrics.

    Usage:
        python evaluate.py \
            --cfg-path recipes/librispeech/baseline.yaml \
            --ckpt <checkpoint_path> \
            --split test
    """
    main()