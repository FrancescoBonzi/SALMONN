import argparse
import json
import os
import re
import string
from typing import Any
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from config import Config
from models.salmonn import SALMONN, MutorSALMONN
from dataset import SALMONNDataset


def string_match(answer, prediction, choices):
    # Function to normalize and tokenize text
    def tokenize(text):
        # Convert to lowercase and find all word tokens
        return set(re.findall(r'\b\w+\b', text.lower()))
    
    # Tokenize prediction and answer
    prediction_tokens = tokenize(prediction)
    answer_tokens = tokenize(answer)
    
    if not prediction_tokens:
        return False
    
    # Tokenize incorrect choices and exclude tokens present in the answer
    incorrect_tokens = set()
    for choice in choices:
        choice_tokens = tokenize(choice)
        if choice_tokens != answer_tokens:
            incorrect_tokens.update(choice_tokens - answer_tokens)
    
    # Condition 1: All tokens of the answer are in the prediction
    cond1 = answer_tokens.issubset(prediction_tokens)
    
    # Condition 2: Prediction does not contain any tokens from incorrect choices (excluding shared words)
    cond2 = prediction_tokens.isdisjoint(incorrect_tokens)
    
    return cond1 and cond2


def official_mmau_evaluation(input_data: list[dict]):

    corr, total = 0, 0

    # Track metrics for different categories:
    task_metrics = {'sound': [0, 0], 'music': [0, 0], 'speech': [0, 0]}
    diff_metrics = {'easy': [0, 0], 'hard': [0, 0], 'medium': [0, 0]}
    
    # Here is the new dict for sub-category metrics
    subcat_metrics = {}

    output_key = 'model_output' # The key that contains model output
    no_pred_count = 0
    matched_outputs = []
    new_data = []

    for idx, sample in enumerate(tqdm(input_data)):
        
        # If there's no model output key, skip
        if output_key not in sample:
            continue
        
        if output_key not in sample:
            _prediction = ''
            no_pred_count += 1
        else:
            _prediction = sample[output_key]

        _answer = sample['answer']
        task = sample['task']
        difficulty = sample['difficulty']
        choices = sample['choices']
        
        # Get the sub-category
        subcat = sample.get('sub-category', None)
        if subcat is not None:
            # If we haven't seen this sub-category before, initialize
            if subcat not in subcat_metrics:
                subcat_metrics[subcat] = [0, 0]

        match_result = string_match(_answer, _prediction, choices)

        if match_result:
            task_metrics[task][0] += 1
            diff_metrics[difficulty][0] += 1
            if subcat is not None:
                subcat_metrics[subcat][0] += 1
            matched_outputs.append([_answer, _prediction])
            corr += 1
            sample['match'] = 1
        else:
            sample['match'] = 0

        total += 1
        new_data.append(sample)
        task_metrics[task][1] += 1
        diff_metrics[difficulty][1] += 1
        if subcat is not None:
            subcat_metrics[subcat][1] += 1


    # Print results:
    print("*"*30)
    print("Task-wise Accuracy:")
    for task in task_metrics:
        n_correct, n_total = task_metrics[task]
        acc = (n_correct / n_total) * 100 if n_total > 0 else 0
        print(f"{task} : {acc:.2f}% over {n_total} samples")
    
    print("*"*30)
    print("Difficulty-wise Accuracy:")
    for diff in diff_metrics:
        n_correct, n_total = diff_metrics[diff]
        acc = (n_correct / n_total) * 100 if n_total > 0 else 0
        print(f"{diff} : {acc:.2f}% over {n_total} samples")
    
    print("*"*30)
    print("Sub-category-wise Accuracy:")
    for subcat in subcat_metrics:
        n_correct, n_total = subcat_metrics[subcat]
        acc = (n_correct / n_total) * 100 if n_total > 0 else 0
        print(f"{subcat} : {acc:.2f}% over {n_total} samples")

    print("*"*30)
    print(f"Total Accuracy: {(corr/total) * 100:.2f}% over {total} samples")
    print("*"*30)
    print(f"No prediction count: {no_pred_count}")

    return corr, total


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SALMONN MMAU")
    parser.add_argument("--cfg-path", type=str, required=True, help="Path to config file")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
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
    print("SALMONN MMAU Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {args.ckpt}")
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
    ann_path = data_config.test_ann_path
    
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
    metadata = {
        item["path"]: {k: v for k, v in item.items() if k != "path"}
        for item in metadata_list
    }

    # Run inference
    print("Running inference...")
    results = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move to device
            batch["spectrogram"] = batch["spectrogram"].to(args.device)
            batch["raw_wav"] = batch["raw_wav"].to(args.device)
            batch["padding_mask"] = batch["padding_mask"].to(args.device)

            # Create prompts for batch
            prompts = []
            for i in range(len(batch["id"])):
                id = batch["id"][i]
                question = metadata[id]["question"]
                if not question.endswith("?") and not question.endswith("."):
                    if question.startswith(("Which", "What", "Who", "When", "Where", "Why", "How", "Are", "Is")):
                        question += "?"
                    else:
                        question += "."
                choices = "\n".join([
                    f"{choice}" 
                    for letter, choice in zip(
                        string.ascii_uppercase[:len(metadata[id]["choices"])], metadata[id]["choices"]
                    )
                ])
                prompt = f"<Speech><SpeechHere></Speech> {question}\n{choices}\nAnswer with the choice directly."
                prompts.append(cfg.config.model.prompt_template.format(prompt))
            
            # Generate transcriptions
            with torch.amp.autocast('cuda', dtype=torch.float16):
                hypotheses = model.generate(batch, cfg.config.generate, prompts=prompts)
            
            references = batch["text"]
            ids = batch["id"]
            
            # Store results
            for i, (ref, hyp, uid) in enumerate(zip(references, hypotheses, ids)):
                # Clean up hypothesis (remove special tokens, extra whitespace)
                hyp_clean = hyp.replace("</s>", "").replace("<s>", "").replace("<unk>", "").strip()
                
                results.append({
                    "id": uid,
                    "model_prediction": hyp_clean,
                    "answer": ref,
                    "prompt": prompts[i],
                    "choices": metadata[uid]["choices"],
                    "difficulty": metadata[uid]["difficulty"],
                    "task": metadata[uid]["task"],
                    "sub-category": metadata[uid]["sub-category"] if metadata[uid]["sub-category"] is not None else None,
                })

            # Free batch GPU tensors to avoid fragmentation and OOM over many iterations
            del batch["spectrogram"], batch["raw_wav"], batch["padding_mask"]
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

    # Save results
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {os.path.join(args.output_dir, 'results.json')}")

    # Compute metrics
    corr, total = official_mmau_evaluation(results)
    print(f"Official MMAU Accuracy: {(corr/total) * 100:.2f}% over {total} samples")


if __name__ == "__main__":
    """
    Evaluate SALMONN MMAU performance.

    Usage:
        python evaluate_mmau.py \
            --cfg-path recipes/mmau/salmonn.yaml \
            --ckpt <checkpoint_path> \
    """
    main()