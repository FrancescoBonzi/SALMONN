"""
Evaluate Qwen2.5-Omni on MMAU benchmark.
Supports both pretrained zero-shot and fine-tuned checkpoints.
"""
import argparse
import json
import os
import re
import string
import sys
from io import StringIO
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from config import Config
from models import load_model
from dataset import Qwen2AudioDataset
from utils import get_prompts


def string_match(answer, prediction, choices):
    def tokenize(text):
        return set(re.findall(r'\b\w+\b', text.lower()))

    prediction_tokens = tokenize(prediction)
    answer_tokens = tokenize(answer)

    if not prediction_tokens:
        return False

    incorrect_tokens = set()
    for choice in choices:
        choice_tokens = tokenize(choice)
        if choice_tokens != answer_tokens:
            incorrect_tokens.update(choice_tokens - answer_tokens)

    cond1 = answer_tokens.issubset(prediction_tokens)
    cond2 = prediction_tokens.isdisjoint(incorrect_tokens)
    return cond1 and cond2


def official_mmau_evaluation(input_data: list[dict]):
    corr, total = 0, 0
    task_metrics = {'sound': [0, 0], 'music': [0, 0], 'speech': [0, 0]}
    diff_metrics = {'easy': [0, 0], 'hard': [0, 0], 'medium': [0, 0]}
    subcat_metrics = {}
    output_key = 'model_output'
    no_pred_count = 0

    for sample in tqdm(input_data):
        if output_key not in sample:
            continue

        _prediction = sample[output_key]
        _answer = sample['answer']
        task = sample['task']
        difficulty = sample['difficulty']
        choices = sample['choices']
        subcat = sample.get('sub-category', None)

        if subcat is not None and subcat not in subcat_metrics:
            subcat_metrics[subcat] = [0, 0]

        match_result = string_match(_answer, _prediction, choices)

        if match_result:
            task_metrics[task][0] += 1
            diff_metrics[difficulty][0] += 1
            if subcat is not None:
                subcat_metrics[subcat][0] += 1
            corr += 1
            sample['match'] = 1
        else:
            sample['match'] = 0

        total += 1
        task_metrics[task][1] += 1
        diff_metrics[difficulty][1] += 1
        if subcat is not None:
            subcat_metrics[subcat][1] += 1

    print("*" * 30)
    print("Task-wise Accuracy:")
    for task in task_metrics:
        n_correct, n_total = task_metrics[task]
        acc = (n_correct / n_total) * 100 if n_total > 0 else 0
        print(f"{task} : {acc:.2f}% over {n_total} samples")
    print("*" * 30)
    print("Difficulty-wise Accuracy:")
    for diff in diff_metrics:
        n_correct, n_total = diff_metrics[diff]
        acc = (n_correct / n_total) * 100 if n_total > 0 else 0
        print(f"{diff} : {acc:.2f}% over {n_total} samples")
    print("*" * 30)
    print("Sub-category-wise Accuracy:")
    for subcat in subcat_metrics:
        n_correct, n_total = subcat_metrics[subcat]
        acc = (n_correct / n_total) * 100 if n_total > 0 else 0
        print(f"{subcat} : {acc:.2f}% over {n_total} samples")
    print("*" * 30)
    print(f"Total Accuracy: {(corr/total) * 100:.2f}% over {total} samples")
    print("*" * 30)
    print(f"No prediction count: {no_pred_count}")

    return corr, total


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Qwen2.5-Omni on MMAU")
    parser.add_argument("--cfg-path", type=str, required=True, help="Path to config file")
    parser.add_argument("--ckpt", type=str, default="", help="Path to checkpoint (optional for zero-shot)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for inference")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--output-file", type=str, default="outputs/mmau/qwen_results.json", help="Output file")
    parser.add_argument("--prompt-type", type=str, default="official", help="Prompt type")
    parser.add_argument("--options", nargs="+", help="Override config options in key=value format")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config(args)

    print("=" * 60)
    print("Qwen2.5-Omni MMAU Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {args.ckpt or '(pretrained zero-shot)'}")
    print(f"Device: {args.device}")
    print("=" * 60)

    cfg.config.model.ckpt = args.ckpt
    
    model_type = cfg.config.model.get("model_type", "qwen25_omni")
    print(f"Loading {model_type} model...")
    model = load_model(cfg.config.model)
    model.to(args.device)
    model.eval()

    data_config = cfg.config.datasets
    ann_path = data_config.test_ann_path
    qwen_path = cfg.config.model.get("qwen25_omni_path", "Qwen/Qwen2.5-Omni-7B")

    print(f"Loading dataset from: {ann_path}")
    dataset = Qwen2AudioDataset(ann_path, qwen_path)
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

    print("Running inference...")
    results = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch["input_features"] = batch["input_features"].to(args.device)
            batch["feature_attention_mask"] = batch["feature_attention_mask"].to(args.device)

            prompts = get_prompts(batch, metadata, cfg, prompt_type=args.prompt_type)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                hypotheses = model.generate(batch, cfg.config.generate, prompts=prompts)

            references = batch["text"]
            ids = batch["id"]

            for i, (ref, hyp, uid) in enumerate(zip(references, hypotheses, ids)):
                hyp_clean = hyp.replace("</s>", "").replace("<s>", "").replace("<unk>", "").strip()
                conclusion_tags = re.search(r"<CONCLUSION>(.*?)</CONCLUSION>", hyp_clean)
                if conclusion_tags is not None:
                    conclusion = conclusion_tags.group(1).strip()
                else:
                    conclusion = hyp_clean

                results.append({
                    "id": uid,
                    "model_output": conclusion,
                    "hyp": hyp,
                    "answer": ref,
                    "prompt": prompts[i],
                    "choices": metadata[uid]["choices"],
                    "difficulty": metadata[uid]["difficulty"],
                    "task": metadata[uid]["task"],
                    "sub-category": metadata[uid].get("sub-category"),
                })

            del batch["input_features"], batch["feature_attention_mask"]
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    answers_path = args.output_file.replace(".json", "_answers.json")
    with open(answers_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {answers_path}")

    old_stdout = sys.stdout
    sys.stdout = StringIO()
    corr, total = official_mmau_evaluation(results)
    metrics_output = sys.stdout.getvalue()
    sys.stdout = old_stdout
    print(metrics_output)
    print(f"Official MMAU Accuracy: {(corr/total) * 100:.2f}% over {total} samples")

    metrics_path = args.output_file.replace(".json", "_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(metrics_output)
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
