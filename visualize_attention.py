#!/usr/bin/env python3
"""
Visualize attention maps for MutorBERTConclusionSALMONN.

Usage:
  python visualize_attention.py --cfg-path recipes/afthink/debug.yaml
"""

import argparse
import json
import os
import torch
from tqdm import tqdm

from config import Config
from models import load_model
from dataset import SALMONNDataset
from torch.utils.data import DataLoader


def extract_regress_sequence_attentions(
    full_attentions: torch.Tensor,
    mask_4d: torch.Tensor,
    start_regress: int,
    speech_len: int,
) -> torch.Tensor:
    """
    Extract the attention the sequence to regress put on the speech tokens.
    """
    active_positions = 1.0 - mask_4d[:, 0, :, :].float()

    # Get the first layer's attention and average over heads
    attentions = full_attentions[0].mean(dim=1)
    # Extract the attention for the speech tokens
    to_regress_attention = attentions[:, start_regress:, 1:1+speech_len]
    # Multiply by the number of valid positions
    to_regress_attention *= active_positions[:, start_regress:, :].sum(dim=-1).unsqueeze(-1)
    # Rescale by the mean of valid positions
    to_regress_attention /= active_positions[:, start_regress:, :].sum(dim=-1).mean(dim=-1).unsqueeze(-1).unsqueeze(-1)

    return to_regress_attention.sum(dim=-1).mean()

def extract_register_attentions(
    reg_token_id: int,
    full_attentions: torch.Tensor,
    to_regress_tokens: torch.Tensor,
    mask_4d: torch.Tensor,
    start_regress: int,
    speech_len: int,
) -> torch.Tensor:
    """
    Extract the attention the registers have on the speech tokens.
    """
    batch_size = to_regress_tokens.shape[0]
    active_positions = 1.0 - mask_4d[:, 0, :, :].float()
    attentions = full_attentions[0].mean(dim=1)
    # Get the attention for the speech tokens
    register_attentions = []
    for i in range(batch_size):
        reg_ids = start_regress + torch.where(to_regress_tokens[i] == reg_token_id)[0]
        ra = attentions[i, reg_ids, 1:1+speech_len]
        ra *= active_positions[i, reg_ids, :].sum(dim=-1).unsqueeze(-1)
        ra /= active_positions[i, reg_ids, :].sum(dim=-1).mean(dim=-1).unsqueeze(-1).unsqueeze(-1)
        register_attentions.append(ra.mean(dim=0).sum(dim=0))
    register_attentions = torch.stack(register_attentions, dim=0)

    return register_attentions.mean(dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-path", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for figures")
    parser.add_argument("--layers", type=str, default=None, help="Comma-separated layer indices, e.g. 0,15,31")
    parser.add_argument(
        "--options",
        nargs="*",
        default=[],
        help="Config overrides in key=value format, e.g. model.ckpt=path/to/ckpt.pth",
    )

    args = parser.parse_args()
    cfg = Config(args)
    
    model_config = cfg.config.model
    data_config = cfg.config.datasets

    # Load model
    model = load_model(model_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    # Load dataset and iterate over all batches
    dataset = SALMONNDataset(data_config.valid_ann_path, data_config.whisper_path)
    loader = DataLoader(
        dataset,
        batch_size=cfg.config.run.batch_size_eval,
        shuffle=False,
        collate_fn=dataset.collater,
        num_workers=0,
    )

    num_batches = 0
    for batch in tqdm(loader, desc="Extracting attention"):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        with torch.no_grad():
            output = model(batch, verbose=False, output_attentions=True)

        if "attentions" not in output:
            print("Model did not return attentions. Ensure you use MutorBERTConclusionSALMONN with output_attentions=True.")
            continue

        meta = output["attentions_meta"]
        regress_sequence_sum_attentions = extract_regress_sequence_attentions(
            output["attentions"],
            meta["mask_4d"],
            meta["start_regress"],
            meta["speech_len"]
        )

        register_sum_attentions = extract_register_attentions(
            model.llama_tokenizer.convert_tokens_to_ids("<reg>"),
            output["attentions"],
            meta["to_regress_tokens"],
            meta["mask_4d"],
            meta["start_regress"],
            meta["speech_len"]
        )

        num_batches += 1

    if num_batches == 0:
        print("No batches with attentions. Exiting.")
        return

    output_dir = cfg.config.run.output_dir if args.output_dir is None else args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    # Save summarized metrics as JSON
    metrics = {
        "regress_sequence_attention": {
            "mean": float(regress_sequence_sum_attentions),
        },
        "register_attention": {
            "mean": float(register_sum_attentions),
        },
    }
    with open(os.path.join(output_dir, "attention_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved attention metrics ({num_batches} batches)")


if __name__ == "__main__":
    main()
