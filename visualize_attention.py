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


def _attention_rollout(full_attentions: torch.Tensor, mask_4d: torch.Tensor) -> torch.Tensor:
    """
    Compute attention rollout per Abnar & Zuidema (2020) https://arxiv.org/pdf/2005.00928.
    A = 0.5*W_att + 0.5*I (residual), then rollout = A_L @ A_{L-1} @ ... @ A_1.
    """
    active_positions = 1.0 - mask_4d[:, 0, :, :].float()
    add_mask = torch.where(active_positions > 0.5, 0.0, float(torch.finfo(active_positions.dtype).min))
    attentions = full_attentions.mean(dim=2)
    seq_len = attentions.shape[-1]
    device, dtype = attentions.device, attentions.dtype

    # Consider skip-connections
    I = torch.eye(seq_len, device=device, dtype=dtype)
    I = I.unsqueeze(0).unsqueeze(0).expand(attentions.shape[0], attentions.shape[1], -1, -1)
    A = 0.5 * attentions + 0.5 * I
    A = torch.softmax(A + add_mask.unsqueeze(0), dim=-1)

    # Rollout
    rollout = A[0]
    for l in range(1, A.shape[0]):
        rollout = torch.matmul(rollout, A[l])

    return rollout


def extract_regress_sequence_attentions(
    full_attentions: torch.Tensor,
    mask_4d: torch.Tensor,
    start_regress: int,
    start_speech: int,
    speech_len: int,
) -> torch.Tensor:
    """
    Extract the attention the sequence to regress put on the speech tokens.
    Returns excess over uniform: (actual - expected) / expected, where expected
    is the attention to speech under uniform distribution over valid keys.
    Value of 0 = uniform, >0 = more attention to speech than chance.
    """
    active_positions = 1.0 - mask_4d[:, 0, :, :].float()
    rollout_attention = _attention_rollout(full_attentions, mask_4d)

    # Extract the attention for the speech tokens (positions [start_speech, start_speech+speech_len))
    to_regress_attention = rollout_attention[:, start_regress:, start_speech : start_speech + speech_len]

    # Compute excess over uniform
    actual = to_regress_attention.sum(dim=-1)
    num_valid_keys = active_positions[:, start_regress:, :].sum(dim=-1).clamp(min=1)
    num_valid_speech = active_positions[:, start_regress:, start_speech : start_speech + speech_len].sum(dim=-1)
    expected = num_valid_speech / num_valid_keys
    # Only include positions where we have valid speech keys to avoid degenerate cases
    valid_mask = num_valid_speech > 0
    if valid_mask.any():
        excess = (actual - expected) / expected.clamp(min=1e-6)
        return excess[valid_mask].mean()
    return torch.tensor(0.0, device=rollout_attention.device, dtype=rollout_attention.dtype)

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
    Returns excess over uniform: (actual - expected) / expected, where expected
    is the attention to speech under uniform distribution over valid keys.
    Value of 0 = uniform, >0 = more attention to speech than chance.
    """
    start_speech = start_regress - speech_len
    batch_size = to_regress_tokens.shape[0]
    active_positions = 1.0 - mask_4d[:, 0, :, :].float()
    rollout_attention = _attention_rollout(full_attentions, mask_4d)

    # Compute excess over uniform
    register_excess = []
    for i in range(batch_size):
        reg_ids = start_regress + torch.where(to_regress_tokens[i] == reg_token_id)[0]
        if len(reg_ids) == 0:
            continue
        ra = rollout_attention[i, reg_ids, start_speech : start_speech + speech_len]
        actual = ra.sum(dim=-1)
        num_valid_keys = active_positions[i, reg_ids, :].sum(dim=-1).clamp(min=1)
        num_valid_speech = active_positions[i, reg_ids, start_speech : start_speech + speech_len].sum(dim=-1)
        expected = num_valid_speech / num_valid_keys
        valid_mask = num_valid_speech > 0
        if valid_mask.any():
            excess = (actual - expected) / expected.clamp(min=1e-6)
            register_excess.append(excess[valid_mask].mean())
    if len(register_excess) == 0:
        return torch.tensor(0.0, device=rollout_attention.device, dtype=rollout_attention.dtype)
    return torch.stack(register_excess, dim=0).mean()


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

    # Set output directory and path
    output_dir = cfg.config.run.output_dir if args.output_dir is None else args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "attention_metrics.json")
    print(f"Saving attention metrics to {output_path}")
    
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
    regress_sequence_sum = 0.0
    register_sum = 0.0
    regress_sequence_count = 0
    register_count = 0

    for batch in tqdm(loader, desc="Extracting attention"):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        with torch.no_grad():
            output = model(batch, verbose=False, output_attentions=True)

        if "attentions" not in output:
            print("Model did not return attentions. Ensure you set output_attentions=True in the model configuration.")
            continue

        meta = output["attentions_meta"]
        batch_size = meta["mask_4d"].shape[0]

        regress_val = extract_regress_sequence_attentions(
            output["attentions"],
            meta["mask_4d"],
            meta["start_regress"],
            meta["start_speech"],
            meta["speech_len"]
        )
        regress_sequence_sum += float(regress_val) * batch_size
        regress_sequence_count += batch_size

        to_regress_tokens = meta.get("to_regress_tokens")
        if to_regress_tokens is not None:
            register_val = extract_register_attentions(
                model.llama_tokenizer.convert_tokens_to_ids("<reg>"),
                output["attentions"],
                to_regress_tokens,
                meta["mask_4d"],
                meta["start_regress"],
                meta["speech_len"]
            )
            register_sum += float(register_val) * batch_size
            register_count += batch_size

        num_batches += 1

    if num_batches == 0:
        print("No batches with attentions. Exiting.")
        return

    # Average over all batches (weighted by batch size)
    regress_sequence_mean = regress_sequence_sum / regress_sequence_count if regress_sequence_count > 0 else 0.0
    register_mean = register_sum / register_count if register_count > 0 else 0.0

    # Save summarized metrics as JSON
    metrics = {
        "regress_sequence_attention": {
            "excess_over_uniform": regress_sequence_mean,
        },
        "register_attention": {
            "excess_over_uniform": register_mean,
        },
    }
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved attention metrics ({num_batches} batches)")
    print(f"Saved attention metrics to {output_path}")


if __name__ == "__main__":
    main()
