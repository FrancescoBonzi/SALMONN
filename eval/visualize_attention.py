#!/usr/bin/env python3
"""
Visualize attention maps for BERTConclusionSPARE.

Usage:
  python eval/visualize_attention.py --cfg-path recipes/afthink/debug.yaml
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import os
import torch
from tqdm import tqdm

from config import Config
from models import load_model
from dataset import SALMONNDataset
from torch.utils.data import DataLoader


def _attention_rollout(
    full_attentions: torch.Tensor,
    active_positions: torch.Tensor,
    use_residual: bool = True,
) -> torch.Tensor:
    """
    Compute attention rollout per Abnar & Zuidema (2020) https://arxiv.org/pdf/2005.00928.
    With residual: A = 0.5*W_att + 0.5*I. Without: A = W_att.
    Rollout = A_L @ A_{L-1} @ ... @ A_1.
    """
    add_mask = torch.where(active_positions > 0.5, 0.0, float(torch.finfo(active_positions.dtype).min))
    attentions = full_attentions.mean(dim=2)
    seq_len = attentions.shape[-1]
    device, dtype = attentions.device, attentions.dtype

    if use_residual:
        I = torch.eye(seq_len, device=device, dtype=dtype)
        I = I.unsqueeze(0).unsqueeze(0).expand(attentions.shape[0], attentions.shape[1], -1, -1)
        A = 0.5 * attentions + 0.5 * I
    else:
        A = attentions
    A = torch.softmax(A + add_mask.unsqueeze(0), dim=-1)

    rollout = A[0]
    for l in range(1, A.shape[0]):
        rollout = torch.matmul(rollout, A[l])

    return rollout


def _excess_over_uniform(
    attention_slice: torch.Tensor,
    active_slice: torch.Tensor,
    start_speech: int,
    speech_end: int,
) -> torch.Tensor:
    """Excess over uniform. attention_slice: (..., num_rows, speech_len), active_slice: (..., num_rows, seq_len)."""
    actual = attention_slice.sum(dim=-1)
    num_valid_keys = active_slice.sum(dim=-1).clamp(min=1)
    num_valid_speech = active_slice[..., start_speech:speech_end].sum(dim=-1)
    expected = num_valid_speech / num_valid_keys
    return (actual - expected).mean()


def _extract_from_rollout(
    rollout_attention: torch.Tensor,
    active_positions: torch.Tensor,
    start_regress: int,
    start_speech: int,
    speech_end: int,
) -> torch.Tensor:
    """Excess over uniform for regress sequence."""
    to_regress_attention = rollout_attention[:, start_regress:, start_speech:speech_end]
    active_regress = active_positions[:, start_regress:, :]
    return _excess_over_uniform(to_regress_attention, active_regress, start_speech, speech_end)


def _raw_attention(
    full_attentions: torch.Tensor,
    active_positions: torch.Tensor,
    layer_idx: int,
) -> torch.Tensor:
    """Raw attention from a single layer. layer_idx: 0 for first, -1 for last."""
    add_mask = torch.where(active_positions > 0.5, 0.0, float(torch.finfo(active_positions.dtype).min))
    attentions = full_attentions.mean(dim=2)
    att = attentions[layer_idx]
    return torch.softmax(att + add_mask, dim=-1)


def extract_regress_sequence_attentions(
    full_attentions: torch.Tensor,
    mask_4d: torch.Tensor,
    start_regress: int,
    start_speech: int,
    speech_end: int,
    use_residual: bool = True,
    attention_map: torch.Tensor = None,
) -> torch.Tensor:
    """
    Extract the attention the sequence to regress put on the speech tokens.
    Returns excess over uniform. If attention_map is provided, use it; else compute rollout.
    """
    active_positions = 1.0 - mask_4d[:, 0, :, :].float()
    if attention_map is not None:
        return _extract_from_rollout(
            attention_map, active_positions, start_regress, start_speech, speech_end
        )
    rollout_attention = _attention_rollout(full_attentions, active_positions, use_residual=use_residual)
    return _extract_from_rollout(
        rollout_attention, active_positions, start_regress, start_speech, speech_end
    )


def extract_register_attentions(
    reg_token_id: int,
    full_attentions: torch.Tensor,
    to_regress_tokens: torch.Tensor,
    mask_4d: torch.Tensor,
    start_regress: int,
    start_speech: int,
    speech_end: int,
    use_residual: bool = True,
    attention_map: torch.Tensor = None,
) -> torch.Tensor:
    """
    Extract the attention the registers have on the speech tokens.
    Returns excess over uniform. If attention_map is provided, use it; else compute rollout.
    """
    batch_size = to_regress_tokens.shape[0]
    active_positions = 1.0 - mask_4d[:, 0, :, :].float()
    if attention_map is not None:
        map_to_use = attention_map
    else:
        map_to_use = _attention_rollout(full_attentions, active_positions, use_residual=use_residual)

    register_excess = []
    for i in range(batch_size):
        reg_ids = start_regress + torch.where(to_regress_tokens[i] == reg_token_id)[0]
        if len(reg_ids) == 0:
            continue
        att_slice = map_to_use[i : i + 1, reg_ids, start_speech:speech_end]
        active_slice = active_positions[i : i + 1, reg_ids, :]
        register_excess.append(_excess_over_uniform(att_slice, active_slice, start_speech, speech_end))

    # Return 0 if no registers were found
    if len(register_excess) == 0:
        return torch.tensor(0.0, device=map_to_use.device, dtype=map_to_use.dtype)
    
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
    method_keys = ["rollout_residual", "rollout_no_residual", "raw_first_layer", "raw_last_layer"]
    regress_sequence_sum = {k: 0.0 for k in method_keys}
    register_sum = {k: 0.0 for k in method_keys}
    regress_sequence_count = {k: 0 for k in method_keys}
    register_count = {k: 0 for k in method_keys}

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
        start_regress = meta["start_regress"]
        speech_len = meta["speech_len"]
        start_speech = meta["start_speech"]
        speech_end = min(start_speech + speech_len, start_regress)

        to_regress_tokens = meta.get("to_regress_tokens")
        active_positions = 1.0 - meta["mask_4d"][:, 0, :, :].float()

        for use_residual, suffix in [(True, "rollout_residual"), (False, "rollout_no_residual")]:
            regress_val = extract_regress_sequence_attentions(
                output["attentions"],
                meta["mask_4d"],
                start_regress,
                start_speech,
                speech_end,
                use_residual=use_residual,
            )
            regress_sequence_sum[suffix] += float(regress_val) * batch_size
            regress_sequence_count[suffix] += batch_size

            if to_regress_tokens is not None:
                register_val = extract_register_attentions(
                    model.llama_tokenizer.convert_tokens_to_ids("<reg>"),
                    output["attentions"],
                    to_regress_tokens,
                    meta["mask_4d"],
                    start_regress,
                    start_speech,
                    speech_end,
                    use_residual=use_residual,
                )
                register_sum[suffix] += float(register_val) * batch_size
                register_count[suffix] += batch_size

        for layer_idx, suffix in [(0, "raw_first_layer"), (-1, "raw_last_layer")]:
            raw_att = _raw_attention(output["attentions"], active_positions, layer_idx)
            regress_val = extract_regress_sequence_attentions(
                output["attentions"],
                meta["mask_4d"],
                start_regress,
                start_speech,
                speech_end,
                attention_map=raw_att,
            )
            regress_sequence_sum[suffix] += float(regress_val) * batch_size
            regress_sequence_count[suffix] += batch_size

            if to_regress_tokens is not None:
                register_val = extract_register_attentions(
                    model.llama_tokenizer.convert_tokens_to_ids("<reg>"),
                    output["attentions"],
                    to_regress_tokens,
                    meta["mask_4d"],
                    start_regress,
                    start_speech,
                    speech_end,
                    attention_map=raw_att,
                )
                register_sum[suffix] += float(register_val) * batch_size
                register_count[suffix] += batch_size

        num_batches += 1

    if num_batches == 0:
        print("No batches with attentions. Exiting.")
        return

    # Save summarized metrics as JSON
    metrics = {}
    for suffix in method_keys:
        rc = regress_sequence_count[suffix]
        rcc = register_count[suffix]
        metrics[suffix] = {
            "regress_sequence_excess_over_uniform": regress_sequence_sum[suffix] / rc if rc > 0 else 0.0,
            "register_excess_over_uniform": register_sum[suffix] / rcc if rcc > 0 else 0.0,
        }
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved attention metrics ({num_batches} batches) to {output_path}")


if __name__ == "__main__":
    main()
