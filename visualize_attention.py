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


# Not using this anymore
def _attention_rollout(
    full_attentions: torch.Tensor,
    active_positions: torch.Tensor,
    use_residual: bool = True,
) -> torch.Tensor:
    """
    Compute attention rollout per Abnar & Zuidema (2020) https://arxiv.org/pdf/2005.00928.
    With residual: A = W_att + I. Without: A = W_att.
    Then row-normalize A and compose across layers.
    """
    attentions = full_attentions.mean(dim=2)
    seq_len = attentions.shape[-1]
    num_layers, batch_size = attentions.shape[0], attentions.shape[1]
    device, dtype = attentions.device, attentions.dtype

    # Enforce valid attention support (causal + padding mask) before normalization.
    valid = (active_positions > 0.5).to(dtype).unsqueeze(0)  # [1, B, Q, K]
    A = attentions * valid

    if use_residual:
        I = torch.eye(seq_len, device=device, dtype=dtype)
        I = I.unsqueeze(0).unsqueeze(0).expand(num_layers, batch_size, -1, -1)
        A = A + I

    # Row-normalization
    A = A / A.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(dtype).eps)

    rollout = A[0]
    for l in range(1, A.shape[0]):
        rollout = torch.matmul(rollout, A[l])

    return rollout


def _audio_excess_over_uniform(
    attention_slice: torch.Tensor,
    active_slice: torch.Tensor,
    start_speech: int,
    speech_end: int,
) -> torch.Tensor:
    """Excess over uniform for audio tokens."""
    actual = attention_slice.sum(dim=-1)
    num_valid_keys = active_slice.sum(dim=-1).clamp(min=1)
    num_valid_speech = active_slice[..., start_speech:speech_end].sum(dim=-1)
    expected = num_valid_speech / num_valid_keys
    return (actual - expected).mean()


def _text_excess_over_uniform(
    attention_slice: torch.Tensor,
    active_slice: torch.Tensor,
    start_speech: int,
    speech_end: int,
) -> torch.Tensor:
    """Excess over uniform for text tokens."""
    actual = attention_slice.sum(dim=-1)
    num_valid_keys = active_slice.sum(dim=-1).clamp(min=1)
    num_valid_text = active_slice[..., :start_speech].sum(dim=-1) + active_slice[..., speech_end:].sum(dim=-1)
    expected = num_valid_text / num_valid_keys
    return (actual - expected).mean()


def extract_regress_sequence_attentions(
    full_attentions: torch.Tensor,
    mask_4d: torch.Tensor,
    start_regress: int,
    start_speech: int,
    speech_end: int,
    layer_idx: int,
) -> torch.Tensor:
    """Excess over uniform for one layer."""
    layer_attention = full_attentions.mean(dim=2)[layer_idx]
    active_positions = 1.0 - mask_4d[:, 0, :, :].float()
    active_regress = active_positions[:, start_regress:, :]

    # Audio attention
    audio_attention = layer_attention[:, start_regress:, start_speech:speech_end]
    audio_excess = _audio_excess_over_uniform(audio_attention, active_regress, start_speech, speech_end)

    # Text attention
    text_attention = torch.cat(
        [
            layer_attention[:, start_regress:, :start_speech],
            layer_attention[:, start_regress:, speech_end:],
        ],
        dim=-1,
    )
    text_excess = _text_excess_over_uniform(text_attention, active_regress, start_speech, speech_end)

    return audio_excess.item(), text_excess.item()


def extract_register_attentions(
    reg_token_id: int,
    full_attentions: torch.Tensor,
    to_regress_tokens: torch.Tensor,
    mask_4d: torch.Tensor,
    start_regress: int,
    start_speech: int,
    speech_end: int,
    layer_idx: int,
) -> tuple[float, float]:
    """
    Extract the attention the registers have on the speech tokens.
    Returns excess over uniform. If attention_map is provided, use it; else compute rollout.
    """
    batch_size = to_regress_tokens.shape[0]
    layer_attention = full_attentions.mean(dim=2)[layer_idx]
    active_positions = 1.0 - mask_4d[:, 0, :, :].float()

    register_audio_excess = []
    register_text_excess = []
    for i in range(batch_size):
        reg_ids = start_regress + torch.where(to_regress_tokens[i] == reg_token_id)[0]
        if len(reg_ids) == 0:
            continue

        # Register audio attention
        att_slice = layer_attention[i : i + 1, reg_ids, start_speech:speech_end]
        active_slice = active_positions[i : i + 1, reg_ids, :]
        register_sample_audio_excess = _audio_excess_over_uniform(att_slice, active_slice, start_speech, speech_end)

        # Register text attention
        att_slice = torch.cat([
            layer_attention[i : i + 1, reg_ids, :start_speech],
            layer_attention[i : i + 1, reg_ids, speech_end:],
        ], dim=-1)
        active_slice = active_positions[i : i + 1, reg_ids, :]
        register_sample_text_excess = _text_excess_over_uniform(att_slice, active_slice, start_speech, speech_end)

        register_audio_excess.append(register_sample_audio_excess)
        register_text_excess.append(register_sample_text_excess)

    # Return 0 if no registers were found
    if len(register_audio_excess) == 0:
        return 0.0, 0.0
    
    return torch.stack(register_audio_excess, dim=0).mean().item(), torch.stack(register_text_excess, dim=0).mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-path", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for figures")
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
    num_layers = None
    metrics = {
        "audio_excess_over_uniform": {},
        "text_excess_over_uniform": {},
        "register_audio_excess_over_uniform": {},
        "register_text_excess_over_uniform": {},
    }
    for batch in tqdm(loader, desc="Extracting attention"):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        with torch.no_grad():
            output = model(batch, verbose=False, output_attentions=True)
            if "attentions" not in output or "attentions_meta" not in output:
                print("Model did not return attentions metadata for this batch. Skipping.")
                continue
            meta = output["attentions_meta"]

        num_layers = output["attentions"].shape[0]
        speech_end = min(meta["start_speech"] + meta["speech_len"], meta["start_regress"])
        for layer_idx in range(num_layers):
            # Regress sequence attention
            audio_excess, text_excess = extract_regress_sequence_attentions(
                output["attentions"],
                meta["mask_4d"],
                meta["start_regress"],
                meta["start_speech"],
                speech_end,
                layer_idx,
            )
            if layer_idx not in metrics["audio_excess_over_uniform"]:
                metrics["audio_excess_over_uniform"][layer_idx] = []
            if layer_idx not in metrics["text_excess_over_uniform"]:
                metrics["text_excess_over_uniform"][layer_idx] = []
            metrics["audio_excess_over_uniform"][layer_idx].append(audio_excess)
            metrics["text_excess_over_uniform"][layer_idx].append(text_excess)

            # Register attention
            reg_token_id = model.llama_tokenizer.convert_tokens_to_ids("<reg>")
            register_audio_excess, register_text_excess = extract_register_attentions(
                reg_token_id,
                output["attentions"],
                meta["to_regress_tokens"],
                meta["mask_4d"],
                meta["start_regress"],
                meta["start_speech"],
                speech_end,
                layer_idx,
            )
            if layer_idx not in metrics["register_audio_excess_over_uniform"]:
                metrics["register_audio_excess_over_uniform"][layer_idx] = []
            if layer_idx not in metrics["register_text_excess_over_uniform"]:
                metrics["register_text_excess_over_uniform"][layer_idx] = []
            metrics["register_audio_excess_over_uniform"][layer_idx].append(register_audio_excess)
            metrics["register_text_excess_over_uniform"][layer_idx].append(register_text_excess)

        num_batches += 1

    if num_batches == 0 or num_layers is None:
        print("No batches with attentions. Exiting.")
        return

    metrics["audio_excess_over_uniform"] = {
        layer_idx: float(torch.tensor(metrics["audio_excess_over_uniform"][layer_idx], dtype=torch.float32).mean().item())
        for layer_idx in range(num_layers)
    }
    metrics["text_excess_over_uniform"] = {
        layer_idx: float(torch.tensor(metrics["text_excess_over_uniform"][layer_idx], dtype=torch.float32).mean().item())
        for layer_idx in range(num_layers)
    }
    metrics["register_audio_excess_over_uniform"] = {
        layer_idx: float(torch.tensor(metrics["register_audio_excess_over_uniform"][layer_idx], dtype=torch.float32).mean().item())
        for layer_idx in range(num_layers)
    }
    metrics["register_text_excess_over_uniform"] = {
        layer_idx: float(torch.tensor(metrics["register_text_excess_over_uniform"][layer_idx], dtype=torch.float32).mean().item())
        for layer_idx in range(num_layers)
    }

    # Save summarized metrics as JSON
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved attention metrics ({num_batches} batches) to {output_path}")


if __name__ == "__main__":
    main()
