#!/usr/bin/env python3
"""
Visualize attention maps for MutorBERTConclusionSALMONN.

Usage:
  # Or with python directly
  python visualize_attention.py --cfg-path recipes/afthink/debug.yaml
"""

import argparse
import json
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from config import Config
from models import load_model
from dataset import SALMONNDataset
from torch.utils.data import DataLoader


def extract_conclusion_attentions(
    full_attentions: torch.Tensor,
    to_regress_tokens: torch.Tensor,
    mask_4d: torch.Tensor,
    start_regress: int,
    speech_len: int,
    pad_token_id: int,
    token_before_conclusion: int = 13,
) -> tuple[list, torch.Tensor]:
    """
    Extract conclusion→speech and register→speech attention from raw LLaMA attentions.
    Causal: query can only attend to key positions <= query. Speech is before conclusion,
    so we need conclusion→speech (query=conclusion, key=speech), not speech→conclusion.

    Returns:
        conclusion_attentions: list of (num_layers, num_heads, conclusion_len, speech_len) per batch
        register_attentions: (num_layers, batch, num_heads, 1, speech_len)
    """
    batch_size = to_regress_tokens.shape[0]
    seq_len = full_attentions.shape[-1]
    speech_key_slice = slice(1, 1 + speech_len)

    def _masked_mean(attn: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Weight attention by valid (non-padded) positions and average over query/key dims."""
        valid_counts = (~mask.bool()).sum(dim=-1).unsqueeze(-1)
        return (attn * valid_counts).mean(dim=(-2, -1))

    conclusion_attentions = []
    for i in range(batch_size):
        # Find conclusion span: from token after last token_before_conclusion to first pad
        last_marker = torch.where(to_regress_tokens[i] == token_before_conclusion)[0][-1]
        conclusion_start_rel = last_marker.item() + 1
        pad_positions = torch.where(to_regress_tokens[i] == pad_token_id)[0]
        pads_after = pad_positions[pad_positions >= conclusion_start_rel]
        conclusion_end_rel = pads_after[0].item() if pads_after.numel() > 0 else to_regress_tokens.shape[1]

        conclusion_start = start_regress + conclusion_start_rel
        conclusion_end = start_regress + conclusion_end_rel
        query_slice = slice(conclusion_start, min(conclusion_end, seq_len))

        attn = full_attentions[:, i, :, query_slice, speech_key_slice]
        mask = mask_4d[i, :, query_slice, :]
        conclusion_attentions.append(_masked_mean(attn, mask))

    conclusion_attentions = torch.stack(conclusion_attentions, dim=0).mean(dim=0).mean(dim=1)
    return conclusion_attentions


def extract_register_attentions(
    model_type: str,
    full_attentions: torch.Tensor,
    to_regress_tokens: torch.Tensor,
    mask_4d: torch.Tensor,
    start_regress: int,
    speech_len: int,
) -> torch.Tensor:
    """
    Extract register→speech attention from raw LLaMA attentions.

    Returns:
        register_attentions: (num_layers, num_heads)
    """
    batch_size = to_regress_tokens.shape[0]
    num_layers, _, num_heads, _, _ = full_attentions.shape
    speech_key_slice = slice(1, 1 + speech_len)

    def _masked_mean(attn: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Weight attention by valid (non-padded) positions and average over query/key dims."""
        valid_counts = (~mask.bool()).sum(dim=-1).unsqueeze(-1)
        return (attn * valid_counts).mean(dim=(-2, -1))

    register_attentions = []
    for i in range(batch_size):
        if model_type == "mutor_bert_conclusion":
            register_query_slice = slice(start_regress, start_regress + 1)
            register_attn = full_attentions[:, i, :, register_query_slice, speech_key_slice]
            register_mask = mask_4d[i, :, register_query_slice, :]
            register_attentions.append(_masked_mean(register_attn, register_mask))
        elif model_type == "salmonn":
            register_attentions.append(torch.zeros(num_layers, num_heads, device=full_attentions.device))

    register_attentions = torch.stack(register_attentions, dim=0).mean(dim=0).mean(dim=1)
    return register_attentions


def plot_attention_heatmaps(
    conclusion_attentions: torch.Tensor,
    register_attentions: torch.Tensor,
    out_dir: str = "attention_maps",
):
    """
    Plot conclusion and register attention per layer (averaged over heads).
    Inputs shape [num_layers, num_heads]; plots show 1D per-layer values.
    """
    os.makedirs(out_dir, exist_ok=True)
    num_layers = conclusion_attentions.shape[0]
    vmin, vmax = 0.9, 1.1

    c = conclusion_attentions.detach().cpu().numpy()
    r = register_attentions.detach().cpu().numpy()

    # Conclusion attention per layer
    fig, ax = plt.subplots(figsize=(max(8, num_layers * 0.4), 4))
    ax.bar(np.arange(num_layers), c, color="steelblue", edgecolor="navy", alpha=0.8)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.7)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Attention (avg over heads)")
    ax.set_xticks(np.arange(num_layers))
    ax.set_ylim(vmin, vmax)
    ax.set_title("Conclusion → speech attention (avg over dataset & heads)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "conclusion_attention.png"), dpi=150)
    plt.close()

    # Register attention per layer
    fig, ax = plt.subplots(figsize=(max(8, num_layers * 0.4), 4))
    ax.bar(np.arange(num_layers), r, color="darkorange", edgecolor="darkred", alpha=0.8)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.7)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Attention (avg over heads)")
    ax.set_xticks(np.arange(num_layers))
    ax.set_ylim(vmin, vmax)
    ax.set_title("Register → speech attention (avg over dataset & heads)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "register_attention.png"), dpi=150)
    plt.close()

    # Save summarized metrics as JSON
    metrics = {
        "conclusion_attention": {
            "mean": float(np.mean(c)),
            "std": float(np.std(c)),
            "min": float(np.min(c)),
            "max": float(np.max(c)),
            "per_layer": c.tolist(),
        },
        "register_attention": {
            "mean": float(np.mean(r)),
            "std": float(np.std(r)),
            "min": float(np.min(r)),
            "max": float(np.max(r)),
            "per_layer": r.tolist(),
        },
        "shape": {"num_layers": int(num_layers)},
    }
    with open(os.path.join(out_dir, "attention_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-path", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for figures")
    parser.add_argument("--layers", type=str, default=None, help="Comma-separated layer indices, e.g. 0,15,31")
    parser.add_argument("options", nargs="*", help="Config overrides, e.g. model.model_type=mutor_bert_conclusion")
    args = parser.parse_args()

    # Load config
    class Args:
        cfg_path = args.cfg_path
        options = args.options or []

    cfg = Config(Args())
    model_config = cfg.config.model
    data_config = cfg.config.datasets

    # Load model
    model = load_model(model_config)
    if model_config.get("model_type") != "mutor_bert_conclusion":
        print("Warning: Use model_type: mutor_bert_conclusion for attention visualization.")
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

    sum_conclusion = None
    sum_register = None
    num_batches = 0

    for batch in loader:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        with torch.no_grad():
            output = model(batch, verbose=False, output_attentions=True)

        if "attentions" not in output:
            print("Model did not return attentions. Ensure you use MutorBERTConclusionSALMONN with output_attentions=True.")
            continue

        meta = output["attentions_meta"]
        conclusion_attentions = extract_conclusion_attentions(
            output["attentions"],
            meta["to_regress_tokens"],
            meta["mask_4d"],
            meta["start_regress"],
            meta["speech_len"],
            model.llama_tokenizer.pad_token_id,
        )

        register_attentions = extract_register_attentions(
            model_config.model_type,
            output["attentions"],
            meta["to_regress_tokens"],
            meta["mask_4d"],
            meta["start_regress"],
            meta["speech_len"],
        )
        # register_attentions: (num_layers, batch, num_heads) -> average to (num_layers, num_heads)
        if register_attentions.ndim == 3:
            register_attentions = register_attentions.mean(dim=1)

        if sum_conclusion is None:
            sum_conclusion = conclusion_attentions.clone()
            sum_register = register_attentions.clone()
        else:
            sum_conclusion += conclusion_attentions
            sum_register += register_attentions
        num_batches += 1

    if num_batches == 0:
        print("No batches with attentions. Exiting.")
        return

    avg_conclusion = sum_conclusion / num_batches
    avg_register = sum_register / num_batches
    output_dir = cfg.config.run.output_dir if args.output_dir is None else args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    plot_attention_heatmaps(avg_conclusion, avg_register, out_dir=output_dir)
    print(f"Saved averaged attention maps ({num_batches} batches)")


if __name__ == "__main__":
    main()
