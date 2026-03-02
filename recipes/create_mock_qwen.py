"""
Create a tiny mock Qwen2.5-Omni Thinker model for debugging purposes.
This creates a randomly initialized model with minimal dimensions.
Similar to create_mock_llm.py for LLaMA/Vicuna.
Includes a minimal preprocessor (no network required after first run).
"""
import json
from pathlib import Path

from transformers import (
    Qwen2_5OmniThinkerForConditionalGeneration,
    Qwen2_5OmniProcessor,
)
from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import (
    Qwen2_5OmniThinkerConfig,
    Qwen2_5OmniAudioEncoderConfig,
    Qwen2_5OmniVisionEncoderConfig,
    Qwen2_5OmniTextConfig,
)


def create_mock_qwen(output_dir: str = "pretrained/mock-qwen"):
    """Create a tiny Qwen2.5-Omni Thinker model for debugging."""

    # Get project root (one level up from recipes/)
    project_root = Path(__file__).resolve().parent.parent

    output_path = project_root / output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    # Minimal hidden size (must be consistent across text, audio, vision)
    hidden_size = 128

    # Minimal text config (1 layer, 2 heads, head_dim=64)
    # rope_scaling.mrope_section: doubled in modeling, must sum to head_dim/2 = 32
    text_config = Qwen2_5OmniTextConfig(
        vocab_size=152064,  # Keep original vocab for tokenizer compatibility
        hidden_size=hidden_size,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=2048,
        max_window_layers=1,
        rope_scaling={"type": "default", "rope_type": "default", "mrope_section": [8, 12, 12]},
    )

    # Minimal audio config (output_dim must match text hidden_size)
    # num_mel_bins=80 to match whisper-tiny (used in qwen_debug.yaml)
    audio_config = Qwen2_5OmniAudioEncoderConfig(
        num_mel_bins=80,
        d_model=64,
        encoder_layers=1,
        encoder_attention_heads=2,
        encoder_ffn_dim=128,
        output_dim=hidden_size,
        max_source_positions=1500,
        n_window=100,
    )

    # Minimal vision config (out_hidden_size must match text hidden_size)
    vision_config = Qwen2_5OmniVisionEncoderConfig(
        depth=1,
        hidden_size=64,
        intermediate_size=128,
        num_heads=2,
        out_hidden_size=hidden_size,
        fullatt_block_indexes=[0],
        window_size=14,
    )

    # Thinker config
    thinker_config = Qwen2_5OmniThinkerConfig(
        audio_config=audio_config,
        vision_config=vision_config,
        text_config=text_config,
    )

    print("Creating mock Qwen2.5-Omni Thinker with config:")
    print(f"  text hidden_size: {text_config.hidden_size}, layers: {text_config.num_hidden_layers}")
    print(f"  audio output_dim: {audio_config.output_dim}, layers: {audio_config.encoder_layers}")
    print(f"  vision out_hidden_size: {vision_config.out_hidden_size}, depth: {vision_config.depth}")

    # Create randomly initialized model
    print("Initializing model (random weights)...")
    model = Qwen2_5OmniThinkerForConditionalGeneration(thinker_config)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {num_params:,} ({num_params / 1e6:.1f}M)")

    # Save model
    print(f"Saving model to {output_path}...")
    model.save_pretrained(output_path)

    # Add minimal preprocessor (required for Qwen2_5OmniProcessor.from_pretrained)
    _save_mock_preprocessor(output_path)

    print(f"\n✓ Mock Qwen2.5-Omni saved to: {output_path}")
    print(f"\nTo use, update qwen_debug.yaml:")
    print(f'  qwen25_omni_path: "{output_dir}"')


def _save_mock_preprocessor(output_path: Path):
    """Save preprocessor files so Qwen2_5OmniProcessor can load from mock path."""
    hub_id = "Qwen/Qwen2.5-Omni-7B"

    # Download processor files from hub (excludes model weights; requires network on first run)
    print("Downloading processor files from Qwen/Qwen2.5-Omni-7B...")
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=hub_id,
            local_dir=output_path,
            local_dir_use_symlinks=False,
            ignore_patterns=["model*.safetensors", "*.md", "LICENSE", ".gitattributes"],
            allow_patterns=[
                "preprocessor_config.json",
                "tokenizer*.json",
                "vocab.json",
                "merges.txt",
                "special_tokens_map.json",
                "added_tokens.json",
                "chat_template.json",
            ],
        )
        print("  Processor files downloaded successfully")
    except Exception as e:
        print(f"Could not download processor files: {e}")
        print("Creating minimal tokenizer files...")
        _save_minimal_tokenizer(output_path)


def _save_minimal_tokenizer(output_path: Path):
    """Save minimal preprocessor/tokenizer files for offline mock (no network)."""
    # Minimal preprocessor_config.json (required for processor to load)
    preprocessor_config = {
        "chunk_length": 300,
        "dither": 0.0,
        "feature_extractor_type": "WhisperFeatureExtractor",
        "feature_size": 128,
        "hop_length": 160,
        "image_mean": [0.48145466, 0.4578275, 0.40821073],
        "image_processor_type": "Qwen2VLImageProcessor",
        "image_std": [0.26862954, 0.26130258, 0.27577711],
        "max_pixels": 12845056,
        "merge_size": 2,
        "min_pixels": 3136,
        "n_fft": 400,
        "n_samples": 4800000,
        "nb_max_frames": 30000,
        "padding_side": "right",
        "padding_value": 0.0,
        "patch_size": 14,
        "processor_class": "Qwen2_5OmniProcessor",
        "return_attention_mask": True,
        "sampling_rate": 16000,
        "temporal_patch_size": 2,
    }
    (output_path / "preprocessor_config.json").write_text(
        json.dumps(preprocessor_config, indent=2)
    )

    # Minimal tokenizer config and special tokens
    tokenizer_config = {
        "auto_map": {"AutoTokenizer": ["transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer", None]},
        "bos_token": None,
        "eos_token": "<|im_end|>",
        "model_max_length": 32768,
        "pad_token": "<|endoftext|>",
        "tokenizer_class": "Qwen2TokenizerFast",
        "unk_token": None,
    }
    (output_path / "tokenizer_config.json").write_text(
        json.dumps(tokenizer_config, indent=2)
    )

    special_tokens = {
        "additional_special_tokens": [
            "<|im_start|>", "<|im_end|>", "<|AUDIO|>", "<|audio_bos|>", "<|audio_eos|>",
            "<|vision_bos|>", "<|vision_eos|>", "<|IMAGE|>", "<|VIDEO|>",
        ],
        "eos_token": {"content": "<|im_end|>", "rstrip": False, "lstrip": False, "normalized": False, "single_word": False},
        "pad_token": {"content": "<|endoftext|>", "rstrip": False, "lstrip": False, "normalized": False, "single_word": False},
    }
    (output_path / "special_tokens_map.json").write_text(
        json.dumps(special_tokens, indent=2)
    )

    # Minimal vocab and merges (required for BPE) - use GPT-2 style minimal set
    # Qwen2TokenizerFast needs tokenizer.json; fallback: create minimal vocab
    vocab = {"<|endoftext|>": 0, "<|im_start|>": 1, "<|im_end|>": 2}
    for i in range(3, 1000):
        vocab[f"<|token_{i}|>"] = i
    (output_path / "vocab.json").write_text(json.dumps(vocab))
    (output_path / "merges.txt").write_text("# minimal merges\n")
    (output_path / "added_tokens.json").write_text('{"<|im_start|>": 1, "<|im_end|>": 2, "<|endoftext|>": 0}')
    print("  Created minimal tokenizer (vocab may not match model - use hub download for full compatibility)")
    print("  Run with network to download full processor: python recipes/create_mock_qwen.py")
    print("  Or set qwen_debug.yaml whisper_path to Qwen/Qwen2.5-Omni-7B for processor fallback in models/qwen.py")


if __name__ == "__main__":
    create_mock_qwen()
