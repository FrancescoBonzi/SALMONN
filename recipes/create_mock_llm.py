"""
Create a tiny mock LLaMA model for debugging purposes.
This creates a randomly initialized model with minimal dimensions.
"""
import os
from pathlib import Path

from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer


def create_mock_llm(output_dir: str = "pretrained/mock-llama"):
    """Create a tiny LLaMA model for debugging."""
    
    # Get project root (one level up from recipes/)
    project_root = Path(__file__).resolve().parent.parent
    
    output_path = project_root / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create a minimal LLaMA config (no GQA for compatibility with custom modeling_llama.py)
    config = LlamaConfig(
        vocab_size=32000,           # Standard LLaMA vocab size
        hidden_size=256,            # Tiny (normally 4096)
        intermediate_size=512,      # Tiny (normally 11008)
        num_hidden_layers=4,        # Tiny (normally 32)
        num_attention_heads=4,      # Tiny (normally 32)
        num_key_value_heads=4,      # Same as attention heads (no GQA for compatibility)
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    
    print(f"Creating mock LLaMA with config:")
    print(f"  hidden_size: {config.hidden_size}")
    print(f"  num_layers: {config.num_hidden_layers}")
    print(f"  num_heads: {config.num_attention_heads}")
    
    # Create randomly initialized model
    print("Initializing model (random weights)...")
    model = LlamaForCausalLM(config)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {num_params:,} ({num_params / 1e6:.1f}M)")
    
    # Save model
    print(f"Saving model to {output_path}...")
    model.save_pretrained(output_path)
    config.save_pretrained(output_path)
    
    # Copy tokenizer from a known source or create a simple one
    # We'll use the Vicuna tokenizer if available, otherwise create a basic one
    vicuna_path = project_root / "pretrained/vicuna-7b-v1.1"
    if vicuna_path.exists():
        print("Copying tokenizer from vicuna-7b-v1.1...")
        tokenizer = LlamaTokenizer.from_pretrained(vicuna_path)
        tokenizer.save_pretrained(output_path)
    else:
        # Download tokenizer from HuggingFace
        print("Downloading LLaMA tokenizer...")
        try:
            tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-7b")
            tokenizer.save_pretrained(output_path)
        except Exception as e:
            print(f"Could not download tokenizer: {e}")
            print("You may need to manually copy a LLaMA tokenizer to the output directory.")
    
    print(f"\n✓ Mock LLaMA saved to: {output_path}")
    print(f"\nTo use, update debug.yaml:")
    print(f'  llama_path: "{output_dir}"')


if __name__ == "__main__":
    create_mock_llm()

