#!/usr/bin/env python3
"""Dump cross-attention golden values from HuggingFace Whisper.

This script extracts reference weights and intermediate outputs from the
HuggingFace Whisper model for bit-for-bit comparison against whisper.apr.

Usage:
    uv run --with transformers --with torch --with numpy scripts/dump_cross_attn_golden.py

Output:
    debug_data/*.npy - NumPy arrays with reference values
"""

import os
import numpy as np
import torch
from transformers import WhisperModel


def save_tensor(name: str, tensor: torch.Tensor, output_dir: str = "debug_data") -> None:
    """Save tensor as uncompressed numpy for easy loading in Rust."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.npy")
    np.save(path, tensor.detach().cpu().numpy())
    print(f"Saved {name}: {tuple(tensor.shape)}")


def main() -> None:
    print("=== Cross-Attention Golden Value Extraction ===\n")

    # 1. Load Reference Model (whisper-tiny to match our target)
    model_name = "openai/whisper-tiny"
    print(f"Loading {model_name}...")
    model = WhisperModel.from_pretrained(model_name).eval()
    decoder_layer = model.decoder.layers[0]
    cross_attn = decoder_layer.encoder_attn

    # 2. Dump Weights (The suspected culprits)
    print("\n--- Dumping Weights ---")
    save_tensor("ref_q_weight", cross_attn.q_proj.weight)
    save_tensor("ref_q_bias", cross_attn.q_proj.bias)
    save_tensor("ref_k_weight", cross_attn.k_proj.weight)
    # Note: k_proj typically has no bias in Whisper
    if cross_attn.k_proj.bias is not None:
        save_tensor("ref_k_bias", cross_attn.k_proj.bias)
    else:
        print("ref_k_bias: None (no bias in k_proj)")
    save_tensor("ref_v_weight", cross_attn.v_proj.weight)
    save_tensor("ref_v_bias", cross_attn.v_proj.bias)
    save_tensor("ref_out_weight", cross_attn.out_proj.weight)
    save_tensor("ref_out_bias", cross_attn.out_proj.bias)

    # 3. Generate Deterministic Inputs
    print("\n--- Generating Test Inputs ---")
    torch.manual_seed(42)
    batch_size, seq_len_dec, dim = 1, 1, 384  # Decoder step 0, tiny d_model=384
    seq_len_enc = 1500  # Encoder fixed output (30s audio)

    # Hidden states from the Decoder (query source)
    hidden_states = torch.randn(batch_size, seq_len_dec, dim)
    # Hidden states from the Encoder (key/value source)
    encoder_hidden_states = torch.randn(batch_size, seq_len_enc, dim)

    save_tensor("input_decoder_hidden", hidden_states)
    save_tensor("input_encoder_hidden", encoder_hidden_states)

    # 4. Run Reference Forward Pass
    print("\n--- Running Reference Forward Pass ---")
    with torch.no_grad():
        # HF forward signature:
        # forward(hidden_states, key_value_states=encoder_hidden_states, ...)
        attn_output, attn_weights, _ = cross_attn(
            hidden_states,
            key_value_states=encoder_hidden_states,
            output_attentions=True,
        )

    save_tensor("ref_attn_output", attn_output)
    save_tensor("ref_attn_weights", attn_weights)

    # 5. Also dump intermediate projections for debugging
    print("\n--- Dumping Intermediate Values ---")
    with torch.no_grad():
        # Q projection (from decoder hidden states)
        q = cross_attn.q_proj(hidden_states)
        save_tensor("ref_q_projected", q)

        # K projection (from encoder hidden states)
        k = cross_attn.k_proj(encoder_hidden_states)
        save_tensor("ref_k_projected", k)

        # V projection (from encoder hidden states)
        v = cross_attn.v_proj(encoder_hidden_states)
        save_tensor("ref_v_projected", v)

    # 6. Print weight shapes for verification
    print("\n--- Weight Shape Summary ---")
    print(f"q_proj.weight: {tuple(cross_attn.q_proj.weight.shape)} (should be [384, 384])")
    print(f"k_proj.weight: {tuple(cross_attn.k_proj.weight.shape)} (should be [384, 384])")
    print(f"v_proj.weight: {tuple(cross_attn.v_proj.weight.shape)} (should be [384, 384])")
    print(f"out_proj.weight: {tuple(cross_attn.out_proj.weight.shape)} (should be [384, 384])")

    print("\n--- CRITICAL: Weight Layout ---")
    print("HuggingFace nn.Linear stores weights as [out_features, in_features]")
    print("If your Rust matmul expects [in, out], you MUST transpose during loading!")

    # 7. Print some statistics for sanity checking
    print("\n--- Weight Statistics ---")
    for name, param in [
        ("q_proj", cross_attn.q_proj.weight),
        ("k_proj", cross_attn.k_proj.weight),
        ("v_proj", cross_attn.v_proj.weight),
        ("out_proj", cross_attn.out_proj.weight),
    ]:
        print(
            f"{name}: mean={param.mean().item():.6f}, "
            f"std={param.std().item():.6f}, "
            f"norm={param.norm().item():.4f}"
        )

    print("\n✅ Golden values saved to debug_data/")
    print("\nNext: Run 'cargo run --example verify_cross_attn_values' to compare")


if __name__ == "__main__":
    main()
