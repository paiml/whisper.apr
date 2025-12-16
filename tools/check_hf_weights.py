#!/usr/bin/env python3
"""
Check layer norm weights from HuggingFace Whisper model.

Usage:
    pip install transformers torch
    python3 tools/check_hf_weights.py
"""

try:
    from transformers import WhisperModel
    import torch
except ImportError:
    print("Please install: pip install transformers torch")
    exit(1)

def main():
    print("=" * 60)
    print("HUGGINGFACE WHISPER LAYER NORM WEIGHTS")
    print("=" * 60)
    print()

    print("Loading openai/whisper-tiny...")
    model = WhisperModel.from_pretrained("openai/whisper-tiny")

    print("\n=== Decoder Layer Norm (Final) ===\n")

    # The final layer norm in decoder
    ln = model.decoder.layer_norm

    weight = ln.weight.detach().numpy()
    bias = ln.bias.detach().numpy()

    print("decoder.layer_norm.weight:")
    print(f"  shape: {weight.shape}")
    print(f"  mean:  {weight.mean():.6f}")
    print(f"  range: [{weight.min():.4f}, {weight.max():.4f}]")
    print(f"  first 5: {weight[:5]}")
    print()

    print("decoder.layer_norm.bias:")
    print(f"  mean:  {bias.mean():.6f}")
    print(f"  first 5: {bias[:5]}")
    print()

    # Encoder final layer norm
    print("\n=== Encoder Layer Norm (Final) ===\n")

    enc_ln = model.encoder.layer_norm
    enc_weight = enc_ln.weight.detach().numpy()

    print("encoder.layer_norm.weight:")
    print(f"  mean:  {enc_weight.mean():.6f}")
    print(f"  range: [{enc_weight.min():.4f}, {enc_weight.max():.4f}]")
    print(f"  first 5: {enc_weight[:5]}")
    print()

    # Check some decoder block layer norms
    print("\n=== Decoder Block Layer Norms ===\n")

    for i, block in enumerate(model.decoder.layers):
        ln1 = block.self_attn_layer_norm.weight.detach().numpy()
        ln2 = block.encoder_attn_layer_norm.weight.detach().numpy()
        ln3 = block.final_layer_norm.weight.detach().numpy()

        print(f"Layer {i}:")
        print(f"  self_attn_layer_norm.weight mean: {ln1.mean():.4f}")
        print(f"  encoder_attn_layer_norm.weight mean: {ln2.mean():.4f}")
        print(f"  final_layer_norm.weight mean: {ln3.mean():.4f}")

    # Encoder block layer norms
    print("\n=== Encoder Block Layer Norms ===\n")

    for i, block in enumerate(model.encoder.layers):
        ln1 = block.self_attn_layer_norm.weight.detach().numpy()
        ln2 = block.final_layer_norm.weight.detach().numpy()

        print(f"Layer {i}:")
        print(f"  self_attn_layer_norm.weight mean: {ln1.mean():.4f}")
        print(f"  final_layer_norm.weight mean: {ln2.mean():.4f}")

    print("\n" + "=" * 60)
    print("EXPECTED: All layer norm weights should have mean ~1.0")
    print("=" * 60)


if __name__ == "__main__":
    main()
