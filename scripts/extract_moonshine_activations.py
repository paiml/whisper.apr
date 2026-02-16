#!/usr/bin/env python3
"""One-shot Moonshine activation extraction for parity debugging.

Registers PyTorch forward hooks at every layer boundary in HuggingFace
Moonshine, runs inference on test audio, and outputs JSON in ProbeOutput
format compatible with `whisper-apr apr parity`.

Usage:
    uv run --with transformers --with torch --with soundfile \
        scripts/extract_moonshine_activations.py \
        demos/test-audio/test-speech-1.5s.wav \
        test_data/moonshine_tiny_activations.json

Output JSON schema matches whisper-apr's ProbeOutput:
    {
        "model": "moonshine-tiny (HuggingFace)",
        "audio": "test-speech-1.5s.wav",
        "model_family": "moonshine",
        "checkpoints": [
            {
                "name": "conv_stem.conv1_out",
                "shape": [seq_len, d_model],
                "l2": float,
                "mean": float,
                "std_dev": float,
                "min": float,
                "max": float,
                "first_n": [float...],
                "full_data": null
            },
            ...
        ]
    }
"""

import json
import sys
from pathlib import Path

import torch


def tensor_stats(name: str, t: torch.Tensor, first_n: int = 8) -> dict:
    """Compute activation statistics matching whisper-apr's ActivationSnapshot."""
    data = t.detach().float().cpu().flatten()
    return {
        "name": name,
        "shape": list(t.shape),
        "l2": float(torch.norm(data, p=2).item()),
        "mean": float(data.mean().item()),
        "std_dev": float(data.std().item()) if data.numel() > 1 else 0.0,
        "min": float(data.min().item()),
        "max": float(data.max().item()),
        "first_n": data[:first_n].tolist(),
        "full_data": None,
    }


def make_hook(name: str, checkpoints: list):
    """Create a forward hook that records activation stats."""
    def hook_fn(module, input, output):
        t = output if isinstance(output, torch.Tensor) else output[0]
        checkpoints.append(tensor_stats(name, t))
    return hook_fn


def _hook_first_attr(module, attr_names: list, hook_name: str, checkpoints: list):
    """Register a hook on the first matching attribute of module."""
    for attr in attr_names:
        if hasattr(module, attr):
            getattr(module, attr).register_forward_hook(make_hook(hook_name, checkpoints))
            return


def _hook_conv_stem(encoder, checkpoints: list):
    """Register hooks on the conv stem layers."""
    _hook_first_attr(encoder, ['conv1'], "conv_stem.conv1_out", checkpoints)
    _hook_first_attr(encoder, ['group_norm', 'groupnorm'], "conv_stem.groupnorm_out", checkpoints)
    _hook_first_attr(encoder, ['conv2'], "conv_stem.conv2_out", checkpoints)
    _hook_first_attr(encoder, ['conv3'], "conv_stem.conv3_out", checkpoints)
    _hook_first_attr(encoder, ['layer_norm', 'ln'], "conv_stem.layernorm_out", checkpoints)


def _hook_encoder_blocks(encoder, checkpoints: list):
    """Register hooks on encoder transformer blocks."""
    encoder_layers = None
    for attr in ['layers', 'blocks', 'encoder_layers']:
        if hasattr(encoder, attr):
            encoder_layers = getattr(encoder, attr)
            break

    if encoder_layers is None:
        return

    for i, block in enumerate(encoder_layers):
        prefix = f"encoder.block_{i}"
        _hook_first_attr(block, ['input_layernorm', 'ln1', 'layer_norm1'], f"{prefix}.ln1_out", checkpoints)
        _hook_first_attr(block, ['self_attn', 'attention', 'mha'], f"{prefix}.self_attn_out", checkpoints)
        _hook_first_attr(block, ['post_attention_layernorm', 'ln2', 'layer_norm2'], f"{prefix}.ln2_out", checkpoints)
        _hook_first_attr(block, ['mlp', 'ffn', 'feed_forward'], f"{prefix}.ffn_out", checkpoints)
        block.register_forward_hook(make_hook(f"{prefix}.residual_2", checkpoints))

    _hook_first_attr(encoder, ['final_layer_norm', 'ln_post', 'layer_norm'], "encoder.ln_post_out", checkpoints)


def _hook_decoder_blocks(decoder, checkpoints: list):
    """Register hooks on decoder transformer blocks."""
    _hook_first_attr(decoder, ['embed_tokens', 'token_embedding', 'wte'], "decoder.token_emb", checkpoints)

    decoder_layers = None
    for attr in ['layers', 'blocks', 'decoder_layers']:
        if hasattr(decoder, attr):
            decoder_layers = getattr(decoder, attr)
            break

    if decoder_layers is None:
        return

    for i, block in enumerate(decoder_layers):
        prefix = f"decoder.block_{i}"
        _hook_first_attr(block, ['input_layernorm', 'ln1', 'self_attn_layer_norm'], f"{prefix}.ln1_out", checkpoints)
        _hook_first_attr(block, ['self_attn', 'attention'], f"{prefix}.self_attn_out", checkpoints)
        _hook_first_attr(block, ['encoder_attn_layer_norm', 'ln_cross', 'cross_attn_layer_norm'], f"{prefix}.ln_cross_out", checkpoints)
        _hook_first_attr(block, ['encoder_attn', 'cross_attn', 'cross_attention'], f"{prefix}.cross_attn_out", checkpoints)
        _hook_first_attr(block, ['final_layer_norm', 'ln2', 'ffn_layer_norm'], f"{prefix}.ln2_out", checkpoints)
        _hook_first_attr(block, ['mlp', 'ffn', 'feed_forward', 'fc'], f"{prefix}.ffn_out", checkpoints)
        block.register_forward_hook(make_hook(f"{prefix}.residual_3", checkpoints))

    _hook_first_attr(decoder, ['final_layer_norm', 'ln_post', 'layer_norm'], "decoder.ln_post_out", checkpoints)


def load_audio(audio_path: str):
    """Load and resample audio to 16kHz."""
    import soundfile as sf
    audio, sr = sf.read(audio_path)
    if sr != 16000:
        import torchaudio
        audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
        audio_tensor = torchaudio.functional.resample(audio_tensor, sr, 16000)
        audio = audio_tensor.squeeze(0).numpy()
    return audio


def extract_activations(audio_path: str, output_path: str, model_name: str = "usefulsensors/moonshine-tiny"):
    """Extract activations from HuggingFace Moonshine model."""
    from transformers import AutoModel, AutoProcessor

    print(f"Loading model: {model_name}")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model.eval()

    audio = load_audio(audio_path)
    print(f"Audio: {audio_path} ({len(audio)} samples, {len(audio)/16000:.2f}s)")

    checkpoints = []

    _hook_conv_stem(model.encoder, checkpoints)
    _hook_encoder_blocks(model.encoder, checkpoints)

    decoder = model.decoder if hasattr(model, 'decoder') else None
    if decoder is not None:
        _hook_decoder_blocks(decoder, checkpoints)

    print("Running inference with hooks...")
    with torch.no_grad():
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=1)

    print(f"Captured {len(checkpoints)} checkpoints")

    output = {
        "model": f"{model_name} (HuggingFace)",
        "audio": str(Path(audio_path).name),
        "model_family": "moonshine",
        "checkpoints": checkpoints,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Written to: {output_path}")
    print("\nCheckpoint summary:")
    for cp in checkpoints:
        print(f"  {cp['name']:<40} L2={cp['l2']:.4f}  mean={cp['mean']:.6f}  shape={cp['shape']}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <audio.wav> <output.json> [model_name]")
        print(f"  model_name defaults to usefulsensors/moonshine-tiny")
        sys.exit(1)

    audio_file = sys.argv[1]
    output_file = sys.argv[2]
    model = sys.argv[3] if len(sys.argv) > 3 else "usefulsensors/moonshine-tiny"

    extract_activations(audio_file, output_file, model)
