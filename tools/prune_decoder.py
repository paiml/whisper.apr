#!/usr/bin/env python3
"""
Prune a Hugging Face Whisper model's decoder to a smaller number of layers.
This is the first step for model distillation. By default, it extracts a 2-layer decoder
by keeping the first and last decoder layers of the original model.

Usage:
    pip install transformers torch
    python3 tools/prune_decoder.py --model openai/whisper-tiny --save-dir models/whisper-tiny-2L
"""

import argparse
import copy
import os

try:
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
except ImportError:
    print("Please install required packages: pip install transformers torch")
    exit(1)

def main():
    parser = argparse.ArgumentParser(description="Prune Whisper decoder layers for distillation")
    parser.add_argument("--model", type=str, default="openai/whisper-tiny", help="Source model name or path")
    parser.add_argument("--save-dir", type=str, required=True, help="Directory to save the pruned model")
    parser.add_argument("--layers", type=int, default=2, help="Number of decoder layers to keep")
    args = parser.parse_args()

    print(f"Loading source model: {args.model}")
    processor = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model)

    original_layers = model.config.decoder_layers
    if args.layers >= original_layers:
        print(f"Error: Requested layers ({args.layers}) must be less than original decoder layers ({original_layers})")
        return

    print(f"Original decoder layers: {original_layers}")
    print(f"Target decoder layers: {args.layers}")

    # Determine which layers to keep
    if args.layers == 1:
        keep_layers = [0]
    elif args.layers == 2:
        keep_layers = [0, original_layers - 1]
    else:
        # Spread layers evenly, always including the first and last if possible
        step = (original_layers - 1) / (args.layers - 1)
        keep_layers = [int(round(i * step)) for i in range(args.layers)]

    print(f"Keeping decoder layers: {keep_layers}")

    # Create new config
    config = copy.deepcopy(model.config)
    config.decoder_layers = args.layers

    # Initialize new model
    print("Initializing new pruned model...")
    pruned_model = WhisperForConditionalGeneration(config)

    # Copy weights
    state_dict = model.state_dict()
    pruned_state_dict = pruned_model.state_dict()

    for key in state_dict.keys():
        if key.startswith("model.decoder.layers."):
            parts = key.split(".")
            layer_idx = int(parts[3])
            if layer_idx in keep_layers:
                new_idx = keep_layers.index(layer_idx)
                parts[3] = str(new_idx)
                new_key = ".".join(parts)
                pruned_state_dict[new_key] = state_dict[key]
        else:
            if key in pruned_state_dict:
                pruned_state_dict[key] = state_dict[key]
    
    pruned_model.load_state_dict(pruned_state_dict)

    print(f"Saving pruned model to {args.save_dir}")
    os.makedirs(args.save_dir, exist_ok=True)
    pruned_model.save_pretrained(args.save_dir)
    processor.save_pretrained(args.save_dir)
    
    print("Done! You can now use this model for distillation or fine-tuning.")
    print("To convert for whisper.apr, run:")
    print(f"    cargo run --release --bin convert -- --model {args.save_dir} --output {args.save_dir}/model.bin")

if __name__ == "__main__":
    main()
