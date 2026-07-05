# Whisper Model Distillation & Pruning Workflow

This document outlines the workflow for extracting a smaller decoder from a pre-trained Whisper model, specifically targeting a 2-layer decoder extraction.

## 1. Prune the Decoder

Use the provided Python script `tools/prune_decoder.py` to load a standard Hugging Face Whisper model and prune its decoder layers. By default, it preserves the first and last decoder layers, which is a common strategy for 2-layer distillation.

### Requirements

```bash
pip install transformers torch
```

### Execution

Run the script to create a pruned model:

```bash
python3 tools/prune_decoder.py \
    --model openai/whisper-tiny \
    --save-dir models/whisper-tiny-2L \
    --layers 2
```

This will:
- Load the `openai/whisper-tiny` model.
- Copy the encoder as-is.
- Copy the first and last layers of the decoder.
- Save the new model and processor configuration to `models/whisper-tiny-2L`.

## 2. Convert to `.bin` format for `whisper.apr`

Once the model is pruned and saved in the Hugging Face format, convert it to the format required by `whisper.apr`:

```bash
cargo run --release --bin convert -- \
    --model models/whisper-tiny-2L \
    --output models/whisper-tiny-2L/model.bin
```

## 3. Fine-tuning (Optional but Recommended)

A pruned model will typically have degraded performance initially. It is highly recommended to fine-tune the pruned model on your target dataset (using knowledge distillation from the original model or standard fine-tuning) to recover accuracy.

## 4. Usage in Inference

You can now use the converted `model.bin` with `whisper.apr` as usual. The inference engine will dynamically adapt to the smaller number of decoder layers.

```bash
cargo run --release --bin apr -- models/whisper-tiny-2L/model.bin audio.wav
```
