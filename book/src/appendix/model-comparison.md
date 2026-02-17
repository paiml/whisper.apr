# Whisper Model Comparison

## Model Overview

whisper.apr supports two model families: OpenAI Whisper and Moonshine ASR.

## Whisper Models

| Model | Params | d_model | Heads | Enc Layers | Dec Layers | Mels | Vocab |
|-------|--------|---------|-------|------------|------------|------|-------|
| tiny | 39M | 384 | 6 | 4 | 4 | 80 | 51,865 |
| tiny.en | 39M | 384 | 6 | 4 | 4 | 80 | 51,864 |
| base | 74M | 512 | 8 | 6 | 6 | 80 | 51,865 |
| base.en | 74M | 512 | 8 | 6 | 6 | 80 | 51,864 |
| small | 244M | 768 | 12 | 12 | 12 | 80 | 51,865 |
| small.en | 244M | 768 | 12 | 12 | 12 | 80 | 51,864 |
| medium | 769M | 1024 | 16 | 24 | 24 | 80 | 51,865 |
| large | 1.5B | 1280 | 20 | 32 | 32 | 128 | 51,866 |
| large-v3-turbo | 809M | 1280 | 20 | 32 | 4 | 128 | 51,866 |

### English-Only vs Multilingual

- **English-only** (`.en` suffix): GPT-2 tokenizer, EOT=50256, 51,864 vocab
- **Multilingual**: Extended tokenizer, EOT=50257, 51,865+ vocab, 99 languages

### Large v3 Turbo

The Turbo variant uses 32 encoder layers but only 4 decoder layers, reducing inference cost while maintaining encoder quality. Best for tasks where encoder representations matter most.

## Moonshine Models

| Model | Params | d_model | Heads | Enc Layers | Dec Layers |
|-------|--------|---------|-------|------------|------------|
| moonshine-tiny | 27M | 288 | 6 | 6 | 6 |
| moonshine-base | 61M | 416 | 8 | 8 | 8 |

### Moonshine vs Whisper

| Feature | Moonshine | Whisper |
|---------|-----------|--------|
| **Encoder** | ConvStem (lightweight) | Conv1D + Transformer |
| **Decoder** | Grouped Query Attention (GQA) | Multi-Head Attention (MHA) |
| **Positional Encoding** | RoPE (Rotary) | Sinusoidal |
| **Model Size** | 27-61M | 39M-1.5B |
| **Best For** | Low-latency, edge devices | High accuracy, multilingual |

## Performance Comparison

| Model | Relative Speed | Memory | Accuracy (WER) |
|-------|---------------|--------|-----------------|
| moonshine-tiny | Fastest | ~50 MB | Good for English |
| tiny | Fast | ~150 MB | Good |
| moonshine-base | Fast | ~100 MB | Better for English |
| base | Moderate | ~300 MB | Better |
| small | Slower | ~800 MB | High |
| large-v3-turbo | Slow | ~1.5 GB | Very High |
| large | Slowest | ~3 GB | Highest |

## Model Format Support

| Model | .apr | GGUF | SafeTensors |
|-------|------|------|-------------|
| Whisper (all sizes) | Yes | Yes | Convert to .apr |
| Moonshine | Yes | No | Convert to .apr |
| Large v3 Turbo | Yes | Yes | Convert to .apr |

## Choosing a Model

1. **Edge/mobile deployment**: moonshine-tiny (27M, fastest)
2. **Browser deployment**: tiny or base (39-74M, good balance)
3. **Server deployment**: small or large-v3-turbo (best accuracy/speed)
4. **Maximum accuracy**: large (1.5B, highest WER)
5. **Multilingual**: Any non-`.en` Whisper model (99 languages)
6. **English only**: `.en` variants or Moonshine models
