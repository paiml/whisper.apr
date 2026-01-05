# Model Card: Whisper.apr

## Model Details

### Model Description

whisper.apr provides WASM-optimized implementations of OpenAI's Whisper speech recognition models.

- **Developed by:** PAIML
- **Model type:** Automatic Speech Recognition (ASR) - Encoder-Decoder Transformer
- **Original source:** [OpenAI Whisper](https://github.com/openai/whisper)
- **License:** MIT (both whisper.apr and original Whisper)

### Model Sources

| Model | Original Source | HuggingFace |
|-------|-----------------|-------------|
| tiny | openai/whisper | [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny) |
| base | openai/whisper | [openai/whisper-base](https://huggingface.co/openai/whisper-base) |
| small | openai/whisper | [openai/whisper-small](https://huggingface.co/openai/whisper-small) |

## Uses

### Direct Use

- Audio transcription (speech-to-text)
- Multilingual speech recognition (99 languages)
- Audio translation to English

### Out-of-Scope Use

- Speaker identification (not supported)
- Emotion detection (not designed for this)
- Real-time low-latency (<100ms) applications

## Bias, Risks, and Limitations

### Known Limitations

1. **Hallucinations**: May generate plausible but incorrect text for unclear audio
2. **Proper nouns**: May struggle with uncommon names, places, or technical terms
3. **Accents**: Performance varies by accent and dialect
4. **Background noise**: Degrades with significant background noise

### Recommendations

- Use VAD (Voice Activity Detection) for noisy audio
- Review output for critical applications
- Consider beam search for higher accuracy

## Training Details

### Training Data

whisper.apr uses weights from OpenAI's Whisper, trained on:
- 680,000 hours of multilingual and multitask supervised data
- Data collected from the internet

### Training Procedure

No training is performed by whisper.apr. We convert pre-trained weights from OpenAI's release.

## Evaluation

### Testing Data

Ground truth validation uses:
- LibriSpeech test-clean subset
- Custom test audio (`demos/test-audio/`)
- whisper.cpp reference output

### Metrics

| Model | WER (LibriSpeech) | RTF Target |
|-------|-------------------|------------|
| tiny | ~8% | < 2.0x |
| base | ~6% | < 2.5x |
| small | ~4% | < 4.0x |

### Results

See [REPRODUCIBILITY.md](../REPRODUCIBILITY.md) for detailed benchmark results and falsification protocols.

## Environmental Impact

### Carbon Emissions

whisper.apr inference is CPU/WASM-based, with minimal environmental impact:
- No GPU required for inference
- Typical power: < 10W during transcription
- Memory: 150-800 MB depending on model size

## Technical Specifications

### Model Architecture

| Model | Parameters | Layers | Dim | Heads | Vocab |
|-------|------------|--------|-----|-------|-------|
| tiny | 39M | 4 | 384 | 6 | 51,865 |
| base | 74M | 6 | 512 | 8 | 51,865 |
| small | 244M | 12 | 768 | 12 | 51,865 |

### Compute Infrastructure

#### Hardware

- Runs on any CPU with SIMD support (SSE4.2/AVX2/NEON)
- WASM SIMD 128-bit for browser deployment
- No GPU required

#### Software

- Rust 1.75+
- wasm-pack for WASM builds
- See [CONTRIBUTING.md](../CONTRIBUTING.md) for full requirements

## Model Card Contact

- GitHub Issues: https://github.com/paiml/whisper.apr/issues
- Repository: https://github.com/paiml/whisper.apr

## Citation

If you use whisper.apr, please cite:

```bibtex
@software{whisper_apr,
  title = {whisper.apr: WASM-first Whisper Implementation},
  author = {PAIML},
  year = {2024},
  url = {https://github.com/paiml/whisper.apr}
}
```

And the original Whisper paper:

```bibtex
@article{radford2022whisper,
  title={Robust Speech Recognition via Large-Scale Weak Supervision},
  author={Radford, Alec and Kim, Jong Wook and Xu, Tao and Brockman, Greg and McLeavey, Christine and Sutskever, Ilya},
  journal={arXiv preprint arXiv:2212.04356},
  year={2022}
}
```
