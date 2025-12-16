#!/usr/bin/env python3
"""
Compare decoder logits between HuggingFace and our implementation.

Usage:
    uv run --with transformers --with torch --with librosa tools/compare_logits_hf.py
"""

import numpy as np
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

def main():
    print("=" * 60)
    print("LOGITS COMPARISON: HuggingFace vs whisper.apr")
    print("=" * 60)
    print()

    # Load HuggingFace model
    print("Loading HuggingFace whisper-tiny...")
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")

    # Load audio
    import librosa
    audio_path = "demos/test-audio/test-speech-1.5s.wav"
    audio, sr = librosa.load(audio_path, sr=16000)

    print(f"Audio: {len(audio)} samples ({len(audio)/16000:.2f}s)")

    # Process with HuggingFace
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")

    # Check mel spectrogram
    mel = inputs.input_features[0].numpy()
    print(f"Mel spectrogram shape: {mel.shape}")
    print(f"Mel stats (full): mean={mel.mean():.4f}, std={mel.std():.4f}")

    # Compare first portion (actual audio, not padding)
    # 1.5s audio = ~150 frames at hop=160
    audio_frames = min(150, mel.shape[1])
    mel_audio = mel[:, :audio_frames]
    print(f"Mel stats (first {audio_frames} frames): mean={mel_audio.mean():.4f}, std={mel_audio.std():.4f}")

    # Save for comparison
    np.save("/tmp/hf_mel_full.npy", mel)
    np.save("/tmp/hf_mel_audio.npy", mel_audio.T)  # Transpose to (frames, mels)

    # Encode
    with torch.no_grad():
        encoder_outputs = model.get_encoder()(inputs.input_features)

    print(f"\nEncoder output shape: {encoder_outputs.last_hidden_state.shape}")
    enc_np = encoder_outputs.last_hidden_state[0].numpy()
    print(f"Encoder output stats: mean={enc_np.mean():.4f}, std={enc_np.std():.4f}")

    # Initial decoder tokens
    # SOT=50257, lang_en=50258, transcribe=50358, notimestamps=50362
    decoder_input_ids = torch.tensor([[50257, 50258, 50358, 50362]])

    # Get decoder output with past_key_values (for comparison)
    with torch.no_grad():
        outputs = model(
            input_features=inputs.input_features,
            decoder_input_ids=decoder_input_ids,
            return_dict=True,
        )

    # Get logits for last position
    logits = outputs.logits[0, -1, :].numpy()

    print(f"\n=== HuggingFace Logits (last position) ===")
    print(f"Shape: {logits.shape}")
    print(f"Mean:  {logits.mean():.4f}")
    print(f"Std:   {logits.std():.4f}")
    print(f"Min:   {logits.min():.4f}")
    print(f"Max:   {logits.max():.4f}")

    # Top 10 tokens
    top_indices = np.argsort(logits)[::-1][:10]
    print(f"\nTop 10 tokens:")
    for i, idx in enumerate(top_indices):
        print(f"  {i+1}. token {idx}: logit={logits[idx]:.4f}")

    # Check for expected tokens
    print(f"\nExpected token positions:")
    print(f"  Token 440 ' The': logit={logits[440]:.4f}")
    print(f"  Token 464 'The':  logit={logits[464]:.4f}")
    print(f"  Token 220 ' ':    logit={logits[220]:.4f}")

    # Distribution analysis
    print(f"\nLogit distribution:")
    percentiles = [1, 10, 25, 50, 75, 90, 99]
    for p in percentiles:
        val = np.percentile(logits, p)
        print(f"  {p}%: {val:.4f}")

    # Check if all positive (like our implementation)
    print(f"\nAll positive: {(logits > 0).all()}")
    print(f"Negative count: {(logits < 0).sum()}")

    # Save encoder output and logits for comparison with our implementation
    np.save("/tmp/hf_encoder_output.npy", enc_np)
    np.save("/tmp/hf_logits.npy", logits)
    print(f"\nSaved encoder output to /tmp/hf_encoder_output.npy")
    print(f"Saved logits to /tmp/hf_logits.npy")

    # Also save intermediate decoder hidden states
    print("\n=== Decoder Hidden States ===")

    # Get decoder hidden states
    with torch.no_grad():
        decoder_outputs = model.model.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            output_hidden_states=True,
            return_dict=True,
        )

    # Last hidden state (before projection to vocab)
    last_hidden = decoder_outputs.last_hidden_state[0, -1, :].numpy()
    print(f"Last hidden state (before projection):")
    print(f"  Shape: {last_hidden.shape}")
    print(f"  Mean:  {last_hidden.mean():.4f}")
    print(f"  Std:   {last_hidden.std():.4f}")
    print(f"  Min:   {last_hidden.min():.4f}")
    print(f"  Max:   {last_hidden.max():.4f}")
    print(f"  L2:    {np.linalg.norm(last_hidden):.4f}")

    np.save("/tmp/hf_last_hidden.npy", last_hidden)


if __name__ == "__main__":
    main()
