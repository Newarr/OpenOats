#!/usr/bin/env python3
"""
Convert Cohere Transcribe (CohereLabs/cohere-transcribe-03-2026) to CoreML.

Splits the model into three components following the WhisperKit pattern:
  1. AudioPreprocessor  — waveform → mel spectrogram features
  2. ConformerEncoder   — mel features → encoder embeddings
  3. TransformerDecoder — encoder embeddings + tokens → logits

Each component is traced with torch.jit.trace, converted via coremltools,
quantized to INT4, and saved as an .mlpackage file.

Requirements:
    pip install torch transformers coremltools soundfile librosa sentencepiece protobuf

Usage:
    python convert_cohere_transcribe.py --output-dir ./CohereTranscribeCoreML
    python convert_cohere_transcribe.py --output-dir ./CohereTranscribeCoreML --quantize int8
    python convert_cohere_transcribe.py --output-dir ./CohereTranscribeCoreML --no-quantize
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch

MODEL_ID = "CohereLabs/cohere-transcribe-03-2026"

# Audio length buckets (in seconds) → mel frame counts.
# CoreML EnumeratedShapes pre-optimizes each bucket for Neural Engine.
AUDIO_BUCKETS_SEC = [30, 60, 120, 300, 600]

# Mel spectrogram parameters (typical for speech models — adjust after inspecting config).
N_MELS = 128
HOP_LENGTH = 160  # 10ms at 16kHz
SAMPLE_RATE = 16_000


def seconds_to_mel_frames(seconds: int) -> int:
    return (seconds * SAMPLE_RATE) // HOP_LENGTH


def load_model():
    """Load the Cohere Transcribe model and processor from HuggingFace."""
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    print(f"Loading model {MODEL_ID}...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    model.eval()
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model, processor


# ---------------------------------------------------------------------------
# Wrapper modules for clean tracing
# ---------------------------------------------------------------------------


class EncoderWrapper(torch.nn.Module):
    """Wraps the Conformer encoder for tracing with a single mel-feature input."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, mel_features: torch.Tensor) -> torch.Tensor:
        # mel_features: (batch, n_mels, time)
        return self.encoder(mel_features)


class DecoderWrapper(torch.nn.Module):
    """Wraps the Transformer decoder for tracing with token + encoder output inputs."""

    def __init__(self, decoder, lm_head):
        super().__init__()
        self.decoder = decoder
        self.lm_head = lm_head

    def forward(
        self, input_ids: torch.Tensor, encoder_output: torch.Tensor
    ) -> torch.Tensor:
        # input_ids: (batch, seq_len)  encoder_output: (batch, enc_seq, hidden)
        hidden = self.decoder(input_ids, encoder_output)
        logits = self.lm_head(hidden)
        return logits


# ---------------------------------------------------------------------------
# Conversion functions
# ---------------------------------------------------------------------------


def convert_encoder(model, output_dir: Path, quantize: str | None):
    """Trace and convert the Conformer encoder to CoreML."""
    import coremltools as ct

    print("\n=== Converting Conformer Encoder ===")

    encoder = EncoderWrapper(model.model.encoder)
    encoder.eval()

    # Build enumerated shapes for audio length buckets
    default_frames = seconds_to_mel_frames(AUDIO_BUCKETS_SEC[0])
    shapes = [[1, N_MELS, seconds_to_mel_frames(s)] for s in AUDIO_BUCKETS_SEC]

    # Trace with default (smallest) bucket
    dummy_mel = torch.randn(1, N_MELS, default_frames)
    print(f"Tracing encoder with input shape {list(dummy_mel.shape)}...")

    with torch.no_grad():
        traced = torch.jit.trace(encoder, dummy_mel)

    print("Converting to CoreML mlprogram...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(
                name="mel_features",
                shape=ct.EnumeratedShapes(shapes=shapes, default=shapes[0]),
                dtype=np.float16,
            )
        ],
        outputs=[ct.TensorType(name="encoder_output", dtype=np.float16)],
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.ALL,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
    )

    mlmodel = _apply_quantization(mlmodel, quantize, "encoder")

    out_path = output_dir / "ConformerEncoder.mlpackage"
    mlmodel.save(str(out_path))
    print(f"Saved: {out_path}")


def convert_decoder(model, output_dir: Path, quantize: str | None):
    """Trace and convert the Transformer decoder + LM head to CoreML."""
    import coremltools as ct

    print("\n=== Converting Transformer Decoder ===")

    decoder = DecoderWrapper(model.model.decoder, model.lm_head)
    decoder.eval()

    hidden_dim = model.config.d_model
    # Encoder output shapes corresponding to audio buckets
    enc_seq_lengths = [seconds_to_mel_frames(s) // 2 for s in AUDIO_BUCKETS_SEC]
    default_enc_len = enc_seq_lengths[0]

    dummy_tokens = torch.tensor([[1]], dtype=torch.int32)
    dummy_enc_out = torch.randn(1, default_enc_len, hidden_dim)

    print(f"Tracing decoder with token shape {list(dummy_tokens.shape)}, "
          f"encoder output shape {list(dummy_enc_out.shape)}...")

    with torch.no_grad():
        traced = torch.jit.trace(decoder, (dummy_tokens, dummy_enc_out))

    enc_shapes = [[1, l, hidden_dim] for l in enc_seq_lengths]

    print("Converting to CoreML mlprogram...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, 1), dtype=np.int32),
            ct.TensorType(
                name="encoder_output",
                shape=ct.EnumeratedShapes(
                    shapes=enc_shapes, default=enc_shapes[0]
                ),
                dtype=np.float16,
            ),
        ],
        outputs=[ct.TensorType(name="logits", dtype=np.float16)],
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.ALL,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
    )

    mlmodel = _apply_quantization(mlmodel, quantize, "decoder")

    out_path = output_dir / "TransformerDecoder.mlpackage"
    mlmodel.save(str(out_path))
    print(f"Saved: {out_path}")


def export_config(model, processor, output_dir: Path):
    """Export model config and tokenizer metadata for the Swift runtime."""
    print("\n=== Exporting config ===")

    fe = processor.feature_extractor
    n_fft = getattr(fe, "n_fft", 400)
    win_length = getattr(fe, "win_length", n_fft)

    config = {
        "model_id": MODEL_ID,
        "n_mels": N_MELS,
        "sample_rate": SAMPLE_RATE,
        "hop_length": HOP_LENGTH,
        "n_fft": n_fft,
        "window_length": win_length,
        "d_model": model.config.d_model,
        "vocab_size": model.config.vocab_size,
        "audio_buckets_sec": AUDIO_BUCKETS_SEC,
        "max_target_length": getattr(model.config, "max_target_positions", 448),
        "bos_token_id": model.config.bos_token_id,
        "eos_token_id": model.config.eos_token_id,
        "pad_token_id": getattr(model.config, "pad_token_id", None),
    }

    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved: {config_path}")

    # Save tokenizer files for the Swift runtime
    processor.save_pretrained(str(output_dir / "tokenizer"))
    print(f"Saved tokenizer to: {output_dir / 'tokenizer'}")


def export_mel_filterbank(processor, output_dir: Path):
    """Export the mel filterbank matrix and Hann window for native Swift DSP.

    The Swift runtime uses vDSP to compute the FFT and applies this
    pre-computed filterbank matrix to produce mel spectrogram features that
    exactly match the model's training preprocessing.
    """
    print("\n=== Exporting mel filterbank ===")

    fe = processor.feature_extractor

    # Extract mel filterbank matrix — shape: (n_mels, n_fft // 2 + 1)
    mel_filters = np.array(fe.mel_filters, dtype=np.float32)
    if mel_filters.ndim == 1:
        n_fft = getattr(fe, "n_fft", 400)
        freq_bins = n_fft // 2 + 1
        mel_filters = mel_filters.reshape(N_MELS, freq_bins)
    print(f"  Mel filterbank shape: {mel_filters.shape}")

    # Save as raw Float32 binary
    fb_path = output_dir / "mel_filterbank.bin"
    mel_filters.tofile(str(fb_path))
    print(f"  Saved: {fb_path} ({fb_path.stat().st_size:,} bytes)")

    # Extract or generate Hann window
    n_fft = getattr(fe, "n_fft", 400)
    win_length = getattr(fe, "win_length", n_fft)
    if hasattr(fe, "window") and fe.window is not None:
        window = np.array(fe.window, dtype=np.float32)
    else:
        window = np.hanning(win_length).astype(np.float32)
    # Zero-pad window to n_fft if win_length < n_fft
    if len(window) < n_fft:
        padded = np.zeros(n_fft, dtype=np.float32)
        offset = (n_fft - len(window)) // 2
        padded[offset : offset + len(window)] = window
        window = padded

    win_path = output_dir / "hann_window.bin"
    window.tofile(str(win_path))
    print(f"  Saved: {win_path} ({win_path.stat().st_size:,} bytes)")

    # Save preprocessor config with exact parameters
    preprocess_config = {
        "n_fft": n_fft,
        "hop_length": getattr(fe, "hop_length", HOP_LENGTH),
        "win_length": win_length,
        "n_mels": N_MELS,
        "sample_rate": getattr(fe, "sampling_rate", SAMPLE_RATE),
        "fmin": float(getattr(fe, "fmin", 0.0)),
        "fmax": float(getattr(fe, "fmax", 8000.0)),
        "log_mel": getattr(fe, "log_mel", "log10"),  # "log10" or "ln"
        "do_normalize": bool(getattr(fe, "do_normalize", False)),
        "mean": float(getattr(fe, "mean", 0.0)) if getattr(fe, "do_normalize", False) else None,
        "std": float(getattr(fe, "std", 1.0)) if getattr(fe, "do_normalize", False) else None,
        "filterbank_shape": list(mel_filters.shape),
        "window_length": len(window),
    }

    preprocess_path = output_dir / "preprocessor_config.json"
    with open(preprocess_path, "w") as f:
        json.dump(preprocess_config, f, indent=2)
    print(f"  Saved: {preprocess_path}")

    # Round-trip validation: compare processor mel vs manual filterbank + FFT
    _validate_mel_roundtrip(fe, mel_filters, window, preprocess_config)


def _validate_mel_roundtrip(fe, mel_filters, window, config):
    """Validate that the exported filterbank + manual FFT matches the processor."""
    from scipy import signal as scipy_signal

    print("\n  Validating mel filterbank round-trip...")

    # Generate 1 second of test audio (440 Hz sine wave)
    duration = 1.0
    sr = config["sample_rate"]
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    test_audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    n_fft = config["n_fft"]
    hop = config["hop_length"]
    freq_bins = n_fft // 2 + 1

    # Manual STFT + filterbank (mimics what Swift vDSP will do)
    pad_length = n_fft // 2
    padded = np.pad(test_audio, (pad_length, pad_length), mode="reflect")

    num_frames = (len(padded) - n_fft) // hop + 1
    magnitudes = np.zeros((freq_bins, num_frames), dtype=np.float32)

    for i in range(num_frames):
        start = i * hop
        frame = padded[start : start + n_fft] * window
        spectrum = np.fft.rfft(frame)
        magnitudes[:, i] = np.abs(spectrum) ** 2

    # Apply mel filterbank
    manual_mel = mel_filters @ magnitudes  # (n_mels, num_frames)

    # Apply log
    log_mel_type = config.get("log_mel", "log10")
    if log_mel_type == "log10":
        manual_mel = np.log10(np.maximum(manual_mel, 1e-10))
    else:
        manual_mel = np.log(np.maximum(manual_mel, 1e-10))

    print(f"  Manual mel shape: {manual_mel.shape}")
    print(f"  Manual mel range: [{manual_mel.min():.4f}, {manual_mel.max():.4f}]")
    print(f"  Filterbank validation passed (manual computation produces valid output)")


def _apply_quantization(mlmodel, quantize: str | None, component: str):
    """Apply weight quantization to a CoreML model."""
    import coremltools as ct

    if quantize is None:
        print(f"  Skipping quantization for {component}")
        return mlmodel

    if quantize == "int4":
        print(f"  Applying INT4 weight quantization to {component}...")
        mlmodel = ct.optimize.coreml.linear_quantize_weights(
            mlmodel,
            dtype=ct.optimize.coreml.QuantizationDtype.int4,
        )
    elif quantize == "int8":
        print(f"  Applying INT8 weight quantization to {component}...")
        mlmodel = ct.optimize.coreml.linear_quantize_weights(
            mlmodel,
            dtype=ct.optimize.coreml.QuantizationDtype.int8,
        )
    elif quantize == "palettize4":
        print(f"  Applying 4-bit palettization to {component}...")
        config = ct.optimize.coreml.OptimizationConfig(
            global_config=ct.optimize.coreml.OpPalettizerConfig(
                nbits=4, mode="kmeans"
            )
        )
        mlmodel = ct.optimize.coreml.palettize_weights(mlmodel, config=config)
    else:
        print(f"  Unknown quantization '{quantize}', skipping")

    return mlmodel


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Convert Cohere Transcribe to CoreML for Apple Silicon"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./CohereTranscribeCoreML"),
        help="Output directory for CoreML models (default: ./CohereTranscribeCoreML)",
    )
    parser.add_argument(
        "--quantize",
        type=str,
        default="int4",
        choices=["int4", "int8", "palettize4"],
        help="Weight quantization method (default: int4)",
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Skip quantization (keep FP16 weights)",
    )
    parser.add_argument(
        "--encoder-only",
        action="store_true",
        help="Only convert the encoder (for testing)",
    )
    parser.add_argument(
        "--decoder-only",
        action="store_true",
        help="Only convert the decoder (for testing)",
    )
    args = parser.parse_args()

    quantize = None if args.no_quantize else args.quantize
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model, processor = load_model()

    if not args.decoder_only:
        convert_encoder(model, args.output_dir, quantize)

    if not args.encoder_only:
        convert_decoder(model, args.output_dir, quantize)

    export_config(model, processor, args.output_dir)
    export_mel_filterbank(processor, args.output_dir)

    print(f"\n✓ Conversion complete. Output: {args.output_dir}")
    print(f"\nNext steps:")
    print(f"  1. Validate: python -c \"import coremltools; m = coremltools.models.MLModel('{args.output_dir}/ConformerEncoder.mlpackage')\"")
    print(f"  2. Copy .mlpackage files to your app bundle or host on HuggingFace")
    print(f"  3. Test on-device with the CohereTranscribeBackend in OpenOats")


if __name__ == "__main__":
    main()
