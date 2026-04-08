"""
Benchmark TTS accuracy (WER) for Voxtral-4B-TTS via the sgl-omni API server.
Evaluates TTS quality using the seed-tts-eval dataset by:
1. Sending text to the /v1/audio/speech endpoint
2. Transcribing the generated audio with Whisper (EN) or FunASR (ZH)
3. Computing Word Error Rate against the original text
Unlike voice-cloning models, Voxtral TTS does NOT use ref_audio/ref_text.
It takes only text + voice (e.g. "cheerful_female") as input.
Prerequisites
-------------
    # 1. Start the server
    export CUDA_VISIBLE_DEVICES=1
    sgl-omni serve \\
        --model-path mistralai/Voxtral-4B-TTS-2603 \\
        --port 8000
    # 2. Download dataset
    huggingface-cli download zhaochenyang20/seed-tts-eval \\
        --repo-type dataset --local-dir seedtts_testset
Usage
-----
    # English evaluation
    python benchmarks/accuracy/tts/benchmark_voxtral_tts_wer.py \\
        --meta seedtts_testset/en/meta.lst \\
        --output-dir results/voxtral_en \\
        --lang en --max-samples 10
    # Chinese evaluation
    python benchmarks/accuracy/tts/benchmark_voxtral_tts_wer.py \\
        --meta seedtts_testset/zh/meta.lst \\
        --output-dir results/voxtral_zh \\
        --lang zh --max-samples 10
    # With custom voice and server address
    python benchmarks/accuracy/tts/benchmark_voxtral_tts_wer.py \\
        --meta seedtts_testset/en/meta.lst \\
        --output-dir results/voxtral_en \\
        --lang en --max-samples 10 \\
        --voice cheerful_female --port 8000
"""

from __future__ import annotations

import argparse
import logging
import os
import time

import requests
from tqdm import tqdm

from tts_wer_utils import (
    DEFAULT_DATASET_DIR,
    DEFAULT_WHISPER_EN,
    DEFAULT_WHISPER_ZH,
    SUMMARY_LABEL_WIDTH,
    SUMMARY_LINE_WIDTH,
    WAV_HEADER_SIZE,
    SampleOutput,
    calculate_metrics,
    compute_wer,
    get_wav_duration,
    load_asr_model,
    parse_meta_lst,
    print_wer_metrics,
    save_results,
    transcribe,
    wait_for_service,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def generate_speech_via_api(
    text: str,
    base_url: str,
    model_name: str,
    voice: str = "cheerful_female",
    max_new_tokens: int = 4096,
    timeout: int = 120,
) -> tuple[bytes, float]:
    """Call /v1/audio/speech and return (wav_bytes, latency_s).
    Voxtral TTS does not use ref_audio/ref_text (no voice cloning).
    It only needs text + voice name.
    """
    api_url = f"{base_url}/v1/audio/speech"
    payload = {
        "model": model_name,
        "input": text,
        "voice": voice,
        "response_format": "wav",
        "max_new_tokens": max_new_tokens,
    }

    t0 = time.perf_counter()
    response = requests.post(api_url, json=payload, timeout=timeout)
    latency = time.perf_counter() - t0

    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:500]}")

    wav_bytes = response.content
    if len(wav_bytes) <= WAV_HEADER_SIZE:
        raise ValueError(f"Empty or invalid audio response ({len(wav_bytes)} bytes)")

    return wav_bytes, latency


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def print_summary(metrics: dict, args: argparse.Namespace) -> None:
    lw = SUMMARY_LABEL_WIDTH
    w = SUMMARY_LINE_WIDTH
    print(f"\n{'=' * w}")
    print(f"{'Voxtral TTS WER Benchmark Result':^{w}}")
    print(f"{'=' * w}")
    print(f"  {'Model:':<{lw}} {args.model}")
    print(f"  {'Voice:':<{lw}} {args.voice}")
    print(f"  {'Language:':<{lw}} {args.lang}")
    asr_label = args.asr_model or (
        DEFAULT_WHISPER_ZH if args.lang == "zh" else DEFAULT_WHISPER_EN
    )
    print(f"  {'ASR model:':<{lw}} {asr_label}")
    print(f"  {'Completed samples:':<{lw}} {metrics['completed']}")
    print(f"  {'Failed samples:':<{lw}} {metrics['failed']}")
    print_wer_metrics(metrics)


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------


def benchmark(args: argparse.Namespace) -> None:
    # Resolve meta path from --dataset-dir and --lang if not explicitly set
    if args.meta is None:
        args.meta = os.path.join(args.dataset_dir, args.lang, "meta.lst")
        logger.info("Resolved meta path: %s", args.meta)

    if not os.path.isfile(args.meta):
        logger.error("Meta file not found: %s", args.meta)
        return

    base_url = args.base_url or f"http://{args.host}:{args.port}"

    # Wait for server to be ready
    wait_for_service(base_url)

    samples = parse_meta_lst(args.meta, args.max_samples)
    logger.info("Loaded %d samples from %s", len(samples), args.meta)

    # Pre-load ASR model
    load_asr_model(args.lang, device=args.asr_device, whisper_model=args.asr_model)

    # Create audio output dir
    audio_dir = os.path.join(args.output_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    outputs: list[SampleOutput] = []
    for sample in tqdm(samples, desc="Generating & evaluating"):
        output = SampleOutput(
            sample_id=sample.sample_id,
            target_text=sample.target_text,
        )

        try:
            # Generate speech via API
            wav_bytes, latency = generate_speech_via_api(
                text=sample.target_text,
                base_url=base_url,
                model_name=args.model,
                voice=args.voice,
                max_new_tokens=args.max_new_tokens,
            )
            output.latency = latency

            output.audio_duration = get_wav_duration(wav_bytes)

            # Save audio
            audio_path = os.path.join(audio_dir, f"{sample.sample_id}.wav")
            with open(audio_path, "wb") as f:
                f.write(wav_bytes)

            # ASR transcribe
            logger.info("[ASR] Transcribing %s on %s", audio_path, args.asr_device)
            hypothesis = transcribe(
                audio_path,
                args.lang,
                asr_device=args.asr_device,
                whisper_model=args.asr_model,
            )
            output.hypothesis = hypothesis

            # Compute WER
            output.wer = compute_wer(sample.target_text, hypothesis, args.lang)
            output.is_success = True

            logger.info(
                "[%s] WER=%.4f | ref=%r | hyp=%r",
                sample.sample_id,
                output.wer,
                sample.target_text[:80],
                hypothesis[:80],
            )
        except Exception as e:
            output.error = str(e)
            logger.error("Error on sample %s: %s", sample.sample_id, e, exc_info=True)

        outputs.append(output)

    # Compute and print metrics
    metrics = calculate_metrics(outputs)
    print_summary(metrics, args)
    save_results(
        outputs,
        metrics,
        output_dir=args.output_dir,
        config={
            "model": args.model,
            "voice": args.voice,
            "meta": args.meta,
            "lang": args.lang,
            "base_url": args.base_url or f"http://{args.host}:{args.port}",
            "max_new_tokens": args.max_new_tokens,
            "max_samples": args.max_samples,
            "asr_device": args.asr_device,
            "asr_model": args.asr_model,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Voxtral TTS accuracy (WER) using seed-tts-eval."
    )
    parser.add_argument(
        "--meta",
        type=str,
        default=None,
        help=(
            "Path to seed-tts-eval meta.lst file. If not set, "
            "resolved from --dataset-dir and --lang."
        ),
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=DEFAULT_DATASET_DIR,
        help=(
            "Path to seed-tts-eval dataset root (contains zh/ and en/ subdirs). "
            f"Default: {DEFAULT_DATASET_DIR}"
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Voxtral-4B-TTS-2603",
        help="Model name for the API request.",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default="cheerful_female",
        help="Voice to use for synthesis (e.g. cheerful_female).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/voxtral_tts_wer",
        help="Directory to save results and audio files.",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        choices=["zh", "en"],
        help="Language for ASR and text normalization.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Server host.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Base URL (e.g. http://localhost:8000). Overrides --host/--port.",
    )
    parser.add_argument(
        "--asr-device",
        type=str,
        default="cpu",
        help="Device for ASR model (cpu recommended to save GPU memory).",
    )
    parser.add_argument(
        "--asr-model",
        type=str,
        default=None,
        help=(
            "Whisper model size for ASR (e.g. tiny, base.en, small, medium, "
            "large-v3). Default: base.en for EN, medium for ZH."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=4096,
        help="Maximum number of new tokens for AR generation.",
    )
    args = parser.parse_args()

    benchmark(args)


if __name__ == "__main__":
    main()
