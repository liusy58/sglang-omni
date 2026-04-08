"""
Benchmark TTS accuracy (WER) for Ming-flash-omni's MingOmniTalker.

Evaluates voice-cloning TTS quality using the seed-tts-eval dataset by:
1. Loading MingOmniTalker + AudioVAE directly on a single GPU (~4GB VRAM)
2. For each sample: voice-clone from ref_audio/ref_text, generate speech for
   target_text, ASR-transcribe, and compute WER
3. Reporting corpus-level micro-average WER + per-sample stats

Dataset: seed-tts-eval

    Download:
        huggingface-cli download zhaochenyang20/seed-tts-eval \
            --repo-type dataset --local-dir seedtts_testset

Usage:
    python benchmarks/accuracy/tts/benchmark_ming_tts_wer.py \
        --dataset-dir seedtts_testset \
        --model-path inclusionAI/Ming-flash-omni-2.0 \
        --output-dir /tmp/results_ming_zh \
        --lang zh --device cuda:0 --max-samples 50

    Or specify meta.lst directly:
    python benchmarks/accuracy/tts/benchmark_ming_tts_wer.py \
        --meta /tmp/seedtts_testset/zh/meta.lst \
        --model-path inclusionAI/Ming-flash-omni-2.0 \
        --output-dir /tmp/results_ming_zh \
        --lang zh --device cuda:0 --max-samples 50
"""

from __future__ import annotations

import argparse
import logging
import os
import time

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

from tts_wer_utils import (
    DEFAULT_DATASET_DIR,
    DEFAULT_WHISPER_EN,
    DEFAULT_WHISPER_ZH,
    SUMMARY_LABEL_WIDTH,
    SUMMARY_LINE_WIDTH,
    SampleInput,
    SampleOutput,
    calculate_metrics,
    compute_wer,
    load_asr_model,
    parse_meta_lst,
    print_wer_metrics,
    save_results,
    transcribe,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _resolve_model_path(model_path: str) -> str:
    """Resolve a HF repo ID or local path to a local directory."""
    if os.path.isdir(model_path):
        return model_path
    # Treat as HF repo ID — download/resolve to local cache
    from huggingface_hub import snapshot_download

    logger.info("Resolving HF repo %s to local cache...", model_path)
    return snapshot_download(model_path)


def load_ming_talker(model_path: str, device: str = "cuda"):
    """Load MingOmniTalker + AudioVAE."""
    from transformers import AutoTokenizer

    from sglang_omni.models.ming_omni.talker import (
        AudioVAE,
        MingOmniTalker,
        MingOmniTalkerConfig,
        SpkembExtractor,
    )
    from sglang_omni.models.weight_loader import load_weights_by_prefix

    local_path = _resolve_model_path(model_path)
    talker_path = os.path.join(local_path, "talker")

    logger.info("Loading MingOmniTalker from %s ...", talker_path)
    t0 = time.time()
    config = MingOmniTalkerConfig.from_pretrained_dir(talker_path)
    talker = MingOmniTalker(config)
    talker.eval()
    weights = load_weights_by_prefix(talker_path, prefix="")
    talker.load_weights(weights.items())
    talker.to(device=device, dtype=torch.bfloat16)
    logger.info("MingOmniTalker loaded in %.1fs", time.time() - t0)

    # Set external dependencies
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(talker_path, "llm"))
    talker.set_tokenizer(tokenizer)

    voice_json_path = os.path.join(talker_path, "data", "voice_name.json")
    if os.path.exists(voice_json_path):
        import json as _json

        with open(voice_json_path, "r") as f:
            voice_dict = _json.load(f)
        for key in voice_dict:
            voice_dict[key]["prompt_wav_path"] = os.path.join(
                talker_path, voice_dict[key]["prompt_wav_path"]
            )
        talker.set_voice_presets(voice_dict)

    campplus_path = os.path.join(talker_path, "campplus.onnx")
    try:
        talker.set_spkemb_extractor(SpkembExtractor(campplus_path))
    except Exception as e:
        logger.warning("SpkembExtractor not available: %s", e)

    try:
        from talker_tn.talker_tn import TalkerTN

        talker.set_normalizer(TalkerTN())
    except ImportError:
        logger.warning("TalkerTN not available, using identity normalizer")

    logger.info("Initializing CUDA graphs...")
    t0g = time.time()
    talker.initial_graph()
    logger.info("CUDA graphs initialized in %.1fs", time.time() - t0g)

    vae_path = os.path.join(talker_path, "vae")
    logger.info("Loading AudioVAE from %s ...", vae_path)
    t0v = time.time()
    vae = AudioVAE.from_pretrained(vae_path, dtype=torch.bfloat16)
    vae.to(device)
    vae.eval()
    logger.info("AudioVAE loaded in %.1fs", time.time() - t0v)

    return talker, vae


# ---------------------------------------------------------------------------
# Speech generation
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate_speech_ming(
    talker,
    vae,
    sample: SampleInput,
) -> tuple[np.ndarray | None, int]:
    """Generate speech using MingOmniTalker with voice cloning.

    Returns:
        Tuple of (waveform as numpy float32, sample_rate).
    """
    all_wavs = []
    for tts_speech, _, _, _ in talker.omni_audio_generation(
        tts_text=sample.target_text,
        voice_name=None,
        prompt_text=sample.ref_text,
        prompt_wav_path=sample.ref_audio,
        audio_detokenizer=vae,
        stream=False,
    ):
        if tts_speech is not None:
            all_wavs.append(tts_speech)

    if not all_wavs:
        return None, 44100

    waveform = torch.cat(all_wavs, dim=-1)
    sample_rate = getattr(vae.config, "sample_rate", 44100)
    return waveform.squeeze().cpu().float().numpy(), sample_rate


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def print_summary(metrics: dict, args: argparse.Namespace) -> None:
    lw = SUMMARY_LABEL_WIDTH
    w = SUMMARY_LINE_WIDTH
    print(f"\n{'=' * w}")
    print(f"{'Ming TTS WER Benchmark Result':^{w}}")
    print(f"{'=' * w}")
    print(f"  {'Model:':<{lw}} {args.model_path}")
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

    samples = parse_meta_lst(args.meta, args.max_samples)
    logger.info("Loaded %d samples from %s", len(samples), args.meta)

    # Load TTS model
    talker, vae = load_ming_talker(args.model_path, device=args.device)

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
            # Generate speech
            t0 = time.time()
            waveform, sample_rate = generate_speech_ming(talker, vae, sample)
            output.latency = time.time() - t0

            if waveform is None or len(waveform) == 0:
                output.error = "Empty waveform"
                outputs.append(output)
                continue

            output.audio_duration = len(waveform) / sample_rate

            # Save audio
            audio_path = os.path.join(audio_dir, f"{sample.sample_id}.wav")
            waveform_tensor = torch.from_numpy(waveform).unsqueeze(0)
            torchaudio.save(audio_path, waveform_tensor, sample_rate)

            # ASR transcribe (runs on --asr-device, default CPU)
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

            logger.debug(
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
            "model_path": args.model_path,
            "meta": args.meta,
            "lang": args.lang,
            "device": args.device,
            "max_samples": args.max_samples,
            "asr_device": args.asr_device,
            "asr_model": args.asr_model,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Ming TTS accuracy (WER) using seed-tts-eval."
    )
    parser.add_argument(
        "--meta",
        type=str,
        default=None,
        help="Path to seed-tts-eval meta.lst file. If not set, resolved from --dataset-dir and --lang.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=DEFAULT_DATASET_DIR,
        help="Path to seed-tts-eval dataset root (contains zh/ and en/ subdirs). "
        f"Default: {DEFAULT_DATASET_DIR}",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="inclusionAI/Ming-flash-omni-2.0",
        help="Path to Ming-flash-omni model (parent dir containing talker/).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/ming_tts_wer",
        help="Directory to save results and audio files.",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="zh",
        choices=["zh", "en"],
        help="Language for ASR and text normalization.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for TTS model.",
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
        help="Whisper model size for ASR (e.g. tiny, base.en, small, medium, "
        "large-v3). Default: base.en for EN, medium for ZH. "
        "Larger models reduce ASR errors but are slower.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process.",
    )
    args = parser.parse_args()

    benchmark(args)


if __name__ == "__main__":
    main()
