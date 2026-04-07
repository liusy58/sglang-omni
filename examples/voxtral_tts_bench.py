# SPDX-License-Identifier: Apache-2.0
"""End-to-end benchmark for Voxtral TTS on sglang-omni.

Mirrors the vllm-omni ``examples/offline_inference/voxtral_tts/end2end.py``
benchmark, but runs on the sglang-omni multi-stage pipeline
(MultiProcessPipelineRunner → Coordinator).

Features
--------
* Non-streaming mode  — ``Coordinator.submit()``
* Streaming mode       — ``Coordinator.stream()``
* Concurrent requests  — wave-based dispatch via ``--concurrency``
* Metrics              — TTFA, RTF, generation time, audio duration, wait rate
* Audio saving         — ``--write-audio``

Usage
-----
    # Non-streaming, single request
    python examples/voxtral_tts_bench.py \\
        --model mistralai/Voxtral-4B-TTS-2603 \\
        --text "Hello, this is a test." --voice cheerful_female

    # Streaming, 4 concurrent requests
    python examples/voxtral_tts_bench.py \\
        --model mistralai/Voxtral-4B-TTS-2603 \\
        --text "Hello, this is a test." --voice cheerful_female \\
        --streaming --num-prompts 4 --concurrency 2

    # Save generated audio
    python examples/voxtral_tts_bench.py \\
        --model mistralai/Voxtral-4B-TTS-2603 \\
        --text "Hello, this is a test." --voice cheerful_female \\
        --write-audio --output-dir output_audio
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import logging
import os
import time
import uuid
from typing import Any

import numpy as np
import soundfile as sf
import torch

from sglang_omni.config.manager import ConfigManager
from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner
from sglang_omni.proto import CompleteMessage, OmniRequest, StreamMessage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

SAMPLE_RATE = 24_000


# ---------------------------------------------------------------------------
# Streaming benchmark
# ---------------------------------------------------------------------------


async def run_streaming(
    runner: MultiProcessPipelineRunner,
    inputs: list[OmniRequest],
    args: argparse.Namespace,
    output_dir: str,
) -> None:
    """Run streaming benchmark using ``Coordinator.stream()``."""
    coordinator = runner.coordinator

    total_audio_dur = 0.0
    total_gen_time = 0.0
    total_ttfa = 0.0
    ttfa_count = 0
    total_waits = 0
    total_chunks = 0
    results_lock = asyncio.Lock()

    async def _generate_one(batch_idx: int, single_input: OmniRequest) -> None:
        nonlocal total_audio_dur, total_gen_time, total_ttfa, ttfa_count
        nonlocal total_waits, total_chunks

        request_id = str(uuid.uuid4())
        all_audio_chunks: list[np.ndarray] = []
        chunk_arrival_times: list[float] = []
        chunk_durations: list[float] = []
        gen_start = time.time()
        ttfa: float | None = None

        async for msg in coordinator.stream(request_id, single_input):
            if isinstance(msg, StreamMessage):
                data = msg.chunk
                if not isinstance(data, dict):
                    continue
                audio_data = data.get("audio_data")
                if audio_data is None:
                    continue

                now = time.time()
                if ttfa is None:
                    ttfa = now - gen_start

                if isinstance(audio_data, torch.Tensor):
                    audio_numpy = audio_data.float().detach().cpu().numpy()
                elif isinstance(audio_data, list):
                    audio_numpy = np.array(audio_data, dtype=np.float32)
                else:
                    audio_numpy = np.asarray(audio_data, dtype=np.float32)

                all_audio_chunks.append(audio_numpy)
                chunk_arrival_times.append(now)
                chunk_durations.append(len(audio_numpy) / SAMPLE_RATE)

            elif isinstance(msg, CompleteMessage):
                # Final completion — extract audio if present
                if msg.result and isinstance(msg.result, dict):
                    audio_data = msg.result.get("audio_data")
                    if audio_data is not None:
                        now = time.time()
                        if ttfa is None:
                            ttfa = now - gen_start
                        if isinstance(audio_data, torch.Tensor):
                            audio_numpy = audio_data.float().detach().cpu().numpy()
                        elif isinstance(audio_data, list):
                            audio_numpy = np.array(audio_data, dtype=np.float32)
                        else:
                            audio_numpy = np.asarray(audio_data, dtype=np.float32)
                        all_audio_chunks.append(audio_numpy)
                        chunk_arrival_times.append(now)
                        chunk_durations.append(len(audio_numpy) / SAMPLE_RATE)

        gen_elapsed = time.time() - gen_start

        # Analyze wait / no-wait per chunk
        chunk_labels: list[str] = []
        accumulated_audio_dur = 0.0
        if chunk_arrival_times:
            first_arrival = chunk_arrival_times[0]
            for i in range(len(chunk_arrival_times)):
                if i == 0:
                    chunk_labels.append("no_wait")
                else:
                    playback_elapsed = chunk_arrival_times[i] - first_arrival
                    buffer_time = accumulated_audio_dur - playback_elapsed
                    chunk_labels.append("wait" if buffer_time <= 0 else "no_wait")
                accumulated_audio_dur += chunk_durations[i]

        req_wait_count = sum(1 for lbl in chunk_labels if lbl == "wait")

        # Concatenate all chunks
        if all_audio_chunks:
            full_audio = np.concatenate(all_audio_chunks)
            output_audio_dur = len(full_audio) / SAMPLE_RATE
            if args.write_audio:
                output_path = os.path.join(output_dir, f"tts_output_{batch_idx}.wav")
                sf.write(output_path, full_audio, SAMPLE_RATE)
                print(
                    f"Request {batch_idx}: saved {len(full_audio)} samples "
                    f"({output_audio_dur:.2f}s) to {output_path}"
                )
            # Per-chunk details
            if chunk_arrival_times:
                first_arrival = chunk_arrival_times[0]
            for i, label in enumerate(chunk_labels):
                dur_ms = chunk_durations[i] * 1000
                arrival_ms = (
                    (chunk_arrival_times[i] - first_arrival) * 1000 if i > 0 else 0.0
                )
                print(
                    f"  Request {batch_idx} chunk {i}: {label} | "
                    f"arrived={arrival_ms:.1f}ms | chunk_dur={dur_ms:.1f}ms"
                )
            req_wait_rate = req_wait_count / len(chunk_labels) if chunk_labels else 0.0
            print(
                f"Request {batch_idx}: "
                f"TTFA={ttfa:.4f}s | "
                f"Generation={gen_elapsed:.4f}s | "
                f"Audio={output_audio_dur:.2f}s | "
                f"RTF={output_audio_dur / gen_elapsed:.4f} | "
                f"WaitRate={req_wait_rate:.2%} ({req_wait_count}/{len(chunk_labels)})"
            )
            async with results_lock:
                total_audio_dur += output_audio_dur
                total_gen_time += gen_elapsed
                total_waits += req_wait_count
                total_chunks += len(chunk_labels)
                if ttfa is not None:
                    total_ttfa += ttfa
                    ttfa_count += 1
        else:
            print(f"Request {batch_idx}: no audio produced")

    # Launch requests in waves of ``concurrency``
    concurrency = args.concurrency or len(inputs)
    gen_start_all = time.time()
    for wave_start in range(0, len(inputs), concurrency):
        wave = inputs[wave_start : wave_start + concurrency]
        print(
            f"\n--- Wave {wave_start // concurrency + 1} "
            f"(requests {wave_start}-{wave_start + len(wave) - 1}) ---"
        )
        await asyncio.gather(
            *[_generate_one(wave_start + i, inp) for i, inp in enumerate(wave)]
        )

    generation_time = time.time() - gen_start_all
    avg_ttfa = total_ttfa / ttfa_count if ttfa_count else float("nan")
    overall_wait_rate = total_waits / total_chunks if total_chunks else float("nan")
    print(
        f"\nAll requests: Generation={generation_time:.4f}s | "
        f"TotalAudio={total_audio_dur:.2f}s | "
        f"Concurrency={concurrency} | "
        f"AvgTTFA={avg_ttfa:.4f}s | "
        f"RTF(total)={total_audio_dur / generation_time:.4f} | "
        f"RTF(per-request)={total_audio_dur / total_gen_time:.4f} | "
        f"WaitRate={overall_wait_rate:.2%} ({total_waits}/{total_chunks})"
    )


# ---------------------------------------------------------------------------
# Non-streaming benchmark
# ---------------------------------------------------------------------------


async def run_non_streaming(
    runner: MultiProcessPipelineRunner,
    inputs: list[OmniRequest],
    args: argparse.Namespace,
    output_dir: str,
) -> None:
    """Run non-streaming benchmark using ``Coordinator.submit()``."""
    coordinator = runner.coordinator

    start = time.time()
    results: list[Any] = []
    for i, inp in enumerate(inputs):
        request_id = str(uuid.uuid4())
        result = await coordinator.submit(request_id, inp)
        results.append(result)
    elapsed = time.time() - start
    print(f"Pipeline run time: {elapsed:.4f}s")

    output_audio_dur = 0.0
    for batch_idx, result in enumerate(results):
        audio_data = None
        if isinstance(result, dict):
            audio_data = result.get("audio_data")

        if audio_data is None:
            print(f"Request {batch_idx}: no audio in result")
            continue

        if isinstance(audio_data, torch.Tensor):
            audio_array = audio_data.float().detach().cpu().numpy()
        elif isinstance(audio_data, list):
            audio_array = np.array(audio_data, dtype=np.float32)
        else:
            audio_array = np.asarray(audio_data, dtype=np.float32)

        output_audio_dur += float(len(audio_array)) / SAMPLE_RATE

        if args.write_audio:
            output_path = os.path.join(output_dir, f"tts_output_{batch_idx}.wav")
            sf.write(output_path, audio_array, SAMPLE_RATE)
            print(f"Audio saved to {output_path}")

    print(f"Total audio duration: {output_audio_dur:.2f}s")
    if elapsed > 0:
        print(f"RTF: {output_audio_dur / elapsed:.4f}")


# ---------------------------------------------------------------------------
# Request construction
# ---------------------------------------------------------------------------


def compose_request(text: str, voice: str, max_new_tokens: int) -> OmniRequest:
    """Build an ``OmniRequest`` for Voxtral TTS."""
    return OmniRequest(
        inputs=text,
        params={"max_new_tokens": max_new_tokens, "voice": voice},
        metadata={"tts_params": {"voice": voice}},
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def async_main(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    if args.write_audio:
        os.makedirs(output_dir, exist_ok=True)

    # Build the request
    omni_request = compose_request(
        text=args.text,
        voice=args.voice,
        max_new_tokens=args.max_new_tokens,
    )

    if args.num_prompts > 1:
        inputs = [omni_request] * args.num_prompts
    else:
        inputs = [omni_request]

    if args.concurrency is not None:
        if not args.streaming:
            raise ValueError("--concurrency requires --streaming")
        if args.num_prompts % args.concurrency != 0:
            raise ValueError(
                f"--num-prompts ({args.num_prompts}) must be divisible by "
                f"--concurrency ({args.concurrency})"
            )

    # Build PipelineConfig from model path or config file
    if args.config_file:
        config_manager = ConfigManager.from_file(args.config_file)
    else:
        config_manager = ConfigManager.from_model_path(args.model)

    # Apply any extra CLI overrides
    config = config_manager.config
    logger.info(
        "Pipeline config: entry_stage=%s, stages=%s",
        config.entry_stage,
        [s.name for s in config.stages],
    )

    # Spawn multi-process pipeline
    runner = MultiProcessPipelineRunner(config)
    await runner.start(timeout=300)
    logger.info("Pipeline started")

    try:
        if args.streaming:
            await run_streaming(runner, inputs, args, output_dir)
        else:
            await run_non_streaming(runner, inputs, args, output_dir)
    finally:
        await runner.stop()
        torch.cuda.empty_cache()
        gc.collect()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end benchmark for Voxtral TTS on sglang-omni"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Voxtral-4B-TTS-2603",
        help="Model name or local path.",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default=None,
        help="Path to pipeline config YAML. Auto-resolved from model if not set.",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="This is a test message.",
        help="Text to synthesize.",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default="cheerful_female",
        help="Voice to use for synthesis.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output_audio",
        help="Directory to write output WAV files.",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1,
        help="Number of replicate prompts for measuring performance.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=4096,
        help="Maximum number of new tokens for AR generation.",
    )
    parser.add_argument(
        "--write-audio",
        action="store_true",
        default=False,
        help="Write audio output to WAV files.",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        default=False,
        help="Use streaming generation via Coordinator.stream().",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help=(
            "Max concurrent requests per wave (default: all at once). "
            "Must evenly divide --num-prompts. Requires --streaming."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
