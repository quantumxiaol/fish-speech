import gc
import queue
import time
from typing import Generator

import numpy as np
import torch
from loguru import logger

from fish_speech.device_memory import empty_mps_cache
from fish_speech.inference_engine.reference_loader import ReferenceLoader
from fish_speech.inference_engine.utils import InferenceResult, wav_chunk_header
from fish_speech.inference_engine.vq_manager import VQManager
from fish_speech.models.dac.modded_dac import DAC
from fish_speech.models.text2semantic.inference import (
    GenerateRequest,
    GenerateResponse,
    WrappedGenerateResponse,
    log_device_memory,
)
from fish_speech.utils import autocast_exclude_mps, set_seed
from fish_speech.utils.schema import ServeTTSRequest


class TTSInferenceEngine(ReferenceLoader, VQManager):
    def __init__(
        self,
        llama_queue: queue.Queue,
        decoder_model: DAC,
        precision: torch.dtype,
        compile: bool,
        perf_detail: bool = False,
        perf_sample_frames: int = 16,
    ) -> None:
        super().__init__()

        self.llama_queue = llama_queue
        self.decoder_model = decoder_model
        self.precision = precision
        self.compile = compile
        self.perf_detail = perf_detail
        self.perf_sample_frames = perf_sample_frames

    @torch.inference_mode()
    def inference(self, req: ServeTTSRequest) -> Generator[InferenceResult, None, None]:
        """
        Main inference function:
        - Loads the reference audio and text.
        - Calls the LLAMA model for inference.
        - Decodes the VQ tokens to audio.
        """

        device = self.decoder_model.device
        if hasattr(self.decoder_model, "spec_transform"):
            sample_rate = self.decoder_model.spec_transform.sample_rate
        else:
            sample_rate = self.decoder_model.sample_rate

        request_start = time.perf_counter()
        reference_seconds = 0.0
        llama_queue_wait_ms = 0.0
        semantic_wall_ms = 0.0
        semantic_accelerator_ms = 0.0
        semantic_accelerator_available = False
        semantic_frames = 0
        dac_and_host_seconds = 0.0
        cleanup_seconds = 0.0
        time_to_first_audio_ms = 0.0
        segments = []
        request_had_error = False
        memory_stats = None
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        log_device_memory("at request start", device)

        try:
            ref_id: str | None = req.reference_id
            prompt_tokens, prompt_texts = [], []
            reference_start = time.perf_counter()

            if ref_id is not None:
                prompt_tokens, prompt_texts = self.load_by_id(
                    ref_id, req.use_memory_cache
                )
            elif req.references:
                prompt_tokens, prompt_texts = self.load_by_hash(
                    req.references, req.use_memory_cache
                )

            reference_seconds = time.perf_counter() - reference_start

            if req.seed is not None:
                set_seed(req.seed)
                logger.warning(f"set seed: {req.seed}")

            response_queue = self.send_Llama_request(
                req, prompt_tokens, prompt_texts
            )

            if req.streaming:
                yield InferenceResult(
                    code="header",
                    audio=(
                        sample_rate,
                        np.array(wav_chunk_header(sample_rate=sample_rate)),
                    ),
                    error=None,
                )

            while True:
                wrapped_result: WrappedGenerateResponse = response_queue.get()
                if wrapped_result.status == "error":
                    yield InferenceResult(
                        code="error",
                        audio=None,
                        error=(
                            wrapped_result.response
                            if isinstance(wrapped_result.response, Exception)
                            else Exception("Unknown error")
                        ),
                    )
                    request_had_error = True
                    break

                if not isinstance(wrapped_result.response, GenerateResponse):
                    raise TypeError(
                        "Expected GenerateResponse, got "
                        f"{type(wrapped_result.response).__name__}"
                    )

                result: GenerateResponse = wrapped_result.response
                if result.action == "next":
                    break

                if result.metrics is not None:
                    llama_queue_wait_ms = max(
                        llama_queue_wait_ms,
                        result.metrics.get("llama_queue_wait_ms", 0.0),
                    )
                    semantic_wall_ms += result.metrics.get(
                        "semantic_wall_ms", 0.0
                    )
                    if "semantic_accelerator_ms" in result.metrics:
                        semantic_accelerator_ms += result.metrics[
                            "semantic_accelerator_ms"
                        ]
                        semantic_accelerator_available = True
                    semantic_frames += int(
                        result.metrics.get("semantic_frames", 0.0)
                    )

                dac_start = time.perf_counter()
                segment = self.get_audio_segment(result)
                dac_and_host_seconds += time.perf_counter() - dac_start
                if not segments:
                    time_to_first_audio_ms = (
                        time.perf_counter() - request_start
                    ) * 1000

                if req.streaming:
                    yield InferenceResult(
                        code="segment",
                        audio=(sample_rate, segment),
                        error=None,
                    )
                segments.append(segment)
        finally:
            cleanup_start = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
            elif device.type == "mps":
                # Keep cleanup outside the Llama worker so it cannot overlap
                # decoder work on another thread. The finally block also covers
                # client cancellation and decode failures.
                memory_stats = empty_mps_cache(device)
            cleanup_seconds = time.perf_counter() - cleanup_start

            if memory_stats is not None:
                gib = 1024**3
                logger.info(
                    "MPS memory cleanup: tensors {:.2f} -> {:.2f} GiB, "
                    "driver {:.2f} -> {:.2f} GiB, released {:.2f} GiB",
                    memory_stats.current_before / gib,
                    memory_stats.current_after / gib,
                    memory_stats.driver_before / gib,
                    memory_stats.driver_after / gib,
                    max(memory_stats.driver_before - memory_stats.driver_after, 0)
                    / gib,
                )

            request_seconds = time.perf_counter() - request_start
            semantic_seconds = semantic_wall_ms / 1000
            audio_seconds = (
                sum(segment.size for segment in segments) / sample_rate
                if sample_rate > 0
                else 0.0
            )
            rtf = request_seconds / audio_seconds if audio_seconds > 0 else 0.0
            semantic_rtf = (
                semantic_seconds / audio_seconds if audio_seconds > 0 else 0.0
            )
            semantic_fps = (
                semantic_frames / semantic_seconds
                if semantic_seconds > 0
                else 0.0
            )
            logger.info(
                "TTS request metrics: llama_queue_wait={:.3f}ms, "
                "reference_wall={:.3f}ms, "
                "semantic_wall={:.3f}ms, semantic_accelerator={}, "
                "semantic_frames={}, semantic_frames/s={:.2f}, "
                "dac+device_to_host_wall={:.3f}ms, cleanup_wall={:.3f}ms, "
                "time_to_first_audio={:.3f}ms, "
                "engine_end_to_end={:.3f}ms, audio={:.3f}s, "
                "semantic_RTF={:.3f}, engine_RTF={:.3f}",
                llama_queue_wait_ms,
                reference_seconds * 1000,
                semantic_wall_ms,
                (
                    f"{semantic_accelerator_ms:.3f}ms"
                    if semantic_accelerator_available
                    else "n/a"
                ),
                semantic_frames,
                semantic_fps,
                dac_and_host_seconds * 1000,
                cleanup_seconds * 1000,
                time_to_first_audio_ms,
                request_seconds * 1000,
                audio_seconds,
                semantic_rtf,
                rtf,
            )
            log_device_memory("after request cleanup", device)

        if request_had_error:
            return None

        if len(segments) == 0:
            yield InferenceResult(
                code="error",
                audio=None,
                error=RuntimeError("No audio generated, please check the input text."),
            )
        else:
            audio = np.concatenate(segments, axis=0)
            yield InferenceResult(
                code="final",
                audio=(sample_rate, audio),
                error=None,
            )

        return None

    def send_Llama_request(
        self, req: ServeTTSRequest, prompt_tokens: list, prompt_texts: list
    ) -> queue.Queue:
        """
        Send a request to the LLAMA model to generate the symbolic tokens.
        """

        # Prepare the request
        request = dict(
            device=self.decoder_model.device,
            max_new_tokens=req.max_new_tokens,
            text=req.text,
            top_p=req.top_p,
            repetition_penalty=req.repetition_penalty,
            temperature=req.temperature,
            compile=self.compile,
            perf_detail=getattr(self, "perf_detail", False),
            perf_sample_frames=getattr(self, "perf_sample_frames", 16),
            iterative_prompt=req.iterative_prompt,
            chunk_length=req.chunk_length,
            prompt_tokens=prompt_tokens,
            prompt_text=prompt_texts,
        )

        # Create a queue to get the response
        response_queue = queue.Queue()

        # Send the request to the LLAMA model
        self.llama_queue.put(
            GenerateRequest(
                request=request,
                response_queue=response_queue,
            )
        )

        return response_queue

    def get_audio_segment(self, result: GenerateResponse) -> np.ndarray:
        """
        Decode the VQ tokens to audio.
        """

        # Don't use autocast on MPS devices
        with autocast_exclude_mps(
            device_type=self.decoder_model.device.type, dtype=self.precision
        ):
            # Decode the symbolic tokens to audio
            segment = self.decode_vq_tokens(codes=result.codes)

        # Convert the audio to numpy
        return segment.float().cpu().numpy()
