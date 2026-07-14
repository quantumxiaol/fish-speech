import queue
import unittest
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import numpy as np
import torch

from fish_speech.device_memory import MPSMemoryCleanupStats
from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.dac.modded_dac import DAC
from fish_speech.models.text2semantic.inference import (
    GenerateResponse,
    WrappedGenerateResponse,
)
from fish_speech.utils.schema import ServeTTSRequest


def fake_decoder(device: str) -> DAC:
    """Build the minimum decoder surface needed without loading model weights."""
    return cast(
        DAC,
        SimpleNamespace(device=torch.device(device), sample_rate=44100),
    )


class InferenceEngineMemoryTest(unittest.TestCase):
    def test_completed_request_clears_cache_once_at_engine_boundary(self) -> None:
        response_queue: queue.Queue = queue.Queue()
        response_queue.put(
            WrappedGenerateResponse(
                status="success",
                response=GenerateResponse(action="sample"),
            )
        )
        response_queue.put(
            WrappedGenerateResponse(
                status="success",
                response=GenerateResponse(action="next"),
            )
        )

        engine = object.__new__(TTSInferenceEngine)
        engine.decoder_model = fake_decoder("mps")

        stats = MPSMemoryCleanupStats(
            current_before=8,
            current_after=7,
            driver_before=12,
            driver_after=8,
        )
        with (
            patch.object(engine, "send_Llama_request", return_value=response_queue),
            patch.object(
                engine,
                "get_audio_segment",
                return_value=np.zeros(16, dtype=np.float32),
            ),
            patch(
                "fish_speech.inference_engine.torch.cuda.is_available",
                return_value=False,
            ),
            patch(
                "fish_speech.inference_engine.empty_mps_cache", return_value=stats
            ) as empty_cache,
        ):
            results = list(engine.inference(ServeTTSRequest(text="test")))

        empty_cache.assert_called_once_with(engine.decoder_model.device)
        self.assertEqual([result.code for result in results], ["final"])

    def test_cuda_cleanup_behavior_is_unchanged(self) -> None:
        response_queue: queue.Queue = queue.Queue()
        response_queue.put(
            WrappedGenerateResponse(
                status="success",
                response=GenerateResponse(action="next"),
            )
        )

        engine = object.__new__(TTSInferenceEngine)
        engine.decoder_model = fake_decoder("cuda:0")

        with (
            patch.object(engine, "send_Llama_request", return_value=response_queue),
            patch(
                "fish_speech.inference_engine.torch.cuda.is_available",
                return_value=True,
            ),
            patch(
                "fish_speech.inference_engine.torch.cuda.empty_cache"
            ) as cuda_empty_cache,
            patch("fish_speech.inference_engine.gc.collect") as collect,
            patch("fish_speech.inference_engine.empty_mps_cache") as mps_empty_cache,
        ):
            results = list(engine.inference(ServeTTSRequest(text="test")))

        cuda_empty_cache.assert_called_once_with()
        collect.assert_called_once_with()
        mps_empty_cache.assert_not_called()
        self.assertEqual([result.code for result in results], ["error"])


if __name__ == "__main__":
    unittest.main()
