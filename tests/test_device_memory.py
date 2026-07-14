import unittest
from unittest.mock import patch

import torch

from fish_speech.device_memory import MPSMemoryCleanupStats, empty_mps_cache


class EmptyDeviceCacheTest(unittest.TestCase):
    def test_mps_synchronizes_collects_and_reports_memory(self) -> None:
        calls: list[str] = []

        with (
            patch(
                "fish_speech.device_memory.torch.backends.mps.is_available",
                return_value=True,
            ),
            patch(
                "fish_speech.device_memory.torch.mps.synchronize",
                side_effect=lambda: calls.append("synchronize"),
            ),
            patch(
                "fish_speech.device_memory.torch.mps.current_allocated_memory",
                side_effect=lambda: (
                    calls.append("current")
                    or [8 * 1024**3, 7 * 1024**3][calls.count("current") - 1]
                ),
            ),
            patch(
                "fish_speech.device_memory.torch.mps.driver_allocated_memory",
                side_effect=lambda: (
                    calls.append("driver")
                    or [12 * 1024**3, 8 * 1024**3][calls.count("driver") - 1]
                ),
            ),
            patch(
                "fish_speech.device_memory.gc.collect",
                side_effect=lambda: calls.append("gc"),
            ),
            patch(
                "fish_speech.device_memory.torch.mps.empty_cache",
                side_effect=lambda: calls.append("empty_cache"),
            ),
        ):
            stats = empty_mps_cache(torch.device("mps"))

        self.assertEqual(
            calls,
            [
                "synchronize",
                "current",
                "driver",
                "gc",
                "empty_cache",
                "current",
                "driver",
            ],
        )
        self.assertEqual(
            stats,
            MPSMemoryCleanupStats(
                current_before=8 * 1024**3,
                current_after=7 * 1024**3,
                driver_before=12 * 1024**3,
                driver_after=8 * 1024**3,
            ),
        )

    def test_cpu_does_not_clear_accelerator_caches(self) -> None:
        with (
            patch(
                "fish_speech.device_memory.torch.cuda.empty_cache"
            ) as cuda_empty_cache,
            patch("fish_speech.device_memory.torch.mps.empty_cache") as mps_empty_cache,
        ):
            stats = empty_mps_cache(torch.device("cpu"))

        self.assertIsNone(stats)
        cuda_empty_cache.assert_not_called()
        mps_empty_cache.assert_not_called()

    def test_unavailable_mps_does_not_call_mps_api(self) -> None:
        with (
            patch(
                "fish_speech.device_memory.torch.backends.mps.is_available",
                return_value=False,
            ),
            patch("fish_speech.device_memory.torch.mps.empty_cache") as mps_empty_cache,
        ):
            stats = empty_mps_cache(torch.device("mps"))

        self.assertIsNone(stats)
        mps_empty_cache.assert_not_called()


if __name__ == "__main__":
    unittest.main()
