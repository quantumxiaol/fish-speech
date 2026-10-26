import unittest
from time import sleep
from unittest.mock import MagicMock, call, patch

import torch

from fish_speech.device_memory import (
    AcceleratorTimer,
    CUDAMemorySnapshot,
    MPSMemoryCleanupStats,
    MPSMemorySnapshot,
    empty_mps_cache,
    get_cuda_memory_snapshot,
    get_mps_memory_snapshot,
    synchronize_device,
)


class EmptyDeviceCacheTest(unittest.TestCase):
    def test_cpu_timer_uses_wall_time_without_accelerator_events(self) -> None:
        timer = AcceleratorTimer(torch.device("cpu"))
        timer.start()
        sleep(0.001)
        timer.stop()
        timer.synchronize()

        self.assertGreater(timer.wall_elapsed_ms(), 0)
        self.assertIsNone(timer.accelerator_elapsed_ms())

    def test_mps_timer_uses_wall_time_without_unsafe_events(self) -> None:
        with (
            patch(
                "fish_speech.device_memory.torch.backends.mps.is_available",
                return_value=True,
            ),
            patch(
                "fish_speech.device_memory.torch.mps.event.Event",
            ) as event_factory,
            patch(
                "fish_speech.device_memory.torch.mps.synchronize"
            ) as global_synchronize,
        ):
            timer = AcceleratorTimer(torch.device("mps"))
            timer.start()
            timer.stop()
            timer.synchronize()
            elapsed = timer.accelerator_elapsed_ms()

        event_factory.assert_not_called()
        global_synchronize.assert_not_called()
        self.assertGreaterEqual(timer.wall_elapsed_ms(), 0)
        self.assertIsNone(elapsed)

    def test_cuda_timer_uses_cuda_events_without_global_synchronize(self) -> None:
        start_event = MagicMock()
        end_event = MagicMock()
        start_event.elapsed_time.return_value = 8.0

        with (
            patch(
                "fish_speech.device_memory.torch.cuda.is_available",
                return_value=True,
            ),
            patch(
                "fish_speech.device_memory.torch.cuda.Event",
                side_effect=[start_event, end_event],
            ) as event_factory,
            patch(
                "fish_speech.device_memory.torch.cuda.synchronize"
            ) as global_synchronize,
        ):
            timer = AcceleratorTimer(torch.device("cuda:0"))
            timer.start()
            timer.stop()
            timer.synchronize()
            elapsed = timer.accelerator_elapsed_ms()

        self.assertEqual(
            event_factory.call_args_list,
            [call(enable_timing=True), call(enable_timing=True)],
        )
        end_event.synchronize.assert_called_once_with()
        global_synchronize.assert_not_called()
        self.assertEqual(elapsed, 8.0)

    def test_cuda_snapshot_reports_allocator_peaks_and_headroom(self) -> None:
        gib = 1024**3
        with (
            patch(
                "fish_speech.device_memory.torch.cuda.is_available",
                return_value=True,
            ),
            patch(
                "fish_speech.device_memory.torch.cuda.memory_allocated",
                return_value=8 * gib,
            ),
            patch(
                "fish_speech.device_memory.torch.cuda.memory_reserved",
                return_value=9 * gib,
            ),
            patch(
                "fish_speech.device_memory.torch.cuda.max_memory_allocated",
                return_value=10 * gib,
            ),
            patch(
                "fish_speech.device_memory.torch.cuda.max_memory_reserved",
                return_value=11 * gib,
            ),
            patch(
                "fish_speech.device_memory.torch.cuda.mem_get_info",
                return_value=(1 * gib, 12 * gib),
            ),
        ):
            snapshot = get_cuda_memory_snapshot(torch.device("cuda:0"))

        self.assertEqual(
            snapshot,
            CUDAMemorySnapshot(
                allocated=8 * gib,
                reserved=9 * gib,
                peak_allocated=10 * gib,
                peak_reserved=11 * gib,
                free=1 * gib,
                total=12 * gib,
            ),
        )
        self.assertEqual(snapshot.reserved_not_allocated, 1 * gib)

    def test_mps_snapshot_reports_allocator_and_recommended_memory(self) -> None:
        with (
            patch(
                "fish_speech.device_memory.torch.backends.mps.is_available",
                return_value=True,
            ),
            patch(
                "fish_speech.device_memory.torch.mps.current_allocated_memory",
                return_value=8 * 1024**3,
            ),
            patch(
                "fish_speech.device_memory.torch.mps.driver_allocated_memory",
                return_value=12 * 1024**3,
            ),
            patch(
                "fish_speech.device_memory.torch.mps.recommended_max_memory",
                return_value=36 * 1024**3,
            ),
        ):
            snapshot = get_mps_memory_snapshot(torch.device("mps"))

        self.assertEqual(
            snapshot,
            MPSMemorySnapshot(
                current=8 * 1024**3,
                driver=12 * 1024**3,
                recommended=36 * 1024**3,
            ),
        )
        self.assertEqual(snapshot.cache_and_driver_overhead, 4 * 1024**3)
        self.assertAlmostEqual(snapshot.driver_ratio, 1 / 3)

    def test_synchronize_uses_resolved_device_type(self) -> None:
        with (
            patch(
                "fish_speech.device_memory.torch.backends.mps.is_available",
                return_value=True,
            ),
            patch(
                "fish_speech.device_memory.torch.mps.synchronize"
            ) as mps_synchronize,
            patch(
                "fish_speech.device_memory.torch.cuda.synchronize"
            ) as cuda_synchronize,
        ):
            synchronize_device(torch.device("mps"))

        mps_synchronize.assert_called_once_with()
        cuda_synchronize.assert_not_called()

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
