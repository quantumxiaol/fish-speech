import gc
import time
from dataclasses import dataclass
from typing import Any

import torch


class AcceleratorTimer:
    """Low-overhead wall timer with CUDA event timing when it is safe."""

    def __init__(self, device: str | torch.device) -> None:
        self.device = torch.device(device)
        self.start_event: Any | None = None
        self.end_event: Any | None = None
        self.wall_start = 0.0
        self.wall_end: float | None = None

        # torch.mps.event.Event.elapsed_time() can wait indefinitely with the
        # threaded Llama worker on PyTorch 2.8/macOS. Keep production MPS
        # metrics synchronization-free and use Instruments signposts for
        # device-side timing instead.
        if self.device.type == "cuda" and torch.cuda.is_available():
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)

    def start(self) -> None:
        self.wall_start = time.perf_counter()
        self.wall_end = None
        if self.start_event is not None:
            self.start_event.record()

    def stop(self) -> None:
        if self.end_event is not None:
            self.end_event.record()
        self.wall_end = time.perf_counter()

    def synchronize(self) -> None:
        if self.end_event is not None:
            self.end_event.synchronize()
            self.wall_end = time.perf_counter()

    def wall_elapsed_ms(self) -> float:
        wall_end = self.wall_end if self.wall_end is not None else time.perf_counter()
        return (wall_end - self.wall_start) * 1000

    def accelerator_elapsed_ms(self) -> float | None:
        if self.start_event is None or self.end_event is None:
            return None
        return float(self.start_event.elapsed_time(self.end_event))


@dataclass(frozen=True)
class MPSMemorySnapshot:
    current: int
    driver: int
    recommended: int

    @property
    def cache_and_driver_overhead(self) -> int:
        return max(self.driver - self.current, 0)

    @property
    def driver_ratio(self) -> float:
        if self.recommended <= 0:
            return 0.0
        return self.driver / self.recommended


@dataclass(frozen=True)
class MPSMemoryCleanupStats:
    current_before: int
    current_after: int
    driver_before: int
    driver_after: int


@dataclass(frozen=True)
class CUDAMemorySnapshot:
    allocated: int
    reserved: int
    peak_allocated: int
    peak_reserved: int
    free: int
    total: int

    @property
    def reserved_not_allocated(self) -> int:
        return max(self.reserved - self.allocated, 0)


def synchronize_device(device: str | torch.device) -> None:
    """Wait for work submitted to the accelerator actually used by the model."""
    resolved = torch.device(device)
    if resolved.type == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()
    elif resolved.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(resolved)


def get_mps_memory_snapshot(
    device: str | torch.device,
) -> MPSMemorySnapshot | None:
    resolved = torch.device(device)
    if resolved.type != "mps" or not torch.backends.mps.is_available():
        return None

    return MPSMemorySnapshot(
        current=torch.mps.current_allocated_memory(),
        driver=torch.mps.driver_allocated_memory(),
        recommended=torch.mps.recommended_max_memory(),
    )


def get_cuda_memory_snapshot(
    device: str | torch.device,
) -> CUDAMemorySnapshot | None:
    resolved = torch.device(device)
    if resolved.type != "cuda" or not torch.cuda.is_available():
        return None

    free, total = torch.cuda.mem_get_info(resolved)
    return CUDAMemorySnapshot(
        allocated=torch.cuda.memory_allocated(resolved),
        reserved=torch.cuda.memory_reserved(resolved),
        peak_allocated=torch.cuda.max_memory_allocated(resolved),
        peak_reserved=torch.cuda.max_memory_reserved(resolved),
        free=free,
        total=total,
    )


def empty_mps_cache(
    device: str | torch.device,
) -> MPSMemoryCleanupStats | None:
    """Release unused MPS cache for the device used by inference."""
    device_type = torch.device(device).type

    if device_type != "mps" or not torch.backends.mps.is_available():
        return None

    # MPS work is asynchronous. Wait until both semantic generation and decoding
    # have finished before measuring and releasing unused allocator blocks.
    synchronize_device(device)
    current_before = torch.mps.current_allocated_memory()
    driver_before = torch.mps.driver_allocated_memory()

    gc.collect()
    torch.mps.empty_cache()

    return MPSMemoryCleanupStats(
        current_before=current_before,
        current_after=torch.mps.current_allocated_memory(),
        driver_before=driver_before,
        driver_after=torch.mps.driver_allocated_memory(),
    )
