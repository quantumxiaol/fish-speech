import gc
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MPSMemoryCleanupStats:
    current_before: int
    current_after: int
    driver_before: int
    driver_after: int


def empty_mps_cache(
    device: str | torch.device,
) -> MPSMemoryCleanupStats | None:
    """Release unused MPS cache for the device used by inference."""
    device_type = torch.device(device).type

    if device_type != "mps" or not torch.backends.mps.is_available():
        return None

    # MPS work is asynchronous. Wait until both semantic generation and decoding
    # have finished before measuring and releasing unused allocator blocks.
    torch.mps.synchronize()
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
