import os
import queue
import re
import threading
import time
import traceback
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal, Optional, Tuple, Union

import click
import numpy as np
import torch
import torch._inductor.config
from loguru import logger
from tqdm import tqdm

from fish_speech.content_sequence import (
    BasePart,
    TextPart,
    VQPart,
)
from fish_speech.conversation import Conversation, Message
from fish_speech.device_memory import (
    AcceleratorTimer,
    get_cuda_memory_snapshot,
    get_mps_memory_snapshot,
    synchronize_device,
)
from fish_speech.env_config import checkpoint_path, default_device
from fish_speech.tokenizer import IM_END_TOKEN

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True

if hasattr(torch._inductor.config, "fx_graph_cache"):
    torch._inductor.config.fx_graph_cache = True


from torch.nn.attention import SDPBackend, sdpa_kernel

from fish_speech.models.text2semantic.llama import (
    BaseTransformer,
    DualARTransformer,
    NaiveTransformer,
)


def multinomial_sample_one_no_sync(probs_sort):
    q = torch.rand_like(probs_sort)
    q = -torch.log(q)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


RAS_WIN_SIZE = 10  # window for Repetition Aware Sampling
RAS_HIGH_TEMP = 1.0
RAS_HIGH_TOP_P = 0.9


@dataclass(frozen=True)
class PerfDetailSample:
    frame_index: int
    slow_transformer_ms: float
    output_projection_ms: float
    slow_ar_and_output_ms: float
    main_sampler_ms: float
    fast_ar_and_sampler_ms: float
    total_ms: float


def should_sample_perf_detail(
    frame_index: int,
    samples_collected: int,
    max_samples: int,
) -> bool:
    """Sample early frames densely and later context every 16 frames."""
    if max_samples <= 0 or samples_collected >= max_samples:
        return False
    return frame_index < 4 or (frame_index >= 16 and frame_index % 16 == 0)


def summarize_perf_detail(
    samples: list[PerfDetailSample],
) -> dict[str, float]:
    if not samples:
        return {}

    def mean(attribute: str) -> float:
        return sum(getattr(sample, attribute) for sample in samples) / len(samples)

    first_count = min(4, len(samples))
    last_count = min(4, len(samples))
    first_total = sum(sample.total_ms for sample in samples[:first_count]) / first_count
    last_total = sum(sample.total_ms for sample in samples[-last_count:]) / last_count
    total_stage_ms = sum(sample.total_ms for sample in samples)
    slow_transformer_ms = sum(sample.slow_transformer_ms for sample in samples)
    output_projection_ms = sum(sample.output_projection_ms for sample in samples)
    slow_stage_ms = sum(sample.slow_ar_and_output_ms for sample in samples)
    sampler_stage_ms = sum(sample.main_sampler_ms for sample in samples)
    fast_stage_ms = sum(sample.fast_ar_and_sampler_ms for sample in samples)

    return {
        "perf_detail_sample_count": float(len(samples)),
        "perf_slow_transformer_mean_ms": mean("slow_transformer_ms"),
        "perf_output_projection_mean_ms": mean("output_projection_ms"),
        "perf_slow_ar_and_output_mean_ms": mean("slow_ar_and_output_ms"),
        "perf_main_sampler_mean_ms": mean("main_sampler_ms"),
        "perf_fast_ar_and_sampler_mean_ms": mean("fast_ar_and_sampler_ms"),
        "perf_profiled_frame_mean_ms": mean("total_ms"),
        "perf_slow_transformer_share": (
            slow_transformer_ms / total_stage_ms if total_stage_ms > 0 else 0.0
        ),
        "perf_output_projection_share": (
            output_projection_ms / total_stage_ms if total_stage_ms > 0 else 0.0
        ),
        "perf_slow_ar_and_output_share": (
            slow_stage_ms / total_stage_ms if total_stage_ms > 0 else 0.0
        ),
        "perf_main_sampler_share": (
            sampler_stage_ms / total_stage_ms if total_stage_ms > 0 else 0.0
        ),
        "perf_fast_ar_and_sampler_share": (
            fast_stage_ms / total_stage_ms if total_stage_ms > 0 else 0.0
        ),
        "perf_first_frames_mean_ms": first_total,
        "perf_last_frames_mean_ms": last_total,
        "perf_context_latency_ratio": (
            last_total / first_total if first_total > 0 else 0.0
        ),
    }


def log_device_memory(label: str, device: str | torch.device) -> None:
    gib = 1024**3
    mps = get_mps_memory_snapshot(device)
    if mps is not None:
        logger.info(
            "MPS memory {}: tensors={:.2f} GiB, driver={:.2f} GiB, "
            "non_tensor_driver={:.2f} GiB, recommended={:.2f} GiB, "
            "driver/recommended={:.1%}",
            label,
            mps.current / gib,
            mps.driver / gib,
            mps.cache_and_driver_overhead / gib,
            mps.recommended / gib,
            mps.driver_ratio,
        )
        return

    cuda = get_cuda_memory_snapshot(device)
    if cuda is not None:
        logger.info(
            "CUDA memory {}: allocated={:.2f} GiB, reserved={:.2f} GiB, "
            "reserved_not_allocated={:.2f} GiB, peak_allocated={:.2f} GiB, "
            "peak_reserved={:.2f} GiB, free={:.2f}/{:.2f} GiB",
            label,
            cuda.allocated / gib,
            cuda.reserved / gib,
            cuda.reserved_not_allocated / gib,
            cuda.peak_allocated / gib,
            cuda.peak_reserved / gib,
            cuda.free / gib,
            cuda.total / gib,
        )


def logits_to_probs(
    logits,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    top_k: int,
    repetition_penalty: torch.Tensor,
    previous_tokens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if previous_tokens is not None:
        previous_tokens = previous_tokens.long()
        previous_scores = torch.gather(logits, dim=-1, index=previous_tokens)
        penalized_scores = torch.where(
            previous_scores < 0,
            previous_scores * repetition_penalty,
            previous_scores / repetition_penalty,
        )
        # Use the out-of-place form so normal and RAS fallback sampling both
        # start from the same unmodified logits.
        logits = logits.scatter(dim=-1, index=previous_tokens, src=penalized_scores)

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)

    indices = torch.arange(sorted_logits.shape[-1], device=sorted_logits.device)
    top_k_mask = indices >= top_k
    sorted_indices_to_remove = (cum_probs > top_p) | top_k_mask
    sorted_indices_to_remove[0] = False  # 单元素修改问题不大，或者写成 | (indices != 0)

    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )
    logits = torch.where(
        indices_to_remove, float("-Inf"), logits
    )  # 同样替换 masked_fill_ 为 torch.where
    logits = logits / torch.clip(temperature, min=1e-5)

    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def sample(
    logits,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    top_k: int,
    repetition_penalty: torch.Tensor,
    previous_tokens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    probs = logits_to_probs(
        logits=logits[0, -1],
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        previous_tokens=previous_tokens,
    )
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


def decode_one_token_ar(
    model: DualARTransformer,
    x: torch.Tensor,
    input_pos: torch.Tensor,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    top_k: int,
    repetition_penalty: torch.Tensor,
    semantic_logit_bias: torch.Tensor,
    audio_masks: torch.Tensor,
    audio_parts: torch.Tensor,
    previous_tokens: Optional[torch.Tensor] = None,
    kv_len: Optional[int] = None,
    perf_sample: Optional[dict[str, float]] = None,
) -> torch.Tensor:
    if perf_sample is not None:
        # Diagnostic mode only: isolate the three data-dependent stages with
        # explicit boundaries. Normal inference never enters this branch.
        synchronize_device(x.device)
        frame_start = time.perf_counter()
        stage_start = frame_start

    forward_result = model.forward_generate(
        x,
        input_pos,
        audio_masks=audio_masks,
        audio_parts=audio_parts,
        kv_len=kv_len,
        perf_sample=perf_sample,
    )
    logits = forward_result.logits  # (1, 1, vocab_size)
    hidden_states = forward_result.hidden_states

    if perf_sample is not None:
        synchronize_device(x.device)
        stage_end = time.perf_counter()
        perf_sample["slow_ar_and_output_ms"] = (stage_end - stage_start) * 1000
        stage_start = stage_end

    # Apply constrained decoding: only allow semantic tokens + im_end
    biased_logits = logits + semantic_logit_bias

    # Normal sample
    main_token_normal = sample(
        biased_logits,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        previous_tokens=previous_tokens[0] if previous_tokens is not None else None,
    )[0]

    # RAS: also sample with high temp to use as fallback if token repeats
    high_temp = torch.tensor(
        RAS_HIGH_TEMP, device=temperature.device, dtype=temperature.dtype
    )
    high_top_p = torch.tensor(RAS_HIGH_TOP_P, device=top_p.device, dtype=top_p.dtype)
    main_token_high = sample(
        biased_logits,
        temperature=high_temp,
        top_p=high_top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        previous_tokens=previous_tokens[0] if previous_tokens is not None else None,
    )[0]

    # Use high-temp sample if: token is semantic AND token is in previous window
    if previous_tokens is not None:
        in_window = (previous_tokens[0] == main_token_normal).any()
        # Use tensor ops (&, torch.where) instead of Python (and, if) — torch.compile requires no data-dependent branching
        is_semantic = (main_token_normal >= model.config.semantic_begin_id) & (
            main_token_normal <= model.config.semantic_end_id
        )
        should_use_high = in_window & is_semantic
        main_token_normal = torch.where(
            should_use_high, main_token_high, main_token_normal
        )

    codebooks = [main_token_normal]

    if perf_sample is not None:
        synchronize_device(x.device)
        stage_end = time.perf_counter()
        perf_sample["main_sampler_ms"] = (stage_end - stage_start) * 1000
        stage_start = stage_end

    input_pos = torch.tensor([0], device=hidden_states.device, dtype=torch.long)
    model.forward_generate_fast(hidden_states, input_pos)

    a = codebooks[0] - model.config.semantic_begin_id
    a = torch.clamp(a, min=0, max=model.config.codebook_size - 1)

    hidden_states = model.fast_embeddings(a)
    codebooks.append(a)

    for codebook_idx in range(1, model.config.num_codebooks):
        input_pos = torch.tensor(
            [codebook_idx], device=hidden_states.device, dtype=torch.long
        )
        logits = model.forward_generate_fast(hidden_states, input_pos)

        short_logits = logits  # DualAR predicts config.codebook_size number of tokens

        # Convert logits to probs (no constrain for fast codebooks)
        a = sample(
            short_logits,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            previous_tokens=(
                previous_tokens[codebook_idx + 1]
                if previous_tokens is not None
                else None
            ),
        )[0]

        hidden_states = model.fast_embeddings(a)
        codebooks.append(a)

    codebooks = torch.stack(codebooks, dim=1)

    if perf_sample is not None:
        synchronize_device(x.device)
        stage_end = time.perf_counter()
        perf_sample["fast_ar_and_sampler_ms"] = (stage_end - stage_start) * 1000
        perf_sample["total_ms"] = (stage_end - frame_start) * 1000

    # Only delete references, let Python GC handle cleanup
    del logits, hidden_states, forward_result

    return codebooks.T


def decode_n_tokens(
    model: DualARTransformer,
    cur_token: torch.Tensor,
    input_pos: torch.Tensor,
    num_new_tokens: int,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    top_k: int,
    repetition_penalty: torch.Tensor,
    semantic_logit_bias: torch.Tensor,
    audio_masks: torch.Tensor,
    audio_parts: torch.Tensor,
    decode_one_token=decode_one_token_ar,
    kv_start_pos: Optional[int] = None,
    perf_detail: bool = False,
    perf_sample_frames: int = 16,
    perf_samples: Optional[list[PerfDetailSample]] = None,
):
    codebook_dim = model.config.num_codebooks + 1
    if num_new_tokens <= 0:
        return torch.empty(
            (codebook_dim, 0), dtype=cur_token.dtype, device=cur_token.device
        )

    if kv_start_pos is None:
        # Compatibility fallback for direct callers. Production generation
        # passes the Python position explicitly, avoiding a device sync here.
        kv_start_pos = int(input_pos[0].item())

    # RAS and repetition penalty only need set membership, not chronological
    # ordering. Seed the fixed-size ring with the prefill token so token 0 is
    # not treated as fake history for fast codebooks.
    first_token = cur_token.view(codebook_dim, -1)[:, -1]
    previous_tokens = first_token[:, None].expand(-1, RAS_WIN_SIZE).clone()
    # Accumulate all generated tokens (the actual output)
    new_tokens = []

    # [MODIFIED] Pre-fetch ID for efficiency loop
    im_end_id = model.tokenizer.get_token_id(IM_END_TOKEN)

    progress = tqdm(
        range(num_new_tokens),
        desc=f"AR decode (max {num_new_tokens}, stops at EOS)",
        unit="tok",
        # The true output length is unknown until EOS. Do not display a
        # percentage or ETA against max_new_tokens as if it were the target.
        bar_format="{desc}: {n_fmt} tokens [{elapsed}, {rate_fmt}]",
    )
    for i in progress:
        perf_sample = None
        if perf_detail and should_sample_perf_detail(
            frame_index=i,
            samples_collected=len(perf_samples or []),
            max_samples=perf_sample_frames,
        ):
            perf_sample = {}

        with sdpa_kernel(SDPBackend.MATH):
            if perf_sample is None:
                next_token = decode_one_token(
                    model=model,
                    x=cur_token,
                    input_pos=input_pos,
                    kv_len=kv_start_pos + i + 1,
                    previous_tokens=previous_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    semantic_logit_bias=semantic_logit_bias,
                    audio_masks=audio_masks,
                    audio_parts=audio_parts,
                ).clone()
            else:
                next_token = decode_one_token(
                    model=model,
                    x=cur_token,
                    input_pos=input_pos,
                    kv_len=kv_start_pos + i + 1,
                    previous_tokens=previous_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    semantic_logit_bias=semantic_logit_bias,
                    audio_masks=audio_masks,
                    audio_parts=audio_parts,
                    perf_sample=perf_sample,
                ).clone()

        if perf_sample is not None and perf_samples is not None:
            perf_samples.append(
                PerfDetailSample(
                    frame_index=i,
                    slow_transformer_ms=perf_sample["slow_transformer_ms"],
                    output_projection_ms=perf_sample["output_projection_ms"],
                    slow_ar_and_output_ms=perf_sample["slow_ar_and_output_ms"],
                    main_sampler_ms=perf_sample["main_sampler_ms"],
                    fast_ar_and_sampler_ms=perf_sample["fast_ar_and_sampler_ms"],
                    total_ms=perf_sample["total_ms"],
                )
            )

        input_pos += 1
        cur_token = next_token.view(1, model.config.num_codebooks + 1, -1)
        # Reuse the ring buffer instead of allocating a new tensor with roll().
        previous_tokens[:, i % RAS_WIN_SIZE] = next_token.view(codebook_dim, -1)[:, 0]
        new_tokens.append(next_token)

        if cur_token[0, 0, -1] == im_end_id:
            break

    progress.close()
    del cur_token

    return torch.cat(new_tokens, dim=1)


def ensure_model_caches(
    model: DualARTransformer,
    *,
    device: str | torch.device,
) -> None:
    if getattr(model, "_cache_setup_done", False):
        return

    with torch.device(device):
        model.setup_caches(
            max_batch_size=1,
            max_seq_len=model.config.max_seq_len,
            dtype=next(model.parameters()).dtype,
        )
    model._cache_setup_done = True


@torch.no_grad()
@torch.inference_mode()
def generate(
    *,
    model: DualARTransformer,
    prompt: torch.Tensor,
    max_new_tokens: int,
    audio_masks: torch.Tensor,
    audio_parts: torch.Tensor,
    decode_one_token=decode_one_token_ar,
    num_samples: int = 1,
    metrics: Optional[dict[str, float]] = None,
    perf_detail: bool = False,
    perf_sample_frames: int = 16,
    **sampling_kwargs,
):
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
    """

    # create an empty tensor of the expected final shape and fill in the current tokens
    T = prompt.size(1)
    prompt = prompt[None].repeat(num_samples, 1, 1)

    if T >= model.config.max_seq_len:
        raise ValueError(
            f"Input sequence length {T} exceeds max_seq_len {model.config.max_seq_len}"
        )

    if max_new_tokens:
        if T + max_new_tokens > model.config.max_seq_len:
            max_new_tokens = model.config.max_seq_len - T

        T_new = T + max_new_tokens
    else:
        T_new = model.config.max_seq_len
        max_new_tokens = T_new - T

    device = prompt.device
    dtype = next(
        model.parameters()
    ).dtype  # model weight dtype (bfloat16), NOT prompt dtype (int32)

    ensure_model_caches(model, device=device)

    codebook_dim = 1 + model.config.num_codebooks

    # Create new tensor each time, but try to reuse memory
    input_pos = torch.arange(0, T, device=device, dtype=torch.long)
    empty = torch.empty(
        (codebook_dim, model.config.max_seq_len), dtype=prompt.dtype, device=device
    )
    empty[:, :T] = prompt
    seq = empty

    temp_val = sampling_kwargs.get("temperature", 1.0)
    top_p_val = sampling_kwargs.get("top_p", 0.9)
    top_k_val = sampling_kwargs.get("top_k", 30)
    repetition_penalty_val = sampling_kwargs.get("repetition_penalty", 1.1)

    temperature = torch.tensor(temp_val, device=device, dtype=dtype)
    top_p = torch.tensor(top_p_val, device=device, dtype=dtype)
    repetition_penalty = torch.tensor(
        repetition_penalty_val, device=device, dtype=dtype
    )

    # Build semantic logit bias: 0 for semantic tokens + im_end, -inf for all others
    vocab_size = model.config.vocab_size
    semantic_logit_bias = torch.full(
        (1, 1, vocab_size), float("-inf"), device=device, dtype=dtype
    )

    # [MODIFIED] Use config for semantic range
    semantic_logit_bias[
        0, 0, model.config.semantic_begin_id : model.config.semantic_end_id + 1
    ] = 0.0

    # [MODIFIED] Use tokenizer.get_token_id (Wrapper method)
    semantic_logit_bias[0, 0, model.tokenizer.get_token_id(IM_END_TOKEN)] = 0.0

    prefill_decode = decode_one_token_ar

    semantic_timer = AcceleratorTimer(device)
    prefill_timer = AcceleratorTimer(device)
    ar_decode_timer = AcceleratorTimer(device)
    semantic_timer.start()
    prefill_timer.start()
    prefill_perf_sample: dict[str, float] | None = {} if perf_detail else None
    first_token = prefill_decode(
        model,
        prompt.view(1, codebook_dim, -1),
        input_pos,
        temperature,
        top_p,
        top_k_val,
        repetition_penalty,
        semantic_logit_bias,
        audio_masks,
        audio_parts,
        kv_len=T,
        perf_sample=prefill_perf_sample,
    )
    prefill_timer.stop()
    seq[:, T : T + 1] = first_token

    # Recreate input_pos
    input_pos = torch.tensor([T], device=device, dtype=torch.int)

    ar_decode_timer.start()
    perf_samples: list[PerfDetailSample] = []
    x = decode_n_tokens(
        model,
        first_token.view(1, codebook_dim, -1),
        input_pos,
        max_new_tokens - 1,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k_val,
        repetition_penalty=repetition_penalty,
        semantic_logit_bias=semantic_logit_bias,
        audio_masks=audio_masks,
        audio_parts=audio_parts,
        decode_one_token=decode_one_token,
        kv_start_pos=T,
        perf_detail=perf_detail,
        perf_sample_frames=perf_sample_frames,
        perf_samples=perf_samples,
    )
    ar_decode_timer.stop()
    semantic_timer.stop()
    # One wait at the end of the whole semantic phase keeps default metrics
    # accurate without forcing a CPU/GPU handshake between prefill and AR.
    semantic_timer.synchronize()
    if metrics is not None:
        metrics["prefill_wall_ms"] = prefill_timer.wall_elapsed_ms()
        metrics["ar_decode_wall_ms"] = ar_decode_timer.wall_elapsed_ms()
        metrics["semantic_wall_ms"] = semantic_timer.wall_elapsed_ms()
        prefill_accelerator_ms = prefill_timer.accelerator_elapsed_ms()
        ar_decode_accelerator_ms = ar_decode_timer.accelerator_elapsed_ms()
        semantic_accelerator_ms = semantic_timer.accelerator_elapsed_ms()
        if prefill_accelerator_ms is not None:
            metrics["prefill_accelerator_ms"] = prefill_accelerator_ms
        if ar_decode_accelerator_ms is not None:
            metrics["ar_decode_accelerator_ms"] = ar_decode_accelerator_ms
        if semantic_accelerator_ms is not None:
            metrics["semantic_accelerator_ms"] = semantic_accelerator_ms
        if prefill_perf_sample is not None:
            metrics.update(
                {
                    f"perf_prefill_{key}": value
                    for key, value in prefill_perf_sample.items()
                }
            )
        metrics.update(summarize_perf_detail(perf_samples))

    if perf_detail and perf_samples:
        detail_summary = summarize_perf_detail(perf_samples)
        logger.warning(
            "Perf detail uses synchronized sampled frames and is diagnostic "
            "only; do not compare this request's RTF with normal inference."
        )
        logger.info(
            "Perf detail sampled frames: {}",
            ",".join(str(sample.frame_index) for sample in perf_samples),
        )
        logger.info(
            "Perf detail per sampled frame: slow_transformer={:.3f}ms "
            "({:.1%}), output_projection={:.3f}ms ({:.1%}), "
            "slow_ar+output={:.3f}ms ({:.1%}), "
            "main_sampler={:.3f}ms ({:.1%}), "
            "fast_ar+residual_sampler={:.3f}ms ({:.1%}), "
            "profiled_total={:.3f}ms, context first/last={:.3f}/{:.3f}ms "
            "(ratio {:.3f})",
            detail_summary["perf_slow_transformer_mean_ms"],
            detail_summary["perf_slow_transformer_share"],
            detail_summary["perf_output_projection_mean_ms"],
            detail_summary["perf_output_projection_share"],
            detail_summary["perf_slow_ar_and_output_mean_ms"],
            detail_summary["perf_slow_ar_and_output_share"],
            detail_summary["perf_main_sampler_mean_ms"],
            detail_summary["perf_main_sampler_share"],
            detail_summary["perf_fast_ar_and_sampler_mean_ms"],
            detail_summary["perf_fast_ar_and_sampler_share"],
            detail_summary["perf_profiled_frame_mean_ms"],
            detail_summary["perf_first_frames_mean_ms"],
            detail_summary["perf_last_frames_mean_ms"],
            detail_summary["perf_context_latency_ratio"],
        )
    seq = seq[:, : T + 1 + x.size(1)]
    seq[:, T + 1 :] = x

    # Clean up temporary variables
    del first_token, x, prompt, empty, input_pos

    return seq


def init_model(checkpoint_path, device, precision, compile=False, max_seq_len=None):
    model = DualARTransformer.from_pretrained(
        checkpoint_path, load_weights=True, max_length=max_seq_len
    )

    model = model.to(device=device, dtype=precision)
    logger.info(f"Restored model from checkpoint")

    if isinstance(model, DualARTransformer):
        decode_one_token = decode_one_token_ar
        # prefill_n_tokens = decode_one_token_ar
        logger.info("Using DualARTransformer")
    else:
        raise ValueError("Unsupported model type")

    # Pre-create fixed parameter tensors to avoid runtime creation
    model.fixed_temperature = torch.tensor(0.7, device=device, dtype=torch.float)
    model.fixed_top_p = torch.tensor(0.7, device=device, dtype=torch.float)
    model.fixed_repetition_penalty = torch.tensor(1.5, device=device, dtype=torch.float)

    # Mark whether cache has been initialized
    model._cache_setup_done = False

    if compile:
        logger.info("Compiling function...")
        decode_one_token = torch.compile(
            decode_one_token,
            backend="inductor" if torch.cuda.is_available() else "aot_eager",
            mode="default" if torch.cuda.is_available() else None,
            fullgraph=True,
        )

    return model.eval(), decode_one_token


@torch.inference_mode()
def load_codec_model(codec_checkpoint_path, device, precision=torch.bfloat16):
    """Load the DAC codec model for audio encoding/decoding."""
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    config_path = Path(__file__).parent.parent.parent / "configs" / "modded_dac_vq.yaml"
    cfg = OmegaConf.load(str(config_path))
    codec = instantiate(cfg)

    state_dict = torch.load(codec_checkpoint_path, map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if any("generator" in k for k in state_dict):
        state_dict = {
            k.replace("generator.", ""): v
            for k, v in state_dict.items()
            if "generator." in k
        }
    codec.load_state_dict(state_dict, strict=False)
    codec.eval()
    codec.to(device=device, dtype=precision)
    return codec


@torch.inference_mode()
def encode_audio(audio_path, codec, device):
    """Encode an audio file to VQ codes."""
    import torchaudio

    wav, sr = torchaudio.load(str(audio_path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = torchaudio.functional.resample(wav.to(device), sr, codec.sample_rate)[0]

    # Match codec model dtype (e.g. bfloat16)
    model_dtype = next(codec.parameters()).dtype
    audios = wav[None, None].to(dtype=model_dtype)  # (1, 1, T)
    audio_lengths = torch.tensor([len(wav)], device=device, dtype=torch.long)

    indices, feature_lengths = codec.encode(audios, audio_lengths)
    return indices[0, :, : feature_lengths[0]]  # (num_codebooks, T)


@torch.inference_mode()
def decode_to_audio(codes, codec):
    """Decode VQ codes to audio waveform."""
    # codes: (num_codebooks, T) -> (1, num_codebooks, T)
    audio = codec.from_indices(codes[None])
    return audio[0, 0]  # (T,) mono waveform


@dataclass
class GenerateResponse:
    action: Literal["sample", "next"]
    codes: Optional[torch.Tensor] = None
    text: Optional[str] = None
    metrics: Optional[dict[str, float]] = None


def split_text_by_speaker(text: str) -> list[str]:
    """
    Split text into turns based on <|speaker:X|> tags.

    Args:
        text: The full text with speaker tags

    Returns:
        List of speaker turns, each starting with <|speaker:X|>
    """
    pattern = r"(<\|speaker:\d+\|>)"
    parts = re.split(pattern, text)

    turns = []
    i = 0
    while i < len(parts):
        part = parts[i].strip()
        if re.match(pattern, part):
            if i + 1 < len(parts):
                turn = part + parts[i + 1]
                turns.append(turn.strip())
                i += 2
            else:
                turns.append(part)
                i += 1
        else:
            i += 1

    return turns


def group_turns_into_batches(
    turns: list[str], max_speakers: int = 3, max_bytes: int = 300
) -> list[str]:
    """
    Group turns into batches based on speaker count or byte limit.

    Args:
        turns: List of speaker turns
        max_speakers: Maximum number of speakers per batch (default 3)
        max_bytes: Maximum UTF-8 bytes per batch (default 300)

    Returns:
        List of batched text strings
    """
    batches = []
    current_batch = []
    current_bytes = 0

    for turn in turns:
        turn_bytes = len(turn.encode("utf-8"))

        would_exceed_speakers = len(current_batch) >= max_speakers
        would_exceed_bytes = current_bytes + turn_bytes > max_bytes and current_batch

        if would_exceed_speakers or would_exceed_bytes:
            batches.append("\n".join(current_batch))
            current_batch = [turn]
            current_bytes = turn_bytes
        else:
            current_batch.append(turn)
            current_bytes += turn_bytes

    if current_batch:
        batches.append("\n".join(current_batch))

    return batches


def split_text_into_batches(text: str, max_bytes: int) -> list[str]:
    """Split plain text without breaking UTF-8 characters.

    Prefer sentence, clause, and whitespace boundaries. If no boundary exists
    before the byte limit, fall back to a character boundary.
    """
    if max_bytes <= 0:
        raise ValueError("chunk_length must be greater than 0")

    remaining = text.strip()
    if not remaining:
        return [text]

    boundaries = frozenset(".!?。！？,，;；:：、\n\t ")
    batches: list[str] = []

    while len(remaining.encode("utf-8")) > max_bytes:
        current: list[str] = []
        current_bytes = 0
        last_boundary = 0

        for char in remaining:
            char_bytes = len(char.encode("utf-8"))
            if current and current_bytes + char_bytes > max_bytes:
                break
            current.append(char)
            current_bytes += char_bytes
            if char in boundaries:
                last_boundary = len(current)

        cut = last_boundary or len(current)
        if cut == 0:
            # max_bytes is smaller than one character; keep forward progress.
            cut = 1

        chunk = "".join(current[:cut]).strip()
        if chunk:
            batches.append(chunk)
        remaining = ("".join(current[cut:]) + remaining[len(current) :]).strip()

    if remaining:
        batches.append(remaining)

    return batches


def prepare_text_batches(
    text: str,
    *,
    iterative_prompt: bool,
    chunk_length: int,
) -> list[str]:
    if not iterative_prompt:
        return [text]

    turns = split_text_by_speaker(text)
    if turns:
        return group_turns_into_batches(turns, max_speakers=5, max_bytes=chunk_length)

    return split_text_into_batches(text, max_bytes=chunk_length)


def generate_long(
    *,
    model,
    device: Union[str, torch.device],
    decode_one_token: Callable,
    text: str,
    num_samples: int = 1,
    max_new_tokens: int = 0,
    top_p: float = 0.9,
    top_k: int = 30,
    repetition_penalty: float = 1.1,
    temperature: float = 1.0,
    compile: bool = False,
    perf_detail: bool = False,
    perf_sample_frames: int = 16,
    iterative_prompt: bool = True,
    chunk_length: int = 512,
    prompt_text: Optional[Union[str, list[str]]] = None,
    prompt_tokens: Optional[Union[torch.Tensor, list[torch.Tensor]]] = None,
):
    assert 0 < top_p <= 1, "top_p must be in (0, 1]"
    assert 0 < repetition_penalty <= 2, "repetition_penalty must be in (0, 2]"
    assert 0 < temperature < 2, "temperature must be in (0, 2)"
    if perf_detail and compile:
        raise ValueError("perf_detail is not supported together with torch.compile")
    if not 1 <= perf_sample_frames <= 64:
        raise ValueError("perf_sample_frames must be between 1 and 64")

    use_prompt = bool(prompt_text) and bool(prompt_tokens)
    if use_prompt and isinstance(prompt_text, str):
        prompt_text = [prompt_text]
        prompt_tokens = [prompt_tokens]

    if use_prompt:
        assert len(prompt_text) == len(prompt_tokens), (
            "Prompt text and tokens must have the same length"
        )

    if prompt_tokens:
        prompt_tokens = [i.cpu() for i in prompt_tokens]

    tokenizer = model.tokenizer
    max_length = model.config.max_seq_len

    # Build base conversation with system message
    base_conversation = Conversation()

    if use_prompt:
        # Auto-add speaker tags to prompt texts that don't have them
        tagged_prompt_text = []
        for i, t in enumerate(prompt_text):
            if not re.search(r"<\|speaker:\d+\|>", t):
                tagged_prompt_text.append(f"<|speaker:{i}|>{t}")
            else:
                tagged_prompt_text.append(t)

        system_parts: list[BasePart] = [
            TextPart(
                text="convert the provided text to speech reference to the following:\n\nText:\n",
                cal_loss=False,
            ),
        ]
        reference_text = "\n".join(tagged_prompt_text)
        system_parts.append(TextPart(text=reference_text, cal_loss=False))
        system_parts.append(TextPart(text="\n\nSpeech:\n", cal_loss=False))
        all_codes = torch.cat([c for c in prompt_tokens], dim=1)
        system_parts.append(VQPart(codes=all_codes, cal_loss=False))
        # torch.save(all_codes, "debug_vq_codes.pt")
    else:
        system_parts: list[BasePart] = [
            TextPart(text="convert the provided text to speech", cal_loss=False)
        ]

    base_conversation.append(
        Message(
            role="system",
            parts=system_parts,
            cal_loss=False,
            add_im_start=True,
            add_im_end=True,
        )
    )

    batches = prepare_text_batches(
        text,
        iterative_prompt=iterative_prompt,
        chunk_length=chunk_length,
    )
    logger.info(
        "Iterative prompt: {}, grouped text into {} batch(es)",
        iterative_prompt,
        len(batches),
    )

    for sample_idx in range(num_samples):
        # Deep copy base conversation for this sample
        conversation = deepcopy(base_conversation)

        for batch_idx, batch_text in enumerate(batches):
            batch_start = time.perf_counter()
            log_device_memory(f"before batch {batch_idx}", device)
            logger.info(
                f"--- Sample {sample_idx}, Batch {batch_idx} "
                f"({len(batch_text.encode('utf-8'))} bytes) ---"
            )
            logger.info(f"Batch text: {batch_text}")

            prompt_build_start = time.perf_counter()

            # Add user message
            conversation.append(
                Message(
                    role="user",
                    parts=[TextPart(text=batch_text, cal_loss=False)],
                    cal_loss=False,
                    add_im_start=True,
                    add_im_end=True,
                )
            )

            # Deep copy for generation (don't pollute original conversation)
            conversation_gen = deepcopy(conversation)
            conversation_gen.append(
                Message(
                    role="assistant",
                    parts=[],
                    cal_loss=False,
                    modality="voice",
                    add_im_start=True,
                    add_im_end=False,
                )
            )

            logger.info("Visualizing prompt structure:")
            conversation_gen.visualize(
                tokenizer,
                merge_audio_tokens=True,
                merge_semantic_tokens=True,
            )

            encoded, audio_masks, audio_parts = conversation_gen.encode_for_inference(
                tokenizer, num_codebooks=model.config.num_codebooks
            )

            logger.info(f"Encoded prompt shape: {encoded.shape}")
            if audio_parts is not None:
                logger.info(f"Audio parts shape: {audio_parts.shape}")
            if audio_masks is not None:
                logger.info(
                    f"Audio masks non-zero count: {torch.count_nonzero(audio_masks)}"
                )

            if encoded.size(1) > max_length - 2048:
                raise ValueError(
                    f"Prompt is too long: {encoded.size(1)} > {max_length - 2048}"
                )

            encoded = encoded.to(device=device)
            prompt_length = encoded.size(1)
            prompt_build_seconds = time.perf_counter() - prompt_build_start

            generation_metrics: dict[str, float] = {}
            y = generate(
                model=model,
                prompt=encoded,
                max_new_tokens=max_new_tokens,
                audio_masks=audio_masks,
                audio_parts=audio_parts,
                decode_one_token=decode_one_token,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                metrics=generation_metrics,
                perf_detail=perf_detail,
                perf_sample_frames=perf_sample_frames,
            )

            if sample_idx == 0 and batch_idx == 0 and compile:
                logger.info(
                    "First compiled batch wall time: {:.2f} seconds",
                    time.perf_counter() - batch_start,
                )

            batch_seconds = time.perf_counter() - batch_start
            tokens_generated = y.size(1) - prompt_length

            # Extract generated codes
            codes = y[1:, prompt_length:-1].clone()
            assert (codes >= 0).all(), f"Negative code found: {codes}"
            semantic_frames = codes.size(1)
            prefill_wall_ms = generation_metrics.get("prefill_wall_ms", 0.0)
            ar_decode_wall_ms = generation_metrics.get("ar_decode_wall_ms", 0.0)
            prefill_accelerator_ms = generation_metrics.get("prefill_accelerator_ms")
            ar_decode_accelerator_ms = generation_metrics.get(
                "ar_decode_accelerator_ms"
            )
            semantic_accelerator_ms = generation_metrics.get("semantic_accelerator_ms")
            semantic_wall_ms = generation_metrics.get("semantic_wall_ms", 0.0)
            semantic_seconds = semantic_wall_ms / 1000
            frames_per_second = (
                semantic_frames / semantic_seconds if semantic_seconds > 0 else 0.0
            )
            frame_ms = (
                semantic_wall_ms / semantic_frames if semantic_frames > 0 else 0.0
            )
            logger.info(
                "Batch {} metrics: prompt_tokens={}, generated_tokens={}, "
                "semantic_frames={}, prompt_build_wall={:.3f}ms, "
                "prefill_first_frame_wall={:.3f}ms, "
                "ar_decode_wall={:.3f}ms, semantic_wall={:.3f}ms, "
                "prefill_accelerator={}, ar_decode_accelerator={}, "
                "semantic_accelerator={}, "
                "semantic_frame={:.3f}ms, semantic_frames/s={:.2f}, "
                "batch_wall={:.3f}ms",
                batch_idx,
                prompt_length,
                tokens_generated,
                semantic_frames,
                prompt_build_seconds * 1000,
                prefill_wall_ms,
                ar_decode_wall_ms,
                semantic_wall_ms,
                (
                    f"{prefill_accelerator_ms:.3f}ms"
                    if prefill_accelerator_ms is not None
                    else "n/a"
                ),
                (
                    f"{ar_decode_accelerator_ms:.3f}ms"
                    if ar_decode_accelerator_ms is not None
                    else "n/a"
                ),
                (
                    f"{semantic_accelerator_ms:.3f}ms"
                    if semantic_accelerator_ms is not None
                    else "n/a"
                ),
                frame_ms,
                frames_per_second,
                batch_seconds * 1000,
            )
            log_device_memory(f"after batch {batch_idx}", device)

            # Add assistant message with generated codes back to conversation
            conversation.append(
                Message(
                    role="assistant",
                    parts=[VQPart(codes=codes.cpu(), cal_loss=False)],
                    cal_loss=False,
                    modality="voice",
                    add_im_start=True,
                    add_im_end=True,
                )
            )

            yield GenerateResponse(
                action="sample",
                codes=codes,
                text=batch_text,
                metrics=(
                    {
                        "prompt_build_wall_ms": prompt_build_seconds * 1000,
                        "prefill_wall_ms": prefill_wall_ms,
                        "ar_decode_wall_ms": ar_decode_wall_ms,
                        "semantic_wall_ms": semantic_wall_ms,
                        "semantic_frames": float(semantic_frames),
                        "batch_wall_ms": batch_seconds * 1000,
                    }
                    | {
                        key: value
                        for key, value in {
                            "prefill_accelerator_ms": prefill_accelerator_ms,
                            "ar_decode_accelerator_ms": ar_decode_accelerator_ms,
                            "semantic_accelerator_ms": semantic_accelerator_ms,
                        }.items()
                        if value is not None
                    }
                ),
            )

            # Cleanup
            del y, encoded

        if torch.device(device).type == "cuda":
            logger.info(
                f"GPU Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB"
            )

        yield GenerateResponse(action="next")


@dataclass
class WrappedGenerateResponse:
    status: Literal["success", "error"]
    response: Optional[Union[GenerateResponse, Exception]] = None


@dataclass
class GenerateRequest:
    request: dict
    response_queue: queue.Queue
    enqueued_at: float = field(default_factory=time.perf_counter)


def launch_thread_safe_queue(
    checkpoint_path,
    device,
    precision,
    compile: bool = False,
    max_seq_len: int | None = None,
):
    input_queue = queue.Queue()
    init_event = threading.Event()

    def worker():
        model, decode_one_token = init_model(
            checkpoint_path,
            device,
            precision,
            compile=compile,
            max_seq_len=max_seq_len,
        )
        ensure_model_caches(model, device=device)
        init_event.set()

        while True:
            item: GenerateRequest | None = input_queue.get()
            if item is None:
                break

            kwargs = item.request
            response_queue = item.response_queue
            queue_wait_ms = (time.perf_counter() - item.enqueued_at) * 1000
            queue_wait_reported = False

            try:
                for chunk in generate_long(
                    model=model, decode_one_token=decode_one_token, **kwargs
                ):
                    if chunk.metrics is not None and not queue_wait_reported:
                        chunk.metrics["llama_queue_wait_ms"] = queue_wait_ms
                        queue_wait_reported = True
                    response_queue.put(
                        WrappedGenerateResponse(status="success", response=chunk)
                    )

                # Only clear cache after complete request batch
                if torch.device(device).type == "cuda":
                    torch.cuda.empty_cache()

            except Exception as e:
                logger.error(traceback.format_exc())
                response_queue.put(WrappedGenerateResponse(status="error", response=e))
                # Clear cache on error
                if torch.device(device).type == "cuda":
                    torch.cuda.empty_cache()

    threading.Thread(target=worker, daemon=True).start()
    init_event.wait()

    return input_queue


@click.command()
@click.option(
    "--text",
    type=str,
    default="<|speaker:0|>你说的对, 但是原神是一款由米哈游自主研发的开放世界手游.",
)
@click.option("--prompt-text", type=str, default=None, multiple=True)
@click.option(
    "--prompt-tokens",
    type=click.Path(path_type=Path, exists=True),
    default=None,
    multiple=True,
)
@click.option(
    "--prompt-audio",
    type=click.Path(path_type=Path, exists=True),
    default=None,
    multiple=True,
)
@click.option("--output", type=click.Path(path_type=Path), default=None)
@click.option("--num-samples", type=int, default=1)
@click.option("--max-new-tokens", type=int, default=0)
@click.option("--top-p", type=float, default=0.9)
@click.option("--top-k", type=int, default=30)
@click.option("--temperature", type=float, default=1.0)
@click.option(
    "--checkpoint-path",
    type=click.Path(path_type=Path, exists=True),
    default=checkpoint_path(),
)
@click.option("--max-seq-len", type=int, default=None)
@click.option("--device", type=str, default=default_device())
@click.option("--compile/--no-compile", default=False)
@click.option("--seed", type=int, default=42)
@click.option("--half/--no-half", default=False)
@click.option("--iterative-prompt/--no-iterative-prompt", default=True)
@click.option("--chunk-length", type=int, default=300)
@click.option("--output-dir", type=Path, default="output")
def main(
    text: str,
    prompt_text: Optional[tuple[str, ...]],
    prompt_tokens: Optional[tuple[Path, ...]],
    prompt_audio: Optional[tuple[Path, ...]],
    output: Optional[Path],
    num_samples: int,
    max_new_tokens: int,
    top_p: float,
    top_k: int,
    temperature: float,
    checkpoint_path: Path,
    max_seq_len: Optional[int],
    device: str,
    compile: bool,
    seed: int,
    half: bool,
    iterative_prompt: bool,
    chunk_length: int,
    output_dir: Path,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    precision = torch.half if half else torch.bfloat16

    if prompt_text and not prompt_audio and not prompt_tokens:
        raise ValueError(
            "--prompt-text requires either --prompt-audio or --prompt-tokens"
        )
    if prompt_text and prompt_tokens and len(prompt_text) != len(prompt_tokens):
        raise ValueError(
            f"Number of prompt text ({len(prompt_text)}) and prompt tokens ({len(prompt_tokens)}) should be the same"
        )
    if prompt_text and prompt_audio and len(prompt_text) != len(prompt_audio):
        raise ValueError(
            f"Number of prompt text ({len(prompt_text)}) and prompt audio ({len(prompt_audio)}) should be the same"
        )

    logger.info("Loading model ...")
    t0 = time.time()
    model, decode_one_token = init_model(
        checkpoint_path,
        device,
        precision,
        compile=compile,
        max_seq_len=max_seq_len,
    )
    with torch.device(device):
        model.setup_caches(
            max_batch_size=1,
            max_seq_len=model.config.max_seq_len,
            dtype=next(model.parameters()).dtype,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    logger.info(f"Time to load model: {time.time() - t0:.02f} seconds")

    codec = None
    codec_checkpoint = checkpoint_path / "codec.pth"

    # Handle prompt: --prompt-audio takes priority over --prompt-tokens
    prompt_tokens_list = None
    if prompt_audio:
        logger.info("Loading codec model for audio encoding...")
        codec = load_codec_model(codec_checkpoint, device, precision)
        prompt_tokens_list = [
            encode_audio(p, codec, device).cpu() for p in prompt_audio
        ]
        logger.info(f"Encoded {len(prompt_audio)} audio file(s) to VQ codes")
    elif prompt_tokens is not None:
        prompt_tokens_list = [torch.from_numpy(np.load(p)) for p in prompt_tokens]

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    generator = generate_long(
        model=model,
        device=device,
        decode_one_token=decode_one_token,
        text=text,
        num_samples=num_samples,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        compile=compile,
        iterative_prompt=iterative_prompt,
        chunk_length=chunk_length,
        prompt_text=list(prompt_text) if prompt_text else None,
        prompt_tokens=prompt_tokens_list,
    )

    idx = 0
    codes = []

    for response in generator:
        if response.action == "sample":
            codes.append(response.codes)
            logger.info(f"Sampled text: {response.text}")
        elif response.action == "next":
            if codes:
                merged_codes = torch.cat(codes, dim=1)
                codes_npy_path = os.path.join(output_dir, f"codes_{idx}.npy")
                np.save(codes_npy_path, merged_codes.cpu().numpy())
                logger.info(f"Saved codes to {codes_npy_path}")

                # Decode to wav if --output is specified
                if output:
                    if codec is None:
                        logger.info("Loading codec model for audio decoding...")
                        codec = load_codec_model(codec_checkpoint, device, precision)
                    audio = decode_to_audio(merged_codes.to(device), codec)
                    import soundfile as sf

                    out_path = (
                        str(output)
                        if num_samples == 1
                        else str(output.with_stem(f"{output.stem}_{idx}"))
                    )
                    sf.write(out_path, audio.cpu().float().numpy(), codec.sample_rate)
                    logger.info(f"Saved audio to {out_path}")

            logger.info(f"Next sample")
            codes = []
            idx += 1
        else:
            logger.error(f"Error: {response}")


if __name__ == "__main__":
    main()
