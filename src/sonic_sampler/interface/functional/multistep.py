# SPDX-License-Identifier: Apache-2.0
# 
# Copyright (c) 2026 SonicSampler Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import torch

from typing import Tuple

from torch import Tensor

from triton import cdiv     # noqa.

from sonic_sampler.base.sampler import Verification, MAX_TOP_LP
from sonic_sampler.ops.base import next_power_of_2, MAX_K
from sonic_sampler.interface.functional.base import (
    I32,
    collapse_2d,
    conditional_stride,
    resolve_block,
    resolve_scratchpad,
    validate_count_update,
    validate_noise,
)
from sonic_sampler.interface.base import TopKStrategy
from sonic_sampler.interface.dispatch import ThreeStageWarpConfig
from sonic_sampler.ops.prologue import bitpacked_reduction_kernel
from sonic_sampler.ops.multistep import cumulative_unpack_kernel
from sonic_sampler.ops.verify import chain_speculative_verification_kernel


BF16: torch.dtype = torch.bfloat16
FP32: torch.dtype = torch.float32

TINY: float = torch.finfo(BF16).tiny


def resolve_batch_vocab(
    logits: Tensor,
    cu_seqlens: Tensor | None,
    lookahead: int | None,
) -> Tuple[bool, int, int, int]:

    varlen = cu_seqlens is not None

    match logits.ndim:

        case 2:

            n_rows, vocab_size = logits.shape

            if varlen:

                batch_size = cu_seqlens.size(0) - 1

                if not lookahead:

                    raise ValueError("maximum lookahead must be defined for varlen logits")

                else:

                    return varlen, batch_size, lookahead, vocab_size

            elif not lookahead:

                raise ValueError("found 2-D logits with zero lookahead")

            else:

                batch_size, margin = divmod(n_rows, lookahead + 1)

                if margin:

                    raise ValueError("not varlen but total rows not a multiple of lookahead + 1")

                else:

                    return varlen, batch_size, lookahead, vocab_size

        case 3:

            if varlen:

                raise ValueError("expected varlen but logits has a fixed timestep dimension")

            else:

                batch_size, timesteps, vocab_size = logits.shape

                return varlen, batch_size, timesteps - 1, vocab_size

        case _:

            raise ValueError(f"unsupported logits shape: {logits.shape}")


def resolve_unpack_buffers(
    logits: Tensor,
    values: Tensor | None,
    indices: Tensor | None,
) -> Tuple[Tensor, Tensor]:

    n_rows = logits.size(0) if logits.ndim == 2 else logits.flatten(0, 1).size(0)

    if (values_buffer := values) is None:

        values_buffer = logits.new_empty((n_rows, MAX_K), dtype=FP32)

    if (indices_buffer := indices) is None:

        indices_buffer = logits.new_empty((n_rows, MAX_K), dtype=I32)

    return values_buffer, indices_buffer


def resolve_gumbel(
    noise: Tensor | None,
    logits: Tensor,
    batch_size: int,
    timesteps: int,
    seeds: Tensor | None,
) -> Tensor | None:

    if (weights := noise) is None and seeds is None:

        weights = (
            logits.new_empty((batch_size, timesteps, logits.size(-1)))
            .exponential_(1.0)
            .log_()
        )

    return weights


def resolve_uniform(
    noise: Tensor | None,
    logits: Tensor,
    batch_size: int,
    lookahead: int,
    seeds: Tensor | None,
) -> Tensor | None:

    if (weights := noise) is None and seeds is None:

        weights = (
            logits.new_empty((batch_size, lookahead))
            .uniform_(TINY, 1)
        )

    return weights


def resolve_target_stride(
    gumbel: Tensor | None,
    probabilities: Tensor | None,
    counts: Tensor | None,
    bias: Tensor | None,
    target_vocab: int,
) -> int:

    if gumbel is not None:

        return collapse_2d(gumbel).stride(0)

    if probabilities is not None:

        return probabilities.stride(1)

    stride = conditional_stride(counts, bias)

    return target_vocab if stride is None else stride


def resolve_tokens(drafted: Tensor, output: Tensor | None) -> Tensor:

    return output if output is not None else drafted.clone()


def conditional_collapsed_stride(data: Tensor | None) -> int | None:

    if data is not None:

        return collapse_2d(data).stride(0)


def fused_multistep(
    # Required.
    logits: Tensor,
    indicators: Tensor,
    drafted_tokens: Tensor,
    # Model Specific(s).
    local_vocab: int | None = None,
    shard_size: int | None = None,
    world_size: int = 1,
    local_rank: int = 0,
    lookahead: int | None = None,
    # Conditional(s).
    enable_pdl: bool = True,
    update_counts: bool = False,
    return_logprobs: bool = False,
    # Kernel Specific(s).
    block_n: int = 4_096,
    # Buffer(s).
    scratchpad: Tensor | None = None,
    values: Tensor | None = None,
    indices: Tensor | None = None,
    # Support(s).
    draft_probabilities: Tensor | None = None,
    # Logit Manipulator(s).
    slot_mapping: Tensor | None = None,
    cu_seqlens: Tensor | None = None,
    grammar: Tensor | None = None,
    context_counts: Tensor | None = None,
    decode_counts: Tensor | None = None,
    repetition_penalties: Tensor | None = None,
    frequency_penalties: Tensor | None = None,
    presence_penalties: Tensor | None = None,
    logit_bias: Tensor | None = None,
    temperature: Tensor | None = None,
    top_k: Tensor | None = None,
    top_p: Tensor | None = None,
    min_p: Tensor | None = None,
    # Selection Parameter(s).
    top_k_logprobs: Tensor | None = None,
    gumbel_noise: Tensor | None = None,
    uniform_noise: Tensor | None = None,
    noise_seeds: Tensor | None = None,
    noise_offsets: Tensor | None = None,
    # Output Buffer(s).
    output_tokens: Tensor | None = None,
    # Top-K Strategy & Tuning Configuration.
    topk_strategy: TopKStrategy | None = None,
    warp_config: ThreeStageWarpConfig | None = None,
) -> Verification:

    # Validation(s).

    validate_count_update(update_counts, decode_counts)
    validate_noise(noise_seeds, noise_offsets, gumbel_noise, uniform_noise)

    # Resolve batch size, lookahead, varlen, and vocab size.

    varlen, batch_size, gamma, vocab_size = resolve_batch_vocab(
        logits=logits,
        cu_seqlens=cu_seqlens,
        lookahead=lookahead,
    )

    vocab_size = min(vocab_size, local_vocab or vocab_size)

    # Resolve sharding offset(s).

    shard_size = shard_size or vocab_size
    shard_offset = local_rank * shard_size

    # Define and resolve conditional(s).

    slotted = slot_mapping is not None
    sharded = world_size > 1

    # Define kernel grid(s) and resolve scratchpad.

    grid_2d, grid_1d, bitpacked, vocab_blocks = resolve_scratchpad(
        logits=logits,
        block_n=block_n,
        buffer=scratchpad,
    )

    grid_batch = (batch_size,)

    # Resolve the values and indices buffer(s).

    values_buffer, indices_buffer = resolve_unpack_buffers(
        logits=logits,
        values=values,
        indices=indices,
    )

    # Define block size(s).

    block_q = resolve_block(cu_seqlens, dim=0)

    block_g = next_power_of_2(gamma + 1)
    block_z = next_power_of_2(MAX_TOP_LP)

    # Resolve gumbel noise weight(s).

    gumbel_weights = resolve_gumbel(gumbel_noise, logits, batch_size, gamma + 1, noise_seeds)
    uniform_weights = resolve_uniform(uniform_noise, logits, batch_size, gamma, noise_seeds)

    # Resolve output buffer(s).

    verification = Verification.from_tokens(
        tokens=resolve_tokens(drafted_tokens, output_tokens),
        batch_size=batch_size,
        timesteps=gamma + 1,
        logprobs=return_logprobs,
        varlen=varlen,
    )

    token_logprobs = top_logprobs = top_tokens = None

    if vl := verification.logprobs:

        token_logprobs = vl.selected

        top_logprobs = collapse_2d((vt := vl.top_k).logprobs)
        top_tokens = vt.tokens

    # Resolve sharding specific(s).

    total_rows = world_size * (block_v := next_power_of_2(vocab_blocks))
    total_cols = total_rows * MAX_K

    # Resolve top-k strategy and tuning configuration(s).

    config = warp_config or ThreeStageWarpConfig.default(gamma=gamma)
    k_strategy = topk_strategy or TopKStrategy()

    # Resolve required tensor stride(s).

    stride_x = collapse_2d(logits).stride(0)
    stride_y = bitpacked.stride(0)

    stride_n = conditional_stride(gumbel_weights)
    stride_u = conditional_stride(uniform_weights)

    stride_c = resolve_target_stride(
        gumbel_weights,
        draft_probabilities,
        decode_counts,
        logit_bias,
        world_size * shard_size,
    )

    # Conditionally resolve optional tensor stride(s).

    stride_d = conditional_stride(draft_probabilities)
    stride_z = conditional_stride(top_logprobs)

    stride_g = conditional_collapsed_stride(grammar)

    # Stage 1: 2D-Tiled Bitonic / Adaptive Bit-Packed Reduction.

    bitpacked_reduction_kernel[grid_2d](
        # Input Data Pointer(s).
        logits,
        indicators,
        slot_mapping,
        cu_seqlens,
        None,
        grammar,
        context_counts,
        decode_counts,
        drafted_tokens,
        repetition_penalties,
        frequency_penalties,
        presence_penalties,
        logit_bias,
        temperature,
        # Output Data Pointer(s).
        bitpacked,
        # Model Specific(s).
        vocab_size,
        shard_offset,
        gamma,
        MAX_K,
        k_strategy.scale,
        batch_size,
        # Conditional(s).
        False,
        True,
        slotted,
        False,
        varlen,
        k_strategy.is_adaptive,
        k_strategy.radix,
        enable_pdl,
        # Stride(s).
        stride_x,
        stride_g,
        stride_c,
        None,
        stride_y,
        # Kernel Specific(s).
        block_n,
        block_q,
        block_g,
        # Tuning Configuration.
        num_warps=config.first,
    )

    # Stage 2: 1D-Tiled Merge Unpack & Probability Truncation.

    cumulative_unpack_kernel[grid_1d](
        # Input Data Pointer(s).
        bitpacked,
        indicators,
        slot_mapping,
        cu_seqlens,
        top_k,
        top_p,
        min_p,
        # Output Data Pointer(s).
        values_buffer,
        indices_buffer,
        # Model Specific(s).
        vocab_blocks,
        world_size,
        shard_size,
        gamma,
        MAX_K,
        k_strategy.inverse,
        batch_size,
        total_rows,
        total_cols,
        # Conditional(s).
        sharded,
        slotted,
        varlen,
        k_strategy.is_adaptive,
        k_strategy.is_alternating,
        enable_pdl,
        # Stride(s).
        stride_y,
        None,
        None,
        # Kernel Specific(s).
        block_n,
        block_v,
        block_q,
        # Tuning Configuration.
        num_warps=config.second,
    )

    # Stage 3: Chain Speculative Verification.

    chain_speculative_verification_kernel[grid_batch](
        # Input Data Pointer(s).
        values_buffer,
        indices_buffer,
        indicators,
        slot_mapping,
        cu_seqlens,
        drafted_tokens,
        draft_probabilities,
        uniform_weights,
        gumbel_weights,
        noise_seeds,
        noise_offsets,
        decode_counts,
        top_k_logprobs,
        # Output Data Pointer(s).
        verification.tokens,
        verification.offsets,
        token_logprobs,
        top_logprobs,
        top_tokens,
        # Model Specific(s).
        gamma,
        MAX_K,
        # Conditional(s).
        slotted,
        varlen,
        enable_pdl,
        update_counts,
        return_logprobs,
        # Stride(s).
        stride_d,
        stride_n,
        stride_u,
        stride_c,
        stride_z,
        # Kernel Specific(s).
        block_g,
        block_z,
        # Tuning Configuration.
        num_warps=config.third,
    )

    return verification
