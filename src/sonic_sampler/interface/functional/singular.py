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

from torch import Tensor

from sonic_sampler.base.sampler import Selection, MAX_TOP_LP
from sonic_sampler.ops.base import next_power_of_2, MAX_K
from sonic_sampler.interface.base import TopKStrategy
from sonic_sampler.interface.dispatch import TwoStageWarpConfig
from sonic_sampler.interface.functional.base import (
    validate_count_update,
    validate_noise,
    resolve_scratchpad,
    conditional_stride,
    resolve_block,
    I32,
)
from sonic_sampler.ops.prologue import bitpacked_reduction_kernel
from sonic_sampler.ops.singular import cumulative_selection_kernel


def resolve_gumbel(noise: Tensor | None, logits: Tensor, seeds: Tensor | None) -> Tensor | None:

    if (weights := noise) is None and seeds is None:

        weights = (
            torch.empty_like(logits)
            .exponential_(1.0)
            .log_()
        )

    return weights


def resolve_tokens(buffer: Tensor | None, batch_size: int, source: Tensor) -> Tensor:

    if (tokens := buffer) is None:

        tokens = source.new_empty(batch_size, 1, dtype=I32)

    return tokens


def resolve_probabilities(
    precondition: bool,
    source: Tensor,
    batch_size: int,
    vocab_size: int | None = None,
    buffer: Tensor | None = None,
) -> Tensor | None:

    if (probabilities := buffer) is None and precondition:

        probabilities = source.new_zeros((batch_size, vocab_size or source.size(1)))

    return probabilities


def fused_singular(
    # Required.
    logits: Tensor,
    indicators: Tensor,
    # Model Specific(s).
    local_vocab: int | None = None,
    shard_size: int | None = None,
    world_size: int = 1,
    local_rank: int = 0,
    lookahead: int = 0,
    target_vocab: int | None = None,
    # Conditional(s).
    enable_pdl: bool = True,
    is_prefill: bool = False,
    update_counts: bool = False,
    return_logprobs: bool = False,
    return_probabilities: bool = False,
    # Kernel Specific(s).
    block_n: int = 4_096,
    # Buffer(s).
    scratchpad: Tensor | None = None,
    # Logit Manipulator(s).
    slot_mapping: Tensor | None = None,
    d2t_mapping: Tensor | None = None,
    grammar: Tensor | None = None,
    context_counts: Tensor | None = None,
    decode_counts: Tensor | None = None,
    drafted_tokens: Tensor | None = None,
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
    noise_seeds: Tensor | None = None,
    noise_offsets: Tensor | None = None,
    # Output Buffer(s).
    output_tokens: Tensor | None = None,
    draft_probabilities: Tensor | None = None,
    # Top-K Strategy & Tuning Configuration.
    topk_strategy: TopKStrategy | None = None,
    warp_config: TwoStageWarpConfig | None = None,
) -> Selection:

    # Validation(s).

    validate_count_update(update_counts, decode_counts)
    validate_noise(noise_seeds, noise_offsets, gumbel_noise)

    # Extract and resolve batch size, local vocab size(s), and shard offset(s).

    batch_size, vocab_size = logits.shape
    vocab_size = min(vocab_size, local_vocab or vocab_size)

    shard_size = shard_size or vocab_size
    shard_offset = local_rank * shard_size

    # Define and resolve conditional(s).

    indirection = d2t_mapping is not None
    sharded = world_size > 1

    tokens = drafted_tokens is not None
    slotted = slot_mapping is not None

    # Define kernel grid(s) and resolve scratchpad.

    grid_2d, grid_1d, bitpacked, vocab_blocks = resolve_scratchpad(
        logits=logits,
        block_n=block_n,
        buffer=scratchpad,
    )

    # Define block size(s).

    block_t = resolve_block(drafted_tokens, dim=1)
    block_z = next_power_of_2(MAX_TOP_LP)

    # Resolve gumbel noise weight(s).

    gumbel_weights = resolve_gumbel(gumbel_noise, logits, noise_seeds)

    # Resolve output buffer(s).

    final_tokens = resolve_tokens(buffer=output_tokens, batch_size=batch_size, source=logits)

    probabilities = resolve_probabilities(
        precondition=return_probabilities,
        source=logits,
        batch_size=indicators.size(0),
        vocab_size=target_vocab,
        buffer=draft_probabilities,
    )

    selection = Selection.from_buffers(final_tokens, probabilities, return_logprobs)
    token_logprobs = top_logprobs = top_tokens = None

    if sl := selection.logprobs:

        token_logprobs = sl.selected

        top_logprobs = (st := sl.top_k).logprobs
        top_tokens = st.tokens

    # Resolve sharding specific(s).

    total_rows = world_size * (block_v := next_power_of_2(vocab_blocks))
    total_cols = total_rows * MAX_K

    # Resolve top-k strategy and tuning configuration(s).

    config = warp_config or TwoStageWarpConfig.default()
    k_strategy = topk_strategy or TopKStrategy()

    # Resolve tensor stride(s).

    stride_x = logits.stride(0)
    stride_y = bitpacked.stride(0)

    stride_n = conditional_stride(gumbel_weights)
    stride_f = final_tokens.stride(0)

    stride_g = conditional_stride(grammar)
    stride_c = conditional_stride(decode_counts, logit_bias)

    stride_t = conditional_stride(drafted_tokens)
    stride_p = conditional_stride(token_logprobs)

    stride_d = conditional_stride(probabilities)
    stride_z = conditional_stride(top_logprobs)

    # Stage 1: 2D-Tiled Bitonic / Adaptive Bit-Packed Reduction.

    bitpacked_reduction_kernel[grid_2d](
        # Input Data Pointer(s).
        logits,
        indicators,
        slot_mapping,
        None,
        d2t_mapping,
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
        lookahead,
        MAX_K,
        k_strategy.scale,
        batch_size,
        # Conditional(s).
        indirection,
        tokens,
        slotted,
        True,
        False,
        k_strategy.is_adaptive,
        k_strategy.radix,
        enable_pdl,
        # Stride(s).
        stride_x,
        stride_g,
        stride_c,
        stride_t,
        stride_y,
        # Kernel Specific(s).
        block_n,
        None,
        block_t,
        # Tuning Configuration.
        num_warps=config.first,
    )

    # Stage 2: 1D-Tiled Merge Unpack Token Selection.

    cumulative_selection_kernel[grid_1d](
        # Input Data Pointer(s).
        bitpacked,
        indicators,
        slot_mapping,
        d2t_mapping,
        decode_counts,
        gumbel_weights,
        noise_seeds,
        noise_offsets,
        top_k,
        top_p,
        min_p,
        top_k_logprobs,
        # Output Data Pointer(s).
        final_tokens,
        token_logprobs,
        probabilities,
        top_logprobs,
        top_tokens,
        # Model Specific(s).
        vocab_blocks,
        world_size,
        shard_size,
        MAX_K,
        k_strategy.inverse,
        total_rows,
        total_cols,
        # Conditional(s).
        indirection,
        slotted,
        sharded,
        is_prefill,
        update_counts,
        k_strategy.is_adaptive,
        k_strategy.is_alternating,
        enable_pdl,
        return_logprobs,
        return_probabilities,
        # Stride(s).
        stride_y,
        None,
        None,
        stride_f,
        stride_p,
        stride_d,
        stride_c,
        stride_n,
        stride_z,
        # Kernel Specific(s).
        block_n,
        block_v,
        block_z,
        # Tuning Configuration.
        num_warps=config.second,
    )

    return selection
