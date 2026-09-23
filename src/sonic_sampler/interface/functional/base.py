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

from sonic_sampler.ops.base import next_power_of_2, MAX_K


U32: torch.dtype = torch.uint32
I32: torch.dtype = torch.int32


def validate_count_update(flag: bool, buffer: Tensor | None) -> None:

    if flag and buffer is None:

        raise ValueError("`update_counts` is True but `decode_counts` is None")


def validate_noise(
    seeds: Tensor | None,
    offsets: Tensor | None,
    *weights: Tensor | None,
) -> None:

    if (seeds is None) != (offsets is None):

        raise ValueError("`noise_seeds` and `noise_offsets` must be given together")

    if seeds is not None and any(weight is not None for weight in weights):

        raise ValueError("`noise_seeds` replaces the noise weights, which must be None")


def resolve_scratchpad(
    logits: Tensor,
    block_n: int,
    buffer: Tensor | None = None,
) -> Tuple[Tuple[int, int], Tuple[int], Tensor, int]:

    rows, cols = logits.shape if logits.ndim == 2 else logits.flatten(0, 1).shape

    grid_2d = (rows, vocab_blocks := cdiv(cols, block_n))
    grid_1d = (rows,)

    if (scratchpad := buffer) is None:

        scratchpad = logits.new_empty(rows, vocab_blocks * MAX_K, dtype=U32)

    return grid_2d, grid_1d, scratchpad, vocab_blocks


def conditional_stride(*data: Tensor | None) -> int | None:

    for tensor in data:

        if tensor is not None:

            return tensor.stride(0)


def resolve_block(data: Tensor | None, dim: int) -> int | None:

    if data is not None:

        return next_power_of_2(data.size(dim))


def collapse_2d(value: Tensor) -> Tensor:

    return value.flatten(0, -2)
