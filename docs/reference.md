# Reference

Lookup tables for the public surface. See [`concepts.md`](concepts.md) for what the pieces mean.

## Imports

```python
from sonic_sampler.base import Selection, Verification, LogProbabilities, MAX_TOP_LP
from sonic_sampler.core import SamplingBuffers, Indicator, NaiveSampler
from sonic_sampler.core.flags import ScopedIndicators
from sonic_sampler.interface import (
    UnshardedFusedSingular,
    UnshardedFusedMultistep,
    UnshardedBoundedTopK,
    TopKStrategy,
    TwoStageWarpConfig,
    ThreeStageWarpConfig,
    fused_singular,
    fused_multistep,
)
from sonic_sampler.interface.base import TwoStageTiling
from sonic_sampler.ops import MAX_K
```

`UnshardedBoundedTopK` exposes the bounded top-$k$ primitive on its own, for engines that want the
selection stage without the sampling pipeline around it.

## Constants

| Constant | Value | Meaning |
| :--- | :--- | :--- |
| `MAX_K` | $128$ | Upper bound on top-$k$, and the width of the selection block |
| `MAX_TOP_LP` | $5$ | Width of the emitted top-$k$ logprob block |

## Buffers

### `SamplingBuffers`

Constructed once per process, sized to $B$ rows and $\gamma + 1$ timesteps.

| Member | Shape | Dtype | Written by |
| :--- | :--- | :--- | :--- |
| `flags.target` | `[ B ]` | `uint16` | `allocate`, `update` |
| `flags.draft` | `[ B ]` | `uint16` | `allocate`, `update` |
| `flags.packed` | `[ B, 2 ]` | `uint16` | `allocate`, `update` |
| `temperature` | `[ B ]` | `bfloat16` | `allocate`, `update` |
| `top_k` | `[ B ]` | `int32` | `allocate`, `update` |
| `top_p` | `[ B ]` | `bfloat16` | `allocate`, `update` |
| `min_p` | `[ B ]` | `bfloat16` | `allocate`, `update` |
| `top_logprobs` | `[ B ]` | `int32` | `allocate`, `update` |
| `bias` | `[ B, V ]` | `bfloat16` | `allocate`, `update` |
| `repetition.multiplicative` | `[ B ]` | `bfloat16` | `allocate`, `update` |
| `repetition.frequency` | `[ B ]` | `bfloat16` | `allocate`, `update` |
| `repetition.presence` | `[ B ]` | `bfloat16` | `allocate`, `update` |
| `repetition.counts.context` | `[ B, V ]` | `int32` | `allocate`, `update` |
| `repetition.counts.decode` | `[ B, V ]` | `int32` | `update`, and the kernels under `update_counts` |
| `grammar` | `[ B, γ + 1, ⌈V / 32⌉ ]` | `uint32` | `step_grammar`, `allocate` |
| `weights.gumbel.target` | `[ B, γ + 1, V ]` | `bfloat16` | `allocate`, `update`, `step_weights` |
| `weights.gumbel.draft` | `[ B, γ + 1, V ]` | `bfloat16` | `allocate`, `update`, `step_weights` |
| `weights.uniform` | `[ B, γ + 1 ]` | `bfloat16` | `allocate`, `update`, `step_weights` |

Memory, dominated by the noise buffers:

$$
M \approx 2 \cdot B \cdot (\gamma + 1) \cdot V \cdot 2\ \text{bytes}
\;+\; 3 \cdot B \cdot V \cdot 4\ \text{bytes}
$$

### In-flight buffers

Owned by the `TwoStageTiling` subclasses, sized to $B \cdot (\gamma + 1)$ rows.

| Buffer | Shape | Dtype | Present on |
| :--- | :--- | :--- | :--- |
| `scratchpad` | `[ B · (γ + 1), blocks · MAX_K ]` | `uint32` | both interfaces |
| `values` | `[ B · (γ + 1), MAX_K ]` | `float32` | the verifier only |
| `indices` | `[ B · (γ + 1), MAX_K ]` | `int32` | the verifier only |

### Slicing

| Expression | Result |
| :--- | :--- |
| `buffers[:R]` | Prefix view over rows, cached per instance |
| `buffers[:R, :T]` | Prefix view over rows and timesteps |
| `buffers.bisect(pivot)` | The pair `(buffers[:pivot], buffers[pivot:])` |
| `buffers.prepopulate(schedule)` | Warms the slice cache ahead of graph capture |

`schedule` accepts a list of row sizes, a list of `(rows, cols)` pairs, or a `(rows, cols)` pair of
lists denoting a cross product.

## `SamplingBuffers` methods

### `default`

```python
@classmethod
def default(
    cls,
    size: int,
    timesteps: int,
    vocab_size: int,
    device: torch.device | None = None,
    bounded: bool = True,
) -> SamplingBuffers
```

Allocates a neutral instance. `bounded=True` initializes `top_k` to `MAX_K` rather than $V$.

### `random`

```python
@classmethod
def random(
    cls,
    size: int,
    timesteps: int,
    vocab_size: int,
    indicators: Indicator | List[Indicator] | None = None,
    device: torch.device | None = None,
    seed: int | None = None,
    bounded: bool = True,
) -> SamplingBuffers
```

Allocates a randomly populated instance for testing.

### `allocate`

Admits requests at explicit `positions`. All arguments are host-resident.

| Argument | Type | Required |
| :--- | :--- | :--- |
| `encodings` | `List[Tensor]`, int64 prompt ids | yes |
| `multiplicative`, `frequency`, `presence` | `List[float]` | yes |
| `biases` | `List[dict[int, float] \| None]` | yes |
| `temperature`, `top_p`, `min_p` | `List[float]` | yes |
| `top_k`, `top_logprobs` | `List[int]` | yes |
| `positions` | `List[int]` | yes |
| `indicators` | `ScopedIndicators \| None` | no |
| `greedy_drafts` | `bool \| List[bool]` | no |
| `bitmasks` | `List[Tensor \| None] \| None` | no |
| `stream` | `Stream \| None` | no |
| `generator` | `Generator \| List[Generator] \| None` | no |

Returns a recorded `Event` when `stream` is given, otherwise `None`. Implies `prefill=True`.

### `update`

Populates the batch sequentially, or at `positions` when given. Returns an `Event` on the same
terms as `allocate`.

| Argument | Type | Required |
| :--- | :--- | :--- |
| `counts` | `List[Tuple[Tensor, Tensor]]`, `(context, decode)` histograms | yes |
| `target_gumbel` | `List[Tensor \| None]`, `[ γ + 1, V ]` per entry | yes |
| `multiplicative`, `frequency`, `presence` | `List[float]` | yes |
| `biases` | `List[dict[int, float] \| None]` | yes |
| `temperature`, `top_p`, `min_p` | `List[float]` | yes |
| `top_k`, `top_logprobs` | `List[int]` | yes |
| `positions` | `List[int] \| None` | no |
| `indicators` | `List[Tuple[Indicator, Indicator]] \| None` | no |
| `greedy_drafts` | `bool \| List[bool]` | no |
| `prefill` | `bool \| List[bool]` | no |
| `bitmasks` | `List[Tensor \| None] \| None` | no |
| `stream` | `Stream \| None` | no |
| `generator` | `Generator \| List[Generator] \| None` | no |

### `step_weights`

```python
def step_weights(
    self,
    positions: List[int],
    indices: Tensor | None = None,
    generator: Generator | List[Generator] | None = None,
    stream: Stream | None = None,
    event: Event | None = None,
) -> Self
```

Advances the stochastic weights after a forward pass. No-op when every position has a greedy
target. Stages draft into target when `indices`, the device tensor of `positions`, is supplied and
at least one position drafts stochastically.

### `step_grammar`

```python
def step_grammar(
    self,
    bitmasks: List[Tensor | None] | None,
    positions: List[int] | None = None,
    timestep: int | None = None,
    stream: Stream | None = None,
    event: Event | None = None,
) -> Self
```

Copies bitmasks into `grammar`. Entries may be `None`, and the call is a no-op when all are.
Omitting `timestep` writes across all timesteps of the row; omitting `positions` consumes the list
in batch order.

### `transfer`

```python
def transfer(
    self,
    positions: List[int] | None,
    stream: Stream | None = None,
    event: Event | None = None,
) -> Self
```

Re-dispatches the pinned to device copy of the non-repetition parameters. Called internally by
`allocate` and `update`.

### `subset`

```python
def subset(self, host_indices: List[int], device_indices: Tensor) -> SamplingBuffers
```

Returns a gathered `SamplingBuffers` over arbitrary indices, as opposed to a prefix slice.

## Modular interfaces

### `initialize`

```python
@classmethod
def initialize(
    cls,
    vocab_size: int,
    batch_size: int = 32,
    lookahead: int = 0,
    pdl: bool = True,
    block_size: int | None = None,
    arch: int | None = None,
) -> UnshardedFusedSingular
```

```python
@classmethod
def initialize(
    cls,
    vocab_size: int,
    lookahead: int,
    batch_size: int = 32,
    pdl: bool = True,
    block_size: int | None = None,
    arch: int | None = None,
) -> UnshardedFusedMultistep
```

`lookahead` is required for the verifier and defaults to $0$ for the sampler. `block_size` pins the
vocabulary tile width and bypasses the tuned lookup, and `arch` targets a device other than the
current one.

### `UnshardedFusedSingular.autoregressive`

| Argument | Type | Default |
| :--- | :--- | :--- |
| `logits` | `[ R, V ]`, bfloat16 | required |
| `buffers` | `SamplingBuffers` | required |
| `prefill` | `bool` | `False` |
| `update_counts` | `bool` | `False` |
| `logprobs` | `bool` | `False` |
| `slot_mapping` | `Tensor \| None` | `None` |
| `strategy` | `TopKStrategy \| None` | `None` |
| `tuning` | `TwoStageWarpConfig \| None` | `None` |

Reads `flags.target`, `grammar[:, 0]`, and `gumbel.target[:, 0]`.

### `UnshardedFusedSingular.drafting`

| Argument | Type | Default |
| :--- | :--- | :--- |
| `logits` | `[ R, V ]`, bfloat16 | required |
| `buffers` | `SamplingBuffers` | required |
| `timestep` | `int` | `0` |
| `tokens` | `[ R, γ + 1 ]` int32, or `None` | `None` |
| `probabilities` | `[ R, V_t ]` output buffer, or `None` | `None` |
| `slot_mapping` | `Tensor \| None` | `None` |
| `d2t_mapping` | `[ V_d ]` int32, or `None` | `None` |
| `target_vocab` | `int \| None` | `None` |
| `in_place` | `bool` | `False` |
| `strategy` | `TopKStrategy \| None` | `None` |
| `tuning` | `TwoStageWarpConfig \| None` | `None` |

Reads `flags.draft`, `grammar[:, timestep]`, and `gumbel.draft[:, timestep]`. With `in_place=True`
the selection is written to `tokens[:, timestep]`; from `timestep > 0` the same buffer is passed
through as the drafted-token history for the repetition penalties. There is no `logprobs` argument,
since draft distributions are surfaced through `probabilities` instead.

### `UnshardedFusedMultistep.__call__`

| Argument | Type | Default |
| :--- | :--- | :--- |
| `logits` | `[ R, γ + 1, V ]`, bfloat16 | required |
| `tokens` | `[ R, γ + 1 ]`, int32 | required |
| `buffers` | `SamplingBuffers` | required |
| `probabilities` | `[ R, γ, V ]`, or `None` | `None` |
| `slot_mapping` | `Tensor \| None` | `None` |
| `update_counts` | `bool` | `False` |
| `logprobs` | `bool` | `False` |
| `in_place` | `bool` | `False` |
| `strategy` | `TopKStrategy \| None` | `None` |
| `tuning` | `ThreeStageWarpConfig \| None` | `None` |

Reads `flags.target`, the full `grammar`, `gumbel.target`, and `weights.uniform`. With
`in_place=True`, `verification.tokens` is `tokens` itself rather than a clone.

## Outputs

### `Selection`

| Field | Shape | Dtype |
| :--- | :--- | :--- |
| `tokens` | `[ B, 1 ]` | `int32` |
| `probabilities` | `[ B, V_t ]` | `bfloat16` |
| `logprobs.selected` | `[ B, 1 ]` | `bfloat16` |
| `logprobs.top_k.tokens` | `[ B, MAX_TOP_LP ]` | `int32` |
| `logprobs.top_k.logprobs` | `[ B, MAX_TOP_LP ]` | `bfloat16` |

### `Verification`

| Field | Shape | Dtype |
| :--- | :--- | :--- |
| `tokens` | `[ B, γ + 1 ]` | `int32` |
| `offsets` | `[ B, 1 ]` | `int32` |
| `logprobs.selected` | `[ B, γ + 1 ]` | `bfloat16` |
| `logprobs.top_k.tokens` | `[ B, γ + 1, MAX_TOP_LP ]` | `int32` |
| `logprobs.top_k.logprobs` | `[ B, γ + 1, MAX_TOP_LP ]` | `bfloat16` |

`logprobs` is `None` unless the call requested it, and `probabilities` is `None` unless the draft
path produced it.

## Indicators

| Bit | Name | Activates |
| :--- | :--- | :--- |
| `0x001` | `GRAMMAR` | Bitmask application |
| `0x002` | `MULTIPLICATIVE` | Multiplicative repetition penalty |
| `0x004` | `FREQUENCY` | Frequency penalty |
| `0x008` | `PRESENCE` | Presence penalty |
| `0x010` | `BIAS` | Logit bias |
| `0x020` | `TEMPERATURE` | Temperature scaling |
| `0x040` | `GREEDY` | Argmax selection |
| `0x080` | `TOP_K` | Top-$k$ truncation |
| `0x100` | `TOP_P` | Nucleus truncation |
| `0x200` | `MIN_P` | Min-$p$ truncation |
| `0x400` | `TOP_K_LOGPROBS` | Top-$k$ logprob emission |
| `0x800` | `STOCHASTIC_VERIFY_GREEDY_DRAFT` | Masked residual verification |

Helpers: `Indicator.stochastic()` is the temperature, top-$k$, top-$p$, and min-$p$ group;
`Indicator.repetition()` is the three penalty bits; `Indicator.maximal()` is every bit except
`GREEDY`; `indicator.greedy()` clears the stochastic group and sets `GREEDY`.

`ScopedIndicators.from_params(size, timesteps, ..., greedy_drafts)` derives the `(target, draft)`
pair from the sampling parameters. Useful predicates on the result:

| Predicate | Meaning |
| :--- | :--- |
| `has_stochastic_target` | Any row verifies stochastically |
| `has_stochastic_draft` | Any row drafts stochastically |
| `stochastic_selectors` | Per-row `(target, draft)` booleans driving the weight refresh |

## The lower rungs

### `TwoStageTiling`

Tuned table lookup, returning the dispatch summary and the vocabulary tiling:

```python
@classmethod
def resolve(
    cls,
    vocab_size: int,
    k: int = MAX_K,
    block_size: int | None = None,
    arch: int | None = None,
    ensure_multiblock: bool = True,
) -> Tuple[VocabBucket | None, Vocabulary, int]
```

Resolution plus allocation of the in-flight buffers:

```python
@classmethod
def factory(
    cls,
    vocab_size: int,
    batch_size: int = 32,
    lookahead: int = 0,
    k: int = MAX_K,
    block_size: int | None = None,
    arch: int | None = None,
    enable_pdl: bool = True,
    ensure_multiblock: bool = True,
    unpacked_buffers: bool = True,
    values_dtype: torch.dtype = FP32,
) -> Tuple[TwoStageTiling, Tensor | None, Tensor | None]
```

Per-call configuration resolved against the current row count:

```python
def tuning(
    self,
    size: int,
    strategy: TopKStrategy | None = None,
    tuning: TwoStageWarpConfig | None = None,
) -> Tuple[TwoStageWarpConfig, TopKStrategy, int]
```

`resolve` raises `ValueError` when $k$ exceeds `MAX_K`, when $k$ is not a power of two, when no
tuned table exists for the architecture, when no `VocabBucket` matches $V$, or when
`ensure_multiblock` is set and $V$ fits in a single tile.

### `fused_singular` and `fused_multistep`

The functional entry points take every buffer explicitly. Their arguments group as:

| Group | Arguments |
| :--- | :--- |
| Inputs | `logits`, `indicators`, `drafted_tokens` (multistep) |
| Modes | `enable_pdl`, `is_prefill`, `update_counts`, `return_logprobs`, `return_probabilities`, `lookahead` |
| Tiling | `block_n`, `scratchpad`, `values`, `indices`, `topk_strategy`, `warp_config` |
| Manipulators | `slot_mapping`, `d2t_mapping`, `grammar`, `context_counts`, `decode_counts`, `repetition_penalties`, `frequency_penalties`, `presence_penalties`, `logit_bias`, `temperature`, `top_k`, `top_p`, `min_p`, `top_k_logprobs` |
| Weights | `gumbel_noise`, `uniform_noise`, `noise_seeds`, `noise_offsets` |
| Outputs | `output_tokens`, `draft_probabilities` |

Any buffer left `None` is allocated inside the call, which is convenient for one-off use and
incompatible with graph capture. Every manipulator is independently optional: omitting one leaves
the corresponding stage inactive regardless of the indicator bits.

With `noise_seeds` and `noise_offsets`, per-slot integer tensors, the noise weights must be `None`
and the kernels draw the noise instead, see [In-kernel noise](concepts.md#in-kernel-noise).
