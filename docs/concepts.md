# Concepts

The vocabulary, buffer model, and contracts shared by every SonicSampler integration. Read this
once, then follow [`quickstart.md`](quickstart.md) into the integration doc matching your engine's
batching strategy.

## Notation

| Symbol | Meaning                                                                             |
| :--- |:------------------------------------------------------------------------------------|
| $B$ | Buffer capacity, i.e. the maximum concurrent requests the engine admits             |
| $R$ | Active rows in the current forward pass, $R \le B$                                  |
| $V$ | Vocabulary size of the target model                                                 |
| $\gamma$ | Speculation lookahead, i.e. the number of tokens the drafter proposes per iteration |
| $\gamma + 1$ | Timesteps per request, the $\gamma$ drafted positions plus the bonus position       |
| $k$ | Top-$`k`$ truncation bound, capped at $\text{MAX\\_K} = 128$                          |

Non-speculative engines set $\gamma = 0$, which collapses every timestep dimension to $1$.

Buffers are tagged by residency:

| Tag | Meaning |
| :--- | :--- |
| `device` | CUDA-resident, read by the kernels |
| `pinned` | Host-resident page-locked staging for non-blocking H2D transfers |
| `host` | Plain CPU-resident Python or tensor state |

## The two buffer families

An integration owns two disjoint families of buffers, distinguished by lifetime and by what a row
means.

### Lifetime buffers

`SamplingBuffers` holds everything that describes *a request*: its sampling parameters, its token
histograms, its grammar bitmasks, and its stochastic weights. One instance is created during engine
initialization and persists for the process lifetime.

```python
from sonic_sampler.core.buffer import SamplingBuffers

buffers = SamplingBuffers.default(
    size=MAX_BATCH_SIZE,        # B
    timesteps=lookahead + 1,    # γ + 1
    vocab_size=model.vocab_size,
    device=device,
)
```

The instance is a nested structure. Each member is sized to $B$ along its leading dimension:

| Member | Shape | Dtype | Residency |
| :--- | :--- | :--- | :--- |
| `flags.packed` | `[ B, 2 ]` | `uint16` | device |
| `flags.pinned` | `[ B, 2 ]` | `uint16` | pinned |
| `flags.indicators` | 2 × `[ B ]` | `Indicator` | host |
| `temperature` | `[ B ]` | `bfloat16` | device |
| `top_p`, `min_p` | `[ B ]` | `bfloat16` | device |
| `top_k`, `top_logprobs` | `[ B ]` | `int32` | device |
| `bias` | `[ B, V ]` | `bfloat16` | device |
| `repetition.multiplicative` | `[ B ]` | `bfloat16` | device |
| `repetition.frequency`, `repetition.presence` | `[ B ]` | `bfloat16` | device |
| `repetition.counts.context` | `[ B, V ]` | `int32` | device |
| `repetition.counts.decode` | `[ B, V ]` | `int32` | device |
| `grammar` | `[ B, γ + 1, ⌈V / 32⌉ ]` | `uint32` | device |
| `weights.gumbel.target` | `[ B, γ + 1, V ]` | `bfloat16` | device |
| `weights.gumbel.draft` | `[ B, γ + 1, V ]` | `bfloat16` | device |
| `weights.uniform` | `[ B, γ + 1 ]` | `bfloat16` | device |

Every device-resident parameter is shadowed by pinned staging, held under `pinned` and
`repetition.pinned`. Population therefore follows a fixed path: pack the Python values into a host
tensor, copy into pinned memory, then dispatch a non-blocking H2D transfer. The public
`allocate` and `update` methods encapsulate that path end to end, optionally under a dedicated copy
stream, and return a recorded `torch.Event` when one is given.

The dominant term in the memory budget is the pair of Gumbel-Max buffers:

$$
M_{\text{noise}} = 2 \cdot B \cdot (\gamma + 1) \cdot V \cdot 2\ \text{bytes}
$$

For $B = 32$, $\gamma = 3$, and $V = 128{,}256$ that is $\approx 131\ \text{MiB}$. At $\gamma = 0$
it collapses to $\approx 33\ \text{MiB}$.

### In-flight buffers

The kernels also need DRAM-staged intermediates: a bit-packed reduction `scratchpad`, and for
verification the unpacked `values` and `indices`. These describe *a row of logits*, not a request,
and they are sized to $B \cdot (\gamma + 1)$ rows:

| Buffer | Shape | Dtype |
| :--- | :--- | :--- |
| `scratchpad` | `[ B · (γ + 1), blocks · MAX_K ]` | `uint32` |
| `values` | `[ B · (γ + 1), MAX_K ]` | `float32` |
| `indices` | `[ B · (γ + 1), MAX_K ]` | `int32` |

where `blocks` is the number of vocabulary tiles, $\lceil V / \texttt{block\\_n} \rceil$.

The modular interfaces (`UnshardedFusedSingular`, `UnshardedFusedMultistep`) allocate and own these
at `initialize(...)` time, so an integration built on them never handles them directly. The
functional entry points accept them as arguments and fall back to allocating per call when they are
left `None`.

## Addressing a row

Kernels consume logits row by row. What differs between engines is how a row finds its request
state, and that follows from the batching strategy:

- **Slot-based pre-allocation.** Request state lives at a stable slot for the lifetime of the
  request. The kernels receive the full `SamplingBuffers` together with a `slot_mapping` tensor of
  length $R$, so row $i$ reads its parameters at `slot_mapping[i]`. A mapping of `[ 2, 5, 1, 7 ]`
  routes the four active rows to slots $2$, $5$, $1$, and $7$ respectively.

- **Just-in-time batching.** The batch is composed immediately before each forward pass and the
  buffers are populated in the same order, so row $i$ reads its parameters at position $i$. No
  `slot_mapping` is involved. Only the active prefix participates, obtained with `buffers[:R]`.

Prefix slicing is a first-class operation. `buffers[:R]` returns a `SamplingBuffers` whose members
are prefix views of the originals, and `buffers[:R, :T]` narrows the timestep dimension as well.
Slices are cached per instance and reused, so repeated slicing at the same size yields the same
views with the same pointers.

## Stochastic weights

Sampling is Gumbel-Max against pre-drawn noise rather than a draw from the truncated distribution
inside the kernel. The buffers hold $\log \varepsilon$ with $\varepsilon \sim \text{Exp}(1)$, and
the kernel owns the subtraction form:

$$
x^{*} = \arg\max_{v} \bigl( \log p_{v} - \log \varepsilon_{v} \bigr),
\qquad -\log \varepsilon_{v} \sim \text{Gumbel}(0, 1)
$$

Refreshing the noise is the responsibility of `SamplingBuffers`, not the caller. Three buffers
participate:

- `weights.gumbel.draft` supplies the drafter across all $\gamma + 1$ timesteps.
- `weights.gumbel.target` supplies the verifier.
- `weights.uniform` supplies the acceptance test, one $u \sim \mathcal{U}(0, 1]$ per timestep.

Refresh is conditional on the indicators. Weights are only touched when at least one row in the
batch has a stochastic target, and within that batch only the rows that need them:

| Phase | `target` | `draft` |
| :--- | :--- | :--- |
| Prefill | timestep $0$ refreshed | all timesteps refreshed when the draft is stochastic |
| Decode | refreshed, or staged from `draft` | refreshed when the draft is stochastic |

The staging case is what makes verification on-policy. When a request drafts stochastically, the
noise the drafter used must be the noise the verifier sees, so the draft weights are transferred
into the target weights before the next verification. Slot-based integrations get this from
`step_weights(...)`; just-in-time integrations feed the weights back through the `target_gumbel`
argument of `update(...)`. Under stochastic verification with greedy drafting there is nothing to
carry over, and the target weights are simply refreshed.

### In-kernel noise

Given `noise_seeds` and `noise_offsets` instead of the weights, the functional entry points draw the
noise inside the kernels, only at the candidates they read. Each row keys a Philox stream by its
seed and the counter `offset * 256 + timestep`, so an offset that grows every step, such as the
sequence length, gives fresh noise for any lookahead below 256. The noise equals a `float32` weight
buffer written with Philox over the same key, four tokens per round. A stochastic drafter's noise is
not carried into the verifier, so draft-coupled verification still requires the weights.

## Scoped indicators

Per-request behaviour is encoded as an `Indicator`, a bitfield whose bits activate the logit
processors and the selection path. The kernels read the packed tensor form, so an integration
mostly deals with the resolution step rather than the bits themselves.

Indicators are *scoped*: `ScopedIndicators` carries a `target` list for the verifier and a `draft`
list for the speculator, both of length $R$. `flags.target` and `flags.packed[:, 0]` are the same
tensor, as are `flags.draft` and `flags.packed[:, 1]`.

Both are derived from the sampling parameters:

```python
from sonic_sampler.core.flags import ScopedIndicators

scoped = ScopedIndicators.from_params(
    size=len(requests),
    timesteps=lookahead + 1,
    multiplicative=[...],
    frequency=[...],
    presence=[...],
    biases=[...],
    temperature=[...],
    top_k=[...],
    top_p=[...],
    min_p=[...],
    top_logprobs=[...],
    bitmasks=[...],
    greedy_drafts=[...],
)
```

The duality between the two scopes is fixed:

- A greedy target implies a greedy draft. There is nothing to verify stochastically.
- A stochastic target admits either draft mode. `greedy_drafts` selects per request, and a request
  marked greedy receives `indicator.greedy()` as its draft indicator: the stochastic bits are
  cleared and `GREEDY` is set, while grammar, penalty, and bias bits survive.

Resolution is already proxied by `SamplingBuffers.allocate(...)` and `SamplingBuffers.update(...)`,
which construct the scoped indicators from the parameters they are given. Passing `indicators`
explicitly is the specialization path: it lets an engine that already tracks per-request indicators
skip the derivation, and it is the only way to supply indicator combinations the parameters alone
cannot express.

## Outputs

Single-step sampling returns a `Selection`:

| Field | Shape | Dtype | Present when |
| :--- | :--- | :--- | :--- |
| `tokens` | `[ B, 1 ]` | `int32` | always |
| `probabilities` | `[ B, V_t ]` | `bfloat16` | drafting with a stochastic draft |
| `logprobs.selected` | `[ B, 1 ]` | `bfloat16` | `logprobs=True` |
| `logprobs.top_k.tokens` | `[ B, 5 ]` | `int32` | `logprobs=True` |
| `logprobs.top_k.logprobs` | `[ B, 5 ]` | `bfloat16` | `logprobs=True` |

Here $V_t$ is the target vocabulary, which coincides with $V$ except under cross-vocab drafting.

Chain verification returns a `Verification`:

| Field | Shape | Dtype | Present when |
| :--- | :--- | :--- | :--- |
| `tokens` | `[ B, γ + 1 ]` | `int32` | always |
| `offsets` | `[ B, 1 ]` | `int32` | always |
| `logprobs.selected` | `[ B, γ + 1 ]` | `bfloat16` | `logprobs=True` |
| `logprobs.top_k.tokens` | `[ B, γ + 1, 5 ]` | `int32` | `logprobs=True` |
| `logprobs.top_k.logprobs` | `[ B, γ + 1, 5 ]` | `bfloat16` | `logprobs=True` |

`offsets` is the count of accepted draft tokens. Row $i$ therefore emits
$\texttt{offsets}[i] + 1$ tokens, the accepted prefix plus the corrected or bonus token:

```python
span = verification.offsets[i] + 1
emitted = verification.tokens[i, :span]
```

### In-place outputs

Both modular interfaces can write into a caller-owned token buffer instead of allocating one,
which keeps the token tensor pointer stable across iterations and across a captured CUDA graph.

`UnshardedFusedSingular.drafting(...)` takes the `[ B, γ + 1 ]` draft buffer as `tokens`. With
`in_place=True` the selection for timestep $t$ lands in `tokens[:, t]`, and for $t > 0$ the same
buffer doubles as the drafted-token history feeding the repetition penalties. With
`in_place=False` the call allocates a fresh `[ B, 1 ]` tensor and leaves `tokens` untouched.

`UnshardedFusedMultistep.__call__(...)` takes the same draft buffer as `tokens`. With
`in_place=True`, `verification.tokens` *is* that buffer, with rejected positions overwritten by the
correction. Otherwise the buffer is cloned first.

## Contracts

The kernels validate very little at runtime. The following are the integration's responsibility.

**Logits are bfloat16.** The prologue rebinds its working values as `bfloat16`, so a `float32`
logits tensor fails to compile. Cast before the call. Softmax and log-sum-exp are computed
internally in `float32`, and unpacked probabilities are `float32`.

**Logits are unsharded.** The sampling pipeline assumes the logits have already been gathered
across tensor-parallel ranks and presents a single logical vocabulary of width $V$.

**Logits are contiguous over the vocabulary.** Row strides are read from the tensor, but the
vocabulary dimension is expected to be the innermost, unit-stride axis.

**$k \le 128$.** `MAX_K` bounds every top-$`k`$ path. `ScopedIndicators.from_params` raises
`ValueError` for a larger finite $k$, so reject such requests at admission. A request that disables
top-$`k`$ altogether is realized as bounded top-$`128`$ truncation.

**Top-$`k`$ logprobs are capped at 5.** `MAX_TOP_LP` is an internal compile-time constant fixing the
width of the top-$`k`$ logprob block. A larger request must be truncated or rejected by the engine.
The bound is an implementation limit and may be relaxed in a future version.

**Greedy rows carry no logprobs.** The greedy path writes only the token. The reference semantics
in `NaiveSampler` define the selected logprob of a greedy row as $0$, the log of a point mass, with
no top-$`k`$ block. An engine exposing OpenAI-style logprobs at $t = 0$ must compute them itself from
the same post-mask logits.

**Grammar bitmasks are permissive-high.** A set bit admits its token. The default buffer is all
ones, i.e. unconstrained. Masking is applied before selection, so emitted logprobs already reflect
the constrained distribution.

**Weight upkeep is per iteration.** A batch containing any stochastic target requires its
stochastic weights advanced once per decoding iteration, through `step_weights(...)` or through
`update(...)`. Skipping this reuses stale noise and makes verification off-policy.

**Sampling is deterministic given its inputs.** Identical buffers and identical logits select
identical tokens, with or without programmatic dependent launch. Seeding each request's generator
identically across ranks is sufficient for tensor-parallel agreement without a broadcast.
