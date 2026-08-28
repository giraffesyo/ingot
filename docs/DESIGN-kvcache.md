# Design: KV-cache and decode execution

Status: proposed (2026-08-28). Motivating data: bf16 GEMV measured 2×
memory-bound wins on both architectures; decode (M=1) is the shape where
bf16 weights and the flash machinery pay most, and we cannot benchmark it
honestly without incremental attention state.

## Problem

Autoregressive decode runs the model once per token with T=1, attending
over all previous tokens' K/V. Today every Session.Run is stateless: a
T=1 run would recompute nothing useful (attention over one position), and
running the full prefix every token is O(T²) per sequence. Real decode
needs the runtime to carry K/V state across Runs.

## Shape of the feature

1. **Graph annotation.** fuse-sdpa already isolates the attention core
   per block. A `cache=1` attribute variant (`ingot.SDPA` with cache
   slots) marks the K and V inputs as *append streams*: at Run i, the op
   receives K/V for the new positions only and appends them to its slot
   before attending over the full cached range.

2. **Session state object.** `type Decode struct` owned by the caller:
   `d := s.NewDecode(maxT)` allocates one K and one V buffer per cached
   SDPA node ([H, maxT, dh], f32 or bf16), plus a position counter.
   `s.RunDecode(d, feeds)` threads the state through the executor via the
   existing Ctx (a `Ctx.Cache []DecodeSlot` field; ops that don't cache
   ignore it). Multiple Decode objects per Session = concurrent
   sequences; the Session itself stays immutable and concurrent-safe.

3. **The op path at T=1.** scores = q·K[0:pos]ᵀ is a GEMV (1×pos×dh);
   out = p·V[0:pos] is a GEMV again. Both are bandwidth-bound: this is
   where bf16 cache storage (halved bytes) and the amd64 bf16 GEMV
   (VDPBF16PS row kernel with a single active row — or a dedicated
   1-row kernel) deliver the measured 2×. Causal masking is implicit
   (attend over [0, pos]); no mask tensor, no flash bookkeeping.

4. **Prefill.** Run 0 processes the prompt at T=prompt length through
   the existing flash path, writing K/V into the cache slots as a side
   effect (the append is the same code path, T positions at once).

5. **Benchmark.** `decodebench`: prefill T₀ then N single-token steps;
   report ms/token vs position, f32 vs bf16 cache, vs ORT with
   equivalent IO binding. gptish is the workload; export needs dynamic
   T (torch.onnx dynamic_axes) or a T=1 sibling sharing weights.

## Non-goals (v1)

- Batched decode (B>1 sequences in one Run) — the state layout allows
  it later ([B, H, maxT, dh]) but v1 keeps B=1.
- Speculative/graph-level scheduling changes. The executor stays
  sequential; only the SDPA ops touch the cache.
- Cache eviction/sliding windows.

## Risks / open questions

- Export shape: gptish is traced at fixed T. Dynamic-axes export needs
  verifying against the runtime's shape handling (executor is
  shape-agnostic; fold-const mask baking must be skipped for the cached
  variant — the causal structure is implicit, so fuse-sdpa's cache form
  simply drops the mask input).
- bf16 cache accuracy: K/V quantized once at append (not per read);
  drift is bounded by the same 7e-3 envelope measured for weights.
- The rec/OCR pipeline is untouched; this is transformer-serving
  surface area.

## Steps

1. `Decode`/slot plumbing in graph (state object, Ctx threading), no op
   changes — mechanical.
2. `ingot.SDPA` cache form + fuse-sdpa variant behind a Compile option
   (`CompileDecode`), gptish dynamic export.
3. GEMV path: bf16 single-row kernel; wire cache storage dtype flag.
4. decodebench + ORT comparison; PERF entry.
