# Design: NCHWc blocked-layout CNN execution

Status: prototyped, parked with data (2026-08-28).

## Motivation

The x86 CNN investigation (PERF 2026-08-27/28) established: mobilenet-class
models plateau at 8–16 workers, ingot's useful compute per run roughly
equals ONNX Runtime's *entire* runtime, and ORT's ~3× on these models
comes from keeping activations in blocked NCHWc layout across the network
(no per-conv repacking, channel-vectorized depthwise) — not from
threading. Fused dw+pw was built and killed (2–3× slower: the
intermediates are LLC-resident at batch 1, so there is no DRAM round trip
to save).

## Prototype result (this document's reason to exist)

A hand-written prototype of one mv2 inverted-residual block
(pw 96→576 → dw 3×3 → pw 576→96 at 14²) compared the current NCHW
pipeline against a channel-last blocked composition built from existing
kernels (SgemmPackedB for the pointwise convs consuming the blocked
layout directly, vek.Mul+Add per tap for the depthwise):

| machine        | NCHW pipeline | NCHWc prototype | ratio |
|----------------|---------------|-----------------|-------|
| Apple M-series | 91 µs         | 94 µs           | 0.97× |
| Zen 5 (32c)    | 103 µs        | 128 µs          | 0.80× |

**Recomposing existing kernels does not deliver the layout win.** The
prototype's depthwise does 2 memory passes per tap (18 per output); a
real NCHWc depthwise holds the 9 taps in registers and makes one pass —
~9× less traffic in the layout's hottest loop. The pointwise side needs
a blocked direct-conv kernel, not a row-major GEMM reinterpretation.

## What a real implementation requires

1. Dedicated NCHWc depthwise kernels (per-arch SIMD: 16-lane channel
   vectors, taps in registers, stride-1/2 variants) — the analog of
   oneDNN's `dw_conv` jit.
2. A blocked pointwise/1×1 kernel consuming [C/16][HW][16] with
   pre-packed weights (close to the existing GEMM micro-kernel but with
   the blocked operand order).
3. A layout-assignment pass (insert NCHW↔NCHWc conversions at graph
   edges and non-conv ops, or teach the elementwise/pool ops the blocked
   layout — they are layout-agnostic per element).
4. The win must be re-proven per step against the pipeline it replaces;
   the fused dw+pw kill and this prototype both show intuition
   mispredicts on cached-activation workloads.

Estimated scope: a multi-round kernel program comparable to the int8 arc.
Not started; the transformer/decode program was prioritized because its
wins were measured larger per unit of work (gptish 3×→1.25× ORT, decode
3.4–17×). CNN latency today: det 8 ms (Apple), mv2 4.9 ms (Zen 5) — vs
ORT 1.5 ms mv2. The gap is real; this document is the map for closing it.
