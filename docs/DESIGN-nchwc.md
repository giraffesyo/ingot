# Design: NCHWc blocked-layout CNN execution

Status: kernel program underway (2026-08-28) — steps 1 (blocked dw) and 2 (blocked pointwise) shipped.

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

1. ~~Dedicated NCHWc depthwise kernels~~ **DONE (vek.DwBlk8S1, 3×3 s1,
   AVX2 + NEON)**: isolated A/B on the mv2 mid-block shape — Zen 5
   44.2 → 6.6 µs (6.7×), Apple 18.4 → 8.1 µs (2.3×), padding included
   on both sides. Stride-2 and 5×5 variants remain.
2. ~~A blocked pointwise/1×1 kernel~~ **DONE (vek.PwBlk6x16, AVX2 +
   NEON)**. Full mv2 inverted-residual block, blocked vs pipeline:
   Zen 5 1.58× (32w) / 2.16× (8w); Apple 1.07× (the CNN gap was always
   an x86 story). Parallelism must chunk (pair × positions) — output
   pairs alone starve wide machines.
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
