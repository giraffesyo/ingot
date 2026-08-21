# Performance log

Machine: Apple Silicon (arm64), Go 1.26.7, `CGO_ENABLED=0`. Append a row per
meaningful change. Numbers are `go test -bench` medians, f32.

## GEMM (`kernels/gemm`)

| date | change | sq512 (1T) | sq512 (MT) | sq2048 (MT) | conv 64×16384×576 (MT) |
|---|---|---|---|---|---|
| 2026-08-21 | baseline: packed/blocked Go, MR=NR=4 scalar micro-kernel | 9.1 GFLOPS | 39.5 | 86.0 | 65.2 |

Reference (naive triple loop, 1T): 0.65 GFLOPS.

Targets: M-series core peak f32 FMA ≈ 100+ GFLOPS/core. NEON 8×12 (or 16×6)
micro-kernel should bring 1T to ≥60 GFLOPS; then fix thread scaling (sq512 MT is
only 4.3× 1T — B-panel packing is serialised and tiles are too small to amortise
goroutine spawn).

Known allocations: `par.For` spawns goroutines per region (2–364 allocs/op
depending on block count). Replace with a persistent worker pool.

| 2026-08-21 | arm64 NEON 8×12 micro-kernel (by-element FMLA via WORD), MC=256 KC=512 NC=3072 | 120.5 GFLOPS | 229.3 | 742.5 | 328.5 |

Machine ceiling (pure-register FMLA loop, no loads): 130–138 GFLOPS per performance core,
~1080 GFLOPS aggregate across all cores.
Micro-kernel alone: 131.9 GFLOPS (95% of core peak).

Full 1T table after NEON kernel:

| shape | 1T GFLOPS | MT GFLOPS | note |
|---|---|---|---|
| sq512 | 120.5 | 229.3 | |
| sq1024 | 124.3 | 452.0 | |
| sq2048 | — | 742.5 | 69% of aggregate ceiling |
| conv 64×16384×576 | 71.6 | 328.5 | M small: B packing cost not amortised |
| conv 256×1024×2304 | 113.3 | 110.9 | **single-threaded**: m≤MC and n≤NC → 1 block |
| lin 256×768×192 | 108.9 | 109.8 | single-threaded, same cause |
| attn 256×256×64 | 101.4 | 99.5 | single-threaded, same cause |

| 2026-08-21 | persistent worker pool (par.Run, 0 allocs), BLIS-style parallel packing + 2D (N-panel × M-chunk) macro tasks | 118.7 | 494.5 | 944–1021 | 377–443 |

After intra-block parallelism (MT unless noted; ranges = quiet box vs. ~2 cores busy with other apps):

| shape | 1T | MT | MT (2 cores busy) | note |
|---|---|---|---|---|
| sq512 | 118.7 | 494–616 | | |
| sq1024 | — | 693–765 | | |
| sq2048 | — | 944–1021 | 997 | **~95% of aggregate ceiling** |
| conv 64×16384×576 | 93–99 | 378–443 | | packing now parallel |
| conv 256×1024×2304 | 113 | 614–668 | | was single-threaded (111) |
| lin 256×768×192 | 113–118 | 377–507 | 495 | was 110 |
| attn 256×256×64 | 103–108 | 144–209 | 217 | was 100; region barriers dominate at 58µs |

Observations:
- Small shapes are barrier-bound: ~3 par.Run regions per (jc,pc,ic) block at ~6µs each.
  Fusing pack-A into the macro task (pack-on-demand per M-chunk) would cut one.
- When other processes hold cores, fewer workers beats more (stragglers at barriers).
  Consider an adaptive worker count or letting the caller cap it per call.
- amd64 AVX2/AVX-512 kernel deferred until there is an amd64 box to measure on
  (can cross-build + correctness-test under Rosetta, but not benchmark).

Method note: always sanity-check the box first — `uptime` (load avg) and a
dependent-add loop for effective clock. On 2026-08-21 the first measurements were
taken with the machine under heavy background load, at ~1.4 GHz effective.


## End-to-end models (`graph`)

Machine: Apple Silicon, Go 1.26.7, CGO_ENABLED=0. `BenchmarkModels`, f32.
Reference: ONNX Runtime 1.29 CPU, same host.

| model | ours 1T | ours MT | ORT 1T | ORT MT | ratio (MT) |
|---|---|---|---|---|---|
| mobilenet_v3_small | 8.2 ms | 5.05 ms | 2.71 ms | 1.12 ms | 4.5× |
| tiny_conv | — | 0.090 ms | — | — | |
| tiny_transformer | — | 0.087 ms | — | — | |

Numerical parity vs ORT (max abs err): tiny_conv 1.5e-8, tiny_transformer 2.4e-7,
mobilenet_v3_small 1.2e-5. Correctness is not the gap; speed is.

Op breakdown, mobilenet_v3_small, 1T (`OCR_PROFILE_MODEL=… go test -run TestOpProfile -v`):

| op | count | µs/run | share |
|---|---|---|---|
| Conv | 52 | 5357 | 63% |
| Gemm | 2 | 929 | 11% |
| HardSwish | 19 | 844 | 10% |
| Mul | 9 | 600 | 7% |
| Relu | 14 | 598 | 7% |
| GlobalAveragePool | 10 | 128 | 2% |

Where the 4.5× goes, and the fix (phase 4):
- **Conv (63%)** — im2col+GEMM materialises the column buffer every call and the
  1×1 path is fine, but 3×3 stride-1 convs pay im2col bandwidth. Implicit-GEMM
  conv (pack directly from the input, no col buffer) + a NEON depthwise kernel.
- **Elementwise + pool (26%)** — pure scalar Go. HardSwish/Relu/Mul/Sigmoid/
  softmax/pool want NEON (and AVX2/512) whole-slice kernels; ~5-8× each.
- **Gemm (11%)** — already NEON, near core peak; multi-op fusion (conv/gemm
  epilogue folds bias+activation) removes separate Relu/HardSwish passes entirely.
- Un-cleared pool allocation (done) removed the double-write on every output;
  worth ~0 here because outputs are written immediately, but it matters once
  kernels get faster.

Uncleared-buffer note: `Pool.GetUninit` / `Ctx.NewUninit` skip zeroing for ops
that write every output element.


## NEON elementwise kernels (`kernels/vek`)

Generated NEON f32 kernels (WORD-encoded, like gemm) for the ops layer:
add/sub/mul/div, max/min, relu, hardswish, hardsigmoid, clip, leakyrelu,
add/mul-scalar. Go wrapper does the <4-element tail; Go fallback on non-arm64.

Single-thread, n=200704 (16×112×112, a real early activation), Apple Silicon:

| kernel | scalar Go | NEON | speedup |
|---|---|---|---|
| HardSwish | 2.05 Gelem/s | 16.7 | 8.1× |
| Relu | — | 22.4 | ~7× |
| Mul | — | 14.5 | mem-bound (2R+1W) |
| Add | — | 14.4 | mem-bound |

Wired into ops with parallel chunking + broadcast fast paths (same-shape,
scalar, and per-channel block broadcast for squeeze-excite `[N,C,1,1]·[N,C,H,W]`).

mobilenet_v3_small op breakdown after vek (MT), vs before:

| op | before | after |
|---|---|---|
| HardSwish | 566 µs | 114 µs |
| Relu | 302 µs | 65 µs |
| Mul (SE broadcast) | 685 µs | 75 µs |

End-to-end mobilenet_v3_small: **5.05 → 3.93 ms MT**, **8.2 → 6.5 ms 1T**.
Conv is now **78%** of runtime — next target is implicit-GEMM conv (drop the
im2col buffer) + a NEON depthwise kernel.

| date | change | mnv3 1T | mnv3 MT |
|---|---|---|---|
| 2026-08-21 | im2col+GEMM conv, scalar elementwise | 8.2 ms | 5.05 ms |
| 2026-08-21 | NEON vek elementwise + SE broadcast fast path | 6.5 ms | 3.93 ms |
