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
