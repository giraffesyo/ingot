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

| 2026-08-21 | arm64 NEON 8×12 micro-kernel (by-element FMLA via WORD), MC=256 KC=512 NC=3072 | 30.4 GFLOPS* | 75.3* | 210.4* | 107.0* |

\* Measured while the machine was saturated by 52 unrelated background load
(heavy background load; effective clock ≈1.4 GHz by dependent-add test).
A pure-register FMLA loop peaked at 46 GFLOPS/thread under the same load, so
the micro-kernel is at the ceiling *of that degraded core*. Re-measure on a quiet
machine before drawing conclusions; expect ≥3× higher.

Method note: always sanity-check the box first —
`uptime` (load avg) and a dependent-add loop for effective clock.
