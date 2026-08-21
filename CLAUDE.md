# ocr — pure-Go SOTA OCR on a pure-Go inference runtime

## Mission

Build world-class (PaddleOCR-v5 / SVTRv2 / PARSeq-class accuracy) OCR in **pure Go**:
no cgo, no external runtimes, static binaries, wasm-capable. Accuracy comes from
Python-trained weights; this repo owns **inference**. The runtime is designed as a
general ONNX CPU inference engine — OCR is the first model family and drives op
priorities, but nothing in `tensor/`, `graph/`, `ops/`, `kernels/` may be OCR-specific.

**We are obsessed with performance.** Target: ≤2× ONNX Runtime CPU latency on the
same hardware for DBNet++ and SVTR/PARSeq. Every kernel ships with a benchmark.
No regression merges.

## Non-negotiables

- `CGO_ENABLED=0` must build and pass everything. No cgo, ever. No `unsafe` outside
  `tensor/` and `kernels/`.
- Correctness before speed, but speed is a correctness requirement of the project:
  a kernel without a benchmark and a reference-comparison test does not exist.
- Every op has a pure-Go reference implementation (`*_ref.go`) used as the oracle.
  Fast paths (blocked Go, asm) are tested against the reference with tolerance tied
  to dtype. Oracles accumulate in float64; f32 tolerance is rel 1e-5·√k for a
  reduction of length k (summation order differs between blocked and naive).
  bf16/int8: documented per op.
- Numerical parity with ONNX Runtime / PyTorch on exported models is a test, not a
  goal. `testdata/` holds small exported models + reference outputs.
- Unsupported ONNX op → loud error at load time listing the op and node name.
  Never silently skip.
- Zero allocations in the hot path. Tensors come from an arena/pool; the executor
  pre-plans buffers. `go test -bench . -benchmem` must show 0 allocs/op for kernels.

## Architecture (layers, strict dependency direction: top depends on bottom)

```
cmd/ocr, cmd/onnxrun       CLIs
models/ocr                 detection (DBNet++), recognition (SVTR/PARSeq), pre/post-proc, pipeline
graph                      IR, shape inference, optimizer passes (const-fold, conv+BN fusion,
                           layout), memory planner, executor (goroutine-parallel)
onnx                       protobuf decode of .onnx → graph.IR; op/attribute mapping
ops                        ONNX-semantics ops over tensors (Conv, MatMul, LayerNorm, Attention,
                           Softmax, Gelu, Resize, ...). Thin: dispatch to kernels.
kernels/{gemm,conv,attn,…} hot loops. Per-arch asm (arm64 NEON, amd64 AVX2/AVX-512)
                           with Go fallback. Pure-Go blocked/packed versions are the
                           first fast path; asm is the second.
tensor                     Tensor (dtype, shape, strides, data), DType, arena/pool, views
bench                      cross-cutting benchmarks + perf harness vs reference numbers
```

Layout convention: **NCHW** for conv activations internally (matches ONNX); packing
to blocked layouts happens inside kernels. Weights are pre-packed once at load time
by the optimizer.

## Performance rules

- GEMM is the center of the universe. conv = im2col/implicit-GEMM → GEMM;
  attention = GEMM + softmax; linear = GEMM. Make GEMM fast first, everything
  inherits.
- Blocking: pack A/B panels into L1/L2-sized tiles (Goto/BLIS style: MC/KC/NC, micro-kernel MR×NR).
  Micro-kernel in asm; packing in Go (or asm later).
- Parallelism: goroutines over M/N tiles with a fixed worker pool (`kernels/par`),
  never `go` per tile. Work size ≥ ~64KB per task or don't split.
- Assembly via `avo` (amd64) and hand-written Plan9 (arm64) under `kernels/*/asm_*.s`
  with `//go:build` tags and a `_generic.go` fallback. `go vet` and `-race` must pass
  on the fallback path.
- Data types: f32 baseline; bf16 weights/int8 activations are planned. Quantized kernels
  live next to the f32 ones, same API.
- Profile before optimizing: `make prof PKG=./kernels/gemm`. Commit messages for
  perf changes include before/after GFLOPS or ns/op on this machine (note CPU).
- Memory bandwidth is the usual wall. Fuse elementwise ops into GEMM epilogues
  (bias, activation, residual) rather than separate passes.

## Conventions

- Go 1.26. `gofmt`, `go vet ./...`, `staticcheck` clean. Errors wrapped with context.
- Packages are small and flat. No `util`/`common`.
- Tests: table-driven; reference-vs-fast property tests with random shapes (seeded).
  Benchmarks named `BenchmarkOp/shape=...`.
- Generated asm is checked in; the generator (`kernels/*/gen/`) is `go run`-able.
- `make test`, `make bench`, `make lint`, `make prof`. CI = `CGO_ENABLED=0 make test`.
- Don't add dependencies casually. Allowed: `golang.org/x/*`, `google.golang.org/protobuf`
  (for onnx), `github.com/mmcloughlin/avo` (build-time only). Anything else: justify in PR.

## Roadmap (see docs/ROADMAP.md)

1. tensor + gemm (blocked Go, then NEON/AVX2 micro-kernels) + bench harness
2. onnx loader + graph IR + shape inference + naive executor; run exported MobileNetV3, match logits
3. DBNet++ + CRNN/SVTR end-to-end; eval harness (ICDAR15, TotalText, IIIT5K, Union14M)
4. Perf: asm kernels, fusion, memory planner, int8/bf16; hit ≤2× ORT
5. PARSeq/SVTRv2, layout, reading order, multilingual
