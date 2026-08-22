# ingot — a pure-Go ONNX inference runtime

*ingot: **in**-**Go** **t**ensors — a general-purpose, cgo-free ONNX CPU
inference engine with hand-written near-peak SIMD kernels. OCR is its flagship
consumer, not its purpose.*

## Mission

Build a **general-purpose ONNX CPU inference runtime in pure Go**: no cgo, no
external runtimes, static binaries, wasm-capable. Accuracy comes from Python-trained
weights; this repo owns **inference**. It runs any ONNX model that uses supported
ops — CNNs, ViTs, BERT/transformer encoders, decoder LLM blocks — verified against
ONNX Runtime. OCR (`models/ocr`, DBNet + PP-OCR) is the first and flagship consumer
and drives op priorities, but nothing in `tensor/`, `graph/`, `ops/`, `kernels/`
may be model-specific.

**We are obsessed with performance.** Target: ≤2× ONNX Runtime CPU latency on the
same hardware. Every kernel ships with a benchmark.
No regression merges.

## Non-negotiables

- `CGO_ENABLED=0` must build and pass everything. No cgo, ever. No `unsafe` outside
  `tensor/` and `kernels/`.
- Correctness before speed, but speed is a correctness requirement of the project:
  a kernel without a benchmark and a reference-comparison test does not exist.
- Every op is verified against an independent oracle: kernels have a pure-Go
  reference (`*_ref.go`) that the fast paths (blocked Go, asm) are tested against;
  ops are tested against hand-written or naive oracles in `ops/*_optest_test.go`
  exercising the branches end-to-end model parity does not (dilation, groups,
  autopad, ceil-mode pooling, negative-step Slice, Pad/Resize modes, strided
  Softmax axis, transposed/batched MatMul). Oracles accumulate in float64; f32
  tolerance is rel 1e-5·√k for a reduction of length k. bf16/int8: documented per op.
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
