# Roadmap

## Phase 1 — tensor + GEMM (now)
- [x] module, layout, CLAUDE.md
- [x] `tensor`: DType, Tensor, shape/strides, pool
- [x] `kernels/gemm`: reference, blocked f32 Go kernel, benchmark
- [x] `kernels/par`: persistent worker pool, `Run(Task)` 0-alloc, dynamic scheduling
- [x] arm64 NEON 8×12 micro-kernel (generated asm) — 95% of core peak
- [ ] amd64 AVX2/AVX-512 micro-kernel via avo (deferred: no amd64 box to measure)
- [x] GFLOPS table for common OCR shapes (docs/PERF.md); ORT comparison pending phase 2

## Phase 2 — ONNX → graph → run
- [x] `onnx`: dependency-free protobuf decode → structs
- [x] `graph`: IR, ONNX→IR builder, const folding (Constant nodes), executor with
      pooled refcounted intermediates, per-op profiler. TODO: shape inference pass,
      conv+BN fusion, static memory planner.
- [x] ops: ~80 ops incl. Conv, MatMul/Gemm, elementwise+broadcast, BatchNorm/
      LayerNorm/InstanceNorm, Softmax, Reshape/Transpose/Concat/Slice/Gather/Split/
      Tile, MaxPool/AvgPool/Global*, reductions, ArgMax/Min, Cast/Where/Expand/
      Squeeze/Unsqueeze/Range/ConstantOfShape/compare/logical.
      TODO: Resize (needed for detection), fused Attention, GridSample, NonMaxSuppression.
- [x] conformance: tiny_conv, tiny_transformer, mobilenet_v3_small vs ONNX Runtime
      (≤1.2e-5 max abs err). Harness in graph/conformance_test.go, exporter in tools/export.

## Phase 3 — OCR
- [ ] DBNet++ port (PaddleOCR / OpenOCR export), DB postproc (polygons, unclip)
- [ ] text crop + perspective rectify
- [ ] SVTR recognizer, CTC decode; then PARSeq (AR decode w/ KV cache)
- [ ] eval harness: ICDAR15, TotalText, IIIT5K, SVT, Union14M
- [ ] `cmd/ocr` end-to-end

## Phase 4 — performance
- [ ] asm conv (implicit GEMM), fused attention, softmax/layernorm SIMD
- [ ] int8 GEMM (dot-product instrs: SDOT/UDOT on arm64, VNNI on amd64), bf16
- [ ] operator fusion (GEMM epilogues), in-place, buffer reuse
- [ ] ≤2× ONNX Runtime CPU on DBNet++ + SVTR, documented

## Phase 5 — breadth
- [ ] layout analysis, reading order, tables, multilingual, handwriting
- [ ] wasm target
