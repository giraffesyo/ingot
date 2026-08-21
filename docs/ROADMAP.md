# Roadmap

## Phase 1 — tensor + GEMM (now)
- [x] module, layout, CLAUDE.md
- [x] `tensor`: DType, Tensor, shape/strides, pool
- [x] `kernels/gemm`: reference, blocked f32 Go kernel, benchmark
- [x] `kernels/par`: parallel-for with atomic scheduling (TODO: persistent pool → 0 allocs/op)
- [ ] arm64 NEON micro-kernel (MR×NR f32), amd64 AVX2 via avo
- [ ] `bench/`: GFLOPS table for common OCR shapes, compare vs ORT numbers

## Phase 2 — ONNX → graph → run
- [ ] `onnx`: protobuf decode (vendored onnx.proto), IR conversion
- [ ] `graph`: shape inference, const folding, conv+BN fusion, memory planner, executor
- [ ] ops: Conv, MatMul/Gemm, Add/Mul/Sub/Div, Relu/Gelu/SiLU/HardSwish, BatchNorm,
      LayerNorm, Softmax, Reshape/Transpose/Concat/Slice/Gather, Resize, MaxPool/AvgPool,
      Sigmoid, Attention (fused), Where, Cast, Expand, Squeeze/Unsqueeze
- [ ] conformance: MobileNetV3 + a tiny ViT exported from PyTorch, logits match ≤1e-4

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
