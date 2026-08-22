# Roadmap

## Phase 1 — tensor + GEMM (now)
- [x] module, layout, CLAUDE.md
- [x] `tensor`: DType, Tensor, shape/strides, pool
- [x] `kernels/gemm`: reference, blocked f32 Go kernel, benchmark
- [x] `kernels/par`: persistent worker pool, `Run(Task)` 0-alloc, dynamic scheduling
- [x] arm64 NEON 8×12 micro-kernel (generated asm) — 95% of core peak
- [x] amd64 AVX2 6×16 GEMM micro-kernel (Go asm, verified under Rosetta; native
      bench + tuning pending an x86 box). TODO: AVX2 vek elementwise.
- [x] GFLOPS table for common OCR shapes (docs/PERF.md); ORT comparison pending phase 2

## Phase 2 — ONNX → graph → run
- [x] `onnx`: dependency-free protobuf decode → structs
- [x] `graph`: IR, ONNX→IR builder, const folding (Constant nodes), executor with
      pooled refcounted intermediates, per-op profiler. Optimizer passes
      (graph/optimize.go): HardSwish fusion, conv affine/BN folding, conv
      epilogue (activation + post-affine). TODO: shape inference pass, static
      memory planner.
- [x] ops: ~80 ops incl. Conv, MatMul/Gemm, elementwise+broadcast, BatchNorm/
      LayerNorm/InstanceNorm, Softmax, Reshape/Transpose/Concat/Slice/Gather/Split/
      Tile, MaxPool/AvgPool/Global*, reductions, ArgMax/Min, Cast/Where/Expand/
      Squeeze/Unsqueeze/Range/ConstantOfShape/compare/logical.
      TODO: Resize (needed for detection), fused Attention, GridSample, NonMaxSuppression.
- [x] conformance: tiny_conv, tiny_transformer, mobilenet_v3_small vs ONNX Runtime
      (≤1.2e-5 max abs err). Harness in graph/conformance_test.go, exporter in tools/export.

## Phase 3 — OCR
- [x] PP-OCRv4 detector (DBNet) on the runtime, DB postproc (min-area rect, unclip)
- [x] text crop + perspective rectify (bilinear corner blend)
- [x] PP-OCRv4 recognizer (SVTR-LCNet), greedy CTC decode; TODO: PARSeq (AR
      decode w/ KV cache), beam/LM decode, angle classifier, rec batching
- [x] eval harness: synthetic corpus (P/R/F1, CER) as regression gate;
      TODO: ICDAR15, TotalText, IIIT5K, SVT, Union14M
- [x] `cmd/ocr` end-to-end

## Phase 4 — performance
- [x] NEON elementwise kernels (kernels/vek): relu/hardswish/hardsigmoid/clip/
      leakyrelu/add/sub/mul/div/scalar — 7-8× scalar; SE block-broadcast fast path.
      mnv3 5.05→3.93 ms MT.
- [x] NEON depthwise kernel (stride-1 3×3/5×5, pad-then-convolve): hot dw convs
      480→150 µs, mnv3 6.5→4.5 ms 1T. TODO: stride-2 depthwise (still scalar),
      GEMV for classifier (batch-1 FC), implicit-GEMM for the regular conv.
- [x] GEMM small-M fast path + work-sized task grains (pointwise-conv shapes 9×);
      tiled im2col conv with in-cache epilogue; ConvTranspose as GEMM+col2im
- [x] operator fusion: HardSwish, conv+BN/affine folding, conv epilogue
      (bias+activation+post-affine) — det 330→102 nodes, rec 440→206
- [x] amd64 AVX2 vek + gemm kernels (AVX-512 opt-in; loses on Zen4/Ice Lake)
- [x] ≤2× ONNX Runtime CPU on PP-OCRv4 det + rec, documented (docs/PERF.md):
      MT det 0.58–0.82× ORT, rec ~1.0×; 1T 1.2–1.8× (ORT uses SME on M4+)
- [ ] vek.Exp → softmax/sigmoid/GELU SIMD; fused attention; layernorm SIMD
- [ ] int8 GEMM (SDOT/UDOT on arm64, VNNI on amd64), bf16
- [ ] weight pre-packing at load; in-place ops; static memory planner
- [ ] SME GEMM micro-kernel (Apple M4+ / Armv9 SME2) — the only way to match
      ORT single-threaded on that hardware
- [ ] rec batching in the OCR pipeline

## Coverage (breadth)
- [x] model zoo conformance harness (graph.TestZoo, auto-discovers manifests)
- [x] verified vs ORT: resnet, mobilenet_v2/v3, efficientnet, vit, bert, LLM block,
      segnet (ConvTranspose+Resize). LLM/BERT/ViT run via primitive decomposition.
- [x] external-data loader (onnx.DecodeFile), Resize (nearest/linear), ConvTranspose,
      Pad (constant/reflect/edge), Dropout
- [ ] gaps documented in docs/GAPS.md; OCR-blocking: If/Loop/Scan, GridSample,
      NMS/TopK. Quantization (int8) and LSTM/GRU also open.

## Phase 5 — OCR + breadth
- [ ] layout analysis, reading order, tables, multilingual, handwriting
- [ ] wasm target
