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
      480→150 µs, mnv3 6.5→4.5 ms 1T. Stride-2 depthwise via column
      de-interleave + 3x2/3x1 (5x3/5x2) row kernels; GEMV for batch-1 FC.
      TODO: implicit-GEMM for the regular conv.
- [x] GEMM small-M fast path + work-sized task grains (pointwise-conv shapes 9×);
      tiled im2col conv with in-cache epilogue; ConvTranspose as GEMM+col2im
- [x] operator fusion: HardSwish, conv+BN/affine folding, conv epilogue
      (bias+activation+post-affine) — det 330→102 nodes, rec 440→206
- [x] amd64 AVX2 vek + gemm kernels (AVX-512 opt-in; loses on Zen4/Ice Lake)
- [x] ≤2× ONNX Runtime CPU on PP-OCRv4 det + rec, documented (docs/PERF.md):
      MT det 0.37–0.47× ORT, rec 0.43–0.49×; 1T 1.0–1.5× (ORT uses SME on M4+)
- [x] vek.Exp/Sigmoid (NEON + AVX2) → Softmax, Sigmoid, Exp op (2.8 Gelem/s 1T)
- [x] SiLU/Erf/GELU SIMD kernels + pattern fusion (ingot.SiLU/Gelu); Dot; GEMV
- [x] Winograd F(2×2,3×3), fused per-block pipeline — default-on (yields to
      SME at 1T; OCR_NO_WINOGRAD=1 disables): resnetish MT −20%, det −7%
- [x] hybrid SME+NEON MT GEMM: prototyped, measured, declined (parity at best
      on real shapes; docs/PERF.md)
- [ ] Tanh SIMD; fused attention; layernorm SIMD; direct 3×3 / SIMD MaxPool for
      small spatial convs (resnetish 4× ORT); fused depthwise+pointwise (mnv2 1T)
- [x] int8 phase 1: SMMLA/USMMLA GEMM (311 GOPS 1T / 630 MT), Quantize/
      Dequantize/DynamicQuantize/QLinearConv/QLinearMatMul/MatMulInteger,
      quantized zoo conformance (tiny_conv bit-exact vs ORT)
- [x] int8 phase 2a: tiled/direct QLinearConv paths + vek quant/dequant/requant
      NEON kernels (mnv3_int8 21 → ~5.5 ms MT, conformance bit-identical)
- [x] int8 phase 2b: SMLAL depthwise kernels (vek.QDwRowS1) + qgemm B-pack
      rewrite (sq512 630→1681 GOPS MT; det-head shape 3×)
- [x] int8 SME: SMOPA s8→s32 GEMM (1.3 TOPS/core 1T, 3.4-3.5× NEON int8),
      auto-1T dispatch via gemm.QPackedA
- [x] quantized PP-OCR part 1: exporter + GT-crop calibration (rec 0.33→0.90
      exact — calibration is everything), ORT parity (det 0.0078), corpus knobs
- [x] quantized PP-OCR part 2: QLinearConv driver maturation — lock-free corr,
      pooled I16 depthwise scratch, memcpy qim2col, NEON zip-transpose B-pack
      (det_int8 MT 36→11.8 ms, beats ORT-int8 14.2; 1T 109→63 vs ORT 36)
- [x] quantized PP-OCR part 3: chunk working-set fix, vek.ShiftU8S8, corr
      folded into requant asm, NEON C-tile scatter — det_int8 MT 8.4 ms
      (1.7× faster than ORT-int8, matches our f32 det), 1T 109→44.6 overall
- [x] quantized PP-OCR part 4: Q/DQ island elision (fold-qdq-affine +
      fuse-qlut/TBL) + pooled Run outputs w/ Session.Release — det_int8
      6.2 ms MT (2.2× faster than ORT-int8, 1.3× faster than our f32),
      1T 38.2 vs ORT 36
- [x] det-head tail: vek.Zip2 (2× Resize 1.84→0.75 ms, s2k2 ConvTranspose
      3.27→2.55), per-channel QLut (BatchNorm islands), fuse-layernorm
      (5 sites in rec) — det_int8 1T ≈ ORT-int8 parity
- [x] amd64 VNNI int8 GEMM: VPDPBUSD 8×12 kernels, quad-major B / row-major C,
      453 GOPS sq512 on CI Ice Lake (2.4× same-run f32); pre-VNNI x86 stays
      portable (AVX2 vpmaddubsw saturates — inexact by construction)
- [x] SDOT mid-tier (Apple M1 / Graviton2): 8×12 SDOT-by-element kernels on
      the VNNI layouts + qpackbq quad-pack — within ~10% of the MMLA tier
      (sq512 1448 vs 1410 GOPS 1T same-run)
- [x] quantized depthwise driver SIMD (WidenS8S16, DeinterleaveS16, corr-in-
      requant, overlapped tail) — det_int8 1T 37.5→31.2 ms, faster than
      ORT-int8 (36) on every remaining metric except rec 1T parity
- [x] bf16 kernels + probes: NEON BFMMLA/BFDOT quarter-rate on Apple, SME
      BFMOPA == f32 FMOPA — bf16 is storage-only here (kernel stays in-tree
      for Neoverse-class hardware, gated + unwired)
- [x] bf16 storage path: vek.DotBF16 + gemm.GemvBF16 (shll-widen into f32
      FMLA) — 2.0× MT GEMV at the DRAM wall (105→215 GFLOPS), 1T parity
- [x] fused attention (ingot.MHA): 11-node exported MHA pattern → one op
      (scale in GEMM alpha, transposed-B K, strided output writes); both rec
      blocks fuse, parity/accuracy unchanged
- [x] generic SDPA fusion (ingot.SDPA): torch-export attention core incl.
      masked decoder blocks — 5 zoo blocks fused, llmblock −12% at toy scale
- [x] fold-const pass: any all-const-input node is evaluated at load with the
      same registered op the executor runs (bit-identical), size-guarded;
      unblocks tiny_transformer's SDPA (fold-const 5 + fuse-sdpa 1) and elides
      15-19 per-run shape-chain nodes in rec/cls; exposed + fixed MHA/SDPA
      per-head pool-wake overhead (headGrain: llmblock/bertish/vit −29…−34%)
- [ ] SE-path islands, fused epilogues; bf16 executor wiring; larger-seq
      transformer zoo model to show SDPA at scale
- [x] weight pre-packing (gemm.PackA, cached per conv op); lock-free par hand-off
      (rec_320 MT 5.5 → 2.7 ms — the hand-off was 78% of CPU samples)
- [ ] in-place ops; static memory planner; per-op overhead (~13 µs/node MT)
- [x] SME probes + Sgemm (kernels/sme, pure Go WORD-encoded): FMOPA peak 2.17
      TFLOPS/core; pre-packed Sgemm 700 GFLOPS 1T (7.4× NEON); signal-mask
      guard (ZA dies on signal delivery — GC-storm regression test); dispatch
      via OCR_GEMM_KERNEL=sme → rec_320 1T 8.8 ms vs ORT 11.7 (0.75×).
      Default-on (auto): SME when the pool is single-threaded, NEON at MT.
      TODO: hybrid SME+NEON MT scheduling, linux detection (HWCAP2_SME)
- [x] rec batching in the OCR pipeline (width-grouped, padding-bounded)

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
