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
- [x] gptish zoo model (4 decoder blocks, d=512 T=256, dynamo-export opset 18;
      Trilu op + fold-const mask): fused SDPA had a B·H-parallelism cliff at
      scale (+55%!) — flash-style row tiling fixed it (opt 10.0 vs raw 10.8 ms,
      ≈1.8× ORT CPU end-to-end)
- [x] transformer round 2: powFast (RMSNorm x² was scalar math.Pow — 20×),
      SDPA absorbs head-split transposes (gptish 12→0 Transpose nodes,
      attention = zero data movement), gemm.PackedB pre-packs Linear/MatMul
      const weights (28 packs/run gone) — gptish −25%, llmblock −21%
- [x] subnormal softmax flush: vek.Exp saturates to min-normal → causal-mask
      probabilities fed subnormals into the AV GEMM on x86 (~100 cyc/FMA;
      Apple runs subnormals at speed, dev box blind) — gptish 3.1× on AVX2
      Xeon, now 1.34× ORT same box (llmblock 1.13×, bertish 1.09×)
- [x] x86 int8 epilogues (AVX2 vek: requant/quant/shift/deint/QLut/depthwise
      rows; noescape fix both arches): Zen 5 pod rec_int8 −52%, det_int8 −39%,
      mv3_int8 −54% — x86 int8-vs-f32 gap 2.8× → 1.4×
- [x] NCHWc SHIPPED end to end: kernels + assignBlockedLayout — mv2 −32%
      (beats ORT-32T), effnet −31%; spatial-gated (≤224² static); det/rec
      on pipeline by measurement. Follow-ups in DESIGN-nchwc.md (blocked
      SE, per-shape assignment, mv3 5×5 tuning)
- [x] NCHWc regions: residual Adds run blocked (no kernel needed — same
      permutation both sides); pass grows regions through Adds and fan-out
      instead of linear chains — mv2 16 chains/32 conversions → 1 region/2,
      Zen 5 mv2 −18% (2.53 ms, 1.6× ORT-32T), effnet −6%
- [x] blocked SE: fused SE runs on nChw8c natively (vek.SumBlk8/MulBlk8;
      FC chain is layout-blind) and joins regions; zoo exports now carry
      inferred shapes (mv3 had no value_info → no seeds). effnet −19.6%
      (3.74 ms, ~2.4× faster than ORT's best), mv3_small −19% vs shipped;
      wins on Apple too (effnet −11%, mv3 −15%)
- [x] NCHWc task starvation: ConvDwBlk row strips (was 1 task/channel-block
      → 4 tasks at C=32 on 32 workers), ToBlk8/FromBlk8 spatial chunks —
      mv2 −11%, mv3 −7%, effnet −5% (Zen 5); gate sweep with fixes says
      224² stays right (56²/28² worse). Next: thin-cin pw kernel, stem
- [x] ConvPwBlk bias/epilogue fused into chunk tasks (ragged tail folded
      into predecessor chunk → per-chunk post-processing is race-free):
      mv2 −11% (2.00 ms, 1.38× ORT-16T), mv3_small −26% (1.09 ms — beats
      ORT's best), effnet −8%; non-idempotent-epilogue oracle test
- [x] blocked-SE pool/scale spatial chunking (two-phase partial sums):
      effnet −8.5% (SE was 27% of the model); ZMM pointwise tile
      pwblk6x16z (1 wload+6 bcast+6 FMA per ci, init-probed): mv2 −2.6%,
      effnet/mv3 −1.5%. Day cumulative: mv2 2.48→1.95, effnet 3.76→3.10,
      mv3 1.60→1.05
- [x] PwBlk6x16Tiles (tile loop in asm, AVX2/ZMM/NEON): mv2 −2.3%; CPU-share
      ≠ wall-share lesson — remaining mv2 cost is bandwidth + region churn
- [x] fuse-blk-res: residual Adds fold into ConvPwBlk (post-epilogue,
      per-chunk): mv2 −4.5%, effnet −3%; all zoo residual Adds fold
- [x] pool-width sweep + OCR_WORKERS knob: every model fastest at 8-16
      workers on 32-core Zen 5 (mv3 −24%, resnetish −40% at 8w). Best-of:
      mv2 1.79 (1.23× ORT-16T), effnet 2.80, mv3 0.854
- [x] SetInputShape + propagateShapes: blocked layout for dynamic models —
      rec declared on amd64 (rec_b8 −15% Zen 5; Apple +11% → arm64 stays
      pipeline); det measured +4-8% blocked → stays pipeline. SE pooling
      chunks now deterministic (batch bit-exactness)
- [ ] rec_batch_test fails on amd64 (PRE-EXISTING, discovered when the OCR
      suite first ran on x86): single-vs-batch low-bit drift + padding
      flips borderline chars; arm64 passes
- [x] pool-width default: min(GOMAXPROCS, 12) on amd64 (Zen 5 rigorous sweep:
      w12 wins/ties every model; Zen 4 agreed in Aug; Apple scales to full
      width so arm64 keeps GOMAXPROCS) — out-of-box effnet −6.7%, gptish
      −4.5%, mv3 −11%, tiny −30%. NEGATIVE: adaptive per-region width
      (spin lease) measured worse — width is a machine property (4th entry
      on the par do-not-retry list)
- [x] SE islands fused (ingot.SE: det 10, mv3 9, effnet 16 — effnet −9%,
      mv3 −8% on Zen 5); sigmoid subnormal flush (amd64, preventive)
- [x] amd64 bf16 kernel (VDPBF16PS, BYTE-encoded): Zen 5 peak probe 1.45× f32
      ALU; Bgemm sq512 1484 GFLOPS = 2.6× f32 Sgemm — bf16 verdict flips on x86
- [x] bf16 weights wired (INGOT_BF16=1, MatMul): rows-kernel v2 (no A pack) —
      gptish −8% on Zen 5, ≈1.27× ORT; accuracy 7e-3 documented, serving-only
- [x] bf16 Gemm wiring (post-added beta*C; bertish −3%, toy-scale flat)
- [x] gptish_1k (T=1024) zoo model: bf16 −19%, 1.47× ORT-16T, beats ORT-32T;
      fold-const cap 16 MiB for runtime-built masks
- [x] flash SDPA (online softmax, causal block skip, Bk=128, ≥4-block gate):
      gptish_1k SDPA −29%, model −12%; +bf16 = 26.0 ms, 1.25× ORT-16T
- [x] KV-cache steps 1-2: cached ingot.SDPA + Session.RunDecode, property-
      tested vs dense causal oracle (2e-5)
- [x] decode e2e: CompileDecode (causal-chain mask proof + DCE), gptish_dyn
      dynamic-T export, decodebench — 1.71 ms/token flat (1.10 bf16), 3.4-17×
      over naive recompute
- [x] decode polish: bf16 K/V cache + SIMD DotBF16/AxpyBF16 — 1.042 ms/token
      (−39% vs f32 decode); DotBF16 no longer scalar on amd64
      remaining f32 GEMM efficiency vs MLAS at transformer shapes (AVX-512
      µkernel auto-probed; paired-panel 6×32 kernel: 1T at machine peak
      +25%, no packing changes — pairing gated by worker feed)
- [x] weight pre-packing (gemm.PackA, cached per conv op); lock-free par hand-off
      (rec_320 MT 5.5 → 2.7 ms — the hand-off was 78% of CPU samples)
- [x] allocation-free run loop: inline tensor shapes + pooled headers, pooled
      session scratch, buffer custody for views (Reshape & friends now
      zero-copy), pooled par tasks — toy transformers −9…−16%, GC STW 54→19%
      of samples; allocs/run: bertish 444→154, rec 774→254
- [ ] in-place ops; static memory planner; remaining per-op allocs (result
      slice + closure per op; rec_int8 quant chunking closures ~900/run)
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
