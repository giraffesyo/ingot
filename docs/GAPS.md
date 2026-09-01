# Known gaps

What the runtime does and does not support, as of 2026-09-01. Coverage is driven
by a model zoo run through the conformance harness (`graph.TestZoo`) against ONNX
Runtime. **Everything below "Verified" matches ORT to the noted tolerance; a model
that hits a gap is *skipped*, never silently wrong.**

## Verified models (parity with ONNX Runtime)

| model | nodes | max abs err | exercises |
|---|---|---|---|
| tiny_conv | 14 | 1.5e-8 | Conv/BN/pool/Gemm/softmax |
| tiny_transformer | 35 | 2.4e-7 | LayerNorm/attention/GELU |
| resnetish | 23 | 2.4e-7 | residual Add, MaxPool, BN, GAP |
| mobilenet_v2 | 100 | 4e-15 | depthwise, ReLU6 (Clip) |
| mobilenet_v3_small | 122 | 1.5e-5 | depthwise, SE, HardSwish/HardSigmoid |
| efficientnet_b0 | 239 | — | SiLU (Sigmoid·x), SE |
| vit | 94 | 2.4e-7 | patch embed, cls token, MHA |
| bertish | 91 | 1.5e-7 | Embedding/Gather, LayerNorm, GELU, Tanh |
| llmblock | 39 | 2.4e-7 | RMSNorm, RoPE-free causal attn, SwiGLU |
| segnet | 8 | 3e-8 | ConvTranspose, bilinear Resize |
| opprobe | 14 | 6e-8 | Pad (reflect/edge/constant), Resize nearest+bilinear |
| deconvprobe | 3 | 6e-8 | ConvTranspose stride-2, groups |

Notable: LLM decoder blocks (RMSNorm/SwiGLU/causal attention) and BERT/ViT
encoders run today because they decompose to supported primitives — no fused
Attention/RMSNorm op is required.

## Supported ops (94 + control flow)

Abs Add And ArgMax ArgMin AveragePool BatchNormalization Cast Ceil Clip Concat
Constant ConstantOfShape Conv ConvTranspose CumSum DequantizeLinear Div Dropout
DynamicQuantizeLinear Elu Equal Erf Exp Expand Flatten Floor Gather Gelu Gemm
GlobalAveragePool GlobalMaxPool Greater GreaterOrEqual GridSample GRU HardSigmoid
HardSwish Identity InstanceNormalization LayerNormalization LeakyRelu Less
LessOrEqual Log LogSoftmax LSTM MatMul MatMulInteger Max MaxPool Min Mish Mul Neg
NonMaxSuppression Not Or Pad Pow QLinearConv QLinearMatMul QuantizeLinear Range
Reciprocal ReduceL1 ReduceL2 ReduceMax ReduceMean ReduceMin ReduceProd ReduceSum
ReduceSumSquare Relu Reshape Resize Round Shape Sigmoid Slice Softmax Softplus
Split Sqrt Squeeze Sub Tanh Tile TopK Transpose Trilu Unsqueeze Where Xor

Plus **If** and **Loop** as executor-level control flow (compiled
sub-Sessions; see below). This list is `ops.Supported()` — the previous
hand-curated list here had drifted badly.

## Gaps, by priority

### Blocks OCR (phase 3) — do first
- **Control flow: If / Loop supported** (2026-09-01). Subgraphs build
  recursively with lexical scoping — outer values a subgraph reads become
  explicit extra inputs on the node (Node.Caps), so the optimizer preserves
  them; each subgraph compiles to its own Session (full Optimize pipeline)
  and If/Loop run as executor ops. Loop covers max-trip, data-dependent
  termination (while), loop-carried state, scan outputs (stacked on a new
  leading axis), and zero-trip; verified bit-exact vs ORT (ctrlprobe,
  whileprobe). **Scan** remains a loud error (GRAPHS attributes).
- **GridSample: supported** (2026-09-01) — bilinear/nearest, zeros/border/
  reflection padding, align_corners both; verified vs ORT (gridprobe, 6
  attribute combinations, ≤1.1e-6). Bicubic errors loudly.
- **NonMaxSuppression, TopK: supported** (2026-09-01) — greedy per-class NMS
  (corner + center box formats, score/IoU thresholds) and TopK along any
  axis with ORT tie-breaking; both bit-exact vs ORT (postprobe).

### Common, not yet needed by the zoo
- **Quantization: supported** (2026-08-24) for the onnxruntime-quantizer
  subset: QuantizeLinear, DequantizeLinear, DynamicQuantizeLinear, QLinearConv,
  QLinearMatMul, MatMulInteger — u8/s8 activations, symmetric s8 weights,
  per-channel weight scales (loud errors outside the subset). ConvInteger and
  the com.microsoft QOperator contrib ops (QLinearAdd, QLinearGlobalAveragePool,
  …) remain open, as does making int8 *fast* (PERF.md int8 phase 2).
- **Recurrent: LSTM / GRU supported** (2026-09-01) — layout 0, forward/
  reverse/bidirectional, biases, initial states, both GRU
  linear_before_reset settings; input projection batched into one GEMM
  across timesteps. Verified vs ORT: rnnprobe (torch BiLSTM→GRU stack,
  7.5e-08) and gruprobe (lbr=0 + reverse, 1.8e-07). sequence_lens,
  peepholes, clip, custom activations, and plain RNN error loudly.
- **Einsum.** Some attention/loss formulations export as Einsum.
- **ScatterND / ScatterElements / GatherND / GatherElements.** Advanced indexing.
- **CumSum: supported** (2026-09-01; f32/int64, exclusive/reverse — PARSeq
  needed it). Still open: DepthToSpace / SpaceToDepth, Mod, Sign, Round modes.
- **Resize cubic mode** (nearest and linear are implemented).

### Types & shape
- **f16 / bf16 / f64 are converted to f32 at load; there is no native low-precision
  compute path.** Fine for accuracy, but loses the memory/speed benefit and cannot
  represent a genuinely f16-only model's overflow behaviour.
- **No true dynamic shapes via control flow.** Data-dependent shapes that flow
  through Shape/Slice/Reshape/Gather work (the executor uses actual runtime tensor
  shapes); shapes produced by Loop/If do not.
- **Sequence, Map, and Optional types are unsupported** (rare outside NLP pipelines).

### Platform
- **amd64 has AVX2 kernels for the whole hot path**: GEMM (6×16), elementwise, and
  3×3/5×5 depthwise. Selected at init by CPU detection (falls back to scalar Go on
  pre-Haswell x86). Correctness-verified under Rosetta; native x86 perf unmeasured
  (the CI bench job will produce it). Remaining amd64: AVX-512 (needs runtime
  detection; not all runners have it), and perf tuning on real hardware.

### Loader
- **External data: supported** (this session) — sidecar `.onnx.data` resolved by
  `onnx.DecodeFile`. Offset/length honoured; sidecar files memoised.
- **Training-only fields, sparse initializers, and function (custom op) definitions
  are ignored/unsupported.**

## How to close a gap
1. Add a small module to `tools/export/zoo.py` that uses the op.
2. `\.venv/bin/python zoo.py <name>` — exports model + ORT reference.
3. `go test -run TestZoo/<name>` prints the exact unsupported op(s).
4. Implement in `ops/` with a `_ref`-style oracle where non-trivial; re-run.

Performance gaps (kernels, not coverage) are tracked separately in `docs/PERF.md`.

## Graph optimizer (what gets fused)

`graph.Compile` runs `graph.Optimize` (use `CompileRaw` to skip it). Rewrites
are semantics-preserving and verified by the zoo conformance suite and
`TestOptimizeAB` (optimised vs raw outputs on every model):

- `Add(x,3)→Clip(0,6)→Mul(x,·)→Div(·,6)` ⇒ `ingot.HardSwish` (opset<14 exports)
- `Mul(x, Sigmoid(x))` ⇒ `ingot.SiLU`; `x·0.5·(1+Erf(x/√2))` (any association)
  ⇒ `ingot.Gelu`
- Conv/ConvTranspose followed by scalar or per-channel Mul/Add/Sub, or by
  BatchNormalization ⇒ folded into weights/bias
- Conv/ConvTranspose followed by Relu/HardSwish/SiLU/HardSigmoid/Sigmoid/Clip/
  LeakyRelu ⇒ activation in the conv epilogue (`ingot_act` attrs)
- Conv(act) followed by scalar Mul/Add/Sub/Div ⇒ epilogue scale/shift

Not yet: Gemm/MatMul + bias/activation epilogues, Resize+Add (FPN), GAP into
SE, LayerNorm decomposition re-fusion, fused attention.

## OCR pipeline notes (phase 3, in progress)

Detection works end-to-end: **PP-OCRv4 detector (330 nodes) runs on the pure-Go
runtime, matching ONNX Runtime to 9.5e-5**, and Go DB post-processing
(binarise → connected components → convex hull + rotating-calipers min-area rect
→ unclip → score/size filter) produces correct boxes on the test image. No op
gaps — the detector uses only Conv/BN/Clip/Concat/ConvTranspose/Resize/etc.

Detection gaps / TODO:
- Box post-processing is min-area-rect only; PP-OCR also supports polygon output
  (`det_db_score_mode=slow`, `use_dilation`). Fine for horizontal/rotated text.
- Unclip uses the exact rectangle-offset formula (w,h += 2·area·ratio/perimeter),
  not a general polygon offset (pyclipper). Correct for the rect case we emit.
- No angle classifier yet (cls.onnx copied but unused); needed for 180°-rotated text.

Recognition: **working end-to-end.** PP-OCRv4 rec model (no op gaps, outputs
post-softmax [1,T,6625]) + Go perspective-crop + CTC greedy decode with the
ppocr_keys_v1 dictionary. Full pipeline on the 4-line sample decodes all lines
exactly ("Hello World", "OCR 12345", "pure golang", "DBNet test"; rec conf 0.91-1.00).

## Accuracy — synthetic corpus (`models/ocr.TestCorpus`)

24 generated images / 108 lines with exact ground truth (varied fonts, sizes,
colours, numbers, dates, punctuation, some CJK, a few rotated). Generator:
`tools/export/corpus.py`. Harness scores detection (IoU-matched P/R/F1 @0.5) and
recognition (exact-match + char error rate via Levenshtein) and gates on them.

Baseline 2026-08-21 (PP-OCRv4, greedy CTC):

| metric | value |
|---|---|
| detection P / R / F1 | 0.917 / 0.917 / 0.917 |
| recognition exact-match | 0.919 |
| char accuracy (1-CER) | 0.987 |

The corpus immediately earned itself: it caught a **corner-ordering bug** in DB
post-processing (min-area-rect corners were not canonically ordered, so some crops
were mirrored/reversed — e.g. "Model fox" → "xog Io"). Fixing it took exact-match
from 0.75 to 0.92 and CER from 0.24 to 0.013.

Remaining detection gap: ~9% of boxes are slightly mis-aligned on the left edge,
clipping the first character (e.g. "Date" → box reads "ate"), which drops their
IoU below 0.5. Raising the unclip ratio overshoots the (tight) GT and hurts more
than it helps; the real fix is likely DB score_mode=slow (polygon from contour)
or an asymmetric expansion. Not chased yet.

Recognition gaps / TODO:
- Greedy CTC only; no beam search or language model.
- Perspective crop uses bilinear corner blend (exact for parallelograms); a true
  homography warp would be better for strongly perspective-distorted text.
- No batching: one forward per box. Batching same-width boxes is the throughput win.
- Angle classifier (cls.onnx) still unused — 180°-flipped text won't be corrected.
- det normalization: pipeline uses ImageNet mean/std; PP-OCR config specifies
  0.5/0.5 with limit_type=min/736. Detection is robust to this but exact PP-OCR
  parity needs the config values.
