# Known gaps

What the runtime does and does not support, as of 2026-08-21. Coverage is driven
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

## Supported ops (41)

AveragePool BatchNormalization Cast Clip Concat Constant ConstantOfShape Conv ConvTranspose Dropout Elu Expand Flatten Gather Gelu Gemm GlobalAveragePool GlobalMaxPool HardSigmoid HardSwish Identity InstanceNormalization LayerNormalization LeakyRelu MatMul MaxPool Not Pad Range Relu Reshape Resize Shape Sigmoid Slice Split Squeeze Tile Transpose Unsqueeze Where 

## Gaps, by priority

### Blocks OCR (phase 3) — do first
- **Control flow: If / Loop / Scan.** Subgraph attributes error at graph-build.
  Needed for autoregressive decoding with dynamic length (PARSeq, TrOCR, Nougat).
  Fixed-length AR can be unrolled without these; truly dynamic loops cannot.
- **GridSample.** Perspective/affine rectification of detected text boxes. (Can
  also be done outside the graph in Go pre/post-processing.)
- **NonMaxSuppression, TopK.** Detection post-processing and beam search.

### Common, not yet needed by the zoo
- **Quantization: QuantizeLinear/DequantizeLinear, QLinearConv/MatMul, ConvInteger,
  MatMulInteger, DynamicQuantizeLinear.** No int8 compute kernels exist, so int8
  models are unsupported end-to-end. This is also a *performance* gap (int8 is the
  big CPU speed win) tracked in PERF.md.
- **Recurrent: LSTM / GRU / RNN.** Older OCR recognizers (CRNN) use BiLSTM. A CRNN
  recognizer would need LSTM; SVTR/PARSeq (attention-based) do not.
- **Einsum.** Some attention/loss formulations export as Einsum.
- **ScatterND / ScatterElements / GatherND / GatherElements.** Advanced indexing.
- **CumSum, Trilu, DepthToSpace / SpaceToDepth, Mod, Sign, Round modes.**
- **Resize cubic mode** (nearest and linear are implemented).

### Types & shape
- **f16 / bf16 / f64 are converted to f32 at load; there is no native low-precision
  compute path.** Fine for accuracy, but loses the memory/speed benefit and cannot
  represent a genuinely f16-only model's overflow behaviour.
- **No true dynamic shapes via control flow.** Data-dependent shapes that flow
  through Shape/Slice/Reshape/Gather work (the executor uses actual runtime tensor
  shapes); shapes produced by Loop/If do not.
- **Sequence, Map, and Optional types are unsupported** (rare outside NLP pipelines).

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

Recognition gaps / TODO:
- rec model (ch_PP-OCRv4_rec) copied; **rec character dictionary not yet bundled**
  (RapidOCR keeps it in package resources, not next to the model). Needed for CTC
  decode. Fetch ppocr_keys_v1.txt.
- Need: crop+perspective-rectify each box, resize to 48×W, CTC greedy decode.
