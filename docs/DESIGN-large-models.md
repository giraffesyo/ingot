# Design: large generative models (diffusion transformers, multi-B params)

Status: proposed (2026-09-23). Motivating model: Qwen-Image-2.1 (released
2026-09-20) — 7B single-stream DiT (32 layers), a text encoder, and a VAE;
BF16 weights, PyTorch/diffusers only, no ONNX export. Local PyTorch run on
the M5 Pro: ~3 min/image (GPU, settings TBD). The goal is not this one
model: it is making ingot able to run the *class* — DiT/MMDiT image models,
multi-billion-parameter transformers — without a hand-exported ONNX graph
per resolution.

## Where we stand

| need | today |
|---|---|
| GEMM / attention / LayerNorm / RMSNorm-by-decomposition | near-peak f32; SDPA fused (fuse-sdpa, fuse-mha-packed) |
| flash (tiled) attention | **masked only** (`ops/attention.go`, `flash := mask != nil && …`); unmasked long-T materialises T×T scores |
| bf16 weights | opt-in `INGOT_BF16=1` for MatMul/Gemm; loader still widens every bf16/f16 initializer to f32 |
| KV cache | `Session.RunDecode` (DESIGN-kvcache.md) — the base for prefix-KV reuse |
| weight loading | ONNX protobuf + external `.onnx.data`, copied into f32 tensors |
| Sin / Cos / GroupNormalization / Einsum / ScatterND / GatherND | **missing** |
| accelerators | CPU only (by charter) |

## Workstreams

### 1. Ops (start here — needed on every path)
- `Sin`, `Cos` (timestep sinusoidal embeddings, RoPE tables).
- `GroupNormalization` (SD-family VAEs — not Qwen-Image-2.1, see below; opset 18 per-group and opset 21 per-channel
  scale/bias semantics). Also re-fuse the common
  Reshape→InstanceNorm→Reshape→Mul→Add export pattern to one op.
- `Einsum` (common subset: batched matmul / transpose forms → MatMul),
  `ScatterND`, `GatherND`.
- Model-specific fused ops in the `ingot` domain as patterns emerge:
  2D/3D RoPE apply, adaLN modulate (`x·(1+scale)+shift`, fused into the
  LayerNorm epilogue).
Each lands with a `_ref` oracle, an `*_optest_test.go`, and a benchmark.

### 2. Native weights: safetensors + bf16 storage
- `safetensors` reader (new package; stdlib-only format: JSON header +
  raw little-endian blobs). mmap'd, zero-copy views where dtype allows.
- bf16 stays bf16 in memory: tensor dtype `BF16` for weights end to end,
  with the existing bf16 GEMM/GEMV kernels consuming it directly. On Apple
  silicon bf16 is storage-only (widen to f32 in the kernel — PERF.md), so
  the win is memory: 7B params = 14 GB bf16 vs 28 GB f32. That is the
  difference between fitting in 48 GB with a text encoder or not.
- Later: int8 / fp8 weight-only quantization for the text encoder.

### 3. Native model definitions (`models/<name>`)
Build `graph.Graph` directly in Go from a config + safetensors, like
`models/ocr` wraps its networks. Rationale: diffusers→ONNX export at 7B is
fragile, needs >2 GB external data, and bakes in one resolution. The graph
IR, optimizer, memory planner and kernels are reused untouched — nothing
model-specific leaks below `models/`.
- A small builder API in `graph` (Linear, LayerNorm, Attention, … emitting
  standard nodes) so model files read like the PyTorch module.
- Parity: export *individual blocks* to ONNX (one DiT layer, the VAE
  decoder) and compare against ORT + the user's PyTorch pipeline.

### 4. Long-sequence attention
- Flash path for **unmasked** attention (image tokens: 4K at 1024², 16K at
  2K). Block Q and K, online softmax, never materialise T×T.
- Mixed-granularity attention and prefix-KV reuse (Qwen-Image-2.1 specific
  mechanisms — pin down from the reference code before designing).

### 5. Pipeline (model-specific, lives in `models/`)
Qwen BPE tokenizer in Go, flow-matching scheduler, CFG, tiled VAE decode
for 2K, RGBA image I/O.

### 6. Accelerators: Metal via an in-tree FFI (decided 2026-09-23)
No new dependencies: we write our own minimal equivalent of `purego`
rather than importing it. Shape:
- **FFI layer** (under `kernels/`, so the "`unsafe` only in `tensor/` and
  `kernels/`" rule holds): dlopen/dlsym of system frameworks plus a
  C-ABI call trampoline in Go asm (arm64 first; amd64 later), switching to
  the system stack for the call. Works with `CGO_ENABLED=0`. Only what
  Metal needs: `objc_msgSend`, `objc_getClass`, `sel_registerName`, and
  the handful of Metal/Foundation entry points.
- **Metal backend**: device, command queue, buffers (shared storage on
  Apple silicon — unified memory, so no copies for weights), compute
  pipelines from MSL source compiled at load time. GEMM, attention,
  norms, elementwise first.
- **Executor seam**: a backend interface with per-node placement, device
  buffers in the memory planner, transfers at partition boundaries, CPU
  fallback per op. CPU remains the reference: every GPU kernel is tested
  against the CPU/`_ref` path.

Spike result (2026-09-23, Go 1.27, M5 Pro) — the risks are retired and the
FFI lives in `kernels/metal` (darwin/arm64; a stub elsewhere):
- `syscall.syscalln`, the runtime's libc-call helper, is on the linker's
  blocklist for non-`syscall` packages, and `runtime.cgocall` throws without
  cgo. What works: `runtime.entersyscall`/`exitsyscall` (kept linkable by
  the runtime, "do not change the signature") around a 30-line asm shim
  that switches SP to a pooled 1 MiB mmap'd C stack, loads x0–x7, and
  calls the function; x28 (g) is callee-saved in the C ABI. dlopen/dlsym
  arrive via `//go:cgo_import_dynamic`, which is allowed outside cgo files.
  Works with CGO_ENABLED=0 and under `-race` (cgo on).
- Autorelease pools and command encoding are per OS thread: all Metal work
  runs on one goroutine locked to its thread (first run crashed popping a
  pool on a different thread than it was pushed).
- Measured: runtime MSL compile + dispatch + readback correct (saxpy,
  2²⁰+3 elements, ragged grid); ~148 GB/s at 16M floats; **~270 µs per
  commit+wait round trip** — the backend must encode a whole graph (or
  step) into one command buffer, not wait per kernel.
- Floats and structs ≤16 bytes by value are not supported by the shim;
  Metal's compute API does not need them.
- Charter amended (CLAUDE.md): CPU-first, optional GPU backends, no cgo.

Executor seam, as built (2026-09-24): `graph.GPUSession` (CompileGPU) is
a copy of the CPU run loop with per-node placement, not a backend
interface inside Session — the CPU executor stays untouched.
- Memory: the session pool allocates page-aligned mmap'd slabs, each
  wrapped once as a no-copy Metal buffer; constants and feeds are copied
  in. CPU and GPU ops share tensors directly — no transfers.
- Ordering: GPU nodes append to one open command buffer (Stream, recorded
  on the device thread in batches of 64 — a thread hand-off per node cost
  more than small kernels). A CPU node that reads GPU-written data flushes
  first; views, and integer shape math whose inputs no pending GPU node
  wrote, run without flushing. Those side nodes allocate from their own
  pool, recycled only at a flush, so a queued GPU reader never sees its
  buffer reused; GPU-to-GPU reuse is safe in stream order.
- Contract for GPU ops: prepare validates and allocates (or declines, and
  the node runs on the CPU); the encode closure runs later and captures
  sizes and regions by value. Integer tensors are never GPU outputs, so
  prepare may read indices, shapes and slice bounds on the CPU.
- GPUSession.Profile flushes after every GPU node and totals wall time per
  op type, for finding slow kernels.

## Order

1. Ops: Sin, Cos, GroupNormalization — done (2026-09-23); Einsum,
   ScatterND, GatherND and the GroupNorm export-pattern re-fusion next.
2. ~~Inspect the local Qwen-Image-2.1 configs~~ done (below). Next: dump
   reference activations (one DiT block, VAE decoder) from the PyTorch
   pipeline as parity targets.
3. Unmasked flash attention.
4. safetensors + bf16 storage.
5. Graph builder + `models/qwenimage`, block-by-block parity.
6. GPU: done for Qwen-Image (2026-09-24) — see ROADMAP phase 6. GEMMs use
   Metal 4's matmul2d tensor ops (the matrix units: 7.6 TFLOPS f32×bf16,
   20.6 bf16×bf16 on the M5 Pro); weights are the mapped checkpoint
   wrapped with newBufferWithBytesNoCopy (zero copies, no packing); each
   DiT step / text-encoder pass / VAE decode is one command buffer. The
   Metal path is model-level (models/qwenimage over kernels/metal);
   executor-level placement for arbitrary graphs is still open.

## Qwen-Image-2.1, measured from the checkpoint (2026-09-23)

Read from the local HF snapshot (safetensors headers + configs) and the
diffusers 0.37.0.dev0 reference source.

| component | params | stored | notes |
|---|---|---|---|
| DiT `QwenImage21Transformer2DModel` | 7.115B | bf16, 14.2 GB | 32 single-stream blocks, d=4096, 32 heads × 128, SwiGLU 12288, no biases |
| text encoder Qwen3-VL-8B | 8.767B | bf16, 17.5 GB | LM 36 layers, GQA 32/8, M-RoPE [24,20,20] θ=5e6; vision tower 0.58B (edit mode only); lm_head 0.62B unused |
| VAE `AutoencoderKLQwenImage21` | 0.338B | **f32**, 1.35 GB | Wan-2.2-style, 2D convs only, 16× spatial, z=64 |

**DiT block:** LayerNorm (no affine) → `x·(1+scale)`; Q/K/V projections;
per-head RMSNorm on Q and K; 3-axis RoPE (frame/height/width = 16/56/56,
interleaved pairs); attention; `x + tanh(gate)·out`; same for the MLP
(`out(silu(gate(x)) · proj(x))`). Modulation (`[16384, 4096]`) is **one
projection shared by all 32 blocks**. Timestep embedding: 256-dim cos/sin →
2 linears. Text path: zero-centred RMSNorm → linear → GELU(tanh) → linear.
Text encoder output = last decoder layer's hidden state *before* the final
norm — the final norm and lm_head are never run.

**Attention structure:** block-causal over [text, condition images, target
image] — text causal, each image block bidirectional within itself.
`causal_condition` modulates prefix tokens from t=0, so the prefix is
timestep-independent: step 1 computes the prefix K/V once (per layer,
post-RoPE), steps 2..N run **only the target tokens** as plain unmasked
attention over [cached prefix, target] (plus a key-padding mask). This is
the DESIGN-kvcache.md machinery generalised from T=1 decode to T=4096
"decode" — the flash path for unmasked attention is what it needs.

**VAE decoder:** RMS_norm is a per-pixel L2 normalisation over channels
(`F.normalize(x, dim=1)·√C·γ`), *not* GroupNorm; upsampling is nearest 2×
+ 3×3 conv; residual shortcuts are repeat-interleave + reshape/permute
pixel shuffles; one single-head attention (d=1152) in the mid block at
latent resolution. Every op already exists in `ops/`.

The GPU decoder runs in bands: everything after the mid attention is local
(3×3 convs, nearest upsampling, per-pixel norms), so each up block runs
over horizontal bands of its input with 2·resblocks+1 halo rows either
side, and the rows it keeps are bit-identical to a whole-image decode. At
2048² the whole-image decoder's buffers were 5 × 4.8 GB (the 288-channel
upsampled tensor); banded, GPU scratch is ~3 GB at any size. The mid
attention runs in query chunks. Before any stage runs, the pipeline
checks each GPU stage (weights + planned scratch) against the device's
`recommendedMaxWorkingSetSize`, and it saves the denoised latents before
decoding (`-from-latents` retries a decode).

**Pipeline defaults:** 40 steps, no CFG (`true_cfg_scale=1`: the model is
meant to be sampled without guidance — one DiT pass per step),
FlowMatchEuler with dynamic exponential shift (seq-len 256→8192 maps
shift 0.5→0.9, terminal 0.02). Output resolution 1024 default, 2048
supported.

**Op coverage:** everything the three networks use is already supported
or is a reshape of something supported (the vision tower's Conv3d patch
embed has stride = kernel → reshape + MatMul). GroupNormalization is *not*
needed by this model. The real work is items 2–5 above: native weights
+ model definitions, unmasked flash attention at 4K–16K tokens, prefix-KV
reuse, tokenizer/scheduler.

## Cost envelope (from the real dims)

DiT linear cost: 436 MFLOP per token per block → 13.96 GFLOP/token.
Latent is image/16 per side with patch 1: 1024² → 4096 target tokens,
2048² → 16384.

| case | per step | 40 steps | CPU @ ~0.9 TFLOPS |
|---|---|---|---|
| T2I 1024², ~200 text tokens | 5.7e13 linear + 0.9e13 attn ≈ 6.6e13 | 2.7e15 | ~50 min |
| edit 1024² + one 1024² reference (prefix ≈ 4.3K) | ≈ 7.5e13 | 3.1e15 | ~55 min |
| T2I 2048² | 2.3e14 + 1.4e14 ≈ 3.7e14 | 1.5e16 | ~4.5 h |

Text encoder (~7.6B active × 2 × few-hundred tokens ≈ 5e12) and VAE decode
are seconds, not minutes. The user's PyTorch run (edit, 1024², 40 steps,
DiT on MPS, KV cache on) takes ~3 min ≈ 17 TFLOPS effective on the GPU —
a ~20× gap to CPU at our measured GEMM throughput, which is the case for
the Metal backend. bf16 storage does not speed up CPU compute on Apple
silicon (widen-to-f32 kernel; M=4096 GEMMs are compute-bound), it only
saves memory.

**Memory:** bf16 resident: DiT 14.2 + text encoder 15.1 (text-only, no
lm_head) + VAE 1.35 + prefix KV cache (32 layers × 4.3K tokens × 4096 × 2 ×
f32 = 4.5 GB) ≈ 35 GB with everything loaded. Running the stages
sequentially (encode text → free → DiT → free → VAE) peaks at ~20 GB bf16,
or ~33 GB even with f32 DiT weights — so stage-sequential loading is a
cheaper first milestone than native bf16.
