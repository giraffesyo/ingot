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
- `GroupNormalization` (VAE; opset 18 per-group and opset 21 per-channel
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

Risks to retire with a spike before committing to the design:
- Calling into libSystem without cgo relies on `//go:cgo_import_dynamic`
  and `go:linkname` to runtime internals (the system-stack call path).
  Since Go 1.23 pull-linknames into the runtime are restricted to an
  allowlist; purego's symbols are on it today, but we would depend on
  that list, not on a public API. Verify on Go 1.26 first.
- Callbacks (Metal completion handlers → Go) are harder than calls; use
  blocking `waitUntilCompleted` until they are needed.
- Charter: CLAUDE.md says CPU inference; it must be amended (CPU-first,
  optional GPU backends, still no cgo) before the first GPU kernel lands.

## Order

1. Ops: Sin, Cos, GroupNormalization — done (2026-09-23); Einsum,
   ScatterND, GatherND and the GroupNorm export-pattern re-fusion next.
2. Inspect the local Qwen-Image-2.1 configs → exact dims, memory, FLOPs;
   export one DiT block + VAE decoder to ONNX for a parity target.
3. Unmasked flash attention.
4. safetensors + bf16 storage.
5. Graph builder + `models/qwenimage`, block-by-block parity.
6. GPU: FFI spike (dlopen + one objc_msgSend round trip, CGO_ENABLED=0)
   → Metal backend → executor placement.

## Cost envelope (rough — refine from the real configs)

1024² ≈ 4K image tokens; 2·7e9·4K ≈ 6e13 FLOP/step (+ attention); ×2 CFG
×40 steps ≈ 5e15 FLOP/image. At the measured ~0.75–1 TFLOPS f32 MT GEMM on
the M5 Pro that is ~1 h/image on CPU. The fair comparison is the PyTorch
pipeline forced to `cpu`, not the ~3 min GPU run.
