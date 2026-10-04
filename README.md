# ingot

**A pure-Go ONNX inference runtime.** `ingot` = *in-Go tensors*: a general-purpose,
cgo-free ONNX CPU inference engine with hand-written, near-peak SIMD kernels
(NEON on arm64, AVX2 on amd64). Static binaries, cross-platform, no external runtime.

Verified numerically against ONNX Runtime on a spread of models — ResNet,
MobileNet v2/v3, EfficientNet, ViT, BERT, a decoder-LLM block — and shipped with a
flagship consumer: **OCR** (`models/ocr`, PP-OCRv4 DBNet detection + CRNN/CTC
recognition), running the real PP-OCR models end to end in pure Go.

```go
import "github.com/giraffesyo/ingot/graph"
import "github.com/giraffesyo/ingot/onnx"

m, _ := onnx.DecodeFile("model.onnx")
g, _ := graph.FromONNX(m)
sess, _ := graph.Compile(g)          // errors loudly on any unsupported op
out, _ := sess.Run(map[string]*tensor.Tensor{"input": x})
```

See `CLAUDE.md` for architecture and rules, `docs/ROADMAP.md` for status,
`docs/PERF.md` for benchmarks, `docs/GAPS.md` for what is and isn't supported.

```
make test    # CGO_ENABLED=0 go test -race ./...
make bench   # kernel benchmarks (GFLOPS reported)
make lint    # gofmt, go vet, tidy go.mod, golangci-lint — what CI gates on
make hooks   # once per clone: reject non-conventional commit messages
```

Commits follow [Conventional Commits](https://www.conventionalcommits.org):
`type(scope): description`, e.g. `perf(gemm): 8x12 AVX-512 micro-kernel`. CI checks
every new commit and PR title.

## Install

```
go get github.com/giraffesyo/ingot@latest
```

Prebuilt static binaries of the commands (`ocr`, `onnxrun`, `qwenimage`, …) for
Linux, macOS and Windows on amd64/arm64 are attached to each
[release](https://github.com/giraffesyo/ingot/releases), with `checksums.txt` and a
build provenance attestation (`gh attestation verify <archive> -R giraffesyo/ingot`).
Pushing a `vX.Y.Z` tag on `canary` cuts one.

An [optional offline QwenImage UI](cmd/qwenimage-ui/README.md) provides a local browser
workspace with a stage-batched image queue. Run `go run ./cmd/qwenimage-ui` from the
repository root. The Go workspace connects its separate module; other executables
do not link its UI or unused batching functions.

Status: pure-Go ONNX CPU runtime with ~40 ops, ORT-verified on 12 models; near-peak
GEMM (arm64 NEON ~120 GFLOPS/core 1T; amd64 AVX2 verified on Zen4). Not yet: int8/
int4 quantization, control flow (If/Loop/Scan), GPU. See `docs/GAPS.md`.

## License

[Apache License 2.0](LICENSE). Model weights are not part of this repository and
carry their own licenses.
