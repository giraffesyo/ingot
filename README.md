# ocr

SOTA OCR in pure Go, on a pure-Go ONNX CPU inference runtime. No cgo.

See `CLAUDE.md` for architecture and rules, `docs/ROADMAP.md` for status.

```
make test    # CGO_ENABLED=0 go test -race ./...
make bench   # kernel benchmarks (GFLOPS reported)
```
