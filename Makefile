export CGO_ENABLED=0
PKG ?= ./...
BENCH ?= .

.PHONY: test bench lint prof vet fmt corpus

# The race detector needs cgo except on darwin; CI is CGO_ENABLED=0, so
# -race runs only where it works without cgo.
RACE ?= $(if $(filter darwin,$(shell go env GOOS)),-race,)

test:
	go test $(RACE) -count=1 $(PKG)

bench:
	go test -run=^$$ -bench=$(BENCH) -benchmem $(PKG)

vet:
	go vet ./...

fmt:
	gofmt -l -w .

lint: vet
	@command -v staticcheck >/dev/null && staticcheck ./... || echo "staticcheck not installed (go install honnef.co/go/tools/cmd/staticcheck@latest)"

corpus:
	cd tools/export && .venv/bin/python corpus.py

prof:
	go test -run=^$$ -bench=$(BENCH) -cpuprofile=cpu.prof $(PKG) && go tool pprof -top cpu.prof | head -40
