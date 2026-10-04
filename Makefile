export CGO_ENABLED=0
PKG ?= ./...
BENCH ?= .

.PHONY: test bench lint prof vet fmt corpus vuln hooks dist

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

# What CI gates on: gofmt, a tidy go.mod, vet, golangci-lint (.golangci.yml:
# staticcheck, errcheck, unused, ...).
lint: vet
	@out=$$(gofmt -l .); [ -z "$$out" ] || { echo "not gofmt-clean:"; echo "$$out"; exit 1; }
	go mod tidy -diff
	@command -v golangci-lint >/dev/null || { echo "golangci-lint not installed (https://golangci-lint.run/docs/welcome/install/)"; exit 1; }
	golangci-lint run ./...

vuln:
	go run golang.org/x/vuln/cmd/govulncheck@v1.8.0 ./...

# Reject commit messages that are not Conventional Commits (tools/commitlint.sh).
hooks:
	git config core.hooksPath .githooks

# Release archives for every target into dist/ (what the Release workflow runs).
VERSION ?= $(shell git describe --tags --always --dirty)
dist:
	tools/dist.sh $(VERSION)

corpus:
	cd tools/export && .venv/bin/python corpus.py

prof:
	go test -run=^$$ -bench=$(BENCH) -cpuprofile=cpu.prof $(PKG) && go tool pprof -top cpu.prof | head -40
