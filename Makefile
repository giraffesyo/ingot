export CGO_ENABLED=0
PKG ?= ./...
BENCH ?= .

.PHONY: test bench lint prof vet fmt

test:
	go test -race -count=1 $(PKG)

bench:
	go test -run=^$$ -bench=$(BENCH) -benchmem $(PKG)

vet:
	go vet ./...

fmt:
	gofmt -l -w .

lint: vet
	@command -v staticcheck >/dev/null && staticcheck ./... || echo "staticcheck not installed (go install honnef.co/go/tools/cmd/staticcheck@latest)"

prof:
	go test -run=^$$ -bench=$(BENCH) -cpuprofile=cpu.prof $(PKG) && go tool pprof -top cpu.prof | head -40
