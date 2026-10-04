export CGO_ENABLED=0

.PHONY: build run test vet

build:
	go build -trimpath -ldflags="-s -w" -o qwenimage-ui .

run:
	go run . $(ARGS)

test:
	go test ./...
	cd ../.. && go test -run 'TestBatch|TestGenerateBatch' ./models/qwenimage

vet:
	go vet ./...
