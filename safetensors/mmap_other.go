//go:build !unix

package safetensors

// mapFile reads the whole file where mmap is unavailable (windows, wasm).
func mapFile(path string) ([]byte, func() error, error) { return readFile(path) }

// willNeed is a no-op where the file was read, not mapped.
func willNeed([]byte) {}
