// Command onnxrun loads an ONNX model and runs it on input tensors.
// Useful for conformance testing against ONNX Runtime.
package main

import (
	"fmt"
	"os"
)

func main() {
	fmt.Fprintln(os.Stderr, "onnxrun: loader not implemented yet (see docs/ROADMAP.md phase 2)")
	os.Exit(2)
}
