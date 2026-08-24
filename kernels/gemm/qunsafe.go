//go:build !arm64

package gemm

import "unsafe"

func unsafePointer[T any](p *T) unsafe.Pointer { return unsafe.Pointer(p) }
