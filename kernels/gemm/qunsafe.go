package gemm

import "unsafe"

func unsafePointer[T any](p *T) unsafe.Pointer { return unsafe.Pointer(p) }

func unsafeSliceU8(p *uint8, n int) []uint8 { return unsafe.Slice(p, n) }
func unsafeSliceI8(p *int8, n int) []int8   { return unsafe.Slice(p, n) }
func unsafeSliceI32(p *int32, n int) []int32 {
	return unsafe.Slice(p, n)
}

func f32AsI32(p *float32) *int32 { return (*int32)(unsafe.Pointer(p)) }
