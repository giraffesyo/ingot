//go:build arm64

package sme

func svl() int64
func probePeak(iters int64, src *float32)
func probeLoad(iters int64, a, b *float32)
func probeN(iters int64, src *float32, variant int64)
func outerK(a, b *float32, k int64, out *float32)
func zakernel(kc int64, ap, bp, c *float32, ldc4 int64)
func qzakernel(kg int64, ap, bp *int8, c *int32, ldc4 int64)
