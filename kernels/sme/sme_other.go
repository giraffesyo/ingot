//go:build !arm64

package sme

// Available reports whether the CPU exposes SME2 (never on this platform).
func Available() bool { return false }

// Assembly stubs: unreachable, since Available() is false and every entry
// point either checks it or is only invoked behind the gemm dispatch, which
// requires Available().
func svl() int64                                        { return 0 }
func probePeak(int64, *float32)                         {}
func probeBF16Peak(int64, *uint16)                      {}
func probeLoad(int64, *float32, *float32)               {}
func probeN(int64, *float32, int64)                     {}
func outerK(*float32, *float32, int64, *float32)        {}
func zakernel(kc int64, ap, bp, c *float32, ldc4 int64) { panic("sme: not available") }

func qzakernel(kg int64, ap, bp *int8, c *int32, ldc4 int64) { panic("sme: not available") }
