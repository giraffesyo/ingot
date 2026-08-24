//go:build !arm64

package gemm

// qkernel: portable fallback (amd64 VNNI/VPMADDUBSW kernels are TODO).
var qkernel = qkernelGeneric

func qkernelGeneric(kg int64, ap *uint8, bp *int8, ct *int32) {
	a := (*[1 << 28]uint8)(unsafePointer(ap))[: kg*qMR*qKG : kg*qMR*qKG]
	b := (*[1 << 28]int8)(unsafePointer(bp))[: kg*qNR*qKG : kg*qNR*qKG]
	c := (*[qMR * qNR]int32)(unsafePointer(ct))
	clear(c[:])
	for g := int64(0); g < kg; g++ {
		ag := a[g*qMR*qKG:]
		bg := b[g*qNR*qKG:]
		for r := 0; r < qMR; r++ {
			for j := 0; j < qNR; j++ {
				var s int32
				for o := 0; o < qKG; o++ {
					s += int32(ag[r*qKG+o]) * int32(bg[j*qKG+o])
				}
				// 2×2-block layout: reg = (r/2)*6 + j/2, lane = (r%2)*2 + j%2
				c[((r>>1)*6+j>>1)*4+(r&1)<<1+j&1] += s
			}
		}
	}
}

// qkernelS8: portable fallback.
var qkernelS8 = qkernelS8Generic

func qkernelS8Generic(kg int64, ap *int8, bp *int8, ct *int32) {
	a := (*[1 << 28]int8)(unsafePointer(ap))[: kg*qMR*qKG : kg*qMR*qKG]
	b := (*[1 << 28]int8)(unsafePointer(bp))[: kg*qNR*qKG : kg*qNR*qKG]
	c := (*[qMR * qNR]int32)(unsafePointer(ct))
	clear(c[:])
	for g := int64(0); g < kg; g++ {
		ag := a[g*qMR*qKG:]
		bg := b[g*qNR*qKG:]
		for r := 0; r < qMR; r++ {
			for j := 0; j < qNR; j++ {
				var s int32
				for o := 0; o < qKG; o++ {
					s += int32(ag[r*qKG+o]) * int32(bg[j*qKG+o])
				}
				c[((r>>1)*6+j>>1)*4+(r&1)<<1+j&1] += s
			}
		}
	}
}
