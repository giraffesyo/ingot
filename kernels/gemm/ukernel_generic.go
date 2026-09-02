package gemm

// microKernelGeneric computes a MR×NR tile:
//
//	C[MR×NR] (+)= Ap[kc×MR]^T · Bp[kc×NR]
//
// where Ap is packed so that for each p, Ap[p*MR : p*MR+MR] are the MR values of
// column p of the A panel, and Bp[p*NR : p*NR+NR] are the NR values of row p of
// the B panel. c is row-major with leading dimension ldc. If accumulate is false,
// C is overwritten. bias (nil = none) holds NR values added to every row after
// C — the fused Linear bias; order (C + sum) + bias everywhere.
//
// This is the portable fallback and the oracle for the asm kernels. It is
// written over the arch's MR/NR constants so it stays in sync.
func microKernelGeneric(kc int, ap, bp []float32, c []float32, ldc int, accumulate bool, bias []float32) {
	var acc [MR][NR]float32
	ap = ap[:kc*MR]
	bp = bp[:kc*NR]
	for p := 0; p < kc; p++ {
		a := ap[p*MR : p*MR+MR : p*MR+MR]
		b := bp[p*NR : p*NR+NR : p*NR+NR]
		for i := 0; i < MR; i++ {
			ai := a[i]
			row := &acc[i]
			for j := 0; j < NR; j++ {
				row[j] += ai * b[j]
			}
		}
	}
	for i := 0; i < MR; i++ {
		row := c[i*ldc : i*ldc+NR : i*ldc+NR]
		if accumulate {
			for j := 0; j < NR; j++ {
				row[j] += acc[i][j]
			}
		} else {
			copy(row, acc[i][:])
		}
		if bias != nil {
			b := bias[:NR]
			for j := range row {
				row[j] += b[j]
			}
		}
	}
}
