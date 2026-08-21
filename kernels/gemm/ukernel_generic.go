package gemm

// microKernel computes a MR×NR tile:
//
//	C[MR×NR] (+)= Ap[kc×MR]^T · Bp[kc×NR]
//
// where Ap is packed so that for each p, Ap[p*MR : p*MR+MR] are the MR values of
// column p of the A panel, and Bp[p*NR : p*NR+NR] are the NR values of row p of
// the B panel. c is row-major with leading dimension ldc. If accumulate is false,
// C is overwritten.
//
// This is the portable fallback; an asm version replaces it via the function
// variable so callers are unchanged.
var microKernel = microKernelGeneric

func microKernelGeneric(kc int, ap, bp []float32, c []float32, ldc int, accumulate bool) {
	var (
		c00, c01, c02, c03 float32
		c10, c11, c12, c13 float32
		c20, c21, c22, c23 float32
		c30, c31, c32, c33 float32
	)
	ap = ap[:kc*MR]
	bp = bp[:kc*NR]
	for p := 0; p < kc; p++ {
		a := ap[p*MR : p*MR+MR : p*MR+MR]
		b := bp[p*NR : p*NR+NR : p*NR+NR]
		a0, a1, a2, a3 := a[0], a[1], a[2], a[3]
		b0, b1, b2, b3 := b[0], b[1], b[2], b[3]
		c00 += a0 * b0
		c01 += a0 * b1
		c02 += a0 * b2
		c03 += a0 * b3
		c10 += a1 * b0
		c11 += a1 * b1
		c12 += a1 * b2
		c13 += a1 * b3
		c20 += a2 * b0
		c21 += a2 * b1
		c22 += a2 * b2
		c23 += a2 * b3
		c30 += a3 * b0
		c31 += a3 * b1
		c32 += a3 * b2
		c33 += a3 * b3
	}
	r0 := c[0*ldc : 0*ldc+NR : 0*ldc+NR]
	r1 := c[1*ldc : 1*ldc+NR : 1*ldc+NR]
	r2 := c[2*ldc : 2*ldc+NR : 2*ldc+NR]
	r3 := c[3*ldc : 3*ldc+NR : 3*ldc+NR]
	if accumulate {
		r0[0] += c00
		r0[1] += c01
		r0[2] += c02
		r0[3] += c03
		r1[0] += c10
		r1[1] += c11
		r1[2] += c12
		r1[3] += c13
		r2[0] += c20
		r2[1] += c21
		r2[2] += c22
		r2[3] += c23
		r3[0] += c30
		r3[1] += c31
		r3[2] += c32
		r3[3] += c33
	} else {
		r0[0], r0[1], r0[2], r0[3] = c00, c01, c02, c03
		r1[0], r1[1], r1[2], r1[3] = c10, c11, c12, c13
		r2[0], r2[1], r2[2], r2[3] = c20, c21, c22, c23
		r3[0], r3[1], r3[2], r3[3] = c30, c31, c32, c33
	}
}
