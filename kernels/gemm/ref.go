package gemm

// Ref is the reference implementation. Slow; used as the oracle in tests.
func Ref(m, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int) {
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var acc float32
			for p := 0; p < k; p++ {
				acc += a[i*lda+p] * b[p*ldb+j]
			}
			if beta == 0 {
				c[i*ldc+j] = alpha * acc
			} else {
				c[i*ldc+j] = alpha*acc + beta*c[i*ldc+j]
			}
		}
	}
}
