package gemm

// packAPanel packs one MR-row panel of A: for each p in [0,kc), MR consecutive
// values a[r][p] for r in [0,rows); rows beyond `rows` are zero-filled.
// a points at the panel's first row; lda is its row stride.
func packAPanel(kc, rows int, a []float32, lda int, dst []float32) {
	dst = dst[:kc*MR]
	if rows == MR {
		for p := 0; p < kc; p++ {
			d := dst[p*MR : p*MR+MR : p*MR+MR]
			for r := 0; r < MR; r++ {
				d[r] = a[r*lda+p]
			}
		}
		return
	}
	for p := 0; p < kc; p++ {
		d := dst[p*MR : p*MR+MR : p*MR+MR]
		r := 0
		for ; r < rows; r++ {
			d[r] = a[r*lda+p]
		}
		for ; r < MR; r++ {
			d[r] = 0
		}
	}
}

// packBPanel packs one NR-column panel of B: for each p in [0,kc), NR
// consecutive values b[p][c] for c in [0,cols); cols beyond `cols` are
// zero-filled. b points at the panel's first column; ldb is its row stride.
func packBPanel(kc, cols int, b []float32, ldb int, dst []float32) {
	dst = dst[:kc*NR]
	if cols == NR {
		for p := 0; p < kc; p++ {
			copy(dst[p*NR:p*NR+NR], b[p*ldb:p*ldb+NR])
		}
		return
	}
	for p := 0; p < kc; p++ {
		d := dst[p*NR : p*NR+NR : p*NR+NR]
		copy(d, b[p*ldb:p*ldb+cols])
		clear(d[cols:])
	}
}
