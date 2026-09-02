package gemm

// packAPanel packs one MR-row panel of A: for each p in [0,kc), MR consecutive
// values a[r][p] for r in [0,rows); rows beyond `rows` are zero-filled.
// a points at the panel's first row; lda is its row stride.
func packAPanel(kc, rows int, a []float32, lda int, dst []float32) {
	dst = dst[:kc*MR]
	if rows == MR {
		// Full panel: MR sequential row streams interleaved into dst. Each row
		// is sliced to exactly kc up front so the inner loads carry no bounds
		// checks (this pack was 11% of a small-M transformer's CPU when it
		// gathered a[r*lda+p] per element).
		var rs [MR][]float32
		for r := 0; r < MR; r++ {
			rs[r] = a[r*lda : r*lda+kc : r*lda+kc]
		}
		p := 0
		for ; p+4 <= kc; p += 4 {
			d := dst[p*MR : p*MR+4*MR : p*MR+4*MR]
			for r := 0; r < MR; r++ {
				row := rs[r][p : p+4 : p+4]
				d[r], d[MR+r], d[2*MR+r], d[3*MR+r] = row[0], row[1], row[2], row[3]
			}
		}
		for ; p < kc; p++ {
			d := dst[p*MR : p*MR+MR : p*MR+MR]
			for r := 0; r < MR; r++ {
				d[r] = rs[r][p]
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

// packAPanelT packs one MR-row panel of op(A)=Aᵀ where A is stored [k×m] with
// row stride lda; a points at column i0 of row 0. For each p, the MR values
// A[p][i0..i0+MR) are contiguous.
func packAPanelT(kc, rows int, a []float32, lda int, dst []float32) {
	dst = dst[:kc*MR]
	if rows == MR {
		for p := 0; p < kc; p++ {
			copy(dst[p*MR:p*MR+MR], a[p*lda:p*lda+MR])
		}
		return
	}
	for p := 0; p < kc; p++ {
		d := dst[p*MR : p*MR+MR : p*MR+MR]
		copy(d, a[p*lda:p*lda+rows])
		clear(d[rows:])
	}
}

// packBPanelT packs one NR-column panel of op(B)=Bᵀ where B is stored [n×k]
// with row stride ldb; b points at row j0. For each p, the NR values
// B[j0..j0+NR)[p] are strided by ldb.
func packBPanelT(kc, cols int, b []float32, ldb int, dst []float32) {
	dst = dst[:kc*NR]
	if cols == NR {
		for p := 0; p < kc; p++ {
			d := dst[p*NR : p*NR+NR : p*NR+NR]
			for c := 0; c < NR; c++ {
				d[c] = b[c*ldb+p]
			}
		}
		return
	}
	for p := 0; p < kc; p++ {
		d := dst[p*NR : p*NR+NR : p*NR+NR]
		c := 0
		for ; c < cols; c++ {
			d[c] = b[c*ldb+p]
		}
		for ; c < NR; c++ {
			d[c] = 0
		}
	}
}
