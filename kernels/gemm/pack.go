package gemm

// packA packs an mc×kc block of A (row-major, stride lda) into panels of MR rows:
// for each panel, for each p in [0,kc): MR consecutive values a[i+0..MR-1][p].
// Rows beyond mc in the last panel are zero-filled.
func packA(mc, kc int, a []float32, lda int, dst []float32) {
	di := 0
	for i := 0; i < mc; i += MR {
		rows := min(MR, mc-i)
		for p := 0; p < kc; p++ {
			r := 0
			for ; r < rows; r++ {
				dst[di] = a[(i+r)*lda+p]
				di++
			}
			for ; r < MR; r++ {
				dst[di] = 0
				di++
			}
		}
	}
}

// packB packs a kc×nc block of B (row-major, stride ldb) into panels of NR
// columns: for each panel, for each p in [0,kc): NR consecutive values b[p][j+0..NR-1].
// Columns beyond nc in the last panel are zero-filled.
func packB(kc, nc int, b []float32, ldb int, dst []float32) {
	di := 0
	for j := 0; j < nc; j += NR {
		cols := min(NR, nc-j)
		for p := 0; p < kc; p++ {
			row := b[p*ldb+j : p*ldb+j+cols]
			c := 0
			for ; c < cols; c++ {
				dst[di] = row[c]
				di++
			}
			for ; c < NR; c++ {
				dst[di] = 0
				di++
			}
		}
	}
}
