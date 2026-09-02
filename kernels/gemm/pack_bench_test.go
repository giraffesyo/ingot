package gemm

import "testing"

// BenchmarkPackAPanel: one full MR-row A panel at KC depth (the small-M
// path packs all of A this way before every sweep).
func BenchmarkPackAPanel(b *testing.B) {
	const lda = 1536
	a := make([]float32, MR*lda)
	for i := range a {
		a[i] = float32(i)
	}
	dst := make([]float32, KC*MR)
	b.SetBytes(int64(KC * MR * 4))
	for i := 0; i < b.N; i++ {
		packAPanel(KC, MR, a, lda, dst)
	}
}
