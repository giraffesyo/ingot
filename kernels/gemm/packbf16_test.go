package gemm

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"
)

// TestPackBBF16 checks the bf16 packer is bit-identical to PackB over the
// widened matrix, for both storage orders and ragged k/n (partial KC blocks
// and NR panels).
func TestPackBBF16(t *testing.T) {
	r := rand.New(rand.NewPCG(71, 72))
	for _, sh := range []struct{ k, n int }{{1, 1}, {7, 5}, {KC + 3, 2*NR + 1}, {2*KC - 1, NR}, {300, 97}} {
		for _, transB := range []bool{false, true} {
			rows, cols := sh.k, sh.n
			if transB {
				rows, cols = sh.n, sh.k
			}
			ldb := cols + 3 // exercise a row stride wider than the matrix
			raw := make([]uint16, rows*ldb)
			wide := make([]float32, rows*ldb)
			for i := range raw {
				raw[i] = uint16(math.Float32bits(float32(r.NormFloat64())) >> 16)
				wide[i] = math.Float32frombits(uint32(raw[i]) << 16)
			}
			got := PackBBF16(transB, sh.k, sh.n, raw, ldb)
			want := PackB(transB, sh.k, sh.n, wide, ldb)
			if len(got.data) != len(want.data) {
				t.Fatalf("k=%d n=%d transB=%v: %d packed, want %d", sh.k, sh.n, transB, len(got.data), len(want.data))
			}
			for i := range want.data {
				if math.Float32bits(got.data[i]) != math.Float32bits(want.data[i]) {
					t.Fatalf("k=%d n=%d transB=%v: data[%d] = %g, want %g", sh.k, sh.n, transB, i, got.data[i], want.data[i])
				}
			}
		}
	}
}

// BenchmarkPackBBF16 times packing a DiT projection ([out, in] = 4096², as
// stored by PyTorch, transB) from bf16.
func BenchmarkPackBBF16(b *testing.B) {
	for _, s := range []struct{ k, n int }{{4096, 4096}, {4096, 12288}} {
		raw := make([]uint16, s.k*s.n)
		for i := range raw {
			raw[i] = uint16(0x3f80 + i%64)
		}
		b.Run(fmt.Sprintf("k=%d/n=%d", s.k, s.n), func(b *testing.B) {
			b.SetBytes(int64(2 * len(raw)))
			for i := 0; i < b.N; i++ {
				PackBBF16(true, s.k, s.n, raw, s.k)
			}
		})
	}
}
