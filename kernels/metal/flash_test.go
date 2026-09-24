//go:build darwin && arm64

package metal

import (
	"encoding/binary"
	"math"
	"math/rand/v2"
	"testing"
)

func bf16buf(t *testing.T, d *Device, r *rand.Rand, n int) (*Buffer, []float64) {
	t.Helper()
	b, err := d.NewBuffer(2 * n)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(b.Release)
	v := make([]float64, n)
	raw := b.Bytes()
	for i := range v {
		bits := uint16(math.Float32bits(r.Float32()*2-1) >> 16)
		raw[2*i], raw[2*i+1] = byte(bits), byte(bits>>8)
		v[i] = float64(math.Float32frombits(uint32(bits) << 16))
	}
	return b, v
}

// TestFlash: fused attention over a two-segment key set vs a float64
// oracle (P rounded to bf16 as the kernel does), with ragged query and key
// counts and 2 heads in strided rows.
func TestFlash(t *testing.T) {
	d := openDev(t)
	if err := d.PrepareFlash(); err != nil {
		t.Fatal(err)
	}
	r := rand.New(rand.NewPCG(10, 10))
	const H, dh = 2, 128
	D := H * dh
	for _, c := range []struct {
		tq, n1, n2  int
		blockCausal bool
	}{{64, 0, 64, false}, {100, 31, 150, false}, {70, 5, 3, false}, {150, 150, 0, true}} {
		q, qv := bf16buf(t, d, r, c.tq*D)
		k1, k1v := bf16buf(t, d, r, max(c.n1, 1)*D)
		v1, v1v := bf16buf(t, d, r, max(c.n1, 1)*D)
		k2, k2v := bf16buf(t, d, r, max(c.n2, 1)*D)
		v2, v2v := bf16buf(t, d, r, max(c.n2, 1)*D)
		o := buf(t, d, c.tq*D)
		scale := float32(1 / math.Sqrt(dh))
		// Block-causal keys: rows 0..39 causal text, 40..119 one image block
		// (sees keys [0, 120)), 120.. causal text again.
		kend := make([]int, c.tq)
		var kb *Region
		if c.blockCausal {
			kbuf, _ := d.NewBuffer(4 * c.tq)
			defer kbuf.Release()
			for r := range c.tq {
				kend[r] = r + 1
				if r >= 40 && r < 120 {
					kend[r] = 120
				}
				binary.LittleEndian.PutUint32(kbuf.Bytes()[4*r:], uint32(kend[r]))
			}
			reg := kbuf.At(0)
			kb = &reg
		}
		if err := d.Run(func(e *Encoder) {
			e.Flash(Flash{Q: q.At(0), K1: k1.At(0), V1: v1.At(0), K2: k2.At(0), V2: v2.At(0), O: o.At(0), KeyEnd: kb,
				Tq: c.tq, N1: c.n1, N2: c.n2, Heads: H, LDQ: D, LD1: D, LD2: D, LDO: D, Scale: scale})
		}); err != nil {
			t.Fatal(err)
		}
		of := f32s(o.Bytes())
		n := c.n1 + c.n2
		key := func(j, h, i int) (float64, float64) {
			if j < c.n1 {
				return k1v[j*D+h*dh+i], v1v[j*D+h*dh+i]
			}
			j -= c.n1
			return k2v[j*D+h*dh+i], v2v[j*D+h*dh+i]
		}
		for h := range H {
			for qi := range c.tq {
				nq := n
				if c.blockCausal {
					nq = kend[qi]
				}
				s := make([]float64, nq)
				m := math.Inf(-1)
				for j := range nq {
					for i := range dh {
						kk, _ := key(j, h, i)
						s[j] += qv[qi*D+h*dh+i] * kk
					}
					s[j] *= float64(scale)
					m = math.Max(m, s[j])
				}
				var sum float64
				for j := range s {
					s[j] = math.Exp(s[j] - m)
					sum += s[j]
				}
				for i := range dh {
					var want float64
					for j := range nq {
						_, vv := key(j, h, i)
						want += s[j] * vv
					}
					want /= sum
					if dd := math.Abs(float64(of[qi*D+h*dh+i]) - want); dd > 1e-2*(1+math.Abs(want)) {
						t.Fatalf("%v: O[%d,%d,%d] = %g, want %g", c, qi, h, i, of[qi*D+h*dh+i], want)
					}
				}
			}
		}
	}
}

// BenchmarkAttention compares fused Flash against the per-head GEMM →
// softmax → GEMM chain at a 1024² DiT step's shape (4096 queries, 32
// heads, 31 prefix + 4096 target keys).
func BenchmarkAttention(b *testing.B) {
	d := openDev(b)
	if err := d.PrepareFlash(); err != nil {
		b.Fatal(err)
	}
	const T, P, H, dh = 4096, 31, 32, 128
	D := H * dh
	nb := func(n int) *Buffer {
		x, err := d.NewBuffer(n)
		if err != nil {
			b.Fatal(err)
		}
		return x
	}
	q, kp, vp, kt, vt, o := nb(2*T*D), nb(2*P*D), nb(2*P*D), nb(2*T*D), nb(2*T*D), nb(4*T*D)
	s, s16 := nb(4*T*(P+T)), nb(2*T*(P+T))
	flops := 4 * float64(H) * T * float64(P+T) * dh
	b.Run("flash", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			d.Run(func(e *Encoder) {
				e.Flash(Flash{Q: q.At(0), K1: kp.At(0), V1: vp.At(0), K2: kt.At(0), V2: vt.At(0), O: o.At(0),
					Tq: T, N1: P, N2: T, Heads: H, LDQ: D, LD1: D, LD2: D, LDO: D, Scale: 0.088})
			})
		}
		b.ReportMetric(flops*float64(b.N)/b.Elapsed().Seconds()/1e12, "TFLOPS")
	})
	b.Run("unfused", func(b *testing.B) {
		SW := P + T
		for i := 0; i < b.N; i++ {
			d.Run(func(e *Encoder) {
				for h := range H {
					h2 := 2 * h * dh
					e.Gemm(Gemm{M: T, N: P, K: dh, A: q.At(h2), LDA: D, B: kp.At(h2), LDB: D, C: s.At(0), LDC: SW, TransB: true, BF16: true, ABF16: true})
					e.Gemm(Gemm{M: T, N: T, K: dh, A: q.At(h2), LDA: D, B: kt.At(h2), LDB: D, C: s.At(4 * P), LDC: SW, TransB: true, BF16: true, ABF16: true})
					e.SoftmaxRowsBF16(s.At(0), s16.At(0), T, SW, SW, SW, 0.088)
					e.Gemm(Gemm{M: T, N: dh, K: P, A: s16.At(0), LDA: SW, B: vp.At(h2), LDB: D, C: o.At(4 * h * dh), LDC: D, BF16: true, ABF16: true})
					e.Gemm(Gemm{M: T, N: dh, K: T, A: s16.At(2 * P), LDA: SW, B: vt.At(h2), LDB: D, C: o.At(4 * h * dh), LDC: D, BF16: true, ABF16: true, Accumulate: true})
				}
			})
		}
		b.ReportMetric(flops*float64(b.N)/b.Elapsed().Seconds()/1e12, "TFLOPS")
	})
}
