//go:build darwin && arm64

package metal

import (
	"math"
	"math/rand/v2"
	"testing"
	"unsafe"
)

func decodeDev(t *testing.T) *Device {
	t.Helper()
	d := prepared(t)
	if err := d.PrepareDecode(); err != nil {
		t.Fatal(err)
	}
	return d
}

// TestGemv: bf16 and int8 weights, 1–4 rows, strided operands, bias and
// accumulate, against a float64 oracle over the same (dequantised) weights.
func TestGemv(t *testing.T) {
	d := decodeDev(t)
	r := rand.New(rand.NewPCG(31, 32))
	const N, K, ldx, ldy, ldw = 70, 320, 330, 75, 336
	for _, bf16 := range []bool{true, false} {
		for _, rows := range []int{1, 3, 4} {
			x, y, bias := buf(t, d, rows*ldx), buf(t, d, rows*ldy), buf(t, d, N)
			xf, bf := fill(r, x), fill(r, bias)
			yf := fill(r, y)
			y0 := append([]float32(nil), yf...)
			wide := make([]float64, N*K) // dequantised weights
			var w, s *Buffer
			if bf16 {
				w = buf(t, d, (N*ldw+1)/2)
				wb := unsafe.Slice((*uint16)(unsafe.Pointer(&w.Bytes()[0])), N*ldw)
				for n := range N {
					for k := range K {
						wb[n*ldw+k] = uint16(math.Float32bits(r.Float32()*2-1) >> 16)
						wide[n*K+k] = float64(math.Float32frombits(uint32(wb[n*ldw+k]) << 16))
					}
				}
			} else {
				w, s = buf(t, d, (N*ldw+3)/4), buf(t, d, N*K/64)
				wq := unsafe.Slice((*int8)(unsafe.Pointer(&w.Bytes()[0])), N*ldw)
				sf := f32s(s.Bytes())
				for i := range sf {
					sf[i] = r.Float32() * 0.01
				}
				for n := range N {
					for k := range K {
						wq[n*ldw+k] = int8(r.IntN(255) - 127)
						wide[n*K+k] = float64(sf[n*K/64+k/64]) * float64(wq[n*ldw+k])
					}
				}
			}
			g := Gemv{N: N, K: K, Rows: rows, X: x.At(0), W: w.At(0), Y: y.At(0), Bias: bias.At(0),
				LDX: ldx, LDY: ldy, LDW: ldw, BF16: bf16, Accumulate: true, HasBias: true}
			if s != nil {
				g.S = s.At(0)
			}
			if err := d.Run(func(e *Encoder) { e.Gemv(g) }); err != nil {
				t.Fatal(err)
			}
			for rr := range rows {
				for n := range N {
					want := float64(y0[rr*ldy+n]) + float64(bf[n])
					for k := range K {
						want += float64(xf[rr*ldx+k]) * wide[n*K+k]
					}
					near(t, "gemv", yf[rr*ldy+n], want, 1e-5*math.Sqrt(K))
				}
			}
		}
	}
}

// TestAttnDecode: causal grouped-query attention of several new queries
// over a cache, against a float64 oracle — prefill-style (pos0 = 0, many
// queries) and decode-style (pos0 > 0, one query), dh 64 and 128.
func TestAttnDecode(t *testing.T) {
	d := decodeDev(t)
	r := rand.New(rand.NewPCG(33, 34))
	for _, c := range []struct{ H, KV, dh, pos0, T int }{{4, 2, 64, 0, 13}, {16, 8, 128, 37, 1}, {8, 8, 64, 300, 2}} {
		n := c.pos0 + c.T
		q, out := buf(t, d, c.T*c.H*c.dh), buf(t, d, c.T*c.H*c.dh)
		kc, vc := buf(t, d, n*c.KV*c.dh), buf(t, d, n*c.KV*c.dh)
		qf, kf, vf := fill(r, q), fill(r, kc), fill(r, vc)
		scale := float32(1 / math.Sqrt(float64(c.dh)))
		if err := d.Run(func(e *Encoder) {
			e.AttnDecode(q.At(0), kc.At(0), vc.At(0), out.At(0), c.T, c.H, c.KV, c.dh, c.pos0, 0, 0, scale)
		}); err != nil {
			t.Fatal(err)
		}
		of := f32s(out.Bytes())
		ld := c.KV * c.dh
		for tt := range c.T {
			for h := range c.H {
				kvh := h / (c.H / c.KV)
				keys := c.pos0 + tt + 1
				s := make([]float64, keys)
				m := math.Inf(-1)
				for j := range keys {
					for i := range c.dh {
						s[j] += float64(qf[tt*c.H*c.dh+h*c.dh+i]) * float64(kf[j*ld+kvh*c.dh+i])
					}
					s[j] *= float64(scale)
					m = math.Max(m, s[j])
				}
				var sum float64
				for j := range s {
					s[j] = math.Exp(s[j] - m)
					sum += s[j]
				}
				for i := range c.dh {
					var want float64
					for j := range keys {
						want += s[j] / sum * float64(vf[j*ld+kvh*c.dh+i])
					}
					near(t, "attn", of[tt*c.H*c.dh+h*c.dh+i], want, 1e-5)
				}
			}
		}
	}
}

// TestSampleLogits: argmax ties take the first index; sampled tokens fall
// in the top-k (ties at the k-th value kept) and the draw's cumulative
// interval (index order, float64 oracle) contains u, up to rounding.
func TestSampleLogits(t *testing.T) {
	d := decodeDev(t)
	r := rand.New(rand.NewPCG(35, 36))
	const V = 2048
	lg, codes := buf(t, d, V), buf(t, d, 64)
	lf := f32s(lg.Bytes())
	cf := unsafe.Slice((*uint32)(unsafe.Pointer(&codes.Bytes()[0])), 64)
	for i := range lf {
		lf[i] = r.Float32()*8 - 4
	}
	lf[700], lf[1500] = 9, 9 // tie for the maximum
	lf[11], lf[12] = 5, 5    // ties around the k-th value region
	if err := d.Run(func(e *Encoder) { e.SampleLogits(lg.At(0), codes.At(0), V, 0, true, 1, 0, 3) }); err != nil {
		t.Fatal(err)
	}
	if cf[3] != 700 {
		t.Fatalf("argmax = %d, want 700 (first of the tied maxima)", cf[3])
	}
	for trial, c := range []struct {
		k    int
		temp float32
	}{{50, 0.9}, {0, 1}, {1, 1}, {7, 0.5}} {
		for draw := range 20 {
			u := r.Float32()
			if err := d.Run(func(e *Encoder) { e.SampleLogits(lg.At(0), codes.At(0), V, c.k, false, c.temp, u, draw) }); err != nil {
				t.Fatal(err)
			}
			got := int(cf[draw])
			// float64 oracle over the same scaled logits.
			x := make([]float64, V)
			for i := range x {
				x[i] = float64(lf[i] / c.temp)
			}
			kth := math.Inf(-1)
			if c.k > 0 {
				sorted := append([]float64(nil), x...)
				for i := range sorted { // partial selection sort of the top k
					for j := i + 1; j < len(sorted) && i < c.k; j++ {
						if sorted[j] > sorted[i] {
							sorted[i], sorted[j] = sorted[j], sorted[i]
						}
					}
					if i >= c.k {
						break
					}
				}
				kth = sorted[c.k-1]
			}
			m := math.Inf(-1)
			for _, v := range x {
				if v >= kth {
					m = math.Max(m, v)
				}
			}
			var total float64
			for _, v := range x {
				if v >= kth {
					total += math.Exp(v - m)
				}
			}
			if x[got] < kth {
				t.Fatalf("trial %d: token %d outside the top %d", trial, got, c.k)
			}
			var before float64
			for i := range got {
				if x[i] >= kth {
					before += math.Exp(x[i] - m)
				}
			}
			after := before + math.Exp(x[got]-m)
			target := float64(u) * total
			if slack := 1e-4 * total; target < before-slack || target >= after+slack {
				t.Fatalf("trial %d u=%g: token %d covers [%g, %g), target %g", trial, u, got, before, after, target)
			}
		}
	}
}

// TestEmbedRow: the row a code selects, widened from bf16.
func TestEmbedRow(t *testing.T) {
	d := decodeDev(t)
	const rows, D = 9, 300
	tab, codes, dst := buf(t, d, rows*D/2), buf(t, d, 4), buf(t, d, D)
	tb := unsafe.Slice((*uint16)(unsafe.Pointer(&tab.Bytes()[0])), rows*D)
	for i := range tb {
		tb[i] = uint16(math.Float32bits(float32(i)*0.5) >> 16)
	}
	unsafe.Slice((*uint32)(unsafe.Pointer(&codes.Bytes()[0])), 4)[2] = 7
	if err := d.Run(func(e *Encoder) { e.EmbedRow(tab.At(0), codes.At(0), dst.At(0), D, 2) }); err != nil {
		t.Fatal(err)
	}
	for c, v := range f32s(dst.Bytes()) {
		if want := math.Float32frombits(uint32(tb[7*D+c]) << 16); v != want {
			t.Fatalf("dst[%d] = %g want %g", c, v, want)
		}
	}
}
