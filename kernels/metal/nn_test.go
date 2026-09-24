//go:build darwin && arm64

package metal

import (
	"encoding/binary"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"syscall"
	"testing"
	"unsafe"
)

func prepared(t *testing.T) *Device {
	t.Helper()
	d := openDev(t)
	if err := d.Prepare(); err != nil {
		t.Fatal(err)
	}
	return d
}

func fill(r *rand.Rand, b *Buffer) []float32 {
	f := f32s(b.Bytes())
	for i := range f {
		f[i] = r.Float32()*4 - 2
	}
	return f
}

func buf(t *testing.T, d *Device, floats int) *Buffer {
	t.Helper()
	b, err := d.NewBuffer(4 * floats)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(b.Release)
	return b
}

func near(t *testing.T, what string, got float32, want, tol float64) {
	t.Helper()
	if d := math.Abs(float64(got) - want); d > tol*(1+math.Abs(want)) || math.IsNaN(float64(got)) {
		t.Fatalf("%s = %g, want %g", what, got, want)
	}
}

func TestLayerNormMod(t *testing.T) {
	d := prepared(t)
	r := rand.New(rand.NewPCG(1, 1))
	const rows, cols, ldx, ldy = 5, 4099, 4100, 4105
	x, y, s := buf(t, d, rows*ldx), buf(t, d, rows*ldy), buf(t, d, cols)
	xf, sf := fill(r, x), fill(r, s)
	if err := d.Run(func(e *Encoder) { e.LayerNormMod(x.At(0), y.At(0), s.At(0), rows, cols, ldx, ldy, 1e-6) }); err != nil {
		t.Fatal(err)
	}
	yf := f32s(y.Bytes())
	for rr := range rows {
		var mean, vs float64
		for c := range cols {
			mean += float64(xf[rr*ldx+c])
		}
		mean /= cols
		for c := range cols {
			vs += (float64(xf[rr*ldx+c]) - mean) * (float64(xf[rr*ldx+c]) - mean)
		}
		inv := 1 / math.Sqrt(vs/cols+1e-6)
		for c := range cols {
			near(t, "ln", yf[rr*ldy+c], (float64(xf[rr*ldx+c])-mean)*inv*float64(sf[c]), 1e-4)
		}
	}
}

func TestRMSNormRoPE(t *testing.T) {
	d := prepared(t)
	r := rand.New(rand.NewPCG(2, 2))
	const T, heads, dh, ld, off = 3, 4, 128, 1000, 7 // heads start at column 7
	x, w, cs, sn := buf(t, d, T*ld), buf(t, d, dh), buf(t, d, T*dh/2), buf(t, d, T*dh/2)
	xf, wf, cf, sf := fill(r, x), fill(r, w), fill(r, cs), fill(r, sn)
	orig := append([]float32(nil), xf...)
	if err := d.Run(func(e *Encoder) {
		e.RMSNormRoPE(x.At(4*off), w.At(0), cs.At(0), sn.At(0), T, heads, dh, ld, 1e-6)
	}); err != nil {
		t.Fatal(err)
	}
	for tt := range T {
		for h := range heads {
			base := tt*ld + off + h*dh
			var ss float64
			for i := range dh {
				ss += float64(orig[base+i]) * float64(orig[base+i])
			}
			inv := 1 / math.Sqrt(ss/dh+1e-6)
			for j := range dh / 2 {
				re := float64(orig[base+2*j]) * inv * float64(wf[2*j])
				im := float64(orig[base+2*j+1]) * inv * float64(wf[2*j+1])
				c, s := float64(cf[tt*dh/2+j]), float64(sf[tt*dh/2+j])
				near(t, "rope re", xf[base+2*j], re*c-im*s, 1e-4)
				near(t, "rope im", xf[base+2*j+1], re*s+im*c, 1e-4)
			}
		}
	}
	if xf[off-1] != orig[off-1] || xf[off+heads*dh] != orig[off+heads*dh] {
		t.Fatal("RMSNormRoPE wrote outside its heads")
	}
}

func TestSoftmaxRows(t *testing.T) {
	d := prepared(t)
	r := rand.New(rand.NewPCG(3, 3))
	const rows, cols, ld = 4, 8451, 8460
	x := buf(t, d, rows*ld)
	xf := fill(r, x)
	orig := append([]float32(nil), xf...)
	if err := d.Run(func(e *Encoder) { e.SoftmaxRows(x.At(0), rows, cols, ld, 3) }); err != nil {
		t.Fatal(err)
	}
	for rr := range rows {
		m := math.Inf(-1)
		for c := range cols {
			m = math.Max(m, 3*float64(orig[rr*ld+c]))
		}
		var sum float64
		for c := range cols {
			sum += math.Exp(3*float64(orig[rr*ld+c]) - m)
		}
		for c := range cols {
			near(t, "softmax", xf[rr*ld+c], math.Exp(3*float64(orig[rr*ld+c])-m)/sum, 1e-4)
		}
	}
}

func TestSiLUMulGatedAdd(t *testing.T) {
	d := prepared(t)
	r := rand.New(rand.NewPCG(4, 4))
	const rows, cols, ld = 3, 1000, 2 * 1000 // a and b interleaved as [gate | proj] halves
	ab, o := buf(t, d, rows*ld), buf(t, d, rows*cols)
	x, g := buf(t, d, rows*cols), buf(t, d, cols)
	abf, xf, gf := fill(r, ab), fill(r, x), fill(r, g)
	x0 := append([]float32(nil), xf...)
	if err := d.Run(func(e *Encoder) {
		e.SiLUMul(ab.At(0), ab.At(4*cols), o.At(0), rows, cols, ld, ld, cols)
		e.GatedAdd(x.At(0), g.At(0), o.At(0), rows, cols, cols, cols)
	}); err != nil {
		t.Fatal(err)
	}
	for rr := range rows {
		for c := range cols {
			a, b := float64(abf[rr*ld+c]), float64(abf[rr*ld+cols+c])
			y := a / (1 + math.Exp(-a)) * b
			near(t, "gated add", xf[rr*cols+c], float64(x0[rr*cols+c])+float64(gf[c])*y, 1e-5)
		}
	}
}

// TestGemmStrided runs one attention head the way the DiT step does: S =
// Q_h·K_hᵀ over head-slices of [T, H·dh] buffers (NT, strided, offset), then
// O_h = S·V_h (NN) written into a head slice of the output.
func TestGemmStrided(t *testing.T) {
	d := prepared(t)
	r := rand.New(rand.NewPCG(5, 5))
	const Tq, Tk, H, dh, h = 70, 150, 3, 64, 1
	D := H * dh
	q, k, v, o, s := buf(t, d, Tq*D), buf(t, d, Tk*D), buf(t, d, Tk*D), buf(t, d, Tq*D), buf(t, d, Tq*Tk)
	qf, kf, vf := fill(r, q), fill(r, k), fill(r, v)
	// Split the keys in two as the DiT step does (cached prefix + target):
	// S's column halves from two NT GEMMs, then P·V accumulated over halves.
	const split = 40
	if err := d.Run(func(e *Encoder) {
		e.Gemm(Gemm{M: Tq, N: split, K: dh, A: q.At(4 * h * dh), LDA: D, B: k.At(4 * h * dh), LDB: D, C: s.At(0), LDC: Tk, TransB: true})
		e.Gemm(Gemm{M: Tq, N: Tk - split, K: dh, A: q.At(4 * h * dh), LDA: D, B: k.At(4 * (split*D + h*dh)), LDB: D,
			C: s.At(4 * split), LDC: Tk, TransB: true})
		e.Gemm(Gemm{M: Tq, N: dh, K: split, A: s.At(0), LDA: Tk, B: v.At(4 * h * dh), LDB: D, C: o.At(4 * h * dh), LDC: D})
		e.Gemm(Gemm{M: Tq, N: dh, K: Tk - split, A: s.At(4 * split), LDA: Tk, B: v.At(4 * (split*D + h*dh)), LDB: D,
			C: o.At(4 * h * dh), LDC: D, Accumulate: true})
	}); err != nil {
		t.Fatal(err)
	}
	of := f32s(o.Bytes())
	for i := range Tq {
		sc := make([]float64, Tk)
		for j := range Tk {
			for c := range dh {
				sc[j] += float64(qf[i*D+h*dh+c]) * float64(kf[j*D+h*dh+c])
			}
		}
		for c := range dh {
			var want float64
			for j := range Tk {
				want += sc[j] * float64(vf[j*D+h*dh+c])
			}
			near(t, "O", of[i*D+h*dh+c], want, 1e-4)
		}
		if of[i*D] != 0 || of[i*D+2*dh] != 0 {
			t.Fatal("gemm wrote outside its head slice")
		}
	}
}

// TestWrapMappedFile: a read-only mmap of a file feeds a GEMM with no copy
// (how checkpoint weights reach the GPU), at a non-zero offset.
func TestWrapMappedFile(t *testing.T) {
	d := prepared(t)
	const m, n, k, off = 8, 16, 32, 4096 + 8
	w := make([]float32, n*k)
	raw := make([]byte, off+4*n*k)
	for i := range w {
		w[i] = float32(i%7) - 3
		binary.LittleEndian.PutUint32(raw[off+4*i:], math.Float32bits(w[i]))
	}
	path := filepath.Join(t.TempDir(), "w.bin")
	if err := os.WriteFile(path, raw, 0o644); err != nil {
		t.Fatal(err)
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	mem, err := syscall.Mmap(int(f.Fd()), 0, len(raw), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		t.Fatal(err)
	}
	defer syscall.Munmap(mem)
	wb, err := d.Wrap(mem)
	if err != nil {
		t.Fatal(err)
	}
	defer wb.Release()
	a, c := buf(t, d, m*k), buf(t, d, m*n)
	af := f32s(a.Bytes())
	for i := range af {
		af[i] = float32(i%5) - 2
	}
	if err := d.Run(func(e *Encoder) {
		e.Gemm(Gemm{M: m, N: n, K: k, A: a.At(0), B: wb.At(off), C: c.At(0), TransB: true})
	}); err != nil {
		t.Fatal(err)
	}
	cf := f32s(c.Bytes())
	for i := range m {
		for j := range n {
			var want float64
			for p := range k {
				want += float64(af[i*k+p]) * float64(w[j*k+p])
			}
			near(t, "C", cf[i*n+j], want, 1e-6)
		}
	}
}

func TestSoftmaxMaskedGather(t *testing.T) {
	d := prepared(t)
	r := rand.New(rand.NewPCG(6, 6))
	const rows, cols = 9, 9
	x, mask, g, idx := buf(t, d, rows*cols), buf(t, d, rows*cols), buf(t, d, 3*cols), buf(t, d, 3)
	xf := fill(r, x)
	orig := append([]float32(nil), xf...)
	mf := f32s(mask.Bytes())
	for i := range rows { // causal, and key 7 masked for everyone
		for j := range cols {
			if j > i || j == 7 {
				mf[i*cols+j] = float32(math.Inf(-1))
			}
		}
	}
	iv := unsafe.Slice((*uint32)(unsafe.Pointer(&idx.Bytes()[0])), 3)
	iv[0], iv[1], iv[2] = 8, 0, 4
	if err := d.Run(func(e *Encoder) {
		e.SoftmaxRowsMasked(x.At(0), mask.At(0), rows, cols, cols, cols, 0.5)
		e.GatherRows(x.At(0), g.At(0), idx.At(0), 3, cols, cols, cols)
	}); err != nil {
		t.Fatal(err)
	}
	for i := range rows {
		m, sum := math.Inf(-1), 0.0
		for j := 0; j <= i; j++ {
			if j != 7 {
				m = math.Max(m, 0.5*float64(orig[i*cols+j]))
			}
		}
		for j := 0; j <= i; j++ {
			if j != 7 {
				sum += math.Exp(0.5*float64(orig[i*cols+j]) - m)
			}
		}
		for j := range cols {
			want := 0.0
			if j <= i && j != 7 {
				want = math.Exp(0.5*float64(orig[i*cols+j])-m) / sum
			}
			near(t, "masked softmax", xf[i*cols+j], want, 1e-5)
		}
	}
	gf := f32s(g.Bytes())
	for r, src := range []int{8, 0, 4} {
		for c := range cols {
			if gf[r*cols+c] != xf[src*cols+c] {
				t.Fatalf("gather row %d col %d", r, c)
			}
		}
	}
}

// TestGemmBF16Activations: CastBF16 + bf16×bf16 GEMMs (NT, NN, NN-acc) vs
// a float64 oracle over the same bf16-rounded operands.
func TestGemmBF16Activations(t *testing.T) {
	d := prepared(t)
	r := rand.New(rand.NewPCG(7, 7))
	const M, N, K = 70, 96, 130
	a, a16 := buf(t, d, M*K), buf(t, d, M*K)
	w16, c := buf(t, d, N*K), buf(t, d, M*N)
	af := fill(r, a)
	wv := make([]float64, N*K)
	wb := w16.Bytes()
	for i := range wv {
		bits := uint16(math.Float32bits(r.Float32()*2-1) >> 16)
		wb[2*i], wb[2*i+1] = byte(bits), byte(bits>>8)
		wv[i] = float64(math.Float32frombits(uint32(bits) << 16))
	}
	// The same bf16 values double as an NN operand viewed as [K, N].
	c2 := buf(t, d, M*N)
	if err := d.Run(func(e *Encoder) {
		e.CastBF16(a.At(0), a16.At(0), M, K, K, K)
		e.Gemm(Gemm{M: M, N: N, K: K, A: a16.At(0), B: w16.At(0), C: c.At(0), TransB: true, BF16: true, ABF16: true})
		e.Gemm(Gemm{M: M, N: N, K: K / 2, A: a16.At(0), LDA: K, B: w16.At(0), C: c2.At(0), BF16: true, ABF16: true})
		e.Gemm(Gemm{M: M, N: N, K: K - K/2, A: a16.At(2 * (K / 2)), LDA: K, B: w16.At(2 * (K / 2) * N), C: c2.At(0),
			BF16: true, ABF16: true, Accumulate: true})
	}); err != nil {
		t.Fatal(err)
	}
	a16v := func(i int) float64 { // what CastBF16 produced (round to nearest even)
		b := a16.Bytes()
		return float64(math.Float32frombits(uint32(b[2*i])<<16 | uint32(b[2*i+1])<<24))
	}
	for i := range M * K {
		if got, x := a16v(i), float64(af[i]); math.Abs(got-x) > math.Abs(x)/256 {
			t.Fatalf("cast[%d] = %g from %g", i, got, x)
		}
	}
	cf, c2f := f32s(c.Bytes()), f32s(c2.Bytes())
	for i := range M {
		for j := range N {
			var nt, nn float64
			for p := range K {
				nt += a16v(i*K+p) * wv[j*K+p]
				nn += a16v(i*K+p) * wv[p*N+j]
			}
			near(t, "NT bb", cf[i*N+j], nt, 1e-5*math.Sqrt(K))
			near(t, "NN bb (split, accumulated)", c2f[i*N+j], nn, 1e-5*math.Sqrt(K))
		}
	}
}

func TestRMSNormRowsAndRopeHalf(t *testing.T) {
	d := prepared(t)
	r := rand.New(rand.NewPCG(8, 8))
	const rows, cols = 3, 4096
	x, y, w := buf(t, d, rows*cols), buf(t, d, rows*cols), buf(t, d, cols)
	xf, wf := fill(r, x), fill(r, w)
	const T, heads, dh = 2, 3, 128
	q, qw, cs, sn := buf(t, d, T*heads*dh), buf(t, d, dh), buf(t, d, T*dh/2), buf(t, d, T*dh/2)
	qf, qwf, cf, sf := fill(r, q), fill(r, qw), fill(r, cs), fill(r, sn)
	q0 := append([]float32(nil), qf...)
	if err := d.Run(func(e *Encoder) {
		e.RMSNormRows(x.At(0), y.At(0), w.At(0), rows, cols, cols, cols, 1e-6)
		e.RMSNormRoPEMode(q.At(0), qw.At(0), cs.At(0), sn.At(0), T, heads, dh, heads*dh, 1e-6, RopeHalf)
	}); err != nil {
		t.Fatal(err)
	}
	yf := f32s(y.Bytes())
	for rr := range rows {
		var ss float64
		for c := range cols {
			ss += float64(xf[rr*cols+c]) * float64(xf[rr*cols+c])
		}
		inv := 1 / math.Sqrt(ss/cols+1e-6)
		for c := range cols {
			near(t, "rmsnorm rows", yf[rr*cols+c], float64(xf[rr*cols+c])*inv*float64(wf[c]), 1e-4)
		}
	}
	for tt := range T {
		for h := range heads {
			base := (tt*heads + h) * dh
			var ss float64
			for i := range dh {
				ss += float64(q0[base+i]) * float64(q0[base+i])
			}
			inv := 1 / math.Sqrt(ss/dh+1e-6)
			n := func(i int) float64 { return float64(q0[base+i]) * inv * float64(qwf[i]) }
			for j := range dh / 2 {
				c, s := float64(cf[tt*dh/2+j]), float64(sf[tt*dh/2+j])
				near(t, "rope half lo", qf[base+j], n(j)*c-n(j+dh/2)*s, 1e-4)
				near(t, "rope half hi", qf[base+j+dh/2], n(j+dh/2)*c+n(j)*s, 1e-4)
			}
		}
	}
}
