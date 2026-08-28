package graph

import (
	"math"
	"testing"

	"github.com/giraffesyo/ingot/ops"
	"github.com/giraffesyo/ingot/tensor"
)

// TestRunDecode drives a one-node cached-SDPA graph through prefill plus
// single-token steps and checks against a dense causal recompute.
func TestRunDecode(t *testing.T) {
	const H, dh, prefill, steps, maxT = 2, 8, 5, 4, 16
	total := prefill + steps
	tg := newTestGraph()
	n := tg.node("SDPA", []string{"q", "k", "v"}, "y")
	n.Domain = "ingot"
	n.Attrs = ops.Attrs{
		"scale": {Kind: ops.KindFloat, F: 0.5},
		"cache": {Kind: ops.KindInt, I: 1},
	}
	g := tg.finish([]string{"q", "k", "v"}, []string{"y"})
	g.Opsets["ingot"] = 1
	s, err := CompileRaw(g)
	if err != nil {
		t.Fatal(err)
	}

	mk := func(seed float32, t0, tc int) *tensor.Tensor {
		x := tensor.New(tensor.F32, 1, H, tc, dh)
		f := x.F32()
		for h := 0; h < H; h++ {
			for i := 0; i < tc; i++ {
				for p := 0; p < dh; p++ {
					f[(h*tc+i)*dh+p] = seed * float32((h*total+t0+i)*dh+p%13) / 100
				}
			}
		}
		return x
	}
	full := func(seed float32) []float32 {
		x := mk(seed, 0, total)
		return x.F32()
	}
	qf, kf, vf := full(0.3), full(0.7), full(-0.4)
	ref := func(h, i, p int) float64 {
		s := make([]float64, i+1)
		m := math.Inf(-1)
		for j := 0; j <= i; j++ {
			var d float64
			for pp := 0; pp < dh; pp++ {
				d += float64(qf[(h*total+i)*dh+pp]) * float64(kf[(h*total+j)*dh+pp])
			}
			s[j] = d * 0.5
			if s[j] > m {
				m = s[j]
			}
		}
		var sum float64
		for j := range s {
			s[j] = math.Exp(s[j] - m)
			sum += s[j]
		}
		var acc float64
		for j := 0; j <= i; j++ {
			acc += s[j] / sum * float64(vf[(h*total+j)*dh+p])
		}
		return acc
	}

	d := s.NewDecode(maxT)
	step := func(t0, tc int) {
		res, err := s.RunDecode(d, map[string]*tensor.Tensor{
			"q": mk(0.3, t0, tc), "k": mk(0.7, t0, tc), "v": mk(-0.4, t0, tc),
		}, tc)
		if err != nil {
			t.Fatal(err)
		}
		of := res["y"].F32()
		for h := 0; h < H; h++ {
			for i := 0; i < tc; i++ {
				for p := 0; p < dh; p++ {
					got := float64(of[(h*tc+i)*dh+p])
					want := ref(h, t0+i, p)
					if math.Abs(got-want) > 2e-5 {
						t.Fatalf("pos %d h %d p %d: got %g want %g", t0+i, h, p, got, want)
					}
				}
			}
		}
	}
	step(0, prefill)
	for s0 := prefill; s0 < total; s0++ {
		step(s0, 1)
	}
	if d.Pos() != total {
		t.Fatalf("pos = %d, want %d", d.Pos(), total)
	}
}
