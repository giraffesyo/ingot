package ops

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/giraffesyo/ingot/tensor"
)

// TestSDPACacheIncremental: prefill + single-token decode steps through the
// cached SDPA must reproduce a full dense causal-attention recompute.
func TestSDPACacheIncremental(t *testing.T) {
	const (
		H, dh    = 4, 16
		prefill  = 13
		steps    = 7
		maxT     = 32
		scaleval = 0.25
	)
	r := rand.New(rand.NewPCG(51, 52))
	total := prefill + steps
	// Full sequences of q/k/v per head: [H][total][dh].
	q := make([]float32, H*total*dh)
	k := make([]float32, H*total*dh)
	v := make([]float32, H*total*dh)
	for i := range q {
		q[i], k[i], v[i] = r.Float32()*2-1, r.Float32()*2-1, r.Float32()*2-1
	}
	slice := func(src []float32, t0, tc int) *tensor.Tensor {
		out := tensor.New(tensor.F32, 1, H, tc, dh)
		of := out.F32()
		for h := 0; h < H; h++ {
			copy(of[h*tc*dh:(h+1)*tc*dh], src[(h*total+t0)*dh:(h*total+t0+tc)*dh])
		}
		return out
	}

	// Reference: dense causal attention over the full sequence.
	want := make([]float32, H*total*dh)
	for h := 0; h < H; h++ {
		for i := 0; i < total; i++ {
			s := make([]float64, i+1)
			var m float64 = math.Inf(-1)
			for j := 0; j <= i; j++ {
				var d float64
				for p := 0; p < dh; p++ {
					d += float64(q[(h*total+i)*dh+p]) * float64(k[(h*total+j)*dh+p])
				}
				s[j] = d * scaleval
				if s[j] > m {
					m = s[j]
				}
			}
			var sum float64
			for j := range s {
				s[j] = math.Exp(s[j] - m)
				sum += s[j]
			}
			for p := 0; p < dh; p++ {
				var acc float64
				for j := 0; j <= i; j++ {
					acc += s[j] / sum * float64(v[(h*total+j)*dh+p])
				}
				want[(h*total+i)*dh+p] = float32(acc)
			}
		}
	}

	op, err := Lookup("ingot", "SDPA", 1)
	if err != nil {
		t.Fatal(err)
	}
	inst, err := op(NodeInfo{Name: "attn0", OpType: "SDPA", Domain: "ingot",
		Attrs: Attrs{"scale": {Kind: KindFloat, F: scaleval}, "cache": {Kind: KindInt, I: 1}},
		NumIn: 3, NumOut: 1})
	if err != nil {
		t.Fatal(err)
	}
	st := &DecodeState{MaxT: maxT, Slots: map[string]*DecodeSlot{
		"attn0": {K: make([]float32, H*maxT*dh), V: make([]float32, H*maxT*dh)},
	}}
	ctx := &Ctx{Decode: st}
	check := func(outs []*tensor.Tensor, t0, tc int) {
		of := outs[0].F32()
		for h := 0; h < H; h++ {
			for i := 0; i < tc; i++ {
				for p := 0; p < dh; p++ {
					got := of[(h*tc+i)*dh+p]
					ref := want[(h*total+t0+i)*dh+p]
					if d := math.Abs(float64(got - ref)); d > 2e-5 {
						t.Fatalf("pos %d h %d p %d: got %g want %g", t0+i, h, p, got, ref)
					}
				}
			}
		}
	}
	// Prefill.
	outs, err := inst.Run(ctx, []*tensor.Tensor{slice(q, 0, prefill), slice(k, 0, prefill), slice(v, 0, prefill)})
	if err != nil {
		t.Fatal(err)
	}
	check(outs, 0, prefill)
	st.Pos = prefill
	// Single-token steps.
	for s0 := prefill; s0 < total; s0++ {
		outs, err := inst.Run(ctx, []*tensor.Tensor{slice(q, s0, 1), slice(k, s0, 1), slice(v, s0, 1)})
		if err != nil {
			t.Fatal(err)
		}
		check(outs, s0, 1)
		st.Pos++
	}
}
