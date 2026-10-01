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
	for _, bf16 := range []bool{false, true} {
		st := &DecodeState{MaxT: maxT, BF16: bf16, Slots: map[string]*DecodeSlot{}}
		tol := 2e-5
		if bf16 {
			tol = 2e-2 // bf16 K/V quantization
		}
		ctx := &Ctx{Decode: st}
		check := func(outs []*tensor.Tensor, t0, tc int) {
			of := outs[0].F32()
			for h := 0; h < H; h++ {
				for i := 0; i < tc; i++ {
					for p := 0; p < dh; p++ {
						got := of[(h*tc+i)*dh+p]
						ref := want[(h*total+t0+i)*dh+p]
						if d := math.Abs(float64(got - ref)); d > tol {
							t.Fatalf("bf16=%v pos %d h %d p %d: got %g want %g", bf16, t0+i, h, p, got, ref)
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
}

// TestSDPACacheGQALayouts: the cached form with grouped-query K/V (Hkv <
// H) and every operand/output in [1, T, heads, dh] order against a float64
// oracle, over a prefill then single-token steps.
func TestSDPACacheGQALayouts(t *testing.T) {
	const (
		H, Hk, dh = 6, 2, 8
		prefill   = 5
		steps     = 4
		scaleval  = 0.3
	)
	total := prefill + steps
	r := rand.New(rand.NewPCG(7, 8))
	// Full sequences stored [total][heads][dh].
	gen := func(heads int) []float32 {
		x := make([]float32, total*heads*dh)
		for i := range x {
			x[i] = r.Float32()*2 - 1
		}
		return x
	}
	q, k, v := gen(H), gen(Hk), gen(Hk)
	rows := func(src []float32, heads, t0, tc int, lay int) *tensor.Tensor {
		if lay == 1 {
			out := tensor.New(tensor.F32, 1, tc, heads, dh)
			copy(out.F32(), src[t0*heads*dh:(t0+tc)*heads*dh])
			return out
		}
		out := tensor.New(tensor.F32, 1, heads, tc, dh)
		for h := range heads {
			for i := range tc {
				copy(out.F32()[(h*tc+i)*dh:(h*tc+i+1)*dh], src[((t0+i)*heads+h)*dh:])
			}
		}
		return out
	}
	want := func(h, i, p int) float64 { // causal attention, kv head h/(H/Hk)
		kh := h / (H / Hk)
		s := make([]float64, i+1)
		m := math.Inf(-1)
		for j := range s {
			for c := range dh {
				s[j] += float64(q[(i*H+h)*dh+c]) * float64(k[(j*Hk+kh)*dh+c])
			}
			s[j] *= scaleval
			m = math.Max(m, s[j])
		}
		var sum, acc float64
		for j := range s {
			e := math.Exp(s[j] - m)
			sum += e
			acc += e * float64(v[(j*Hk+kh)*dh+p])
		}
		return acc / sum
	}
	op, _ := Lookup("ingot", "SDPA", 1)
	for _, lay := range []int{0, 1} {
		inst, err := op(NodeInfo{Name: "attn", OpType: "SDPA", Domain: "ingot", NumIn: 3, NumOut: 1,
			Attrs: Attrs{"scale": {Kind: KindFloat, F: scaleval}, "cache": {Kind: KindInt, I: 1},
				"a_layout": {Kind: KindInt, I: int64(lay)}, "k_layout": {Kind: KindInt, I: int64(lay)},
				"v_layout": {Kind: KindInt, I: int64(lay)}, "stride_out": {Kind: KindInt, I: int64(lay)}}})
		if err != nil {
			t.Fatal(err)
		}
		st := &DecodeState{MaxT: total, Slots: map[string]*DecodeSlot{}}
		ctx := &Ctx{Decode: st}
		for t0 := 0; t0 < total; {
			tc := 1
			if t0 == 0 {
				tc = prefill
			}
			outs, err := inst.Run(ctx, []*tensor.Tensor{rows(q, H, t0, tc, lay), rows(k, Hk, t0, tc, lay), rows(v, Hk, t0, tc, lay)})
			if err != nil {
				t.Fatal(err)
			}
			of := outs[0].F32()
			for h := range H {
				for i := range tc {
					for p := range dh {
						oi := (h*tc+i)*dh + p
						if lay == 1 {
							oi = (i*H+h)*dh + p
						}
						if d := math.Abs(float64(of[oi]) - want(h, t0+i, p)); d > 1e-5 {
							t.Fatalf("layout %d pos %d head %d: got %g want %g", lay, t0+i, h, of[oi], want(h, t0+i, p))
						}
					}
				}
			}
			st.Pos += tc
			t0 += tc
		}
	}
}
