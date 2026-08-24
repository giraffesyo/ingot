package ops

import (
	"math/rand/v2"
	"testing"

	"github.com/giraffesyo/ingot/tensor"
)

func BenchmarkConv3x3(b *testing.B) {
	r := rand.New(rand.NewPCG(1, 2))
	for _, c := range []struct {
		name         string
		Cin, H, W, M int
	}{
		{"det58_96x160x160_to24", 96, 160, 160, 24},
		{"det55_96x80x80_to24", 96, 80, 80, 24},
		{"res_16x16x16_to16", 16, 16, 16, 16},
		{"res_64x56x56_to64", 64, 56, 56, 64},
	} {
		x := tensor.New(tensor.F32, 1, c.Cin, c.H, c.W)
		for i := range x.F32() {
			x.F32()[i] = r.Float32()*2 - 1
		}
		w := tensor.New(tensor.F32, c.M, c.Cin, 3, 3)
		for i := range w.F32() {
			w.F32()[i] = r.Float32()*2 - 1
		}
		bias := tensor.New(tensor.F32, c.M)
		for _, mode := range []string{"wino", "im2col"} {
			op := mkOpB(b, "Conv", 1, Attrs{
				"kernel_shape": {Kind: KindInts, Ints: []int64{3, 3}},
				"pads":         {Kind: KindInts, Ints: []int64{1, 1, 1, 1}},
			}, 3, 1)
			co := op.(*convOp)
			ctx := &Ctx{Pool: tensor.NewPool()}
			b.Run(c.name+"/"+mode, func(b *testing.B) {
				old := winogradEnabled
				winogradEnabled = mode == "wino"
				defer func() { winogradEnabled = old }()
				for i := 0; i < b.N; i++ {
					outs, err := co.Run(ctx, []*tensor.Tensor{x, w, bias})
					if err != nil {
						b.Fatal(err)
					}
					ctx.Pool.Put(outs[0])
				}
			})
		}
	}
}

func mkOpB(b *testing.B, name string, ver int, attrs Attrs, numIn, numOut int) Op {
	bl, err := Lookup("", name, ver)
	if err != nil {
		b.Fatal(err)
	}
	op, err := bl(NodeInfo{Name: name, OpType: name, Version: ver, Attrs: attrs, NumIn: numIn, NumOut: numOut})
	if err != nil {
		b.Fatal(err)
	}
	return op
}
