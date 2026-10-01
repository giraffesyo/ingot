//go:build darwin && arm64

package metal

import (
	"testing"
)

// BenchmarkGemvTalker: one 1.7B talker step's GEMVs (28 layers × q, k, v,
// o, gate, up, down at D=2048, I=6144) in one command buffer — the decode
// bandwidth bound, without the other kernels.
func BenchmarkGemvTalker(b *testing.B) {
	d := openDev(b)
	if err := d.Prepare(); err != nil {
		b.Fatal(err)
	}
	if err := d.PrepareDecode(); err != nil {
		b.Fatal(err)
	}
	const D, I, L = 2048, 6144, 28
	shapes := [][2]int{{2048, D}, {1024, D}, {1024, D}, {D, 2048}, {I, D}, {I, D}, {D, I}}
	var bytes int
	for _, s := range shapes {
		bytes += s[0] * s[1]
	}
	x, _ := d.NewBuffer(4 * I)
	y, _ := d.NewBuffer(4 * I)
	for _, bf16 := range []bool{true, false} {
		esz := 1
		if bf16 {
			esz = 2
		}
		var ws, ss []*Buffer
		for range L {
			for _, s := range shapes {
				w, err := d.NewBuffer(esz * s[0] * s[1])
				if err != nil {
					b.Fatal(err)
				}
				sc, _ := d.NewBuffer(4 * s[0] * s[1] / 64)
				ws, ss = append(ws, w), append(ss, sc)
			}
		}
		for _, staged := range []bool{false, true, false, true} {
			gemvStaged = staged
			name := "int8"
			if bf16 {
				name = "bf16"
			}
			if staged {
				name += "-staged"
			}
			b.Run(name, func(b *testing.B) {
				for b.Loop() {
					if err := d.Run(func(e *Encoder) {
						for i := range ws {
							s := shapes[i%len(shapes)]
							e.Gemv(Gemv{N: s[0], K: s[1], Rows: 1, X: x.At(0), W: ws[i].At(0), S: ss[i].At(0), Y: y.At(0), BF16: bf16})
						}
					}); err != nil {
						b.Fatal(err)
					}
				}
				b.ReportMetric(float64(L*bytes*esz)*float64(b.N)/b.Elapsed().Seconds()/1e9, "GB/s")
			})
		}
		gemvStaged = true
		for i := range ws {
			ws[i].Release()
			ss[i].Release()
		}
	}
}
