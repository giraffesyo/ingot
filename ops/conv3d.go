package ops

import (
	"sync"

	"github.com/giraffesyo/ingot/kernels/gemm"
	"github.com/giraffesyo/ingot/kernels/par"
	"github.com/giraffesyo/ingot/tensor"
)

// conv3dOp is 3-D Conv (NCDHW) as vol2col + GEMM, tiled over output (depth,
// row) pairs: each task packs the column matrix for a run of output rows
// into per-worker scratch, runs a serial GEMM into the output slab and adds
// the bias while the slab is hot. 1×1×1/stride-1/no-pad convs skip the
// column matrix and run GEMM on the input directly.
type conv3dOp struct {
	n         NodeInfo
	group     int
	strides   [3]int
	dilations [3]int
	pads      [6]int // front, top, left, back, bottom, right
	autoPad   string
	epi       epilogue

	packMu  sync.Mutex
	packed  []*gemm.PackedA
	packSrc *float32
	packLen int
}

// conv3dColFloats bounds the column scratch per task. Volumetric kernels
// have K = 27·C rows, so the 2-D tile budget would leave GEMM a handful of
// columns per sweep of the whole weight matrix; 16 MB keeps a few hundred.
const conv3dColFloats = 1 << 22

// isConv3D reports whether any spatial attribute is three-dimensional.
func isConv3D(n NodeInfo) bool {
	return len(n.Attrs.Ints("kernel_shape", nil)) == 3 || len(n.Attrs.Ints("strides", nil)) == 3 ||
		len(n.Attrs.Ints("dilations", nil)) == 3 || len(n.Attrs.Ints("pads", nil)) == 6
}

func buildConv3D(n NodeInfo) (*conv3dOp, error) {
	o := &conv3dOp{n: n, group: int(n.Attrs.Int("group", 1)), autoPad: n.Attrs.String("auto_pad", "NOTSET")}
	st := n.Attrs.Ints("strides", []int64{1, 1, 1})
	di := n.Attrs.Ints("dilations", []int64{1, 1, 1})
	pa := n.Attrs.Ints("pads", []int64{0, 0, 0, 0, 0, 0})
	if len(st) != 3 || len(di) != 3 || len(pa) != 6 {
		return nil, n.Errorf("3-D conv needs 3 strides, 3 dilations and 6 pads (strides=%v dilations=%v pads=%v)", st, di, pa)
	}
	for i := range 3 {
		o.strides[i], o.dilations[i] = int(st[i]), int(di[i])
	}
	for i := range 6 {
		o.pads[i] = int(pa[i])
	}
	var err error
	if o.epi, err = parseEpilogue(n); err != nil {
		return nil, err
	}
	return o, nil
}

func (o *conv3dOp) packedWeights(wf []float32, G, Mg, K int) []*gemm.PackedA {
	if !gemm.PackFits(Mg, K) {
		return nil
	}
	o.packMu.Lock()
	defer o.packMu.Unlock()
	if o.packSrc == &wf[0] && o.packLen == len(wf) {
		return o.packed
	}
	pk := make([]*gemm.PackedA, G)
	for g := range G {
		pk[g] = gemm.PackA(false, Mg, K, wf[g*Mg*K:], K)
	}
	o.packed, o.packSrc, o.packLen = pk, &wf[0], len(wf)
	return pk
}

func (o *conv3dOp) resolvePads(in, k [3]int) (pads [6]int, err error) {
	pads = o.pads
	switch o.autoPad {
	case "NOTSET", "":
	case "VALID":
		pads = [6]int{}
	case "SAME_UPPER", "SAME_LOWER":
		for d := range 3 {
			s := o.strides[d]
			out := (in[d] + s - 1) / s
			total := max(0, (out-1)*s+o.dilations[d]*(k[d]-1)+1-in[d])
			a := total / 2
			if o.autoPad == "SAME_LOWER" {
				a = (total + 1) / 2
			}
			pads[d], pads[d+3] = a, total-a
		}
	default:
		return pads, o.n.Errorf("unsupported auto_pad %q", o.autoPad)
	}
	return pads, nil
}

func (o *conv3dOp) Run(ctx *Ctx, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(in) < 2 || in[0] == nil || in[1] == nil {
		return nil, o.n.Errorf("need X and W")
	}
	x, w := in[0], in[1]
	var bias []float32
	if len(in) > 2 && in[2] != nil {
		bias = in[2].F32()
	}
	if x.DType() != tensor.F32 || w.DType() != tensor.F32 {
		return nil, o.n.Errorf("only f32 supported")
	}
	xs, ws := x.Shape(), w.Shape()
	if len(xs) != 5 || len(ws) != 5 {
		return nil, o.n.Errorf("3-D conv needs rank-5 X and W (X %v, W %v)", xs, ws)
	}
	N, C := xs[0], xs[1]
	M, Cg := ws[0], ws[1]
	inS, kS := [3]int{xs[2], xs[3], xs[4]}, [3]int{ws[2], ws[3], ws[4]}
	G := o.group
	if C != Cg*G || M%G != 0 {
		return nil, o.n.Errorf("channel mismatch: X C=%d, W [%d,%d], group=%d", C, M, Cg, G)
	}
	pads, err := o.resolvePads(inS, kS)
	if err != nil {
		return nil, err
	}
	var outS [3]int
	for d := range 3 {
		outS[d] = convOut(inS[d], kS[d], o.strides[d], o.dilations[d], pads[d], pads[d+3])
		if outS[d] <= 0 {
			return nil, o.n.Errorf("non-positive output size %v", outS)
		}
	}
	out := ctx.NewUninit(tensor.F32, N, M, outS[0], outS[1], outS[2])
	xf, wf, of := x.F32(), w.F32(), out.F32()
	Mg := M / G
	K := Cg * kS[0] * kS[1] * kS[2]
	OW := outS[2]
	R := outS[0] * outS[1] // output rows of width OW
	P := R * OW
	V := inS[0] * inS[1] * inS[2]
	pk := o.packedWeights(wf, G, Mg, K)
	pointwise := kS == [3]int{1, 1, 1} && o.strides == [3]int{1, 1, 1} && pads == [6]int{}

	rows := max(1, conv3dColFloats/max(1, K*OW))
	if want := 2 * par.Workers(); N*G*((R+rows-1)/rows) < want && R > 1 {
		rows = max(1, min(rows, (R*N*G+want-1)/want))
	}
	rows = min(R, max(rows, (convTaskMACs+Mg*K*OW-1)/max(1, Mg*K*OW)))
	nChunks := (R + rows - 1) / rows
	bufs := make([]*tensor.Tensor, par.Workers())
	par.For(N*G*nChunks, 1, func(t, wk int) {
		ch := t % nChunks
		ng := t / nChunks
		n, g := ng/G, ng%G
		r0 := ch * rows
		r1 := min(r0+rows, R)
		pc := (r1 - r0) * OW
		xg := xf[(n*C+g*Cg)*V:]
		dst := of[(n*M+g*Mg)*P+r0*OW:]
		cf, ldb := xg[r0*OW:], P
		if !pointwise {
			if bufs[wk] == nil {
				bufs[wk] = ctx.NewUninit(tensor.F32, K, rows*OW)
			}
			cf, ldb = bufs[wk].F32()[:K*pc], pc
			o.vol2col(xg, cf, Cg, inS, kS, outS, r0, r1, pads)
		}
		if pk != nil {
			gemm.SgemmPackedA(pk[g], pc, cf, ldb, 0, dst, P, false)
		} else {
			gemm.SgemmSerial(Mg, pc, K, 1, wf[g*Mg*K:], K, cf, ldb, 0, dst, P)
		}
		o.finishTile(dst, bias, g*Mg, Mg, P, pc)
	})
	if ctx.Pool != nil {
		for _, b := range bufs {
			if b != nil {
				ctx.Pool.Put(b)
			}
		}
	}
	return ctx.Out(out), nil
}

func (o *conv3dOp) finishTile(dst, bias []float32, m0, Mg, P, pc int) {
	c := convOp{epi: o.epi}
	c.finishTile(dst, bias, m0, Mg, P, pc)
}

// vol2col writes the column matrix for output rows [r0, r1), a row being one
// (od, oh) pair: col[((c·KD+kd)·KH+kh)·KW+kw][(r−r0)·OW+ow] =
// x[c][od·s+kd·d−pf][oh·s+kh·d−pt][ow·s+kw·d−pl], zero outside the volume.
func (o *conv3dOp) vol2col(x, col []float32, C int, in, k, out [3]int, r0, r1 int, pads [6]int) {
	D, H, W := in[0], in[1], in[2]
	OH, OW := out[1], out[2]
	sd, sh, sw := o.strides[0], o.strides[1], o.strides[2]
	dd, dh, dw := o.dilations[0], o.dilations[1], o.dilations[2]
	pc := (r1 - r0) * OW
	row := 0
	for c := range C {
		xc := x[c*D*H*W : (c+1)*D*H*W]
		for kd := range k[0] {
			for kh := range k[1] {
				for kw := range k[2] {
					// In-bounds output columns [lo, hi) for this tap.
					start := kw*dw - pads[2]
					lo := 0
					if start < 0 {
						lo = min(OW, (-start+sw-1)/sw)
					}
					hi := OW
					if start+(OW-1)*sw >= W {
						hi = max(lo, (W-1-start)/sw+1)
					}
					cr := col[row*pc : (row+1)*pc]
					row++
					for r := r0; r < r1; r++ {
						dst := cr[(r-r0)*OW : (r-r0+1)*OW]
						id := (r/OH)*sd + kd*dd - pads[0]
						ih := (r%OH)*sh + kh*dh - pads[1]
						if id < 0 || id >= D || ih < 0 || ih >= H {
							clear(dst)
							continue
						}
						src := xc[(id*H+ih)*W : (id*H+ih+1)*W]
						clear(dst[:lo])
						if hi > lo {
							d := dst[lo:hi]
							if sw == 1 {
								copy(d, src[start+lo:])
							} else {
								for j, i := 0, start+lo*sw; j < len(d); j, i = j+1, i+sw {
									d[j] = src[i]
								}
							}
						}
						clear(dst[hi:])
					}
				}
			}
		}
	}
}
