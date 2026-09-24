package ops

import (
	"fmt"
	"strings"

	"github.com/giraffesyo/ingot/kernels/gemm"
	"github.com/giraffesyo/ingot/tensor"
)

// einsumOp: ONNX Einsum over f32, explicit or implicit output, no ellipsis
// (an unsupported equation fails at load). Two-operand contractions where
// every label appears in at least two of {A, B, output} (the matmul family:
// batch labels in all three, M in A+out, N in B+out, K in A+B) run as
// batched GEMMs over permuted operands; everything else — traces, diagonals,
// one-operand reductions, labels summed out of one input — runs through the
// direct evaluator (f64 accumulation).
type einsumOp struct {
	n    NodeInfo
	ins  []string // per-operand labels
	out  string
	gemm bool
}

func parseEinsum(eq string) (ins []string, out string, err error) {
	eq = strings.ReplaceAll(eq, " ", "")
	if strings.Contains(eq, "...") {
		return nil, "", fmt.Errorf("einsum ellipsis is not supported (%q)", eq)
	}
	lhs, rhs, explicit := strings.Cut(eq, "->")
	ins = strings.Split(lhs, ",")
	count := map[rune]int{}
	for _, s := range ins {
		for _, c := range s {
			if !(c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z') {
				return nil, "", fmt.Errorf("einsum label %q in %q", c, eq)
			}
			count[c]++
		}
	}
	if explicit {
		out = rhs
		outSeen := map[rune]bool{}
		for _, c := range out {
			if count[c] == 0 {
				return nil, "", fmt.Errorf("einsum output label %q not in inputs (%q)", c, eq)
			}
			if outSeen[c] {
				return nil, "", fmt.Errorf("einsum output label %q repeated (%q)", c, eq)
			}
			outSeen[c] = true
		}
	} else { // implicit: labels seen once, alphabetical
		var b strings.Builder
		for c := 'A'; c <= 'z'; c++ {
			if count[c] == 1 {
				b.WriteRune(c)
			}
		}
		out = b.String()
	}
	return ins, out, nil
}

// einsumGEMMable reports the matmul-family shape (see einsumOp).
func einsumGEMMable(ins []string, out string) bool {
	if len(ins) != 2 {
		return false
	}
	for _, s := range append(ins, out) {
		seen := map[rune]bool{}
		for _, c := range s {
			if seen[c] {
				return false // repeated label within an operand: a diagonal
			}
			seen[c] = true
		}
	}
	in := func(s string, c rune) bool { return strings.ContainsRune(s, c) }
	for _, s := range []string{ins[0], ins[1], out} {
		for _, c := range s {
			n := 0
			for _, t := range []string{ins[0], ins[1], out} {
				if in(t, c) {
					n++
				}
			}
			if n < 2 {
				return false // summed out of a single input
			}
		}
	}
	return true
}

func (o *einsumOp) Run(ctx *Ctx, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(in) != len(o.ins) {
		return nil, o.n.Errorf("Einsum: %d inputs for %d operands", len(in), len(o.ins))
	}
	dims := map[rune]int{}
	for i, t := range in {
		if t == nil || t.DType() != tensor.F32 {
			return nil, o.n.Errorf("Einsum: operand %d must be f32", i)
		}
		s := t.Shape()
		if len(s) != len(o.ins[i]) {
			return nil, o.n.Errorf("Einsum: operand %d rank %d vs %q", i, len(s), o.ins[i])
		}
		for k, c := range o.ins[i] {
			if d, ok := dims[c]; ok && d != s[k] && d != 1 && s[k] != 1 {
				return nil, o.n.Errorf("Einsum: label %q is %d and %d", c, d, s[k])
			}
			dims[c] = max(dims[c], s[k])
		}
	}
	oshape := make([]int, len(o.out))
	for k, c := range o.out {
		oshape[k] = dims[c]
	}
	if o.gemm && o.noBroadcast(in) {
		return o.runGEMM(ctx, in, dims, oshape)
	}
	out := ctx.NewUninit(tensor.F32, oshape...)
	einsumDirect(o.ins, o.out, in, dims, out.F32())
	return ctx.Out(out), nil
}

// noBroadcast: every operand dim equals its label's extent.
func (o *einsumOp) noBroadcast(in []*tensor.Tensor) bool {
	dims := map[rune]int{}
	for i, t := range in {
		for k, c := range o.ins[i] {
			if d, ok := dims[c]; ok && d != t.Shape()[k] {
				return false
			}
			dims[c] = t.Shape()[k]
		}
	}
	return true
}

func (o *einsumOp) runGEMM(ctx *Ctx, in []*tensor.Tensor, dims map[rune]int, oshape []int) ([]*tensor.Tensor, error) {
	A, B, out := o.ins[0], o.ins[1], o.out
	has := strings.ContainsRune
	var batch, ms, ns, ks []rune
	for _, c := range out {
		switch {
		case has(A, c) && has(B, c):
			batch = append(batch, c)
		case has(A, c):
			ms = append(ms, c)
		default:
			ns = append(ns, c)
		}
	}
	for _, c := range A {
		if !has(out, c) {
			ks = append(ks, c)
		}
	}
	prod := func(ls []rune) int {
		p := 1
		for _, c := range ls {
			p *= dims[c]
		}
		return p
	}
	nb, M, N, K := prod(batch), prod(ms), prod(ns), prod(ks)
	// A → [batch, m, k], B → [batch, k, n] (permuted copies when needed).
	a := permuteTo(ctx, in[0], A, append(append(append([]rune{}, batch...), ms...), ks...))
	b := permuteTo(ctx, in[1], B, append(append(append([]rune{}, batch...), ks...), ns...))
	cbuf := ctx.NewUninit(tensor.F32, nb, M, N)
	af, bf, cf := a.F32(), b.F32(), cbuf.F32()
	for i := 0; i < nb; i++ {
		gemm.SgemmT(false, false, M, N, K, 1, af[i*M*K:], K, bf[i*K*N:], N, 0, cf[i*M*N:], N)
	}
	// [batch, m, n] → output order.
	order := string(append(append(append([]rune{}, batch...), ms...), ns...))
	shp := make([]int, 0, len(order))
	for _, c := range order {
		shp = append(shp, dims[c])
	}
	res := permuteTo(ctx, cbuf.Reshape(shp...), order, []rune(out))
	for _, t := range []*tensor.Tensor{a, b} {
		if t != in[0] && t != in[1] && ctx.Pool != nil {
			ctx.Pool.Put(t)
		}
	}
	if res != cbuf && ctx.Pool != nil {
		ctx.Pool.Put(cbuf)
	}
	return ctx.Out(res), nil
}

// permuteTo returns t (labelled from) with its axes reordered to `to` —
// t itself when the order already matches.
func permuteTo(ctx *Ctx, t *tensor.Tensor, from string, to []rune) *tensor.Tensor {
	perm := make([]int, len(to))
	same := true
	for i, c := range to {
		perm[i] = strings.IndexRune(from, c)
		same = same && perm[i] == i
	}
	if same {
		return t
	}
	s := t.Shape()
	os := make([]int, len(perm))
	for i, p := range perm {
		os[i] = s[p]
	}
	out := ctx.NewUninit(t.DType(), os...)
	transposeBytes(t, out, perm)
	return out
}

// einsumDirect evaluates any (ellipsis-free) equation by iterating every
// label combination, accumulating in f64. Operand dims of 1 broadcast; a
// label repeated within an operand walks its diagonal.
func einsumDirect(ins []string, out string, ts []*tensor.Tensor, dims map[rune]int, of []float32) {
	var labels []rune
	seen := map[rune]bool{}
	for _, c := range out + strings.Join(ins, "") {
		if !seen[c] {
			labels, seen[c] = append(labels, c), true
		}
	}
	pos := make(map[rune]int, len(labels))
	ext := make([]int, len(labels))
	for i, c := range labels {
		pos[c], ext[i] = i, dims[c]
	}
	nOut := len([]rune(out))
	st := make([][]int, len(ins))
	data := make([][]float32, len(ins))
	for i, s := range ins {
		st[i] = make([]int, len(labels))
		shape, strides := ts[i].Shape(), ts[i].Shape().Strides()
		for k, c := range []rune(s) {
			if shape[k] != 1 || dims[c] == 1 {
				st[i][pos[c]] += strides[k]
			}
		}
		data[i] = ts[i].F32()
	}
	coord := make([]int, len(labels))
	nSum := 1
	for _, e := range ext[nOut:] {
		nSum *= e
	}
	for o := range of {
		var acc float64
		for q := 0; q < nSum; q++ {
			p := 1.0
			for i := range ins {
				off := 0
				for l, c := range coord {
					off += c * st[i][l]
				}
				p *= float64(data[i][off])
			}
			acc += p
			incCoord(coord[nOut:], ext[nOut:])
		}
		of[o] = float32(acc)
		incCoord(coord[:nOut], ext[:nOut])
	}
}

func init() {
	Register("", "Einsum", 12, func(n NodeInfo) (Op, error) {
		ins, out, err := parseEinsum(n.Attrs.String("equation", ""))
		if err != nil {
			return nil, n.Errorf("%v", err)
		}
		return &einsumOp{n: n, ins: ins, out: out, gemm: einsumGEMMable(ins, out)}, nil
	})
}
