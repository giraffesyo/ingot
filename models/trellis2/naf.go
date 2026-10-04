package trellis2

import (
	"fmt"
	"math"

	"github.com/giraffesyo/ingot/graph"
	"github.com/giraffesyo/ingot/safetensors"
	"github.com/giraffesyo/ingot/tensor"
)

// The NAF feature upsampler (valeoai/NAF, "neighborhood attention
// filtering"): each pixel of a guide image attends over the 9×9 patch
// features around it, with queries and keys computed from the image alone.
// Pixal3D uses it to sharpen the encoder's patch features before sampling
// them at the voxels' projections.
const (
	nafDim     = 256 // query/key channels
	nafHeads   = 4
	nafKernel  = 9
	nafRopeTop = 100.0 // rotary base period
)

// BuildNAFEncoder builds the upsampler's image encoder: input "image"
// [1, 3, size, size], RGB in [0, 1]; output "q" [out·out, 256] — per
// output pixel (row-major) the query vector, rotary-encoded by position.
// size must be a multiple of out; the encoder runs at the image's
// resolution and is average-pooled down.
func BuildNAFEncoder(f *safetensors.File, size, out int) (g *graph.Graph, err error) {
	defer catch(&err)
	if out <= 0 || size%out != 0 || size > 4*out {
		return nil, fmt.Errorf("trellis2: upsampler image size %d must be a multiple of the output size %d, at most 4×", size, out)
	}
	w := weights{f: f}.scope("image_encoder")
	b := graph.NewBuilder("naf_encoder")
	img := b.Input("image", tensor.F32, 1, 3, size, size)
	x := b.Concat(1, nafBranch(b.Scope("encoder"), img, w.scope("encoder")), nafBranch(b.Scope("sem_encoder"), img, w.scope("sem_encoder")))
	if r := size / out; r > 1 {
		x = b.Op("AveragePool", graph.Attr("kernel_shape", []int{r, r}, "strides", []int{r, r}), x)
	}
	// [1, heads·dh, H, W] → [HW, heads, dh], rotated per position.
	T := int64(out * out)
	x = b.Transpose(b.Reshape(x, nafDim, T), 1, 0)
	dh := nafDim / nafHeads
	cos, sin := nafRope(out, dh)
	x = b.Op("ingot.RoPE", graph.Attr("layout", 1), b.Reshape(x, T, nafHeads, int64(dh)),
		b.Const("rope_cos", tensor.FromF32(cos, out*out, dh/2)), b.Const("rope_sin", tensor.FromF32(sin, out*out, dh/2)))
	b.Output("q", b.Reshape(x, T, nafDim))
	return b.Build()
}

// nafBranch is one of the encoder's two conv stacks (1×1 and 3×3 kernels,
// reflect padding): a conv, then blocks of GroupNorm → SiLU → conv twice.
func nafBranch(b *graph.Builder, x *graph.Value, w weights) *graph.Value {
	conv := func(sb *graph.Builder, x *graph.Value, w weights) *graph.Value {
		wt := w.f32("weight")
		if k := wt.Shape()[2]; k > 1 {
			p := int64(k / 2)
			x = sb.Op("Pad", graph.Attr("mode", "reflect"), x, sb.Ints(0, 0, p, p, 0, 0, p, p))
		}
		return sb.Op("Conv", nil, x, sb.Const("weight", wt), sb.Const("bias", w.f32("bias")))
	}
	norm := func(sb *graph.Builder, x *graph.Value, w weights) *graph.Value {
		return sb.Op("GroupNormalization", graph.Attr("num_groups", 8, "epsilon", 1e-5), x,
			sb.Const("weight", w.f32("weight")), sb.Const("bias", w.f32("bias")))
	}
	x = conv(b.Scope("0"), x, w.scope("0"))
	for i := 1; w.has(fmt.Sprintf("%d.conv1.weight", i)); i++ {
		sb, wb := b.Scope(fmt.Sprint(i)), w.scope(fmt.Sprint(i))
		x = conv(sb.Scope("conv1"), sb.SiLU(norm(sb.Scope("norm1"), x, wb.scope("norm1"))), wb.scope("conv1"))
		x = conv(sb.Scope("conv2"), sb.SiLU(norm(sb.Scope("norm2"), x, wb.scope("norm2"))), wb.scope("conv2"))
	}
	return x
}

// nafRope returns cos/sin [n·n, dh/2] of the upsampler's rotary encoding:
// pixel centres in [-1, 1], dh/4 periods per axis growing geometrically to
// the base.
func nafRope(n, dh int) (cos, sin []float32) {
	half, quarter := dh/2, dh/4
	periods := make([]float32, quarter)
	for i := range periods {
		periods[i] = float32(math.Pow(nafRopeTop, float64(2*float32(i)/float32(half))))
	}
	cos, sin = make([]float32, n*n*half), make([]float32, n*n*half)
	coord := func(i int) float32 { return 2*((float32(i)+0.5)/float32(n)) - 1 }
	for y := range n {
		for x := range n {
			row := (y*n + x) * half
			for ax, p := range [2]float32{coord(y), coord(x)} {
				for i, per := range periods {
					a := float64(float32(2*math.Pi) * p / per)
					cos[row+ax*quarter+i], sin[row+ax*quarter+i] = float32(math.Cos(a)), float32(math.Sin(a))
				}
			}
		}
	}
	return cos, sin
}

// Upsampled is a feature map upsampled by NAF, evaluated on demand: only
// the pixels asked for are computed.
type Upsampled struct {
	q, keys  []float32 // [size², 256], [n², 256]
	values   []float32 // [n², channels]
	size, n  int
	channels int
}

// Upsample prepares the upsampling of feats — patch features [n·n, C],
// row-major — to size×size, guided by q, the encoder's queries for a
// size×size output (BuildNAFEncoder). size must be a multiple of n, n at
// least the 9-patch neighbourhood, and C divisible by the 4 heads.
func Upsample(q, feats *tensor.Tensor, size int) (*Upsampled, error) {
	fs := feats.Shape()
	n := int(math.Round(math.Sqrt(float64(fs[0]))))
	if len(fs) != 2 || n*n != fs[0] || n < nafKernel || size%n != 0 || fs[1]%nafHeads != 0 {
		return nil, fmt.Errorf("trellis2: cannot upsample features %v to %d² (need a square map of at least %d², a divisor of the size)", fs, size, nafKernel)
	}
	if qs := q.Shape(); len(qs) != 2 || qs[0] != size*size || qs[1] != nafDim {
		return nil, fmt.Errorf("trellis2: upsampler queries are %v, want [%d, %d]", qs, size*size, nafDim)
	}
	u := &Upsampled{q: q.F32(), values: feats.F32(), size: size, n: n, channels: fs[1], keys: make([]float32, n*n*nafDim)}
	// Keys: the queries averaged over each patch's pixels.
	d := size / n
	inv := 1 / float32(d*d)
	for cy := range n {
		for cx := range n {
			k := u.keys[(cy*n+cx)*nafDim : (cy*n+cx+1)*nafDim]
			for y := cy * d; y < (cy+1)*d; y++ {
				for x := cx * d; x < (cx+1)*d; x++ {
					for c, v := range u.q[(y*size+x)*nafDim : (y*size+x+1)*nafDim] {
						k[c] += v
					}
				}
			}
			for c := range k {
				k[c] *= inv
			}
		}
	}
	return u, nil
}

// windowStart is the first patch of the 9-wide neighbourhood of patch i in
// a row of n: centred, shifted inwards at the borders so it never leaves
// the map (NATTEN's window rule).
func windowStart(i, n int) int {
	const r = nafKernel / 2
	s := max(i-r, 0)
	if i+r >= n {
		s += n - i - r - 1
	}
	return s
}

// at writes the upsampled feature of pixel (x, y) into out [channels]: per
// head, a softmax over the neighbourhood's key·query scores weights the
// neighbourhood's values.
func (u *Upsampled) at(x, y int, out []float32) {
	d := u.size / u.n
	y0, x0 := windowStart(y/d, u.n), windowStart(x/d, u.n)
	q := u.q[(y*u.size+x)*nafDim : (y*u.size+x+1)*nafDim]
	const dh = nafDim / nafHeads
	vh := u.channels / nafHeads
	scale := float32(1 / math.Sqrt(dh))
	var score [nafKernel * nafKernel]float32
	clear(out)
	for h := range nafHeads {
		qh := q[h*dh : (h+1)*dh]
		top := float32(math.Inf(-1))
		for j := range score {
			cell := (y0+j/nafKernel)*u.n + x0 + j%nafKernel
			k := u.keys[cell*nafDim+h*dh : cell*nafDim+(h+1)*dh]
			var s float32
			for c, v := range qh {
				s += v * k[c]
			}
			score[j] = s * scale
			top = max(top, score[j])
		}
		var sum float32
		for j := range score {
			score[j] = float32(math.Exp(float64(score[j] - top)))
			sum += score[j]
		}
		oh := out[h*vh : (h+1)*vh]
		for j, wt := range score {
			cell := (y0+j/nafKernel)*u.n + x0 + j%nafKernel
			wt /= sum
			for c, v := range u.values[cell*u.channels+h*vh : cell*u.channels+(h+1)*vh] {
				oh[c] += wt * v
			}
		}
	}
}
