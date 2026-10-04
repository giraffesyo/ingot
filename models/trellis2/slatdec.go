package trellis2

import (
	"fmt"

	"github.com/giraffesyo/ingot/graph"
	"github.com/giraffesyo/ingot/safetensors"
	"github.com/giraffesyo/ingot/sparse"
	"github.com/giraffesyo/ingot/tensor"
)

// DecoderConfig is the "args" of a structured-latent decoder's JSON:
// SparseUnetVaeDecoder (texture) or FlexiDualGridVaeDecoder (shape).
type DecoderConfig struct {
	OutChannels    int      `json:"out_channels"`
	ModelChannels  []int    `json:"model_channels"`
	LatentChannels int      `json:"latent_channels"`
	NumBlocks      []int    `json:"num_blocks"`
	BlockType      []string `json:"block_type"`
	UpBlockType    []string `json:"up_block_type"`
	PredSubdiv     bool     `json:"pred_subdiv"`
	VoxelMargin    float64  `json:"voxel_margin"`
}

// LoadDecoderConfig reads a decoder's JSON and rejects variants this
// package does not build.
func LoadDecoderConfig(path string) (DecoderConfig, error) {
	var c struct {
		Name string        `json:"name"`
		Args DecoderConfig `json:"args"`
	}
	c.Args.PredSubdiv, c.Args.VoxelMargin = true, 0.5
	if err := readJSON(path, &c); err != nil {
		return c.Args, err
	}
	a := c.Args
	switch c.Name {
	case "FlexiDualGridVaeDecoder":
		a.OutChannels = 7 // vertex offset (3), edge intersections (3), quad split weight (1)
	case "SparseUnetVaeDecoder":
	default:
		return a, fmt.Errorf("trellis2: %s: model %q is not a structured-latent decoder", path, c.Name)
	}
	if len(a.NumBlocks) != len(a.ModelChannels) || len(a.BlockType) != len(a.NumBlocks) || len(a.UpBlockType) != len(a.NumBlocks)-1 {
		return a, fmt.Errorf("trellis2: %s: level lists disagree in length", path)
	}
	for _, bt := range a.BlockType {
		if bt != "SparseConvNeXtBlock3d" {
			return a, fmt.Errorf("trellis2: %s: block type %q not implemented", path, bt)
		}
	}
	for i, ut := range a.UpBlockType {
		if ut != "SparseResBlockC2S3d" || a.ModelChannels[i]%8 != 0 || a.ModelChannels[i+1]%(a.ModelChannels[i]/8) != 0 {
			return a, fmt.Errorf("trellis2: %s: up block %q (%d → %d channels) not implemented", path, ut, a.ModelChannels[i], a.ModelChannels[i+1])
		}
	}
	return a, nil
}

// Levels is the number of resolution levels; level 0 is the latent grid and
// each further level doubles the resolution.
func (c DecoderConfig) Levels() int { return len(c.NumBlocks) }

// BuildDecoderLevel builds one resolution level of a structured-latent
// decoder over the cells at coords. The decoder cannot be one graph: the
// cells of a level are chosen by the subdivision the previous level
// predicts, so each level is built once its cells are known.
//
// Level 0 takes "latent" [N, latent_channels]. A later level takes the
// previous level's "h" and "x" as "parent_h" [P, 8·C] and "parent_x"
// [P, C_prev], P the previous level's cell count, and src, the row of each
// of this level's cells in their [P·8, ·] view (sparse.Subdivide), and
// finishes the upsampling block there.
//
// Every level but the last then outputs "h" and "x" for the next level,
// plus "subdiv" [N, 8] — the logits of each cell's eight children — when
// the decoder predicts its own subdivision. The last level outputs "out"
// [N, out_channels].
func BuildDecoderLevel(cfg DecoderConfig, f *safetensors.File, level int, coords []sparse.Coord, src []int32) (g *graph.Graph, err error) {
	defer catch(&err)
	if level < 0 || level >= cfg.Levels() {
		return nil, fmt.Errorf("trellis2: decoder level %d out of range [0, %d)", level, cfg.Levels())
	}
	nb, err := sparse.Neighbors(coords, 3, 1)
	if err != nil {
		return nil, fmt.Errorf("trellis2: decoder level %d: %w", level, err)
	}
	w := weights{f: f}
	b := graph.NewBuilder(fmt.Sprintf("trellis2_decoder_level%d", level))
	N, C := len(coords), cfg.ModelChannels[level]
	nbr := b.Const("neighbors", tensor.FromI32(nb, N, 27))
	conv := func(sb *graph.Builder, x *graph.Value, w weights) *graph.Value {
		return sb.Op("ingot.SparseConv", nil, x, nbr, sb.Const("weight", w.f32("weight")), sb.Const("bias", w.f32("bias")))
	}

	var h *graph.Value
	if level == 0 {
		wt, bias := w.linear("from_latent")
		h = b.Scope("from_latent").Linear(b.Input("latent", tensor.F32, N, cfg.LatentChannels), wt, bias)
	} else {
		// SparseResBlockC2S3d, second half: each cell takes its sub-cell's
		// slice of the parent's widened features.
		if len(src) != N {
			return nil, fmt.Errorf("trellis2: decoder level %d: %d source rows for %d cells", level, len(src), N)
		}
		up := level - 1
		P := -1 // the previous level's cell count, fixed at run time
		prev := cfg.ModelChannels[up]
		ub := b.Scope(fmt.Sprintf("blocks.%d.%d", up, cfg.NumBlocks[up]))
		wu := w.scope(fmt.Sprintf("blocks.%d.%d", up, cfg.NumBlocks[up]))
		idx := ub.Const("src", tensor.FromI32(src, N))
		gather := func(v *graph.Value, ch int) *graph.Value {
			return ub.Op("Gather", graph.Attr("axis", 0), ub.Reshape(v, -1, int64(ch)), idx)
		}
		hg := gather(b.Input("parent_h", tensor.F32, P, 8*C), C)
		xg := gather(b.Input("parent_x", tensor.F32, P, prev), prev/8)
		// repeat_interleave over channels: [N, prev/8] → [N, C].
		rep := C / (prev / 8)
		skip := ub.Reshape(ub.Op("Tile", nil, ub.Reshape(xg, int64(N), int64(prev/8), 1), ub.Ints(1, 1, int64(rep))), int64(N), int64(C))
		h = ub.Add(conv(ub.Scope("conv2"), ub.SiLU(ub.LayerNorm(hg, C, nil, nil, 1e-6)), wu.scope("conv2")), skip)
	}

	for j := range cfg.NumBlocks[level] {
		// SparseConvNeXtBlock3d: conv → norm → MLP, residual.
		bb := b.Scope(fmt.Sprintf("blocks.%d.%d", level, j))
		wb := w.scope(fmt.Sprintf("blocks.%d.%d", level, j))
		y := bb.LayerNorm(conv(bb.Scope("conv"), h, wb.scope("conv")), C, wb.f32("norm.weight"), wb.f32("norm.bias"), 1e-6)
		wt, bias := wb.linear("mlp.0")
		y = bb.SiLU(bb.Linear(y, wt, bias))
		wt, bias = wb.linear("mlp.2")
		h = bb.Add(h, bb.Linear(y, wt, bias))
	}

	if level == cfg.Levels()-1 {
		wt, bias := w.linear("output_layer")
		b.Output("out", b.Scope("output_layer").Linear(b.LayerNorm(h, C, nil, nil, 1e-5), wt, bias))
		return b.Build()
	}
	// SparseResBlockC2S3d, first half: predict the subdivision and widen
	// the features eightfold for the children.
	ub := b.Scope(fmt.Sprintf("blocks.%d.%d", level, cfg.NumBlocks[level]))
	wu := w.scope(fmt.Sprintf("blocks.%d.%d", level, cfg.NumBlocks[level]))
	if cfg.PredSubdiv {
		wt, bias := wu.linear("to_subdiv")
		b.Output("subdiv", ub.Scope("to_subdiv").Linear(h, wt, bias))
	}
	y := ub.SiLU(ub.LayerNorm(h, C, wu.f32("norm1.weight"), wu.f32("norm1.bias"), 1e-6))
	b.Output("h", conv(ub.Scope("conv1"), y, wu.scope("conv1")))
	b.Output("x", b.Op("Identity", nil, h))
	return b.Build()
}

// Decoded is a structured-latent decoder's result.
type Decoded struct {
	// Levels holds, per level but the last, the cells and the subdivision
	// that produced the next level's cells.
	Levels []DecodedLevel
	// Coords are the cells of the last level, at 2^(levels−1) times the
	// latent grid's resolution; Out is their [N, out_channels] features.
	Coords []sparse.Coord
	Out    *tensor.Tensor
}

// DecodedLevel is one level's cells and its subdivision: Logits [N, 8]
// when the decoder predicted it (nil when it was guided), Mask the
// children kept.
type DecodedLevel struct {
	Coords []sparse.Coord
	Logits []float32
	Mask   []bool
}

// Guide returns the subdivision masks of every level, for decoding another
// latent over the same cells (the texture decoder follows the shape
// decoder's subdivision).
func (d *Decoded) Guide() [][]bool {
	g := make([][]bool, len(d.Levels))
	for i, l := range d.Levels {
		g[i] = l.Mask
	}
	return g
}

// Decode runs a structured-latent decoder over the latent's cells, level
// by level on device (graph.CompileOn). guide gives each level's
// subdivision for a decoder that does not predict its own; it must be nil
// for one that does. stopAt > 0 stops after that many levels, returning
// only the cells reached (Out is nil) — the cascade's upsampling.
func Decode(cfg DecoderConfig, f *safetensors.File, coords []sparse.Coord, latent *tensor.Tensor, guide [][]bool, device string, stopAt int) (*Decoded, error) {
	if cfg.PredSubdiv != (guide == nil) {
		return nil, fmt.Errorf("trellis2: decoder predicts subdivision = %v but guide given = %v", cfg.PredSubdiv, guide != nil)
	}
	if guide != nil && len(guide) != cfg.Levels()-1 {
		return nil, fmt.Errorf("trellis2: %d guide levels for a %d-level decoder", len(guide), cfg.Levels())
	}
	d := &Decoded{}
	feeds := map[string]*tensor.Tensor{"latent": latent}
	var src []int32
	for level := range cfg.Levels() {
		if stopAt > 0 && level == stopAt {
			break
		}
		g, err := BuildDecoderLevel(cfg, f, level, coords, src)
		if err != nil {
			return nil, err
		}
		r, err := graph.CompileOn(g, device)
		if err != nil {
			return nil, fmt.Errorf("trellis2: decoder level %d: %w", level, err)
		}
		res, err := r.Run(feeds)
		if err != nil {
			return nil, fmt.Errorf("trellis2: decoder level %d: %w", level, err)
		}
		if level == cfg.Levels()-1 {
			d.Coords, d.Out = coords, res["out"]
			return d, nil
		}
		lv := DecodedLevel{Coords: coords}
		if cfg.PredSubdiv {
			lv.Logits = append([]float32(nil), res["subdiv"].F32()...)
			lv.Mask = make([]bool, len(lv.Logits))
			for i, v := range lv.Logits {
				lv.Mask[i] = v > 0
			}
		} else {
			lv.Mask = guide[level]
		}
		d.Levels = append(d.Levels, lv)
		if coords, src, err = sparse.Subdivide(coords, lv.Mask); err != nil {
			return nil, fmt.Errorf("trellis2: decoder level %d: %w", level, err)
		}
		feeds = map[string]*tensor.Tensor{"parent_h": res["h"], "parent_x": res["x"]}
	}
	d.Coords = coords
	return d, nil
}
