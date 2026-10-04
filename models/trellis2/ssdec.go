package trellis2

import (
	"fmt"

	"github.com/giraffesyo/ingot/graph"
	"github.com/giraffesyo/ingot/safetensors"
	"github.com/giraffesyo/ingot/tensor"
)

// StructureDecoderConfig is the "args" of the SparseStructureDecoder JSON.
type StructureDecoderConfig struct {
	OutChannels    int    `json:"out_channels"`
	LatentChannels int    `json:"latent_channels"`
	NumResBlocks   int    `json:"num_res_blocks"`
	NumResMiddle   int    `json:"num_res_blocks_middle"`
	Channels       []int  `json:"channels"`
	NormType       string `json:"norm_type"`
}

// LoadStructureDecoderConfig reads the decoder's JSON.
func LoadStructureDecoderConfig(path string) (StructureDecoderConfig, error) {
	var c struct {
		Name string                 `json:"name"`
		Args StructureDecoderConfig `json:"args"`
	}
	c.Args.NumResMiddle = 2
	if err := readJSON(path, &c); err != nil {
		return c.Args, err
	}
	if c.Name != "SparseStructureDecoder" || (c.Args.NormType != "" && c.Args.NormType != "layer") || len(c.Args.Channels) == 0 {
		return c.Args, fmt.Errorf("trellis2: %s: structure decoder variant not implemented (name=%q norm_type=%q)", path, c.Name, c.Args.NormType)
	}
	return c.Args, nil
}

// BuildStructureDecoder builds SparseStructureDecoder: input "z" [1,
// latent_channels, n, n, n]; output "logits" [1, out_channels, n·2ᵏ, …]
// for k upsampling stages — occupancy logits of the dense voxel grid.
func BuildStructureDecoder(cfg StructureDecoderConfig, f *safetensors.File, n int) (g *graph.Graph, err error) {
	defer catch(&err)
	w := weights{f: f}
	b := graph.NewBuilder("trellis2_structure_decoder")
	h := conv3d(b.Scope("input_layer"), b.Input("z", tensor.F32, 1, cfg.LatentChannels, n, n, n), w.scope("input_layer"))
	for i := range cfg.NumResMiddle {
		h = resBlock3d(b.Scope(fmt.Sprintf("middle_block.%d", i)), h, w.scope(fmt.Sprintf("middle_block.%d", i)), cfg.Channels[0])
	}
	blk := 0
	for i, ch := range cfg.Channels {
		for range cfg.NumResBlocks {
			h = resBlock3d(b.Scope(fmt.Sprintf("blocks.%d", blk)), h, w.scope(fmt.Sprintf("blocks.%d", blk)), ch)
			blk++
		}
		if i < len(cfg.Channels)-1 {
			sb := b.Scope(fmt.Sprintf("blocks.%d", blk))
			h = pixelShuffle3d(sb, conv3d(sb, h, w.scope(fmt.Sprintf("blocks.%d.conv", blk))), cfg.Channels[i+1], n)
			n *= 2
			blk++
		}
	}
	last := cfg.Channels[len(cfg.Channels)-1]
	h = b.SiLU(channelLayerNorm(b.Scope("out_layer"), h, w.scope("out_layer.0"), last))
	b.Output("logits", conv3d(b.Scope("out_layer"), h, w.scope("out_layer.2")))
	return b.Build()
}

// conv3d is a same-padded 3×3×3 nn.Conv3d.
func conv3d(b *graph.Builder, x *graph.Value, w weights) *graph.Value {
	return b.Op("Conv", graph.Attr("strides", []int{1, 1, 1}, "pads", []int{1, 1, 1, 1, 1, 1}),
		x, b.Const("weight", w.f32("weight")), b.Const("bias", w.f32("bias")))
}

// channelLayerNorm is ChannelLayerNorm32: LayerNorm over the channel axis
// of NCDHW x, via channels-last.
func channelLayerNorm(b *graph.Builder, x *graph.Value, w weights, ch int) *graph.Value {
	y := b.LayerNorm(b.Transpose(x, 0, 2, 3, 4, 1), ch, w.f32("weight"), w.f32("bias"), 1e-5)
	return b.Transpose(y, 0, 4, 1, 2, 3)
}

// resBlock3d is ResBlock3d with equal in/out channels (identity skip).
func resBlock3d(b *graph.Builder, x *graph.Value, w weights, ch int) *graph.Value {
	h := conv3d(b.Scope("conv1"), b.SiLU(channelLayerNorm(b.Scope("norm1"), x, w.scope("norm1"), ch)), w.scope("conv1"))
	h = conv3d(b.Scope("conv2"), b.SiLU(channelLayerNorm(b.Scope("norm2"), h, w.scope("norm2"), ch)), w.scope("conv2"))
	return b.Add(h, x)
}

// pixelShuffle3d rearranges x [1, c·8, n, n, n] to [1, c, 2n, 2n, 2n]: the
// 8 channel groups become the 2×2×2 sub-cells of each cell.
func pixelShuffle3d(b *graph.Builder, x *graph.Value, c, n int) *graph.Value {
	N, C := int64(n), int64(c)
	x = b.Transpose(b.Reshape(x, 1, C, 2, 2, 2, N, N, N), 0, 1, 5, 2, 6, 3, 7, 4)
	return b.Reshape(x, 1, C, 2*N, 2*N, 2*N)
}
