package trellis2

import (
	"fmt"
	"path/filepath"
	"testing"

	"github.com/giraffesyo/ingot/sparse"
	"github.com/giraffesyo/ingot/tensor"
)

// cells converts a reference [N, 3] int32 tensor to coordinates.
func cells(t *tensor.Tensor) []sparse.Coord {
	v := t.I32()
	out := make([]sparse.Coord, len(v)/3)
	for i := range out {
		out[i] = sparse.Coord{v[3*i], v[3*i+1], v[3*i+2]}
	}
	return out
}

func sameCells(t *testing.T, what string, got, want []sparse.Coord) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d cells, reference has %d", what, len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("%s: cell %d = %v, reference %v", what, i, got[i], want[i])
		}
	}
}

// TestLatentFlowParity runs the shape structured-latent flow transformer,
// truncated to the reference's block count, over the reference's occupied
// cells — the sparse form of the flow model, positions from coordinates.
func TestLatentFlowParity(t *testing.T) {
	base := filepath.Join(trellisDir(t), "ckpts", "slat_flow_img2shape_dit_1_3B_512_bf16")
	ref := loadRef(t, "slat_flow")
	cfg, err := LoadFlowConfig(base + ".json")
	if err != nil {
		t.Fatal(err)
	}
	cond := ref.tensor(t, "cond")
	g, err := BuildFlow(cfg, openFile(t, base+".safetensors"), cells(ref.tensor(t, "coords")), cond.Shape()[0], int(ref.Meta["blocks"].(float64)), false)
	s := compile(t, g, err)
	out, err := s.Run(map[string]*tensor.Tensor{
		"x": ref.tensor(t, "x"), "cond": cond,
		"t": tensor.FromF32([]float32{float32(ref.Meta["t"].(float64))}, 1),
	})
	if err != nil {
		t.Fatal(err)
	}
	compare(t, "v", out["v"].F32(), ref.tensor(t, "v").F32(), 2e-3)
}

// TestDecoderParity decodes the reference latents through both sparse
// decoders: the shape decoder must predict the reference's subdivision at
// every level (the same cells, in the same order) and its output features;
// the texture decoder, guided by that subdivision, its own.
func TestDecoderParity(t *testing.T) {
	dir := filepath.Join(trellisDir(t), "ckpts")
	shapeRef, texRef := loadRef(t, "shape_dec"), loadRef(t, "tex_dec")
	start := cells(shapeRef.tensor(t, "coords0"))

	cfg, err := LoadDecoderConfig(filepath.Join(dir, "shape_dec_next_dc_f16c32_fp16.json"))
	if err != nil {
		t.Fatal(err)
	}
	shape, err := Decode(cfg, openFile(t, filepath.Join(dir, "shape_dec_next_dc_f16c32_fp16.safetensors")), start, shapeRef.tensor(t, "latent"), nil, "cpu", 0)
	if err != nil {
		t.Fatal(err)
	}
	for i, lv := range shape.Levels {
		sameCells(t, fmt.Sprintf("level %d", i), lv.Coords, cells(shapeRef.tensor(t, fmt.Sprintf("coords%d", i))))
		compare(t, fmt.Sprintf("subdiv%d", i), lv.Logits, shapeRef.tensor(t, fmt.Sprintf("subdiv%d", i)).F32(), 5e-3)
	}
	sameCells(t, "shape cells", shape.Coords, cells(shapeRef.tensor(t, "coords")))
	compare(t, "shape out", shape.Out.F32(), shapeRef.tensor(t, "out").F32(), 5e-3)

	cfg, err = LoadDecoderConfig(filepath.Join(dir, "tex_dec_next_dc_f16c32_fp16.json"))
	if err != nil {
		t.Fatal(err)
	}
	tex, err := Decode(cfg, openFile(t, filepath.Join(dir, "tex_dec_next_dc_f16c32_fp16.safetensors")), start, texRef.tensor(t, "latent"), shape.Guide(), "cpu", 0)
	if err != nil {
		t.Fatal(err)
	}
	sameCells(t, "texture cells", tex.Coords, cells(texRef.tensor(t, "coords")))
	compare(t, "texture out", tex.Out.F32(), texRef.tensor(t, "out").F32(), 5e-3)
}
