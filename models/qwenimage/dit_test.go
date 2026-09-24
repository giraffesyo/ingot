package qwenimage

import (
	"path/filepath"
	"testing"

	"github.com/giraffesyo/ingot/graph"
	"github.com/giraffesyo/ingot/safetensors"
	"github.com/giraffesyo/ingot/tensor"
)

// TestDiTParity runs the 1-layer reference model (real block-0 weights) as
// ingot's prefix + target graphs and compares with diffusers: step 1's text
// rows come from the prefix pass, its image rows from the target pass over
// the prefix K/V; step 2 reuses that K/V at a new timestep and latent — the
// reference's cached mode.
func TestDiTParity(t *testing.T) {
	dir := filepath.Join(snapshotDir(t), "transformer")
	ref := loadRef(t, "dit_l1")
	cfg, err := LoadDiTConfig(dir)
	if err != nil {
		t.Fatal(err)
	}
	set, err := safetensors.OpenDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer set.Close()

	toBools := func(x *tensor.Tensor) []bool {
		out := make([]bool, x.Numel())
		for i, v := range x.U8() {
			out[i] = v != 0
		}
		return out
	}
	hw := ref.Meta["latent_hw"].([]any)
	lh, lw := int(hw[0].(float64)), int(hw[1].(float64))
	l, err := NewDiTLayout(cfg, toBools(ref.tensor(t, "img_mask")), toBools(ref.tensor(t, "encoder_hidden_states_mask")),
		[][3]int{{1, lh, lw}})
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("layout: prefix %d (valid %d), target %d", l.Prefix, len(l.prefixValid), l.Target)

	compile := func(g *graph.Graph, err error) *graph.Session {
		t.Helper()
		if err != nil {
			t.Fatal(err)
		}
		s, err := graph.Compile(g)
		if err != nil {
			t.Fatal(err)
		}
		return s
	}
	pre := compile(BuildDiTPrefix(cfg, set, l, 1, true))
	tgt := compile(BuildDiTTarget(cfg, set, l, 1))

	enc := ref.tensor(t, "encoder_hidden_states")
	pout, err := pre.Run(l.PrefixFeeds(enc.Reshape(enc.Shape()[1:]...), nil))
	if err != nil {
		t.Fatal(err)
	}
	kv := map[string]*tensor.Tensor{"k0": pout["k0"], "v0": pout["v0"]}
	step := func(name string, t0 float32) *tensor.Tensor {
		x := ref.tensor(t, name)
		out, err := tgt.Run(l.TargetFeeds(x.Reshape(x.Shape()[1:]...), t0, kv))
		if err != nil {
			t.Fatal(err)
		}
		return out["out"]
	}
	out1 := step("hidden_states", 0.7)
	out2 := step("hidden_states_step2", 0.4)

	want1 := ref.tensor(t, "out_step1").F32() // [1, prefix+target, 64]
	C := cfg.OutChannels
	compare(t, "step1 prefix rows", pout["out"].F32(), want1[:l.Prefix*C], 2e-4)
	compare(t, "step1 target rows", out1.F32(), want1[l.Prefix*C:], 2e-4)
	compare(t, "step2 cached", out2.F32(), ref.tensor(t, "out_step2_cached").F32(), 2e-4)
}
