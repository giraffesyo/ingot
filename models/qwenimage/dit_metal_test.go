//go:build darwin && arm64

package qwenimage

import (
	"path/filepath"
	"testing"

	"github.com/giraffesyo/ingot/graph"
	"github.com/giraffesyo/ingot/kernels/metal"
	"github.com/giraffesyo/ingot/safetensors"
	"github.com/giraffesyo/ingot/tensor"
)

// TestMetalDiTParity: the GPU target step against the 1-layer diffusers
// reference (step 1 target rows and the cached step 2), with the prefix
// K/V from the CPU prefix graph.
func TestMetalDiTParity(t *testing.T) {
	if !metal.Available() {
		t.Skip("no Metal device")
	}
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
	l, err := NewDiTLayout(cfg, toBools(ref.tensor(t, "img_mask")), toBools(ref.tensor(t, "encoder_hidden_states_mask")),
		[][3]int{{1, int(hw[0].(float64)), int(hw[1].(float64))}})
	if err != nil {
		t.Fatal(err)
	}
	g, err := BuildDiTPrefix(cfg, set, l, 1, false)
	if err != nil {
		t.Fatal(err)
	}
	pre, err := graph.Compile(g)
	if err != nil {
		t.Fatal(err)
	}
	enc := ref.tensor(t, "encoder_hidden_states")
	kv, err := pre.Run(l.PrefixFeeds(enc.Reshape(enc.Shape()[1:]...), nil))
	if err != nil {
		t.Fatal(err)
	}
	m, err := NewMetalDiT(cfg, set, l, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	C := cfg.OutChannels
	// Prefix K/V two ways: from the CPU prefix graph, and computed on the
	// GPU (masked prefix attention, valid-row gather).
	for _, mode := range []string{"cpu prefix", "gpu prefix"} {
		if mode == "cpu prefix" {
			err = m.SetPrefix(kv)
		} else {
			err = m.Prefix(enc.Reshape(enc.Shape()[1:]...), nil)
		}
		if err != nil {
			t.Fatal(err)
		}
		if mode == "gpu prefix" {
			compare(t, "prefix k0 (GPU vs CPU)", f32view(m.kp[0]), kv["k0"].F32(), 1e-4)
			compare(t, "prefix v0 (GPU vs CPU)", f32view(m.vp[0]), kv["v0"].F32(), 1e-4)
		}
		step := func(name string, t0 float32) []float32 {
			x := ref.tensor(t, name)
			out, err := m.Step(x.Reshape(x.Shape()[1:]...), t0)
			if err != nil {
				t.Fatal(err)
			}
			return out.F32()
		}
		compare(t, mode+": step1 target rows", step("hidden_states", 0.7), ref.tensor(t, "out_step1").F32()[l.Prefix*C:], 5e-4)
		compare(t, mode+": step2 cached", step("hidden_states_step2", 0.4), ref.tensor(t, "out_step2_cached").F32(), 5e-4)
	}

	// Fast mode: bf16 GEMM inputs. Error grows to bf16's ~3 significant
	// digits (tolerance 1% of the output range).
	f, err := NewMetalDiT(cfg, set, l, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	f.Fast = true
	if err := f.Prefix(enc.Reshape(enc.Shape()[1:]...), nil); err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct {
		in, want string
		t0       float32
		rows     int
	}{{"hidden_states", "out_step1", 0.7, l.Prefix * C}, {"hidden_states_step2", "out_step2_cached", 0.4, 0}} {
		x := ref.tensor(t, c.in)
		out, err := f.Step(x.Reshape(x.Shape()[1:]...), c.t0)
		if err != nil {
			t.Fatal(err)
		}
		compare(t, "fast (bf16): "+c.want, out.F32(), ref.tensor(t, c.want).F32()[c.rows:], 0.15)
	}
}
