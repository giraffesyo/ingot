package trellis2

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/giraffesyo/ingot/graph"
	"github.com/giraffesyo/ingot/tensor"
)

// TestFlowTiming reports one velocity evaluation's time per device and the
// CPU's per-op split. TRELLIS2_TIMING selects the block count (default 4).
func TestFlowTiming(t *testing.T) {
	if os.Getenv("TRELLIS2_TIMING") == "" {
		t.Skip("set TRELLIS2_TIMING=<blocks> to time the flow model")
	}
	blocks := 4
	if os.Getenv("TRELLIS2_TIMING") == "all" {
		blocks = 30
	}
	base := filepath.Join(trellisDir(t), "ckpts", "ss_flow_img_dit_1_3B_64_bf16")
	cfg, err := LoadFlowConfig(base + ".json")
	if err != nil {
		t.Fatal(err)
	}
	coords := GridCoords(cfg.Resolution)
	const L = 1029
	feeds := map[string]*tensor.Tensor{
		"x": tensor.New(tensor.F32, len(coords), cfg.InChannels), "t": tensor.FromF32([]float32{0.5}, 1),
		"cond": tensor.New(tensor.F32, L, cfg.CondChannels),
	}
	for i := range feeds["x"].F32() {
		feeds["x"].F32()[i] = float32(i%17)/17 - 0.5
	}
	for i := range feeds["cond"].F32() {
		feeds["cond"].F32()[i] = float32(i%13)/13 - 0.5
	}
	var cpuOut []float32
	for _, dev := range []string{"cpu", "gpu", "gpu-bf16"} {
		g, err := BuildFlow(cfg, openFile(t, base+".safetensors"), coords, L, blocks, dev != "cpu")
		if err != nil {
			t.Fatal(err)
		}
		r, err := graph.CompileOn(g, dev)
		if err != nil {
			t.Logf("%s: %v", dev, err)
			continue
		}
		var out []float32
		for i := range 3 {
			start := time.Now()
			res, err := r.Run(feeds)
			if err != nil {
				t.Fatal(err)
			}
			out = append(out[:0], res["v"].F32()...)
			r.Release(res)
			t.Logf("%s run %d: %.2fs (%d blocks)", dev, i, time.Since(start).Seconds(), blocks)
		}
		if dev == "cpu" {
			cpuOut = out
			cs := r.(*graph.Session)
			cs.Profile = true
			if res, err := cs.Run(feeds); err == nil {
				cs.Release(res)
			}
			st := cs.Stats()
			for _, s := range st[:min(8, len(st))] {
				t.Logf("  %-22s %4d nodes %8.2fs", s.OpType, s.Count, s.Total.Seconds())
			}
		} else {
			compare(t, dev+" vs cpu", out, cpuOut, 0.5)
		}
	}
}
