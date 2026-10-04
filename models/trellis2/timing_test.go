package trellis2

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/giraffesyo/ingot/graph"
	"github.com/giraffesyo/ingot/tensor"
)

// TestFlowTiming reports one velocity evaluation's time per device (through
// FlowVelocity, as the sampler runs it), the one-off cost of the
// condition's keys and values, and the per-op split. TRELLIS2_TIMING
// selects the block count (default 4, "all" for the whole model);
// TRELLIS2_TIMING_DEVICES a comma-separated subset of cpu,gpu,gpu-bf16.
func TestFlowTiming(t *testing.T) {
	if os.Getenv("TRELLIS2_TIMING") == "" {
		t.Skip("set TRELLIS2_TIMING=<blocks> to time the flow model")
	}
	blocks := 4
	if os.Getenv("TRELLIS2_TIMING") == "all" {
		blocks = 30
	}
	devices := []string{"cpu", "gpu", "gpu-bf16"}
	if d := os.Getenv("TRELLIS2_TIMING_DEVICES"); d != "" {
		devices = strings.Split(d, ",")
	}
	base := filepath.Join(trellisDir(t), "ckpts", "ss_flow_img_dit_1_3B_64_bf16")
	cfg, err := LoadFlowConfig(base + ".json")
	if err != nil {
		t.Fatal(err)
	}
	coords := GridCoords(cfg.Resolution)
	const L = 1029
	x, cond := tensor.New(tensor.F32, len(coords), cfg.InChannels), tensor.New(tensor.F32, L, cfg.CondChannels)
	for i := range x.F32() {
		x.F32()[i] = float32(i%17)/17 - 0.5
	}
	for i := range cond.F32() {
		cond.F32()[i] = float32(i%13)/13 - 0.5
	}
	start := time.Now()
	kv, err := FlowConditions(cfg, openFile(t, base+".safetensors"), blocks, cond, tensor.New(tensor.F32, cond.Shape()...))
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("condition keys/values, 2 conditions, once per sampling run: %.2fs (%d blocks)", time.Since(start).Seconds(), blocks)
	var cpuOut []float32
	for _, dev := range devices {
		g, err := BuildFlow(cfg, openFile(t, base+".safetensors"), coords, L, blocks, dev != "cpu")
		if err != nil {
			t.Fatal(err)
		}
		r, err := graph.CompileOn(g, dev)
		if err != nil {
			t.Logf("%s: %v", dev, err)
			continue
		}
		vel := FlowVelocity(r, len(coords), cfg.InChannels, kv[0], kv[1], nil)
		runs := 3
		if dev != "cpu" {
			runs = 8
		}
		var out []float32
		best := time.Duration(1 << 62)
		for i := range runs {
			start := time.Now()
			if out, err = vel(x.F32(), 0.5, true); err != nil {
				t.Fatal(err)
			}
			d := time.Since(start)
			best = min(best, d)
			t.Logf("%s run %d: %.3fs (%d blocks)", dev, i, d.Seconds(), blocks)
		}
		t.Logf("%s best of %d: %.3fs", dev, runs, best.Seconds())
		feeds := map[string]*tensor.Tensor{"x": x, "t": tensor.FromF32([]float32{0.5}, 1)}
		for name, v := range kv[0] {
			feeds[name] = v
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
			continue
		}
		if cpuOut != nil {
			compare(t, dev+" vs cpu", out, cpuOut, 0.5)
		}
		gpuProfile(t, r, feeds)
		if c, ok := r.(interface{ Close() }); ok {
			c.Close()
		}
	}
}
