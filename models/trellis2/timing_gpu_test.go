package trellis2

import (
	"os"
	"sort"
	"strings"
	"testing"

	"github.com/giraffesyo/ingot/graph"
	"github.com/giraffesyo/ingot/tensor"
)

// gpuProfile logs a GPU session's placement and per-op GPU time; with
// TRELLIS2_TIMING_NODES set, also every node of block 1.
func gpuProfile(t *testing.T, r graph.Runner, feeds map[string]*tensor.Tensor) {
	gs, ok := r.(*graph.GPUSession)
	if !ok {
		return
	}
	t.Logf("  gpu steps %d, cpu steps %d, flushes %d %v, gpu time %.2fs", gs.GPUSteps, gs.CPUSteps, gs.Flushes, gs.FlushedBy, gs.GPUTime.Seconds())
	gs.Profile = true
	res, err := gs.Run(feeds)
	gs.Profile = false
	if err != nil {
		t.Fatal(err)
	}
	gs.Release(res)
	type kv struct {
		op string
		s  float64
	}
	var st []kv
	var total float64
	for op, d := range gs.OpTime {
		st = append(st, kv{op, d.Seconds()})
		total += d.Seconds()
	}
	sort.Slice(st, func(i, j int) bool { return st[i].s > st[j].s })
	t.Logf("  per-node GPU time, summed: %.3fs", total)
	for _, s := range st[:min(12, len(st))] {
		t.Logf("  %-22s %8.3fs", s.op, s.s)
	}
	if os.Getenv("TRELLIS2_TIMING_NODES") != "" {
		var ns []kv
		for n, d := range gs.NodeTime {
			if strings.Contains(n.Name, "blocks.1/") || strings.Contains(n.Name, "blocks.1.") {
				ns = append(ns, kv{n.String(), d.Seconds()})
			}
		}
		sort.Slice(ns, func(i, j int) bool { return ns[i].s > ns[j].s })
		for _, s := range ns {
			t.Logf("    %-60s %7.2fms", s.op, s.s*1e3)
		}
	}
	gs.OpTime, gs.NodeTime = nil, nil
}
