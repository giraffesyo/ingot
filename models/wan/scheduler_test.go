package wan

import (
	"path/filepath"
	"testing"
)

func TestUniPC(t *testing.T) {
	c := loadRef(t, "scheduler")
	cfg, err := LoadSchedulerConfig(filepath.Join(tinyDir(t), "scheduler"))
	if err != nil {
		t.Fatal(err)
	}
	steps := int(c.metaFloat("steps"))
	u := NewUniPC(cfg, steps)
	compare(t, "sigmas", u.Sigmas, c.tensor(t, "sigmas").F32(), 1e-7)
	ts := c.tensor(t, "timesteps").I64()
	for i, v := range u.Timesteps {
		if v != ts[i] {
			t.Fatalf("timesteps %v, want %v", u.Timesteps, ts)
		}
	}
	x := c.tensor(t, "x0").F32()
	n := len(x)
	outs, want := c.tensor(t, "model_outputs").F32(), c.tensor(t, "samples").F32()
	for i := range steps {
		copy(x, u.Step(outs[i*n:(i+1)*n], x)) // in place, as the pipeline does
		compare(t, "step", x, want[i*n:(i+1)*n], 1e-5)
	}
}
