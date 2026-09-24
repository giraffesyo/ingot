package qwenimage

import (
	"math"
	"path/filepath"
	"testing"
)

// TestScheduleExact: sigmas and timesteps match the reference bit for bit.
func TestScheduleExact(t *testing.T) {
	ref := loadRef(t, "pipeline")
	cfg, err := LoadSchedulerConfig(filepath.Join(snapshotDir(t), "scheduler"))
	if err != nil {
		t.Fatal(err)
	}
	n := ref.tensor(t, "latents0").Shape()[0]
	s := NewSchedule(cfg, len(ref.tensor(t, "timesteps").F32()), n)
	for name, got := range map[string][]float32{"sigmas": s.Sigmas, "timesteps": s.Timesteps} {
		want := ref.tensor(t, name).F32()
		if len(got) != len(want) {
			t.Fatalf("%s: %d values, want %d", name, len(got), len(want))
		}
		for i := range want {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Errorf("%s[%d] = %.9g, want %.9g", name, i, got[i], want[i])
			}
		}
	}
	t.Logf("sigmas %v", s.Sigmas)
}
