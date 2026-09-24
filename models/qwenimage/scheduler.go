package qwenimage

import (
	"fmt"
	"math"
)

// SchedulerConfig is scheduler/scheduler_config.json
// (FlowMatchEulerDiscreteScheduler).
type SchedulerConfig struct {
	NumTrainTimesteps  int     `json:"num_train_timesteps"`
	BaseImageSeqLen    int     `json:"base_image_seq_len"`
	MaxImageSeqLen     int     `json:"max_image_seq_len"`
	BaseShift          float64 `json:"base_shift"`
	MaxShift           float64 `json:"max_shift"`
	ShiftTerminal      float64 `json:"shift_terminal"`
	TimeShiftType      string  `json:"time_shift_type"`
	UseDynamicShifting bool    `json:"use_dynamic_shifting"`
	InvertSigmas       bool    `json:"invert_sigmas"`
	StochasticSampling bool    `json:"stochastic_sampling"`
	UseKarrasSigmas    bool    `json:"use_karras_sigmas"`
	UseExponential     bool    `json:"use_exponential_sigmas"`
	UseBetaSigmas      bool    `json:"use_beta_sigmas"`
}

// LoadSchedulerConfig reads dir/scheduler_config.json.
func LoadSchedulerConfig(dir string) (SchedulerConfig, error) {
	var c SchedulerConfig
	if err := readJSON(dir, "scheduler_config.json", &c); err != nil {
		return c, err
	}
	if !c.UseDynamicShifting || c.TimeShiftType != "exponential" || c.InvertSigmas || c.StochasticSampling ||
		c.UseKarrasSigmas || c.UseExponential || c.UseBetaSigmas {
		return c, fmt.Errorf("qwenimage: scheduler variant not implemented: %+v", c)
	}
	return c, nil
}

// Schedule is a flow-matching Euler schedule: Sigmas has one more entry than
// steps (the trailing 0).
type Schedule struct {
	Sigmas    []float32
	Timesteps []float32 // σ·num_train_timesteps
}

// NewSchedule reproduces QwenImage21Pipeline's schedule: sigmas
// linspace(1, 1/N, N), shifted exponentially by μ (linear in the image
// token count between base and max), stretched to end at shift_terminal.
// Arithmetic follows numpy/torch float32 rounding step by step (explicit
// conversions stop the compiler fusing into FMAs), so the values match the
// reference bit for bit.
func NewSchedule(c SchedulerConfig, steps, imageSeqLen int) Schedule {
	m := (c.MaxShift - c.BaseShift) / float64(c.MaxImageSeqLen-c.BaseImageSeqLen)
	mu := float64(imageSeqLen)*m + (c.BaseShift - m*float64(c.BaseImageSeqLen))
	e := float32(math.Exp(mu))
	sig := make([]float32, steps)
	for i := range steps {
		// np.linspace in float64, then astype(float32).
		var lin float64 = 1
		if steps > 1 {
			lin = 1 + float64(i)*((1/float64(steps)-1)/float64(steps-1))
		}
		t := float32(lin)
		x := float32(float32(1)/t) - 1
		sig[i] = e / float32(e+x)
	}
	if c.ShiftTerminal != 0 {
		scale := float32(1-sig[steps-1]) / float32(1-c.ShiftTerminal)
		for i, t := range sig {
			sig[i] = 1 - float32(float32(1-t)/scale)
		}
	}
	s := Schedule{Sigmas: append(sig, 0), Timesteps: make([]float32, steps)}
	for i, v := range sig {
		s.Timesteps[i] = float32(v * float32(c.NumTrainTimesteps))
	}
	return s
}

// ModelTime is the timestep the pipeline hands the transformer at step i:
// timestep / 1000 in float32 (not σ itself — the round trip can differ in
// the last bit).
func (s Schedule) ModelTime(i int) float32 { return float32(s.Timesteps[i] / 1000) }

// Step applies one Euler step in place: x += (σ[i+1] − σ[i]) · v.
func (s Schedule) Step(i int, x, v []float32) {
	dt := float32(s.Sigmas[i+1] - s.Sigmas[i])
	for j := range x {
		x[j] = x[j] + float32(dt*v[j])
	}
}
