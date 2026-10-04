package trellis2

import "math"

// SamplerParams are a FlowEulerGuidanceIntervalSampler's settings
// (pipeline.json "params" plus the sampler's sigma_min).
type SamplerParams struct {
	Steps            int        `json:"steps"`
	GuidanceStrength float64    `json:"guidance_strength"`
	GuidanceRescale  float64    `json:"guidance_rescale"`
	GuidanceInterval [2]float64 `json:"guidance_interval"`
	RescaleT         float64    `json:"rescale_t"`
	SigmaMin         float64    `json:"-"`
}

// Times returns the Steps+1 flow times from 1 to 0, warped by RescaleT:
// t → r·t / (1 + (r−1)·t).
func (p SamplerParams) Times() []float64 {
	r := p.RescaleT
	if r == 0 {
		r = 1
	}
	ts := make([]float64, p.Steps+1)
	for i := range ts {
		t := 1 - float64(i)/float64(p.Steps)
		ts[i] = r * t / (1 + (r-1)*t)
	}
	return ts
}

// Velocity evaluates the flow model at x and time t in [0, 1], with the
// image condition (cond) or the negative one.
type Velocity func(x []float32, t float32, cond bool) ([]float32, error)

// Sample integrates the flow from noise at t=1 to a sample at t=0 with
// Euler steps. Inside the guidance interval the velocity is the
// classifier-free combination s·v(cond) + (1−s)·v(neg), optionally with its
// predicted x₀ rescaled towards the conditional prediction's standard
// deviation; outside it (or at strength 1) only the conditional model runs.
// step, if not nil, is called after each step.
func Sample(v Velocity, noise []float32, p SamplerParams, step func(i, n int)) ([]float32, error) {
	x := append([]float32(nil), noise...)
	ts := p.Times()
	for i := range p.Steps {
		t, tPrev := ts[i], ts[i+1]
		s := 1.0
		if p.GuidanceInterval[0] <= t && t <= p.GuidanceInterval[1] {
			s = p.GuidanceStrength
		}
		var pred []float32
		var err error
		switch s {
		case 1:
			pred, err = v(x, float32(t), true)
		case 0:
			pred, err = v(x, float32(t), false)
		default:
			var pos, neg []float32
			if pos, err = v(x, float32(t), true); err == nil {
				neg, err = v(x, float32(t), false)
			}
			if err == nil {
				pred = guide(x, pos, neg, t, s, p)
			}
		}
		if err != nil {
			return nil, err
		}
		dt := float32(t - tPrev)
		for j := range x {
			x[j] -= dt * pred[j]
		}
		if step != nil {
			step(i+1, p.Steps)
		}
	}
	return x, nil
}

// guide combines the conditional and negative velocities and applies the
// CFG rescale in x₀ space.
func guide(x, pos, neg []float32, t, s float64, p SamplerParams) []float32 {
	pred := make([]float32, len(x))
	for j := range pred {
		pred[j] = float32(s)*pos[j] + float32(1-s)*neg[j]
	}
	if p.GuidanceRescale <= 0 {
		return pred
	}
	// x₀ = (1−σ)·x − (σ + (1−σ)·t)·v
	a, c := float32(1-p.SigmaMin), float32(p.SigmaMin+(1-p.SigmaMin)*t)
	x0 := func(v []float32, j int) float32 { return a*x[j] - c*v[j] }
	ratio := float32(std(len(x), func(j int) float32 { return x0(pos, j) }) / std(len(x), func(j int) float32 { return x0(pred, j) }))
	r := float32(p.GuidanceRescale)
	for j := range pred {
		xc := x0(pred, j)
		pred[j] = (a*x[j] - (r*xc*ratio + (1-r)*xc)) / c
	}
	return pred
}

// std is the sample standard deviation (n−1) of n values.
func std(n int, at func(int) float32) float64 {
	var sum float64
	for j := range n {
		sum += float64(at(j))
	}
	mean := sum / float64(n)
	var ss float64
	for j := range n {
		d := float64(at(j)) - mean
		ss += d * d
	}
	return math.Sqrt(ss / float64(n-1))
}
