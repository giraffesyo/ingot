package wan

import (
	"fmt"
	"math"
)

// SchedulerConfig is scheduler/scheduler_config.json
// (UniPCMultistepScheduler).
type SchedulerConfig struct {
	NumTrainTimesteps int     `json:"num_train_timesteps"`
	FlowShift         float64 `json:"flow_shift"`
	SolverOrder       int     `json:"solver_order"`
	SolverType        string  `json:"solver_type"`
	PredictionType    string  `json:"prediction_type"`
	PredictX0         bool    `json:"predict_x0"`
	UseFlowSigmas     bool    `json:"use_flow_sigmas"`
	LowerOrderFinal   bool    `json:"lower_order_final"`
	FinalSigmasType   string  `json:"final_sigmas_type"`
	DisableCorrector  []int   `json:"disable_corrector"`
	Thresholding      bool    `json:"thresholding"`
	DynamicShifting   bool    `json:"use_dynamic_shifting"`
	KarrasSigmas      bool    `json:"use_karras_sigmas"`
}

// LoadSchedulerConfig reads dir/scheduler_config.json and rejects variants
// not implemented here.
func LoadSchedulerConfig(dir string) (SchedulerConfig, error) {
	var c SchedulerConfig
	if err := readJSON(dir, "scheduler_config.json", &c); err != nil {
		return c, err
	}
	if !c.UseFlowSigmas || c.PredictionType != "flow_prediction" || !c.PredictX0 || c.SolverType != "bh2" ||
		c.SolverOrder != 2 || c.FinalSigmasType != "zero" || len(c.DisableCorrector) != 0 || c.Thresholding ||
		c.DynamicShifting || c.KarrasSigmas {
		return c, fmt.Errorf("wan: scheduler variant not implemented: %+v", c)
	}
	return c, nil
}

// UniPC is UniPCMultistepScheduler in its flow-matching form (bh2,
// order 2, predicting x0, corrector on): one Step per model evaluation.
type UniPC struct {
	cfg SchedulerConfig
	// Sigmas has one entry per step plus the final 0; float32 values as
	// the reference stores them.
	Sigmas []float32
	// Timesteps is what the transformer is told at each step:
	// int(σ·num_train_timesteps).
	Timesteps []int64

	step       int
	outputs    [2][]float32 // the last two x0 predictions, newest last
	lowerOrder int
	thisOrder  int
	last       []float32 // the sample the last predictor started from
}

// NewUniPC sets up a run of n steps (set_timesteps): sigmas linear from 1
// to 1/num_train_timesteps, shifted by flow_shift.
func NewUniPC(cfg SchedulerConfig, n int) *UniPC {
	u := &UniPC{cfg: cfg}
	N := float64(cfg.NumTrainTimesteps)
	for i := range n {
		// np.linspace(1, 1/N, n+1)[:-1]
		s := 1 + float64(i)*((1/N-1)/float64(n))
		s = cfg.FlowShift * s / (1 + (cfg.FlowShift-1)*s)
		if i == 0 && math.Abs(s-1) < 1e-6 {
			s -= 1e-6
		}
		u.Timesteps = append(u.Timesteps, int64(s*N))
		u.Sigmas = append(u.Sigmas, float32(s))
	}
	u.Sigmas = append(u.Sigmas, 0)
	return u
}

// lambda is log(α) − log(σ) of a flow sigma (α = 1 − σ).
func lambda(s float32) float64 {
	return math.Log(float64(1-s)) - math.Log(float64(s))
}

// Step advances sample by one step given the model's velocity prediction;
// it returns the new sample in a fresh slice (sample may be overwritten
// with it afterwards).
func (u *UniPC) Step(velocity, sample []float32) []float32 {
	sigma := u.Sigmas[u.step]
	// convert_model_output: x0 = x − σ·v.
	x0 := make([]float32, len(sample))
	for i := range x0 {
		x0[i] = sample[i] - sigma*velocity[i]
	}
	if u.step > 0 && u.last != nil {
		sample = u.correct(x0)
	}
	u.outputs[0], u.outputs[1] = u.outputs[1], x0
	order := u.cfg.SolverOrder
	if u.cfg.LowerOrderFinal {
		order = min(order, len(u.Timesteps)-u.step)
	}
	u.thisOrder = min(order, u.lowerOrder+1)
	u.last = append(u.last[:0:0], sample...) // the caller may overwrite sample with the result
	out := u.predict(sample)
	if u.lowerOrder < u.cfg.SolverOrder {
		u.lowerOrder++
	}
	u.step++
	return out
}

// phiCoeffs returns hh = −h, h·φ₁ = expm1(hh), B(h) = expm1(hh) (bh2) and
// the b vector for the given order.
func phiCoeffs(h float64, order int) (hPhi1, bh float64, b []float64) {
	hh := -h
	hPhi1 = math.Expm1(hh)
	hPhiK := hPhi1/hh - 1
	bh = math.Expm1(hh)
	fact := 1.0
	for i := 1; i <= order; i++ {
		b = append(b, hPhiK*fact/bh)
		fact *= float64(i + 1)
		hPhiK = hPhiK/hh - 1/fact
	}
	return hPhi1, bh, b
}

// predict is multistep_uni_p_bh_update.
func (u *UniPC) predict(x []float32) []float32 {
	st, s0 := u.Sigmas[u.step+1], u.Sigmas[u.step]
	at := float64(1 - st)
	h := lambda(st) - lambda(s0)
	hPhi1, bh, _ := phiCoeffs(h, u.thisOrder)
	m0 := u.outputs[1]
	a := float64(st) / float64(s0)
	out := make([]float32, len(x))
	if u.thisOrder == 1 {
		for i := range out {
			out[i] = float32(a*float64(x[i]) - at*hPhi1*float64(m0[i]))
		}
		return out
	}
	// Order 2: D1 = (m₋₁ − m0)/r₁, ρ = 0.5.
	rk := (lambda(u.Sigmas[u.step-1]) - lambda(s0)) / h
	m1 := u.outputs[0]
	c := at * bh * 0.5 / rk
	for i := range out {
		out[i] = float32(a*float64(x[i]) - at*hPhi1*float64(m0[i]) - c*float64(m1[i]-m0[i]))
	}
	return out
}

// correct is multistep_uni_c_bh_update: it recomputes the last step from
// the sample that step started from, with this step's prediction x0 —
// the result replaces the predictor's sample.
func (u *UniPC) correct(x0 []float32) []float32 {
	st, s0 := u.Sigmas[u.step], u.Sigmas[u.step-1]
	at := float64(1 - st)
	h := lambda(st) - lambda(s0)
	order := u.thisOrder
	hPhi1, bh, b := phiCoeffs(h, order)
	m0 := u.outputs[1]
	x := u.last
	a := float64(st) / float64(s0)
	out := make([]float32, len(x))
	if order == 1 {
		const rho = 0.5
		for i := range out {
			base := a*float64(x[i]) - at*hPhi1*float64(m0[i])
			out[i] = float32(base - at*bh*rho*float64(x0[i]-m0[i]))
		}
		return out
	}
	// Order 2: solve [[1, 1], [r₁, 1]]·ρ = b.
	rk := (lambda(u.Sigmas[u.step-2]) - lambda(s0)) / h
	det := 1 - rk
	rho0 := (b[0] - b[1]) / det
	rho1 := (b[1] - rk*b[0]) / det
	m1 := u.outputs[0]
	for i := range out {
		base := a*float64(x[i]) - at*hPhi1*float64(m0[i])
		d1 := float64(m1[i]-m0[i]) / rk
		out[i] = float32(base - at*bh*(rho0*d1+rho1*float64(x0[i]-m0[i])))
	}
	return out
}
