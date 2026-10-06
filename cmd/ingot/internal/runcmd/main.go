// Package runcmd is ingot run: it loads an ONNX model and runs it, on
// inputs from .npy files or seeded random ones — for conformance checks
// against ONNX Runtime / PyTorch (--ref compares outputs with reference .npy
// files) and quick timings.
//
//	ingot run --model m.onnx --in input_values=x.npy --out outdir
//	ingot run --model m.onnx --random --dim 16000 --runs 20 --device gpu
//	ingot run --model m.onnx --in x=x.npy --ref logits=logits_ort.npy
//
// Inputs not given with --in are random when --random is set (f32 normal,
// integers in [0, 10), bools false); their dynamic dims take --dim, or a
// full shape from --shape name=1x3x224x224.
package runcmd

import (
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/spf13/cobra"

	"github.com/giraffesyo/ingot/graph"
	"github.com/giraffesyo/ingot/onnx"
	"github.com/giraffesyo/ingot/tensor"
)

type kv map[string]string

func (m kv) String() string {
	if len(m) == 0 {
		return "" // no "(default map[])" in --help
	}
	return fmt.Sprint(map[string]string(m))
}
func (m kv) Set(s string) error {
	k, v, ok := strings.Cut(s, "=")
	if !ok || k == "" {
		return fmt.Errorf("want name=value, got %q", s)
	}
	m[k] = v
	return nil
}
func (m kv) Type() string { return "name=value" }

type options struct {
	model, device, out string
	ins, shapes, refs  kv
	random             bool
	dim, runs          int
	seed               uint64
}

// Command returns the run subcommand.
func Command() *cobra.Command {
	o := options{ins: kv{}, shapes: kv{}, refs: kv{}}
	cmd := &cobra.Command{
		Use:   "run",
		Short: "Run an ONNX model on .npy or random inputs; compare and time it",
		Args:  cobra.NoArgs,
		RunE: func(*cobra.Command, []string) error {
			return run(o)
		},
	}
	f := cmd.Flags()
	f.StringVar(&o.model, "model", "", "ONNX model (required)")
	f.Var(o.ins, "in", "input from a .npy file: name=path (repeatable)")
	f.Var(o.shapes, "shape", "shape of a random input: name=1x3x224x224 (repeatable)")
	f.Var(o.refs, "ref", "compare an output with a reference .npy: name=path (repeatable)")
	f.BoolVar(&o.random, "random", false, "fill inputs not given with --in with seeded random data")
	f.IntVar(&o.dim, "dim", 1, "value for dynamic dims of random inputs")
	f.Uint64Var(&o.seed, "seed", 1, "random input seed")
	f.StringVar(&o.device, "device", "cpu", "cpu, gpu, gpu-bf16 or auto")
	f.IntVar(&o.runs, "runs", 1, "runs to time (reports the median)")
	f.StringVar(&o.out, "out", "", "directory to write each output as <name>.npy")
	_ = cmd.MarkFlagRequired("model")
	return cmd
}

func run(o options) error {
	t0 := time.Now()
	m, err := onnx.DecodeFile(o.model)
	if err != nil {
		return err
	}
	g, err := graph.FromONNX(m)
	if err != nil {
		return err
	}
	inputs := g.Inputs // the compiled graph's inputs, read before optimisation
	r, err := graph.CompileOn(g, o.device)
	if err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "run: %s loaded on %s in %.2f s\n", filepath.Base(o.model), o.device, time.Since(t0).Seconds())

	rng := rand.New(rand.NewPCG(o.seed, o.seed^0x5eed))
	feeds := map[string]*tensor.Tensor{}
	for name, path := range o.ins {
		t, err := readNPY(path)
		if err != nil {
			return err
		}
		feeds[name] = t
	}
	for _, v := range inputs {
		if v.Const != nil || feeds[v.Name] != nil {
			continue
		}
		if !o.random {
			return fmt.Errorf("input %q (%s %v) not given: pass --in %s=x.npy or --random", v.Name, v.DType, v.Shape, v.Name)
		}
		shape := append([]int(nil), v.Shape...)
		if s, ok := o.shapes[v.Name]; ok {
			if shape, err = parseShape(s); err != nil {
				return err
			}
		}
		for i, d := range shape {
			if d < 0 {
				shape[i] = o.dim
			}
		}
		feeds[v.Name] = randomTensor(rng, v.DType, shape)
	}
	for name := range feeds {
		found := false
		for _, v := range inputs {
			found = found || v.Name == name
		}
		if !found {
			return fmt.Errorf("the model has no input %q", name)
		}
	}

	var res map[string]*tensor.Tensor
	var times []time.Duration
	for range max(1, o.runs) {
		if res != nil {
			r.Release(res)
		}
		t := time.Now()
		if res, err = r.Run(feeds); err != nil {
			return err
		}
		times = append(times, time.Since(t))
	}
	sort.Slice(times, func(i, j int) bool { return times[i] < times[j] })
	fmt.Fprintf(os.Stderr, "run: run %v (median of %d; min %v)\n", times[len(times)/2], len(times), times[0])

	names := make([]string, 0, len(res))
	for n := range res {
		names = append(names, n)
	}
	sort.Strings(names)
	for _, n := range names {
		t := res[n]
		fmt.Printf("%-24s %-4s %-18v %s\n", n, t.DType(), t.Shape(), stats(t))
		if p, ok := o.refs[n]; ok {
			ref, err := readNPY(p)
			if err != nil {
				return err
			}
			fmt.Printf("%-24s vs %s: %s\n", "", filepath.Base(p), compare(t, ref))
		}
		if o.out != "" {
			if err := os.MkdirAll(o.out, 0o755); err != nil {
				return err
			}
			f, err := os.Create(filepath.Join(o.out, sanitize(n)+".npy"))
			if err != nil {
				return err
			}
			if err := writeNPY(f, t); err != nil {
				f.Close()
				return fmt.Errorf("output %s: %w", n, err)
			}
			if err := f.Close(); err != nil {
				return err
			}
		}
	}
	for n := range o.refs {
		if res[n] == nil {
			return fmt.Errorf("--ref %s: no such output", n)
		}
	}
	return nil
}

func parseShape(s string) ([]int, error) {
	var out []int
	for _, p := range strings.Split(s, "x") {
		n, err := strconv.Atoi(p)
		if err != nil || n < 0 {
			return nil, fmt.Errorf("shape %q", s)
		}
		out = append(out, n)
	}
	return out, nil
}

func randomTensor(rng *rand.Rand, dt tensor.DType, shape []int) *tensor.Tensor {
	switch dt {
	case tensor.I64:
		t := tensor.New(tensor.I64, shape...)
		for i := range t.I64() {
			t.I64()[i] = int64(rng.IntN(10))
		}
		return t
	case tensor.I32:
		t := tensor.New(tensor.I32, shape...)
		for i := range t.I32() {
			t.I32()[i] = int32(rng.IntN(10))
		}
		return t
	case tensor.Bool, tensor.U8:
		return tensor.New(dt, shape...)
	}
	t := tensor.New(tensor.F32, shape...)
	for i := range t.F32() {
		t.F32()[i] = float32(rng.NormFloat64())
	}
	return t
}

// stats summarises a tensor's values.
func stats(t *tensor.Tensor) string {
	var v []float64
	switch t.DType() {
	case tensor.F32:
		for _, x := range t.F32() {
			v = append(v, float64(x))
		}
	case tensor.I64:
		for _, x := range t.I64() {
			v = append(v, float64(x))
		}
	case tensor.I32:
		for _, x := range t.I32() {
			v = append(v, float64(x))
		}
	default:
		return ""
	}
	if len(v) == 0 {
		return "(empty)"
	}
	lo, hi, sum, nan := math.Inf(1), math.Inf(-1), 0.0, 0
	for _, x := range v {
		if math.IsNaN(x) {
			nan++
			continue
		}
		lo, hi, sum = math.Min(lo, x), math.Max(hi, x), sum+x
	}
	s := fmt.Sprintf("min %.4g max %.4g mean %.4g", lo, hi, sum/float64(len(v)-nan))
	if nan > 0 {
		s += fmt.Sprintf(" NaN×%d", nan)
	}
	return s
}

// compare reports max abs and max relative (to the reference's largest
// magnitude) differences.
func compare(got, ref *tensor.Tensor) string {
	if got.Numel() != ref.Numel() {
		return fmt.Sprintf("SIZE MISMATCH %v vs %v", got.Shape(), ref.Shape())
	}
	if got.DType() != tensor.F32 || ref.DType() != tensor.F32 {
		if got.DType() == tensor.I64 && ref.DType() == tensor.I64 {
			diff := 0
			for i, x := range got.I64() {
				if x != ref.I64()[i] {
					diff++
				}
			}
			return fmt.Sprintf("%d of %d elements differ", diff, got.Numel())
		}
		return fmt.Sprintf("dtypes %s vs %s", got.DType(), ref.DType())
	}
	var maxAbs, scale float64
	for i, x := range got.F32() {
		y := float64(ref.F32()[i])
		maxAbs = math.Max(maxAbs, math.Abs(float64(x)-y))
		scale = math.Max(scale, math.Abs(y))
	}
	shape := ""
	if !got.Shape().Equal(ref.Shape()) {
		shape = fmt.Sprintf(" (shapes %v vs %v)", got.Shape(), ref.Shape())
	}
	return fmt.Sprintf("max abs err %.3g, rel %.3g%s", maxAbs, maxAbs/math.Max(scale, 1e-30), shape)
}

func sanitize(s string) string {
	return strings.Map(func(r rune) rune {
		if r == '/' || r == ':' || r == ' ' {
			return '_'
		}
		return r
	}, s)
}
