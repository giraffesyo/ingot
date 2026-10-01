// Command onnxrun loads an ONNX model and runs it: on inputs from .npy
// files, or seeded random ones — for conformance checks against ONNX
// Runtime / PyTorch (-ref compares outputs with reference .npy files) and
// quick timings.
//
//	onnxrun -model m.onnx -in input_values=x.npy -out outdir
//	onnxrun -model m.onnx -random -dim 16000 -runs 20 -device gpu
//	onnxrun -model m.onnx -in x=x.npy -ref logits=logits_ort.npy
//
// Inputs not given with -in are random when -random is set (f32 normal,
// integers in [0, 10), bools false); their dynamic dims take -dim, or a
// full shape from -shape name=1x3x224x224.
package main

import (
	"flag"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/giraffesyo/ingot/graph"
	"github.com/giraffesyo/ingot/onnx"
	"github.com/giraffesyo/ingot/tensor"
)

type kv map[string]string

func (m kv) String() string { return fmt.Sprint(map[string]string(m)) }
func (m kv) Set(s string) error {
	k, v, ok := strings.Cut(s, "=")
	if !ok || k == "" {
		return fmt.Errorf("want name=value, got %q", s)
	}
	m[k] = v
	return nil
}

func main() {
	model := flag.String("model", "", "ONNX model (required)")
	ins, shapes, refs := kv{}, kv{}, kv{}
	flag.Var(ins, "in", "input from a .npy file: name=path (repeatable)")
	flag.Var(shapes, "shape", "shape of a random input: name=1x3x224x224 (repeatable)")
	flag.Var(refs, "ref", "compare an output with a reference .npy: name=path (repeatable)")
	random := flag.Bool("random", false, "fill inputs not given with -in with seeded random data")
	dim := flag.Int("dim", 1, "value for dynamic dims of random inputs")
	seed := flag.Uint64("seed", 1, "random input seed")
	device := flag.String("device", "cpu", "cpu, gpu, gpu-bf16 or auto")
	runs := flag.Int("runs", 1, "runs to time (reports the median)")
	out := flag.String("out", "", "directory to write each output as <name>.npy")
	flag.Parse()
	if *model == "" {
		fmt.Fprintln(os.Stderr, "onnxrun: -model is required")
		flag.Usage()
		os.Exit(2)
	}
	t0 := time.Now()
	m, err := onnx.DecodeFile(*model)
	if err != nil {
		fail(err)
	}
	g, err := graph.FromONNX(m)
	if err != nil {
		fail(err)
	}
	inputs := g.Inputs // the compiled graph's inputs, read before optimisation
	r, err := graph.CompileOn(g, *device)
	if err != nil {
		fail(err)
	}
	fmt.Fprintf(os.Stderr, "onnxrun: %s loaded on %s in %.2f s\n", filepath.Base(*model), *device, time.Since(t0).Seconds())

	rng := rand.New(rand.NewPCG(*seed, *seed^0x5eed))
	feeds := map[string]*tensor.Tensor{}
	for name, path := range ins {
		t, err := readNPY(path)
		if err != nil {
			fail(err)
		}
		feeds[name] = t
	}
	for _, v := range inputs {
		if v.Const != nil || feeds[v.Name] != nil {
			continue
		}
		if !*random {
			fail(fmt.Errorf("input %q (%s %v) not given: pass -in %s=x.npy or -random", v.Name, v.DType, v.Shape, v.Name))
		}
		shape := append([]int(nil), v.Shape...)
		if s, ok := shapes[v.Name]; ok {
			if shape, err = parseShape(s); err != nil {
				fail(err)
			}
		}
		for i, d := range shape {
			if d < 0 {
				shape[i] = *dim
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
			fail(fmt.Errorf("the model has no input %q", name))
		}
	}

	var res map[string]*tensor.Tensor
	var times []time.Duration
	for range max(1, *runs) {
		if res != nil {
			r.Release(res)
		}
		t := time.Now()
		if res, err = r.Run(feeds); err != nil {
			fail(err)
		}
		times = append(times, time.Since(t))
	}
	sort.Slice(times, func(i, j int) bool { return times[i] < times[j] })
	fmt.Fprintf(os.Stderr, "onnxrun: run %v (median of %d; min %v)\n", times[len(times)/2], len(times), times[0])

	names := make([]string, 0, len(res))
	for n := range res {
		names = append(names, n)
	}
	sort.Strings(names)
	for _, n := range names {
		t := res[n]
		fmt.Printf("%-24s %-4s %-18v %s\n", n, t.DType(), t.Shape(), stats(t))
		if p, ok := refs[n]; ok {
			ref, err := readNPY(p)
			if err != nil {
				fail(err)
			}
			fmt.Printf("%-24s vs %s: %s\n", "", filepath.Base(p), compare(t, ref))
		}
		if *out != "" {
			if err := os.MkdirAll(*out, 0o755); err != nil {
				fail(err)
			}
			f, err := os.Create(filepath.Join(*out, sanitize(n)+".npy"))
			if err != nil {
				fail(err)
			}
			if err := writeNPY(f, t); err != nil {
				f.Close()
				fail(fmt.Errorf("output %s: %w", n, err))
			}
			if err := f.Close(); err != nil {
				fail(err)
			}
		}
	}
	for n := range refs {
		if res[n] == nil {
			fail(fmt.Errorf("-ref %s: no such output", n))
		}
	}
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

func fail(err error) {
	fmt.Fprintln(os.Stderr, "onnxrun:", err)
	os.Exit(1)
}
