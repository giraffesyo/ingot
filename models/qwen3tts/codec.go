package qwen3tts

import (
	"fmt"
	"math"

	"github.com/giraffesyo/ingot/graph"
	"github.com/giraffesyo/ingot/tensor"
)

// CodecConfig is speech_tokenizer/config.json's decoder_config (plus the
// output rate).
type CodecConfig struct {
	SampleRate int `json:"-"`
	Decoder    struct {
		LatentDim        int     `json:"latent_dim"`
		CodebookDim      int     `json:"codebook_dim"`
		CodebookSize     int     `json:"codebook_size"`
		DecoderDim       int     `json:"decoder_dim"`
		HiddenAct        string  `json:"hidden_act"`
		HiddenSize       int     `json:"hidden_size"`
		IntermediateSize int     `json:"intermediate_size"`
		HeadDim          int     `json:"head_dim"`
		NumHeads         int     `json:"num_attention_heads"`
		NumKVHeads       int     `json:"num_key_value_heads"`
		NumLayers        int     `json:"num_hidden_layers"`
		NumQuantizers    int     `json:"num_quantizers"`
		NumSemantic      int     `json:"num_semantic_quantizers"`
		RMSNormEps       float32 `json:"rms_norm_eps"`
		RopeTheta        float64 `json:"rope_theta"`
		SlidingWindow    int     `json:"sliding_window"`
		UpsampleRates    []int   `json:"upsample_rates"`
		UpsamplingRatios []int   `json:"upsampling_ratios"`
		AttentionBias    bool    `json:"attention_bias"`
	} `json:"decoder_config"`
	Encoder                CodecEncoderConfig `json:"encoder_config"`
	EncoderValidQuantizers int                `json:"encoder_valid_num_quantizers"`
	ModelType              string             `json:"model_type"`
	OutRate                int                `json:"output_sample_rate"`
	Upsample               int                `json:"decode_upsample_rate"`
}

func loadCodecConfig(dir string) (CodecConfig, error) {
	var c CodecConfig
	if err := readJSON(dir, "config.json", &c); err != nil {
		return c, err
	}
	d := c.Decoder
	total := 1
	for _, r := range append(append([]int{}, d.UpsampleRates...), d.UpsamplingRatios...) {
		total *= r
	}
	switch {
	case c.ModelType != "qwen3_tts_tokenizer_12hz":
		return c, fmt.Errorf("qwen3tts: speech tokenizer %q not implemented", c.ModelType)
	case d.HiddenAct != "silu" || d.AttentionBias || d.NumHeads != d.NumKVHeads || d.NumSemantic != 1:
		return c, fmt.Errorf("qwen3tts: codec decoder variant not implemented: %+v", d)
	case total != c.Upsample:
		return c, fmt.Errorf("qwen3tts: codec upsample %d != decode_upsample_rate %d", total, c.Upsample)
	}
	c.SampleRate = c.OutRate
	return c, nil
}

// Codec decodes 16-codebook frames to a waveform: the RVQ lookup runs on
// the host (a sum of codebook rows), everything after it — the two RVQ
// output projections, the causal pre-conv, the sliding-window transformer,
// the ConvNeXt upsamplers and the SnakeBeta/transposed-conv vocoder — is
// one graph over a dynamic frame count.
type Codec struct {
	cfg       CodecConfig
	codebooks [][]float32 // per quantizer [size, dim]: embedding_sum / clamp(usage)
	dim       int
	sess      graph.Runner
	rope      *rope
}

// maxCodecFrames bounds one decode window (RoPE table size); the reference
// decodes in 300-frame chunks with 25 frames of left context.
const (
	codecChunk   = 300
	codecContext = 25
)

// NewCodec builds the decoder graph from the checkpoint's speech tokenizer
// on device ("cpu", or "gpu": Metal, nodes without a GPU kernel on the
// CPU).
func (m *Model) NewCodec(device string) (*Codec, error) { return m.newCodec(-1, device) }

// newCodec with layers >= 0 truncates the transformer (tests).
func (m *Model) newCodec(layers int, device string) (cd *Codec, err error) {
	defer catch(&err)
	c := m.Codec
	d := c.Decoder
	w := weights{set: m.codecSet, prefix: "decoder."}
	cd = &Codec{cfg: c, dim: d.CodebookDim / 2}
	for _, group := range []string{"rvq_first", "rvq_rest"} {
		n := 1
		if group == "rvq_rest" {
			n = d.NumQuantizers - d.NumSemantic
		}
		for i := range n {
			wi := w.scope(fmt.Sprintf("quantizer.%s.vq.layers.%d._codebook", group, i))
			sum, usage := wi.f32("embedding_sum"), wi.f32("cluster_usage")
			if s := sum.Shape(); len(s) != 2 || s[0] != d.CodebookSize || s[1] != cd.dim {
				return nil, fmt.Errorf("qwen3tts: codebook shape %v", s)
			}
			cb := make([]float32, d.CodebookSize*cd.dim)
			for r := range d.CodebookSize {
				u := max(usage.F32()[r], 1e-5)
				for j := range cd.dim {
					cb[r*cd.dim+j] = sum.F32()[r*cd.dim+j] / u
				}
			}
			cd.codebooks = append(cd.codebooks, cb)
		}
	}
	if layers < 0 {
		layers = d.NumLayers
	}
	g, err := buildCodecGraph(c, w, layers)
	if err != nil {
		return nil, err
	}
	if device == "gpu" {
		cd.sess, err = graph.CompileGPU(g)
	} else {
		cd.sess, err = graph.Compile(g)
	}
	if err != nil {
		return nil, err
	}
	cd.rope = newRope(d.RopeTheta, d.HeadDim, codecChunk+codecContext)
	return cd, nil
}

// Decode turns frames [F][16] into a waveform of F·1920 samples in [-1, 1],
// chunked like the reference (300 frames, 25 frames of left context).
func (cd *Codec) Decode(frames [][]int64) ([]float32, error) {
	var wav []float32
	for start := 0; start < len(frames); {
		end := min(start+codecChunk, len(frames))
		ctxN := codecContext
		if start-codecContext <= 0 {
			ctxN = start
		}
		out, err := cd.decodeWindow(frames[start-ctxN : end])
		if err != nil {
			return nil, err
		}
		wav = append(wav, out[ctxN*cd.cfg.Upsample:]...)
		start = end
	}
	return wav, nil
}

// decodeWindow runs the graph over one window of frames.
func (cd *Codec) decodeWindow(frames [][]int64) ([]float32, error) {
	out, err := cd.run(frames)
	if err != nil {
		return nil, err
	}
	wav := append([]float32(nil), out["wav"].F32()...)
	cd.sess.Release(out)
	return wav, nil
}

func (cd *Codec) run(frames [][]int64) (map[string]*tensor.Tensor, error) {
	T := len(frames)
	nq := len(cd.codebooks)
	first := tensor.New(tensor.F32, 1, cd.dim, T)
	rest := tensor.New(tensor.F32, 1, cd.dim, T)
	// Each quantizer's rows, transposed to [dim, T]; the rest group sums in
	// quantizer order (as ResidualVectorQuantization.decode does).
	for t, f := range frames {
		if len(f) != nq {
			return nil, fmt.Errorf("qwen3tts: frame %d has %d codes, want %d", t, len(f), nq)
		}
		for q, code := range f {
			code = max(code, 0) // decode clamps padding (-1) to 0
			if int(code) >= len(cd.codebooks[q])/cd.dim {
				return nil, fmt.Errorf("qwen3tts: frame %d code %d = %d outside codebook", t, q, code)
			}
			row := cd.codebooks[q][int(code)*cd.dim : (int(code)+1)*cd.dim]
			dst := rest.F32()
			if q == 0 {
				dst = first.F32()
			}
			for j, v := range row {
				dst[j*T+t] += v
			}
		}
	}
	d := cd.cfg.Decoder
	mask := tensor.New(tensor.F32, T, T)
	for q := range T {
		for k := range T {
			// causal, window keys (q-window, q]
			if k > q || q-k >= d.SlidingWindow {
				mask.F32()[q*T+k] = float32(math.Inf(-1))
			}
		}
	}
	cos, sin := cd.rope.at(0, T)
	return cd.sess.Run(map[string]*tensor.Tensor{"first": first, "rest": rest, "mask": mask, "cos": cos, "sin": sin})
}

// buildCodecGraph is Qwen3TTSTokenizerV2Decoder.forward after the RVQ
// lookup: inputs "first"/"rest" [1, dim, T] (codebook sums), "mask" [T, T]
// (sliding-window causal), "cos"/"sin" [T, head_dim/2]; outputs "rvq" [1,
// codebook_dim, T], "pre_tf" [1, T, latent] and "wav" [1, 1, T·upsample].
func buildCodecGraph(c CodecConfig, w weights, layers int) (*graph.Graph, error) {
	d := c.Decoder
	b := graph.NewBuilder("qwen3tts_codec")
	half := d.CodebookDim / 2
	first := b.Input("first", tensor.F32, 1, half, -1)
	rest := b.Input("rest", tensor.F32, 1, half, -1)
	mask := b.Input("mask", tensor.F32, -1, -1)
	cosV := b.Input("cos", tensor.F32, -1, d.HeadDim/2)
	sinV := b.Input("sin", tensor.F32, -1, d.HeadDim/2)

	conv := func(b *graph.Builder, x *graph.Value, w weights, k, dil, groups int) *graph.Value {
		// Causal (left-padded) stride-1 conv: CausalConvNet.
		return b.Op("Conv", graph.Attr("pads", []int{(k - 1) * dil, 0}, "dilations", []int{dil}, "group", groups),
			x, b.Const("weight", w.f32("conv.weight")), b.Const("bias", w.f32("conv.bias")))
	}
	transConv := func(b *graph.Builder, x *graph.Value, w weights, k, s int) *graph.Value {
		y := b.Op("ConvTranspose", graph.Attr("strides", []int{s}),
			x, b.Const("weight", w.f32("conv.weight")), b.Const("bias", w.f32("conv.bias")))
		if pad := k - s; pad > 0 { // CausalTransConvNet trims the right
			y = b.Op("Slice", nil, y, b.Ints(0), b.Ints(int64(-pad)), b.Ints(2))
		}
		return y
	}
	snake := func(b *graph.Builder, x *graph.Value, w weights) *graph.Value {
		// SnakeBeta: x + 1/(e^β + 1e-9) · sin²(x·e^α), per channel.
		alpha, beta := w.f32("alpha"), w.f32("beta")
		C := alpha.Numel()
		a, ib := tensor.New(tensor.F32, C), tensor.New(tensor.F32, C)
		for i := range C {
			a.F32()[i] = float32(math.Exp(float64(alpha.F32()[i])))
			ib.F32()[i] = 1 / (float32(math.Exp(float64(beta.F32()[i]))) + 1e-9)
		}
		return b.Snake(x, a, ib)
	}

	// RVQ output projections (1x1 convs, no bias) and their sum.
	proj := func(x *graph.Value, name string) *graph.Value {
		return b.Op("Conv", nil, x, b.Const(name, w.f32("quantizer."+name+".output_proj.weight")))
	}
	x := b.Add(proj(first, "rvq_first"), proj(rest, "rvq_rest"))
	b.Output("rvq", x)
	x = conv(b.Scope("pre_conv"), x, w.scope("pre_conv"), 3, 1, 1) // [1, latent, T]

	// Pre-transformer over [T, hidden].
	pt, bt := w.scope("pre_transformer"), b.Scope("pre_transformer")
	H, dh := d.NumHeads, d.HeadDim
	h := bt.Linear(bt.Reshape(bt.Transpose(x, 0, 2, 1), -1, int64(d.LatentDim)), pt.f32("input_proj.weight"), pt.f32("input_proj.bias"))
	scale := float32(1 / math.Sqrt(float64(dh)))
	for i := range layers {
		bi, wi := bt.Scope(fmt.Sprintf("layers.%d", i)), pt.scope(fmt.Sprintf("layers.%d", i))
		n := bi.RMSNorm(h, wi.f32("input_layernorm.weight"), d.RMSNormEps)
		q := bi.Linear(n, wi.f32("self_attn.q_proj.weight"), nil)
		k := bi.Linear(n, wi.f32("self_attn.k_proj.weight"), nil)
		v := bi.Linear(n, wi.f32("self_attn.v_proj.weight"), nil)
		q = bi.Op("ingot.RoPE", graph.Attr("layout", 1), bi.Reshape(q, -1, int64(H), int64(dh)), cosV, sinV)
		k = bi.Op("ingot.RoPE", graph.Attr("layout", 1), bi.Reshape(k, -1, int64(H), int64(dh)), cosV, sinV)
		o := bi.Op("ingot.SDPA", graph.Attr("scale", scale, "a_layout", 1, "b_layout", 1, "v_layout", 1, "stride_out", 1),
			bi.Reshape(q, 1, -1, int64(H), int64(dh)), bi.Reshape(k, 1, -1, int64(H), int64(dh)),
			bi.Reshape(v, 1, -1, int64(H), int64(dh)), mask)
		o = bi.Linear(bi.Reshape(o, -1, int64(H*dh)), wi.f32("self_attn.o_proj.weight"), nil)
		h = bi.Add(h, bi.Mul(o, bi.Const("attn_scale", wi.f32("self_attn_layer_scale.scale"))))

		n = bi.RMSNorm(h, wi.f32("post_attention_layernorm.weight"), d.RMSNormEps)
		gate := bi.SiLU(bi.Linear(n, wi.f32("mlp.gate_proj.weight"), nil))
		up := bi.Linear(n, wi.f32("mlp.up_proj.weight"), nil)
		m := bi.Linear(bi.Mul(gate, up), wi.f32("mlp.down_proj.weight"), nil)
		h = bi.Add(h, bi.Mul(m, bi.Const("mlp_scale", wi.f32("mlp_layer_scale.scale"))))
	}
	h = bt.RMSNorm(h, pt.f32("norm.weight"), d.RMSNormEps)
	h = bt.Linear(h, pt.f32("output_proj.weight"), pt.f32("output_proj.bias")) // [T, latent]
	b.Output("pre_tf", bt.Reshape(h, 1, -1, int64(d.LatentDim)))
	x = b.Transpose(b.Reshape(h, 1, -1, int64(d.LatentDim)), 0, 2, 1) // [1, latent, T]

	// Upsample: transposed conv (k = s) then a ConvNeXt block, per ratio.
	for i, r := range d.UpsamplingRatios {
		bu, wu := b.Scope(fmt.Sprintf("upsample.%d", i)), w.scope(fmt.Sprintf("upsample.%d", i))
		x = transConv(bu.Scope("0"), x, wu.scope("0"), r, r)
		wc, bc := wu.scope("1"), bu.Scope("1")
		y := conv(bc.Scope("dwconv"), x, wc.scope("dwconv"), 7, 1, d.LatentDim)
		y = bc.Transpose(y, 0, 2, 1) // [1, T, C]
		y = bc.LayerNorm(y, d.LatentDim, wc.f32("norm.weight"), wc.f32("norm.bias"), 1e-6)
		y = bc.Linear(bc.Reshape(y, -1, int64(d.LatentDim)), wc.f32("pwconv1.weight"), wc.f32("pwconv1.bias"))
		y = bc.Op("Gelu", nil, y)
		y = bc.Linear(y, wc.f32("pwconv2.weight"), wc.f32("pwconv2.bias"))
		y = bc.Mul(y, bc.Const("gamma", wc.f32("gamma")))
		y = bc.Transpose(bc.Reshape(y, 1, -1, int64(d.LatentDim)), 0, 2, 1)
		x = bc.Add(x, y)
	}

	// Vocoder: causal conv in, then per rate SnakeBeta → transposed conv
	// (k = 2r, s = r) → three dilated residual units, then SnakeBeta →
	// causal conv to one channel, clamped.
	wd, bd := w.scope("decoder"), b.Scope("decoder")
	x = conv(bd.Scope("0"), x, wd.scope("0"), 7, 1, 1)
	for i, r := range d.UpsampleRates {
		bb, wb := bd.Scope(fmt.Sprintf("%d.block", i+1)), wd.scope(fmt.Sprintf("%d.block", i+1))
		x = snake(bb.Scope("0"), x, wb.scope("0"))
		x = transConv(bb.Scope("1"), x, wb.scope("1"), 2*r, r)
		for j, dil := range []int{1, 3, 9} {
			bj, wj := bb.Scope(fmt.Sprintf("%d", j+2)), wb.scope(fmt.Sprintf("%d", j+2))
			y := snake(bj.Scope("act1"), x, wj.scope("act1"))
			y = conv(bj.Scope("conv1"), y, wj.scope("conv1"), 7, dil, 1)
			y = snake(bj.Scope("act2"), y, wj.scope("act2"))
			y = conv(bj.Scope("conv2"), y, wj.scope("conv2"), 1, 1, 1)
			x = bj.Add(x, y)
		}
	}
	n := len(d.UpsampleRates)
	x = snake(bd.Scope(fmt.Sprint(n+1)), x, wd.scope(fmt.Sprint(n+1)))
	x = conv(bd.Scope(fmt.Sprint(n+2)), x, wd.scope(fmt.Sprint(n+2)), 7, 1, 1)
	b.Output("wav", b.Op("Clip", nil, x, b.Scalar(-1), b.Scalar(1)))
	return b.Build()
}
