package qwenimage

import (
	"context"
	"encoding/binary"
	"errors"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

func TestBatchStagesShareLoadsAndIsolateFailures(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	jobs := []*batchJob{{}, {ctx: context.Background()}, {ctx: ctx}}
	for _, j := range jobs {
		j.result = &Result{Stages: map[string]time.Duration{}}
	}
	var events []string
	var outcomes []error
	active := false
	runBatchStages(jobs, func(i int, err error) {
		if jobs[i].finished {
			t.Fatal("duplicate completion")
		}
		jobs[i].finished = true
		outcomes = append(outcomes, err)
	}, func(stage string, visit func(func(*batchJob) error)) {
		if active {
			t.Fatal("two stages resident together")
		}
		active = true
		events = append(events, "load:"+stage)
		visit(func(j *batchJob) error {
			events = append(events, stage)
			if stage == "text encoder" && j == jobs[1] {
				return errors.New("bad prompt")
			}
			if stage == "denoise" {
				cancel()
			}
			return nil
		})
		events = append(events, "close:"+stage)
		active = false
	})
	want := []string{"load:text encoder", "text encoder", "text encoder", "text encoder", "close:text encoder", "load:vae encode", "close:vae encode", "load:denoise", "denoise", "close:denoise", "load:decode", "decode", "close:decode"}
	if !reflect.DeepEqual(events, want) {
		t.Fatalf("stage order: %v", events)
	}
	if len(outcomes) != 3 || outcomes[0].Error() != "bad prompt" || !errors.Is(outcomes[1], context.Canceled) || outcomes[2] != nil {
		t.Fatalf("outcomes: %v", outcomes)
	}
}

func TestBatchWeightMappingReused(t *testing.T) {
	dir := t.TempDir()
	header := []byte(`{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}`)
	header = append(header, []byte(strings.Repeat(" ", (8-len(header)%8)%8))...)
	data := binary.LittleEndian.AppendUint64(nil, uint64(len(header)))
	data = append(data, header...)
	data = append(data, 0, 0, 128, 63)
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors"), data, 0600); err != nil {
		t.Fatal(err)
	}
	c := &batchCache{}
	defer c.close()
	one, err := c.open(dir)
	if err != nil {
		t.Fatal(err)
	}
	two, err := c.open(dir)
	if err != nil {
		t.Fatal(err)
	}
	if one != two || len(c.sets) != 1 {
		t.Fatal("checkpoint was reopened instead of reused")
	}
}

func TestGenerateBatchParity(t *testing.T) {
	fullModel(t)
	dir := snapshotDir(t)
	for _, device := range []string{"cpu", "gpu"} {
		if device == "gpu" && !metalAvailable() {
			continue
		}
		t.Run(device, func(t *testing.T) {
			opts := []Options{
				{Prompt: "a small red cube", Resolution: 256, Steps: 1, Seed: 7, Device: device, Fast: true},
				{Prompt: "a small red cube", Resolution: 256, Steps: 1, Seed: 8, Device: device, Fast: true},
				{Prompt: "a blue sphere on a white table", Resolution: 256, Steps: 1, Seed: 9, Device: device},
			}
			var want [][]float32
			var requests []BatchRequest
			for _, opt := range opts {
				result, err := Generate(dir, opt)
				if err != nil {
					t.Fatal(err)
				}
				want = append(want, append([]float32(nil), result.Float.F32()...))
				requests = append(requests, BatchRequest{Options: opt})
			}
			completed := 0
			GenerateBatch(dir, requests, func(i int, result *Result, err error) {
				if err != nil {
					t.Error(err)
					return
				}
				compare(t, "batched vs standalone", result.Float.F32(), want[i], 1e-5)
				completed++
			})
			if completed != len(opts) {
				t.Fatalf("completed %d", completed)
			}
		})
	}
}

func TestBatchTextSessionReuseAndShapeChange(t *testing.T) {
	// A zero-layer text encoder is an independent embedding/slice oracle.
	// It verifies cache reuse, output ownership, and invalidation without weights.
	dir := t.TempDir()
	tdir := filepath.Join(dir, "text_encoder")
	if err := os.Mkdir(tdir, 0700); err != nil {
		t.Fatal(err)
	}
	cfg := `{"text_config":{"hidden_size":4,"intermediate_size":8,"num_hidden_layers":0,"num_attention_heads":1,"num_key_value_heads":1,"head_dim":4,"rms_norm_eps":0.000001,"rope_theta":10000,"vocab_size":3,"hidden_act":"silu"}}`
	if err := os.WriteFile(filepath.Join(tdir, "config.json"), []byte(cfg), 0600); err != nil {
		t.Fatal(err)
	}
	header := []byte(`{"model.language_model.embed_tokens.weight":{"dtype":"F32","shape":[3,4],"data_offsets":[0,48]}}`)
	header = append(header, []byte(strings.Repeat(" ", (8-len(header)%8)%8))...)
	data := binary.LittleEndian.AppendUint64(nil, uint64(len(header)))
	data = append(data, header...)
	for i := range 12 {
		data = binary.LittleEndian.AppendUint32(data, math.Float32bits(float32(i)))
	}
	if err := os.WriteFile(filepath.Join(tdir, "model.safetensors"), data, 0600); err != nil {
		t.Fatal(err)
	}
	c := &batchCache{}
	defer c.close()
	one, _, err := c.encodeText(dir, []int64{0, 1}, 1, nil, false)
	if err != nil {
		t.Fatal(err)
	}
	session := c.textSession
	two, _, err := c.encodeText(dir, []int64{0, 2}, 1, nil, false)
	if err != nil {
		t.Fatal(err)
	}
	if session != c.textSession {
		t.Fatal("same shape did not reuse session")
	}
	if !reflect.DeepEqual(one.F32(), []float32{4, 5, 6, 7}) || !reflect.DeepEqual(two.F32(), []float32{8, 9, 10, 11}) {
		t.Fatalf("stale outputs: %v %v", one.F32(), two.F32())
	}
	three, _, err := c.encodeText(dir, []int64{0, 1, 2}, 1, nil, false)
	if err != nil {
		t.Fatal(err)
	}
	if session == c.textSession || !reflect.DeepEqual(three.F32(), []float32{4, 5, 6, 7, 8, 9, 10, 11}) {
		t.Fatal("shape change did not rebuild correctly")
	}
}
