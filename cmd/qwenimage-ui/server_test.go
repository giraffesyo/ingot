package main

import (
	"bytes"
	"context"
	"encoding/json"
	"image"
	"image/color"
	"image/png"
	"io"
	"mime/multipart"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

const testHost = "127.0.0.1:7860"

func testApp(t *testing.T) *app {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	a := newApp(ctx, t.TempDir(), t.TempDir())
	t.Cleanup(func() { cancel(); a.wait.Wait() })
	return a
}

func call(t *testing.T, a *app, method, path string, body io.Reader) *httptest.ResponseRecorder {
	t.Helper()
	r := httptest.NewRequest(method, "http://"+testHost+path, body)
	r.Header.Set("X-Ingot-UI", "1")
	r.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	a.handler(testHost).ServeHTTP(w, r)
	return w
}

func enqueue(t *testing.T, a *app, fields map[string]string, files ...[]byte) *httptest.ResponseRecorder {
	t.Helper()
	values := map[string]string{"prompt": "a fox", "size": "512", "width": "0", "height": "0", "steps": "2", "seed": "42", "device": "auto", "fast": "true", "count": "1"}
	for k, v := range fields {
		values[k] = v
	}
	var body bytes.Buffer
	m := multipart.NewWriter(&body)
	for k, v := range values {
		if err := m.WriteField(k, v); err != nil {
			t.Fatal(err)
		}
	}
	for _, data := range files {
		f, err := m.CreateFormFile("images", "../../reference.png")
		if err != nil {
			t.Fatal(err)
		}
		if _, err := f.Write(data); err != nil {
			t.Fatal(err)
		}
	}
	m.Close()
	r := httptest.NewRequest("POST", "http://"+testHost+"/api/generate", &body)
	r.Header.Set("Content-Type", m.FormDataContentType())
	r.Header.Set("X-Ingot-UI", "1")
	w := httptest.NewRecorder()
	a.handler(testHost).ServeHTTP(w, r)
	return w
}

func readState(t *testing.T, a *app) queueState {
	t.Helper()
	w := call(t, a, "GET", "/api/queue", nil)
	var state queueState
	if err := json.Unmarshal(w.Body.Bytes(), &state); err != nil {
		t.Fatal(err)
	}
	return state
}

func smallPNG(t *testing.T) []byte {
	t.Helper()
	img := image.NewNRGBA(image.Rect(0, 0, 2, 2))
	img.SetNRGBA(0, 0, color.NRGBA{R: 255, A: 255})
	var b bytes.Buffer
	if err := png.Encode(&b, img); err != nil {
		t.Fatal(err)
	}
	return b.Bytes()
}

func TestQueueBatchingPauseCancellationAndDownload(t *testing.T) {
	a := testApp(t)
	started := make(chan []*job, 8)
	release := make(chan struct{}, 8)
	a.process = func(jobs []*job) {
		started <- jobs
		select {
		case <-release:
		case <-a.ctx.Done():
		}
		for _, j := range jobs {
			a.appendLog(j, "stage: decode")
			err := savePNG(filepath.Join(a.output, j.ID, "image.png"), image.NewNRGBA(image.Rect(0, 0, 2, 2)))
			a.finish(j, err)
		}
	}
	a.startWorker()
	w := enqueue(t, a, map[string]string{"count": "4", "seed": "18446744073709551610"}, smallPNG(t))
	if w.Code != 202 {
		t.Fatal(w.Body.String())
	}
	state := readState(t, a)
	if !state.Paused || len(state.Jobs) != 4 || state.Jobs[3].Settings.Seed != "18446744073709551613" {
		t.Fatalf("bad initial state: %+v", state)
	}
	select {
	case <-started:
		t.Fatal("paused queue started")
	default:
	}
	queuedID := state.Jobs[1].ID
	if w := call(t, a, "POST", "/api/cancel/"+queuedID, nil); w.Code != 202 {
		t.Fatal(w.Body.String())
	}
	if w := enqueue(t, a, map[string]string{"device": "cpu"}); w.Code != 202 {
		t.Fatal(w.Body.String())
	}
	if w := call(t, a, "POST", "/api/queue", strings.NewReader(`{"paused":false}`)); w.Code != 200 {
		t.Fatal(w.Body.String())
	}
	var batch []*job
	select {
	case batch = <-started:
	case <-time.After(3 * time.Second):
		t.Fatal("worker did not start")
	}
	if len(batch) != 3 {
		t.Fatalf("batch has %d jobs, wanted three compatible jobs", len(batch))
	}
	for _, j := range batch {
		if _, err := os.Stat(j.inputs[0]); err != nil {
			t.Fatalf("cancelled sibling removed references: %v", err)
		}
	}
	call(t, a, "POST", "/api/queue", strings.NewReader(`{"paused":true,"batch_size":2}`))
	call(t, a, "POST", "/api/cancel/"+batch[0].ID, nil)
	release <- struct{}{}
	deadline := time.Now().Add(3 * time.Second)
	for readState(t, a).Jobs[3].Status != "completed" {
		if time.Now().After(deadline) {
			t.Fatal("batch did not finish")
		}
		time.Sleep(time.Millisecond)
	}
	state = readState(t, a)
	if state.Jobs[0].Status != "cancelled" || state.Jobs[1].Status != "cancelled" || state.Jobs[4].Status != "queued" {
		t.Fatalf("wrong completion state: %+v", state.Jobs)
	}
	select {
	case <-started:
		t.Fatal("paused queue admitted next batch")
	default:
	}
	imageID := state.Jobs[2].ID
	w = call(t, a, "GET", "/api/image/"+imageID, nil)
	if w.Code != 200 {
		t.Fatal(w.Body.String())
	}
	if _, err := png.Decode(w.Body); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(a.output, imageID, "inputs")); !os.IsNotExist(err) {
		t.Fatal("references were not cleaned up")
	}
	call(t, a, "POST", "/api/queue", strings.NewReader(`{"paused":false}`))
	select {
	case batch = <-started:
	case <-time.After(3 * time.Second):
		t.Fatal("queue did not resume")
	}
	if len(batch) != 1 || batch[0].Settings.Device != "cpu" {
		t.Fatal("mixed compute devices in batch")
	}
	release <- struct{}{}
}

func TestRequestValidationAndLocalOnlyAccess(t *testing.T) {
	a := testApp(t)
	for _, fields := range []map[string]string{
		{"prompt": " "}, {"steps": "0"}, {"width": "256"}, {"height": "33", "width": "256"},
		{"seed": "18446744073709551616"}, {"seed": "18446744073709551615", "count": "2"},
		{"count": "9"}, {"device": "remote"}, {"fast": "maybe"},
	} {
		if w := enqueue(t, a, fields); w.Code != 400 {
			t.Fatalf("accepted invalid fields %v: %s", fields, w.Body)
		}
	}
	if w := enqueue(t, a, nil, []byte("not an image")); w.Code != 400 {
		t.Fatal("accepted invalid PNG")
	}
	if w := enqueue(t, a, map[string]string{"device": "cpu"}, smallPNG(t)); w.Code != 400 {
		t.Fatal("accepted CPU editing")
	}
	entries, err := os.ReadDir(a.output)
	if err != nil || len(entries) != 0 {
		t.Fatalf("invalid uploads left directories: %v %v", entries, err)
	}
	for _, testcase := range []struct{ host, origin, header string }{
		{"attacker.example:7860", "", "1"}, {testHost, "https://attacker.example", "1"}, {testHost, "", ""},
	} {
		r := httptest.NewRequest("POST", "http://"+testcase.host+"/api/queue", strings.NewReader(`{"paused":false}`))
		r.Header.Set("Origin", testcase.origin)
		r.Header.Set("X-Ingot-UI", testcase.header)
		w := httptest.NewRecorder()
		a.handler(testHost).ServeHTTP(w, r)
		if w.Code != 403 {
			t.Fatalf("unsafe request accepted: %+v", testcase)
		}
	}
	for _, addr := range []string{"0.0.0.0:7860", ":7860", "192.168.1.1:7860", "example.com:7860"} {
		if _, err := loopbackAddress(addr); err == nil {
			t.Fatalf("accepted %q", addr)
		}
	}
	for _, addr := range []string{"127.0.0.1:7860", "[::1]:7860", "localhost:7860"} {
		if _, err := loopbackAddress(addr); err != nil {
			t.Fatal(err)
		}
	}
}

func TestModelFailureCompletesBatchAndClearPreservesOutputs(t *testing.T) {
	a := testApp(t)
	a.startWorker()
	if w := enqueue(t, a, map[string]string{"count": "2"}); w.Code != 202 {
		t.Fatal(w.Body.String())
	}
	call(t, a, "POST", "/api/queue", strings.NewReader(`{"paused":false}`))
	deadline := time.Now().Add(3 * time.Second)
	for {
		s := readState(t, a)
		if s.Jobs[0].Status == "failed" && s.Jobs[1].Status == "failed" {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("missing model did not complete all jobs")
		}
		time.Sleep(time.Millisecond)
	}
	s := readState(t, a)
	path := filepath.Join(a.output, s.Jobs[0].ID, "image.png")
	if err := os.WriteFile(path, smallPNG(t), 0600); err != nil {
		t.Fatal(err)
	}
	call(t, a, "POST", "/api/queue", strings.NewReader(`{"clear_finished":true}`))
	if len(readState(t, a).Jobs) != 0 {
		t.Fatal("finished history not cleared")
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatal("clearing history deleted an output")
	}
}

func TestConcurrentAdmissionStaysBounded(t *testing.T) {
	a := testApp(t)
	var wg sync.WaitGroup
	codes := make(chan int, 6)
	for range 6 {
		wg.Add(1)
		go func() { defer wg.Done(); codes <- enqueue(t, a, map[string]string{"count": "8"}).Code }()
	}
	wg.Wait()
	close(codes)
	accepted, rejected := 0, 0
	for code := range codes {
		switch code {
		case 202:
			accepted++
		case 409:
			rejected++
		default:
			t.Fatalf("unexpected status %d", code)
		}
	}
	if accepted != 4 || rejected != 2 || len(readState(t, a).Jobs) != 32 {
		t.Fatalf("accepted %d, rejected %d", accepted, rejected)
	}
	entries, err := os.ReadDir(a.output)
	if err != nil || len(entries) != 32 {
		t.Fatalf("rejected jobs left files: %d entries, %v", len(entries), err)
	}
}
