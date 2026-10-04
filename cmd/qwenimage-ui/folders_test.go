package main

import (
	"bytes"
	"encoding/json"
	"image/png"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestQueuedOutputFoldersAndDownloads(t *testing.T) {
	a := testApp(t)
	first := filepath.Join(t.TempDir(), "first output")
	second := filepath.Join(t.TempDir(), "second output")
	for _, output := range []string{first, second} {
		if w := enqueue(t, a, map[string]string{"output": output}, smallPNG(t)); w.Code != 202 {
			t.Fatal(w.Body.String())
		}
	}
	state := readState(t, a)
	if state.Output != a.output || state.Jobs[0].Settings.Output != first || state.Jobs[1].Settings.Output != second {
		t.Fatalf("incorrect destinations: %+v", state)
	}
	img, err := png.Decode(bytes.NewReader(smallPNG(t)))
	if err != nil {
		t.Fatal(err)
	}
	finished := make(chan int, 1)
	a.process = func(jobs []*job) {
		for _, j := range jobs {
			a.finish(j, savePNG(filepath.Join(j.directory(), "image.png"), img))
		}
		finished <- len(jobs)
	}
	a.startWorker()
	call(t, a, "POST", "/api/queue", strings.NewReader(`{"paused":false}`))
	select {
	case n := <-finished:
		if n != 2 {
			t.Fatalf("different output folders split a compatible batch: %d jobs", n)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("jobs did not finish")
	}
	call(t, a, "POST", "/api/queue", strings.NewReader(`{"paused":true,"clear_finished":true}`))
	if len(readState(t, a).Jobs) != 0 {
		t.Fatal("finished jobs were not cleared")
	}
	for i, output := range []string{first, second} {
		id := state.Jobs[i].ID
		if _, err := os.Stat(filepath.Join(output, id, "image.png")); err != nil {
			t.Fatal(err)
		}
		if _, err := os.Stat(filepath.Join(output, id, "inputs")); !os.IsNotExist(err) {
			t.Fatal("temporary references were not removed from selected output folder")
		}
		w := call(t, a, "GET", "/api/image/"+id, nil)
		if w.Code != 200 || !bytes.Equal(w.Body.Bytes(), smallPNG(t)) {
			t.Fatalf("download after clearing history: %d %s", w.Code, w.Body)
		}
	}
	if entries, err := os.ReadDir(a.output); err != nil || len(entries) != 0 {
		t.Fatalf("jobs wrote to the default instead of their selected folders: %v %v", entries, err)
	}
	// An arbitrary PNG placed on disk is not exposed as a generated image.
	id := strings.Repeat("a", 32)
	if err := os.MkdirAll(filepath.Join(a.output, id), 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(a.output, id, "image.png"), smallPNG(t), 0600); err != nil {
		t.Fatal(err)
	}
	if w := call(t, a, "GET", "/api/image/"+id, nil); w.Code != 404 {
		t.Fatal("served an unregistered image")
	}
}

func TestOutputFolderCancellationAndValidation(t *testing.T) {
	a := testApp(t)
	output := t.TempDir()
	if w := enqueue(t, a, map[string]string{"output": output}, smallPNG(t)); w.Code != 202 {
		t.Fatal(w.Body.String())
	}
	id := readState(t, a).Jobs[0].ID
	call(t, a, "POST", "/api/cancel/"+id, nil)
	if _, err := os.Stat(filepath.Join(output, id, "inputs")); !os.IsNotExist(err) {
		t.Fatal("cancellation left references in selected output folder")
	}
	file := filepath.Join(t.TempDir(), "file")
	if err := os.WriteFile(file, []byte("keep"), 0600); err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{file, filepath.Join(file, "child"), "bad\x00path"} {
		if w := enqueue(t, a, map[string]string{"output": path}); w.Code < 400 {
			t.Fatalf("accepted unusable output path: %q", path)
		}
	}
	if len(readState(t, a).Jobs) != 1 {
		t.Fatal("invalid output path was admitted to queue")
	}
	if data, err := os.ReadFile(file); err != nil || string(data) != "keep" {
		t.Fatal("existing file was altered")
	}
}

func TestFolderBrowser(t *testing.T) {
	a := testApp(t)
	root := t.TempDir()
	for _, name := range []string{"a & b", "z folder"} {
		if err := os.Mkdir(filepath.Join(root, name), 0700); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(filepath.Join(root, "private.txt"), []byte("not listed"), 0600); err != nil {
		t.Fatal(err)
	}
	w := call(t, a, "GET", "/api/folders?path="+url.QueryEscape(root), nil)
	var listing folderListing
	if err := json.Unmarshal(w.Body.Bytes(), &listing); err != nil || w.Code != 200 {
		t.Fatalf("folder listing: %d %s %v", w.Code, w.Body, err)
	}
	if listing.Path != root || listing.Parent != filepath.Dir(root) || strings.Join(listing.Folders, ",") != "a & b,z folder" {
		t.Fatalf("incorrect folder listing: %+v", listing)
	}
	for _, path := range []string{filepath.Join(root, "missing"), filepath.Join(root, "private.txt")} {
		if w := call(t, a, "GET", "/api/folders?path="+url.QueryEscape(path), nil); w.Code != 400 {
			t.Fatalf("invalid folder accepted: %q", path)
		}
	}
	r := httptest.NewRequest("GET", "http://"+testHost+"/api/folders", nil)
	w = httptest.NewRecorder()
	a.handler(testHost).ServeHTTP(w, r)
	if w.Code != 403 {
		t.Fatal("folder listing accepted without UI header")
	}
}

func TestResolveFolder(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)
	for _, path := range []string{"~/pictures", `~\pictures`} {
		resolved, err := resolveFolder(path, "unused")
		if err != nil || resolved != filepath.Join(home, "pictures") {
			t.Fatalf("resolve %q: %q %v", path, resolved, err)
		}
	}
	resolved, err := resolveFolder("", home)
	if err != nil || resolved != home {
		t.Fatalf("default path: %q %v", resolved, err)
	}
	resolved, err = resolveFolder("relative output", home)
	expected, _ := filepath.Abs("relative output")
	if err != nil || resolved != expected {
		t.Fatalf("relative path: %q %v", resolved, err)
	}
}
