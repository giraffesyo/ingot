package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"image/png"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/giraffesyo/ingot/models/qwenimage"
)

func terminal(status string) bool {
	return status == "completed" || status == "failed" || status == "cancelled"
}

func (a *app) notify() {
	select {
	case a.wake <- struct{}{}:
	default:
	}
}

func (a *app) configureQueue(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Paused    *bool `json:"paused"`
		BatchSize *int  `json:"batch_size"`
		Clear     bool  `json:"clear_finished"`
	}
	r.Body = http.MaxBytesReader(w, r.Body, 1024)
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		apiError(w, http.StatusBadRequest, "Invalid queue settings.")
		return
	}
	if req.BatchSize != nil && (*req.BatchSize < 1 || *req.BatchSize > 8) {
		apiError(w, http.StatusBadRequest, "Batch size must be 1–8.")
		return
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if req.Paused != nil {
		a.paused = *req.Paused
	}
	if req.BatchSize != nil {
		a.batchSize = *req.BatchSize
	}
	if req.Clear {
		kept := make([]*job, 0, len(a.jobs))
		for _, j := range a.jobs {
			if !terminal(j.Status) {
				kept = append(kept, j)
			}
		}
		a.jobs = kept
	}
	writeJSON(w, http.StatusOK, a.state())
	a.notify()
}

// One persistent worker admits a bounded FIFO batch. Pause stops the next
// admission; an admitted batch finishes together to avoid extra model swaps.
func (a *app) startWorker() {
	a.wait.Add(1)
	go func() {
		defer a.wait.Done()
		for {
			if a.ctx.Err() != nil {
				return
			}
			a.mu.Lock()
			var batch []*job
			if !a.paused {
				for _, j := range a.jobs {
					if j.Status != "queued" {
						continue
					}
					if len(batch) > 0 && j.Settings.Device != batch[0].Settings.Device {
						break
					}
					j.Status, j.Started = "running", time.Now()
					batch = append(batch, j)
					if len(batch) == a.batchSize {
						break
					}
				}
			}
			a.mu.Unlock()
			if len(batch) == 0 {
				select {
				case <-a.ctx.Done():
					return
				case <-a.wake:
				}
				continue
			}
			a.process(batch)
		}
	}()
}

func (a *app) appendLog(j *job, line string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	j.Log += line + "\n"
	if len(j.Log) > 16<<10 {
		j.Log = j.Log[len(j.Log)-(16<<10):]
	}
}

func (a *app) finish(j *job, err error) {
	// Remove only private reference files, preserving results and recovery latents.
	_ = os.RemoveAll(filepath.Join(j.directory(), "inputs"))
	a.mu.Lock()
	defer a.mu.Unlock()
	j.Finished = time.Now()
	switch {
	case j.ctx.Err() != nil:
		j.Status = "cancelled"
	case err != nil:
		j.Status, j.Error = "failed", err.Error()
		j.Log += "error: " + err.Error() + "\n"
		if errors.Is(err, os.ErrNotExist) && strings.Contains(err.Error(), a.model) {
			j.Error = "Model files are missing. Restart the UI with -model pointing to a complete local Qwen-Image-2.1 snapshot. See the run log for the missing file."
		}
	default:
		j.Status, j.Image = "completed", "/api/image/"+j.ID
		a.images[j.ID] = filepath.Join(j.directory(), "image.png")
	}
	j.cancel()
}

func (a *app) processBatch(jobs []*job) {
	requests := make([]qwenimage.BatchRequest, 0, len(jobs))
	valid := make([]*job, 0, len(jobs))
	for _, j := range jobs {
		if err := j.ctx.Err(); err != nil {
			a.finish(j, err)
			continue
		}
		images, err := loadReferences(j.inputs)
		if err != nil {
			a.finish(j, err)
			continue
		}
		seed, _ := strconv.ParseUint(j.Settings.Seed, 10, 64)
		opt := qwenimage.Options{
			Prompt: j.Settings.Prompt, Images: images, Resolution: j.Settings.Size,
			Width: j.Settings.Width, Height: j.Settings.Height, Steps: j.Settings.Steps,
			Seed: seed, Device: j.Settings.Device, Fast: j.Settings.Fast,
			SaveLatents: filepath.Join(j.directory(), "image.png.latents"),
			Log:         func(format string, args ...any) { a.appendLog(j, fmt.Sprintf(format, args...)) },
		}
		a.appendLog(j, fmt.Sprintf("Batch of %d · shared stage weights", len(jobs)))
		valid = append(valid, j)
		requests = append(requests, qwenimage.BatchRequest{Context: j.ctx, Options: opt})
	}
	qwenimage.GenerateBatch(a.model, requests, func(i int, result *qwenimage.Result, err error) {
		j := valid[i]
		if err == nil && j.ctx.Err() == nil {
			path := filepath.Join(j.directory(), "image.png")
			err = savePNG(path, result.Image)
			if err == nil {
				_ = os.Remove(path + ".latents")
			}
		}
		a.finish(j, err)
	})
}

func loadReferences(paths []string) ([]image.Image, error) {
	var images []image.Image
	for _, path := range paths {
		f, err := os.Open(path)
		if err != nil {
			return nil, err
		}
		img, _, err := image.Decode(f)
		f.Close()
		if err != nil {
			return nil, fmt.Errorf("read reference: %w", err)
		}
		images = append(images, img)
	}
	return images, nil
}

func savePNG(path string, img image.Image) error {
	f, err := os.OpenFile(path+".tmp", os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	err = png.Encode(f, img)
	closeErr := f.Close()
	if err == nil {
		err = closeErr
	}
	if err == nil {
		err = os.Rename(path+".tmp", path)
	}
	if err != nil {
		_ = os.Remove(path + ".tmp")
	}
	return err
}
