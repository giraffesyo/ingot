package main

import (
	"context"
	"crypto/rand"
	"embed"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"io/fs"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"
)

//go:embed web/*
var assets embed.FS

const maxUpload = 64 << 20

type settings struct {
	Prompt string `json:"prompt"`
	Size   int    `json:"size"`
	Width  int    `json:"width"`
	Height int    `json:"height"`
	Steps  int    `json:"steps"`
	Seed   string `json:"seed"` // Keep all 64 bits across JavaScript's number boundary.
	Device string `json:"device"`
	Fast   bool   `json:"fast"`
	Output string `json:"output"`
}

type job struct {
	ID       string    `json:"id"`
	Status   string    `json:"status"`
	Started  time.Time `json:"started,omitzero"`
	Finished time.Time `json:"finished,omitzero"`
	Settings settings  `json:"settings"`
	Log      string    `json:"log"`
	Error    string    `json:"error,omitempty"`
	Image    string    `json:"image,omitempty"`
	ctx      context.Context
	inputs   []string
	cancel   context.CancelFunc
}

type app struct {
	ctx           context.Context
	model, output string
	mu            sync.Mutex
	jobs          []*job
	images        map[string]string // Keep downloads available after clearing history.
	paused        bool
	batchSize     int
	wake          chan struct{}
	wait          sync.WaitGroup
	process       func([]*job)
}

type queueState struct {
	Jobs      []*job `json:"jobs"`
	Paused    bool   `json:"paused"`
	BatchSize int    `json:"batch_size"`
	Output    string `json:"output"`
}

func newApp(ctx context.Context, model, output string) *app {
	a := &app{ctx: ctx, model: model, output: output, paused: true, batchSize: 4, wake: make(chan struct{}, 1), jobs: []*job{}, images: make(map[string]string)}
	a.process = a.processBatch
	return a
}

func (a *app) state() queueState { return queueState{a.jobs, a.paused, a.batchSize, a.output} }

func (j *job) directory() string { return filepath.Join(j.Settings.Output, j.ID) }

func (a *app) handler(addr string) http.Handler {
	mux := http.NewServeMux()
	web, _ := fs.Sub(assets, "web")
	mux.Handle("GET /", http.FileServer(http.FS(web)))
	mux.HandleFunc("GET /api/queue", func(w http.ResponseWriter, r *http.Request) {
		a.mu.Lock()
		defer a.mu.Unlock()
		writeJSON(w, http.StatusOK, a.state())
	})
	mux.HandleFunc("POST /api/queue", a.configureQueue)
	mux.HandleFunc("POST /api/generate", a.generate)
	mux.HandleFunc("POST /api/cancel/{id}", a.cancel)
	mux.HandleFunc("GET /api/image/{id}", a.image)
	mux.HandleFunc("GET /api/folders", a.folders)
	origin := http.NewCrossOriginProtection().Handler(mux)
	_, port, _ := net.SplitHostPort(addr)
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Security-Policy", "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' blob:; connect-src 'self'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'")
		w.Header().Set("X-Content-Type-Options", "nosniff")
		w.Header().Set("Referrer-Policy", "no-referrer")
		w.Header().Set("Cache-Control", "no-store")
		// Reject DNS rebinding even for read-only endpoints.
		if r.Host != addr && r.Host != net.JoinHostPort("localhost", port) {
			apiError(w, http.StatusForbidden, "Use the localhost URL printed by qwenimage-ui.")
			return
		}
		if r.Method == http.MethodPost && r.Header.Get("X-Ingot-UI") != "1" {
			apiError(w, http.StatusForbidden, "Missing UI request header.")
			return
		}
		origin.ServeHTTP(w, r)
	})
}

func (a *app) generate(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, maxUpload)
	err := r.ParseMultipartForm(1 << 20)
	if r.MultipartForm != nil {
		defer r.MultipartForm.RemoveAll()
	}
	if err != nil {
		apiError(w, http.StatusBadRequest, "Could not read the upload. Use PNG or JPEG files totaling less than 64 MB.")
		return
	}
	opt, err := parseSettings(r)
	if err != nil {
		apiError(w, http.StatusBadRequest, err.Error())
		return
	}
	files := r.MultipartForm.File["images"]
	if len(files) > 10 {
		apiError(w, http.StatusBadRequest, "Use at most 10 reference images.")
		return
	}
	if len(files) > 0 && opt.Device == "cpu" {
		apiError(w, http.StatusBadRequest, "Image editing requires Metal on Apple Silicon. Choose Auto or Metal.")
		return
	}
	count, err := strconv.Atoi(r.FormValue("count"))
	if err != nil || count < 1 || count > 8 {
		apiError(w, http.StatusBadRequest, "Choose 1–8 images to add.")
		return
	}
	seed, _ := strconv.ParseUint(opt.Seed, 10, 64)
	if seed > ^uint64(0)-uint64(count-1) {
		apiError(w, http.StatusBadRequest, "Seed is too large for this many consecutive variations.")
		return
	}
	opt.Output, err = resolveFolder(r.FormValue("output"), a.output)
	if err != nil {
		apiError(w, http.StatusBadRequest, "Choose a valid output folder: "+err.Error())
		return
	}
	pending := make([]*job, 0, count)
	accepted := false
	defer func() {
		if !accepted {
			for _, j := range pending {
				j.cancel()
				_ = os.RemoveAll(j.directory())
			}
		}
	}()
	for i := 0; i < count; i++ {
		id := make([]byte, 16)
		if _, err := rand.Read(id); err != nil {
			apiError(w, http.StatusInternalServerError, "Could not create a generation ID.")
			return
		}
		ctx, cancel := context.WithCancel(a.ctx)
		j := &job{ID: hex.EncodeToString(id), Status: "queued", Settings: opt, ctx: ctx, cancel: cancel}
		j.Settings.Seed = strconv.FormatUint(seed+uint64(i), 10)
		pending = append(pending, j)
		if err := os.MkdirAll(filepath.Join(j.directory(), "inputs"), 0700); err != nil {
			apiError(w, http.StatusInternalServerError, "Could not create output directory: "+err.Error())
			return
		}
	}
	inputs := filepath.Join(pending[0].directory(), "inputs")
	var pixels int64
	for i, header := range files {
		src, err := header.Open()
		if err != nil {
			apiError(w, http.StatusBadRequest, "Could not open reference image.")
			return
		}
		cfg, format, err := image.DecodeConfig(src)
		pixels += int64(cfg.Width) * int64(cfg.Height)
		if err != nil || (format != "png" && format != "jpeg") || cfg.Width < 1 || cfg.Height < 1 ||
			cfg.Width > 8192 || cfg.Height > 8192 || int64(cfg.Width)*int64(cfg.Height) > 16_000_000 || pixels > 40_000_000 {
			src.Close()
			apiError(w, http.StatusBadRequest, "Use PNG/JPEG references up to 16 megapixels each (40 megapixels total, maximum side 8192).")
			return
		}
		path := filepath.Join(inputs, fmt.Sprintf("%02d.%s", i+1, format))
		_, err = src.Seek(0, io.SeekStart)
		if err == nil {
			err = copyUpload(path, src)
		}
		src.Close()
		if err != nil {
			apiError(w, http.StatusInternalServerError, "Could not save reference image: "+err.Error())
			return
		}
		pending[0].inputs = append(pending[0].inputs, path)
	}
	for _, j := range pending[1:] {
		for _, path := range pending[0].inputs {
			dest := filepath.Join(j.directory(), "inputs", filepath.Base(path))
			// Separate hard links keep each job's references alive independently.
			if err := os.Link(path, dest); err != nil {
				src, openErr := os.Open(path)
				if openErr != nil {
					apiError(w, 500, openErr.Error())
					return
				}
				err = copyUpload(dest, src)
				src.Close()
				if err != nil {
					apiError(w, 500, err.Error())
					return
				}
			}
			j.inputs = append(j.inputs, dest)
		}
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	unfinished := 0
	for _, j := range a.jobs {
		if !terminal(j.Status) {
			unfinished++
		}
	}
	if unfinished+count > 32 || len(a.jobs)+count > 100 {
		apiError(w, http.StatusConflict, "Queue limit reached (32 pending, 100 in history). Clear finished jobs or wait for space.")
		return
	}
	a.jobs = append(a.jobs, pending...)
	accepted = true
	writeJSON(w, http.StatusAccepted, a.state())
	a.notify()
}

func copyUpload(path string, src io.Reader) error {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	_, err = io.Copy(f, src)
	closeErr := f.Close()
	if err != nil {
		return err
	}
	return closeErr
}

func parseSettings(r *http.Request) (settings, error) {
	opt := settings{Prompt: strings.TrimSpace(r.FormValue("prompt")), Device: r.FormValue("device"), Seed: r.FormValue("seed")}
	if opt.Prompt == "" || len(opt.Prompt) > 16_000 {
		return opt, fmt.Errorf("Enter a prompt of 1–16,000 bytes.")
	}
	for _, field := range []struct {
		name           string
		dst            *int
		min, max, step int
	}{
		{"size", &opt.Size, 256, 2048, 32}, {"width", &opt.Width, 0, 2048, 32},
		{"height", &opt.Height, 0, 2048, 32}, {"steps", &opt.Steps, 1, 100, 1},
	} {
		n, err := strconv.Atoi(r.FormValue(field.name))
		if err != nil || n < field.min || n > field.max || n%field.step != 0 {
			return opt, fmt.Errorf("%s must be %d–%d in increments of %d.", field.name, field.min, field.max, field.step)
		}
		*field.dst = n
	}
	if (opt.Width == 0) != (opt.Height == 0) || (opt.Width != 0 && (opt.Width < 256 || opt.Height < 256)) {
		return opt, fmt.Errorf("Set both dimensions to 0 for automatic sizing, or use 256–2048 for each.")
	}
	if _, err := strconv.ParseUint(opt.Seed, 10, 64); err != nil {
		return opt, fmt.Errorf("Seed must be a whole number from 0 to 18446744073709551615.")
	}
	if opt.Device != "auto" && opt.Device != "gpu" && opt.Device != "cpu" {
		return opt, fmt.Errorf("Choose Auto, Metal, or CPU.")
	}
	var err error
	opt.Fast, err = strconv.ParseBool(r.FormValue("fast"))
	if err != nil {
		return opt, fmt.Errorf("Fast mode must be true or false.")
	}
	return opt, nil
}

func (a *app) cancel(w http.ResponseWriter, r *http.Request) {
	a.mu.Lock()
	defer a.mu.Unlock()
	for _, j := range a.jobs {
		if j.ID != r.PathValue("id") {
			continue
		}
		if terminal(j.Status) {
			apiError(w, http.StatusConflict, "This job has already finished.")
			return
		}
		j.cancel()
		if j.Status == "queued" {
			j.Status, j.Finished = "cancelled", time.Now()
			_ = os.RemoveAll(filepath.Join(j.directory(), "inputs"))
		} else {
			j.Status = "cancelling"
		}
		writeJSON(w, http.StatusAccepted, a.state())
		return
	}
	apiError(w, http.StatusNotFound, "Job not found.")
}

func (a *app) image(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	b, err := hex.DecodeString(id)
	if err != nil || len(b) != 16 {
		http.NotFound(w, r)
		return
	}
	// Only generated PNGs are served; never a user-supplied file path.
	a.mu.Lock()
	path, ok := a.images[id]
	a.mu.Unlock()
	if !ok {
		http.NotFound(w, r)
		return
	}
	w.Header().Set("Content-Type", "image/png")
	w.Header().Set("Content-Disposition", fmt.Sprintf(`inline; filename="qwenimage-%s.png"`, id[:8]))
	http.ServeFile(w, r, path)
}

func writeJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}

func apiError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]string{"error": msg})
}
