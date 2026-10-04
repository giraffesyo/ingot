# Optional QwenImage UI

A local browser workspace for Ingot's Qwen-Image-2.1: prompts, reference images,
seed variations, a pauseable queue, previews, run logs, cancellation, and PNG
downloads. Plain HTML/CSS/JS embedded in one Go executable. No npm, webview,
CDN, network fonts, telemetry, or model downloads.

From the repository root:

```sh
go run ./cmd/qwenimage-ui
# Open http://127.0.0.1:7860
```

To select a model snapshot explicitly:

```sh
go run ./cmd/qwenimage-ui -model /path/to/snapshot
```

The repository's `go.work` connects the separate modules. No build tags or
environment settings are required. `cd cmd/qwenimage-ui && go run .` also works.
To produce a standalone binary, use `make -C cmd/qwenimage-ui build` and then
`./cmd/qwenimage-ui/qwenimage-ui` (with the same optional flags). Builds also work
with `CGO_ENABLED=0`.

Omit `-model` to discover existing weights via `HF_HUB_CACHE`, `HF_HOME/hub`, or
`~/.cache/huggingface/hub`. The server accepts only loopback addresses. Change
the port with `-listen 127.0.0.1:7861`; set the initial output folder with `-output`.
Once built and supplied with local weights, the app runs entirely offline.
Building offline also requires Ingot's existing Go dependencies in the Go cache.

## Queue and weight reuse

The queue starts paused. Add one or more prompts; **Images to add** makes up to
eight variations with consecutive seeds. Click **Start queue** when ready.
Keep adding jobs while it runs. Select a row to inspect its progress, cancel it,
download its output, or copy its settings back into the form.

The worker admits up to **Batch size** jobs (four by default, maximum eight),
in FIFO order, without mixing compute devices. It processes the whole batch
through each model stage before moving on:

```text
encode every prompt → encode references → denoise every job → decode every image
```

Each stage opens its checkpoint once for the batch. GPU text/VAE objects are
reused; matching DiT layouts and precision reuse the DiT object, and matching
CPU shapes reuse compiled sessions. Shape changes replace that session's
scratch/compiled state while the checkpoint remains mapped. Large model stages
are released before the next stage loads, so this does not require keeping all
models resident. Completed latents/embeddings remain in host memory between stages.

For four compatible text-to-image jobs, the three stages open their checkpoint
sets **three times per batch**, compared with twelve across four standalone CLI
runs. Actual speedup depends on disk caching, device, shapes, and model compute;
this is not a measured latency claim. Vision preparation for edits still runs
per request. The worker releases model weights after each batch rather than
keeping them resident while idle.

Larger batches reduce repeated loads per image but delay the first preview and
retain more intermediate tensors and references. Use batch size one for quickest
first results. New arrivals enter the next batch, preventing an endless queue
from postponing decode. Pause takes effect **after the admitted batch**.
Cancellation is cooperative: queued jobs cancel immediately, while running
jobs finish their current stage or denoising step before releasing resources.

## Storage and limits

- PNGs are saved under `outputs/<job-id>/image.png` relative to the launch directory
  by default. Use **Output folder → Browse…** to choose a directory, or type a
  path (including `~/…` or a path relative to the launch directory). New folders
  are created when jobs are queued. Each job remembers its destination, so
  changing the field affects only new jobs. Different destinations can share
  a batch. Clearing finished rows only clears history; it does not delete outputs.
- Queue/history live in server memory. Browser refreshes reconnect to them;
  server restarts reset them. Run logs, prompts, and settings are not saved to
  disk. The output field starts at the `-output` default on a page reload;
  **Use these settings** restores a selected job's destination with its settings.
- Up to 32 unfinished jobs and 100 history entries. Clear finished history when
  the limit is reached. Only one bounded batch runs at a time.
- Up to 10 PNG/JPEG references per job, under 64 MB total, 16 megapixels each,
  40 megapixels combined, and 8192 pixels on either side. Temporary references
  are removed when jobs finish/cancel; abrupt termination can leave input files.
- Image editing requires Metal on Apple Silicon, as in the existing pipeline.
  The UI inherits QwenImage's substantial model-memory requirements.
- A failed decode retains `image.png.latents`. Recover using the normal CLI:
  `qwenimage -model /path/to/snapshot -from-latents outputs/<id>/image.png.latents -out recovered.png`.

## Build isolation and checks

This is a separate Go module. The root module never imports the UI or embeds its
assets. Its `go.mod`, `go.sum`, normal build/test targets, and release workflow
are unchanged. No new external packages are introduced: the optional module
uses the local Ingot module plus its existing transitive dependencies.
The root `go.work` enables explicit commands such as `go run ./cmd/qwenimage-ui`;
root `go build ./...` and `go test ./...` still exclude the nested UI module.

The batch implementation in `models/qwenimage/batch*.go` is a normal library API.
Go's linker removes the unused batch functions from executables that do not
call them; no build tags are needed. Its cancellation interface accepts a
`context.Context` without adding context initialization to the standalone CLI.
The existing CLI and single-image pipeline are unchanged. QwenImage and OCR
binary sizes were verified unchanged using `-trimpath -buildvcs=false`.

```sh
make -C cmd/qwenimage-ui test
make -C cmd/qwenimage-ui vet
CGO_ENABLED=0 go test -race ./cmd/qwenimage-ui/... # macOS
```

Tests cover queue admission, pause/resume, device boundaries, cancellation,
uploads, per-job output folders, folder browsing, image download, errors, stage order, mapping/session reuse, and CPU
output ownership. The optional full-checkpoint parity test compares batched
generation against standalone generation on CPU and Metal:

```sh
QWENIMAGE_FULL=1 CGO_ENABLED=0 go test \
  -run '^TestGenerateBatchParity$' -timeout 60m ./models/qwenimage
```

That parity test needs local weights and substantial memory/time; it skips by
default. Full-checkpoint inference and performance have not been measured for
this UI. Desktop/mobile browser checks use real queue APIs and a missing-model
failure, without loading model weights.
