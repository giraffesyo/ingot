# QwenImage UI

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

The user selected a local browser UI with no frontend dependencies. A nested Go
module embeds plain HTML, CSS, and JavaScript and calls Ingot's optional batch
pipeline. No new third-party dependencies or UI assets enter the root module.
The root Go workspace enables `go run ./cmd/qwenimage-ui`; plain `go run .` also works
in this module. Root `./...` build/test patterns still exclude the nested module.
Go's linker omits unused batch functions from other executables; build tags are
not required.

## Product Purpose

A small, optional, offline interface for generating and editing images with
Ingot's Qwen-Image-2.1 implementation and locally available model weights.

## Capabilities and Constraints

Prompt, ordered reference images, seed variations, generation settings, preview,
cancellation, run log, and PNG download. A persistent queue worker processes one
bounded batch at a time, sharing stage weights and compatible model sessions.
The user explicitly requested queueing and minimizing weight unloads/reloads.
Pause/resume, batch size, and individual cancellation are exposed. Loopback only.
No CDN, network fonts, package manager, telemetry, or automatic model downloads.
Editing inherits the CLI's Metal requirement. Outputs persist locally; the UI
shows the current server session's queue and history. Queue history is in memory;
it survives browser refreshes but not server restarts. Cancellation waits for a
running stage or denoising step to finish safely. Model weights unload after each
stage; all stages are never held simultaneously just to keep the queue warm.

The output folder can be entered directly or selected by browsing local folders.
Each queued job keeps its chosen destination, which appears in job details and
is restored by settings reuse. Reloading the page resets the output field to the
server's command-line default.

## Users

Assumption from the request: local Ingot users who want to operate QwenImage
without assembling a command for each image.
