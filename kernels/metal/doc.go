// Package metal drives Apple GPUs from pure Go: no cgo, no dependencies,
// CGO_ENABLED=0 binaries.
//
// C functions are reached through an in-tree FFI (ffi_darwin_arm64.*):
// dlopen/dlsym come in through //go:cgo_import_dynamic, and each call runs
// on a private C stack between runtime.entersyscall/exitsyscall — symbols
// the runtime keeps linkable for exactly this use — so the goroutine is
// never preempted, moved or shrunk while C code runs. Objective-C is
// objc_msgSend over that FFI; Metal is loaded with dlopen at first use.
//
// Metal work (and autorelease pools, which are per OS thread) runs on one
// goroutine locked to its thread; the API funnels every call through it.
// Only integer/pointer arguments are supported (up to 8, plus structs over
// 16 bytes, which the arm64 ABI passes by reference) — all Metal's compute
// API needs.
//
// On other platforms Open returns an error and Available reports false.
package metal
