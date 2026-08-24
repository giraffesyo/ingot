// Package sme probes the Arm Scalable Matrix Extension (SME/SME2) present on
// Apple M4-class and newer cores. ONNX Runtime's single-thread GEMM runs on
// this unit at ~1 TFLOPS f32 (via KleidiAI), ~8× the NEON FMA peak of a core —
// the entire remaining single-thread gap documented in docs/PERF.md.
//
// This package is a research spike, not yet a GEMM: it measures what a
// WORD-encoded Go assembly kernel can get out of FMOPA (outer-product
// accumulate into the ZA tile), and verifies a real K-step outer-product
// against a Go reference. The safety story: each assembly function brackets
// its own SMSTART/SMSTOP with no Go calls in between; Go never async-preempts
// assembly functions, and the OS preserves SME state across context switches
// and signals, so streaming mode is never observable from Go code.
package sme
