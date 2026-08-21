// Package vek provides NEON-accelerated f32 elementwise vector kernels used by
// the ops layer: binary arithmetic, ReLU, HardSwish, HardSigmoid, Clip,
// LeakyReLU, and scalar add/mul.
//
// Each public function has a pure-Go implementation (the oracle in tests) and,
// on arm64, a generated NEON kernel (vek_arm64.s, via ./gen) that processes the
// bulk in 4-lane vectors while the Go wrapper handles the <4 element tail.
// Callers pass equal-length slices; dst may alias src/a.
package vek
