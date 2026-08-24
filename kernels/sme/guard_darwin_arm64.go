package sme

import "runtime"

// ZA/streaming state survives ordinary context switches, but signal delivery
// destroys it on darwin (measured: a GC storm — Go's SIGURG preemption pings —
// corrupts ZA accumulation; serial runs without signals are clean; see
// stress_test.go). guard runs fn with the goroutine pinned to its OS thread
// and all blockable signals masked; masked signals are delivered after
// unmasking, so nothing is lost — the thread just cannot be async-preempted
// for fn's duration, which is already true of assembly functions.
//
// The mask is set with the __pthread_sigmask syscall issued directly (SVC;
// guard_darwin_arm64.s): Go's runtime does not expose per-thread signal
// masking, cgo is banned, and a go:linkname pull of runtime.sigprocmask
// stopped linking under Go 1.27's hardening. The syscall number (329) has
// been stable across macOS releases; TestGuardMask fails loudly if it ever
// returns an error, and the GC-storm stress test fails if masking stops
// protecting ZA.

func pthreadSigmask(how uint32, new, old *uint32) int64

const sigSetmask = 3 // SIG_SETMASK

func guard(fn func()) {
	runtime.LockOSThread()
	all := ^uint32(0)
	var old uint32
	pthreadSigmask(sigSetmask, &all, &old)
	fn()
	pthreadSigmask(sigSetmask, &old, nil)
	runtime.UnlockOSThread()
}
