package sme

import (
	"runtime"
	_ "unsafe" // go:linkname
)

// ZA/streaming state survives ordinary context switches, but signal delivery
// destroys it on darwin (measured: a GC storm — Go's SIGURG preemption pings —
// corrupts ZA accumulation; serial runs without signals are clean). guard runs
// fn with the goroutine pinned to its OS thread and all blockable signals
// masked; masked signals are delivered after unmasking, so nothing is lost —
// the thread just cannot be async-preempted for fn's duration, which is
// already true of assembly functions.

//go:linkname runtime_sigprocmask runtime.sigprocmask
func runtime_sigprocmask(how uint32, new *uint32, old *uint32)

const sigSetmask = 3 // SIG_SETMASK on darwin

func guard(fn func()) {
	runtime.LockOSThread()
	all := ^uint32(0)
	var old uint32
	runtime_sigprocmask(sigSetmask, &all, &old)
	fn()
	runtime_sigprocmask(sigSetmask, &old, nil)
	runtime.UnlockOSThread()
}
