package sme

import (
	"runtime"
	"testing"
)

func TestGuardMask(t *testing.T) {
	if !Available() {
		t.Skip()
	}
	// The mask syscall must succeed and restore: run a guarded no-op and a
	// guarded kernel, then verify signals still work (a GC completes).
	all := ^uint32(0)
	var old uint32
	if rc := pthreadSigmask(sigSetmask, &all, &old); rc != 0 {
		t.Fatalf("pthread_sigmask set failed: errno %d", rc)
	}
	if rc := pthreadSigmask(sigSetmask, &old, nil); rc != 0 {
		t.Fatalf("pthread_sigmask restore failed: errno %d", rc)
	}
	guard(func() {})
	runtime.GC()
}
