// __pthread_sigmask via direct syscall — see guard_darwin_arm64.go.

//go:build darwin && arm64

#include "textflag.h"

// func pthreadSigmask(how uint32, new, old *uint32) int64
// Returns 0 on success, errno on failure (carry flag set by the kernel).
TEXT ·pthreadSigmask(SB), NOSPLIT, $0-32
	MOVWU how+0(FP), R0
	MOVD  new+8(FP), R1
	MOVD  old+16(FP), R2
	MOVD  $329, R16 // SYS___pthread_sigmask
	SVC   $0x80
	BCC   ok
	MOVD  R0, ret+24(FP)
	RET
ok:
	MOVD  ZR, ret+24(FP)
	RET
