//go:build arm64

#include "textflag.h"

// func bfdotProbe(iters int64, src *uint16)
// 4 independent BFDOT chains per iteration (16 FLOPs each at .4s).
TEXT ·bfdotProbe(SB), NOSPLIT, $0-16
	MOVD iters+0(FP), R0
	MOVD src+8(FP), R1
	VLD1 (R1), [V0.B16, V1.B16, V2.B16, V3.B16]
bfdloop:
	WORD $0x6e41fc04 // bfdot v4.4s, v0.8h, v1.8h
	WORD $0x6e42fc25 // bfdot v5.4s, v1.8h, v2.8h
	WORD $0x6e43fc46 // bfdot v6.4s, v2.8h, v3.8h
	WORD $0x6e40fc67 // bfdot v7.4s, v3.8h, v0.8h
	SUBS $1, R0, R0
	BNE  bfdloop
	RET
