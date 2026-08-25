//go:build arm64

#include "textflag.h"

// qscatter writes one full 8×12 int32 accumulator tile from the MMLA
// kernels' 2×2-block layout (reg bi*6+bj, lanes [r0c0 r0c1 r1c0 r1c1],
// r0=2bi, c0=2bj) into row-major C. Viewing each reg as .2d, the low
// halves are the even row's column pairs and the high halves the odd
// row's — so a row pair is three uzp1 + three uzp2. Encodings
// clang-verified.
//
// func qscatter(ct *int32, c *int32, ldc int64)
TEXT ·qscatter(SB), NOSPLIT, $0-24
	MOVD ct+0(FP), R0
	MOVD c+8(FP), R1
	MOVD ldc+16(FP), R2
	LSL  $2, R2, R2 // ldc in bytes
	MOVD $4, R3     // four bi row-pairs

loop:
	VLD1.P 64(R0), [V0.S4, V1.S4, V2.S4, V3.S4]
	VLD1.P 32(R0), [V4.S4, V5.S4]

	WORD $0x4ec11808 // uzp1.2d v8, v0, v1
	WORD $0x4ec31849 // uzp1.2d v9, v2, v3
	WORD $0x4ec5188a // uzp1.2d v10, v4, v5
	WORD $0x4ec1580b // uzp2.2d v11, v0, v1
	WORD $0x4ec3584c // uzp2.2d v12, v2, v3
	WORD $0x4ec5588d // uzp2.2d v13, v4, v5

	VST1 [V8.S4, V9.S4, V10.S4], (R1)
	ADD  R2, R1
	VST1 [V11.S4, V12.S4, V13.S4], (R1)
	ADD  R2, R1

	SUBS $1, R3, R3
	BNE  loop
	RET
