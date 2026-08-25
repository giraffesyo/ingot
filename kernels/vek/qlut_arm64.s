//go:build arm64

#include "textflag.h"

// qlut_asm: dst[i] = tab[src[i]] via four 64-entry TBL/TBX stages — the
// 256-byte table lives in v16-v31 for the whole call; out-of-range indexes
// fall through TBL (zero) into the TBX stages (leave unchanged), so the
// four stages compose exactly. n must be a multiple of 16.
// Encodings clang-verified.
//
// func qlut_asm(dst *uint8, src *uint8, n int64, tab *uint8)
TEXT ·qlut_asm(SB), NOSPLIT, $0-32
	MOVD dst+0(FP), R0
	MOVD src+8(FP), R1
	MOVD n+16(FP), R2
	MOVD tab+24(FP), R3

	VLD1.P 64(R3), [V16.B16, V17.B16, V18.B16, V19.B16]
	VLD1.P 64(R3), [V20.B16, V21.B16, V22.B16, V23.B16]
	VLD1.P 64(R3), [V24.B16, V25.B16, V26.B16, V27.B16]
	VLD1   (R3), [V28.B16, V29.B16, V30.B16, V31.B16]
	WORD   $0x4f02e407 // movi.16b v7, #0x40

loop:
	VLD1.P 16(R1), [V0.B16]
	WORD   $0x4e006201 // tbl.16b  v1, {v16-v19}, v0
	WORD   $0x6e278402 // sub.16b  v2, v0, v7
	WORD   $0x4e027281 // tbx.16b  v1, {v20-v23}, v2
	WORD   $0x6e278442 // sub.16b  v2, v2, v7
	WORD   $0x4e027301 // tbx.16b  v1, {v24-v27}, v2
	WORD   $0x6e278442 // sub.16b  v2, v2, v7
	WORD   $0x4e027381 // tbx.16b  v1, {v28-v31}, v2
	VST1.P [V1.B16], 16(R0)
	SUB    $16, R2, R2
	CBNZ   R2, loop
	RET
