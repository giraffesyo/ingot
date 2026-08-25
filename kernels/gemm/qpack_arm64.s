//go:build arm64

#include "textflag.h"

// qpackb packs `groups` full k-groups of a 12-column B panel into group-major
// [group][j][o] layout (bp[g*96+j*8+o] = B[g*8+o][j]): per group it loads 8
// consecutive B rows (16 bytes each — 4 bytes past column 11, caller
// guarantees j0+16 <= ldb) and transposes 8×12 bytes with three zip rounds.
// After round 3 each register holds two full output columns (col c rows 0-7,
// col c+1 rows 0-7), so v20-v25 are the 96 output bytes in order.
// Encodings clang-verified (zip1/zip2 .16b/.8h/.4s).
//
// func qpackb(dst *int8, src *int8, ldb int64, groups int64)
TEXT ·qpackb(SB), NOSPLIT, $0-32
	MOVD dst+0(FP), R0
	MOVD src+8(FP), R1
	MOVD ldb+16(FP), R2
	MOVD groups+24(FP), R3
	LSL  $3, R2, R6 // 8*ldb: src advance per group

loop:
	MOVD R1, R4
	VLD1 (R4), [V0.B16]
	ADD  R2, R4
	VLD1 (R4), [V1.B16]
	ADD  R2, R4
	VLD1 (R4), [V2.B16]
	ADD  R2, R4
	VLD1 (R4), [V3.B16]
	ADD  R2, R4
	VLD1 (R4), [V4.B16]
	ADD  R2, R4
	VLD1 (R4), [V5.B16]
	ADD  R2, R4
	VLD1 (R4), [V6.B16]
	ADD  R2, R4
	VLD1 (R4), [V7.B16]

	// round 1: interleave row pairs (bytes)
	WORD $0x4e013808 // zip1.16b v8, v0, v1
	WORD $0x4e033849 // zip1.16b v9, v2, v3
	WORD $0x4e05388a // zip1.16b v10, v4, v5
	WORD $0x4e0738cb // zip1.16b v11, v6, v7
	WORD $0x4e01780c // zip2.16b v12, v0, v1
	WORD $0x4e03784d // zip2.16b v13, v2, v3
	WORD $0x4e05788e // zip2.16b v14, v4, v5
	WORD $0x4e0778cf // zip2.16b v15, v6, v7

	// round 2: interleave row quads (16-bit units = col × 2 rows)
	WORD $0x4e493910 // zip1.8h v16, v8, v9
	WORD $0x4e497911 // zip2.8h v17, v8, v9
	WORD $0x4e4b3952 // zip1.8h v18, v10, v11
	WORD $0x4e4b7953 // zip2.8h v19, v10, v11
	WORD $0x4e4d399a // zip1.8h v26, v12, v13
	WORD $0x4e4f39db // zip1.8h v27, v14, v15

	// round 3: full columns (32-bit units = col × 4 rows)
	WORD $0x4e923a14 // zip1.4s v20, v16, v18 (cols 0,1)
	WORD $0x4e927a15 // zip2.4s v21, v16, v18 (cols 2,3)
	WORD $0x4e933a36 // zip1.4s v22, v17, v19 (cols 4,5)
	WORD $0x4e937a37 // zip2.4s v23, v17, v19 (cols 6,7)
	WORD $0x4e9b3b58 // zip1.4s v24, v26, v27 (cols 8,9)
	WORD $0x4e9b7b59 // zip2.4s v25, v26, v27 (cols 10,11)

	VST1.P [V20.B16, V21.B16, V22.B16, V23.B16], 64(R0)
	VST1.P [V24.B16, V25.B16], 32(R0)

	ADD  R6, R1
	SUBS $1, R3, R3
	BNE  loop
	RET
