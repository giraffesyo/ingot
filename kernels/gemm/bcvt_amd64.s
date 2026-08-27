//go:build amd64

#include "textflag.h"

// func bcvt32(dst *uint16, src *float32, n int64)
// dst[i] = bf16(src[i]) for i in [0,n), n a multiple of 32.
// VCVTNE2PS2BF16 (EVEX.512.F2.0F38.W0 72 /r): dst.lo16 = cvt(src2),
// dst.hi16 = cvt(src1). BYTE-encoded — the Go assembler has no
// AVX512-BF16 mnemonics.
TEXT ·bcvt32(SB), NOSPLIT, $0-24
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ n+16(FP), CX
loop:
	TESTQ CX, CX
	JE   done
	VMOVUPS 64(SI), Z1 // floats 16..31 -> high half
	VMOVUPS (SI), Z2   // floats 0..15  -> low half
	BYTE $0x62; BYTE $0xF2; BYTE $0x77; BYTE $0x48; BYTE $0x72; BYTE $0xC2 // VCVTNE2PS2BF16 Z2, Z1, Z0
	VMOVDQU32 Z0, (DI)
	ADDQ $128, SI
	ADDQ $64, DI
	SUBQ $32, CX
	JMP  loop
done:
	VZEROUPPER
	RET
