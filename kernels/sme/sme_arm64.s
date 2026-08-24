// SME (Scalable Matrix Extension) probes — WORD-encoded, encodings verified
// against clang (-march=armv9.2-a+sme2); see kernels/sme/doc.go.
// All functions enter/leave streaming mode themselves (SMSTART/SMSTOP) and
// make no Go calls in between; assembly functions are not async-preempted,
// so no Go code ever runs in streaming mode.

//go:build arm64

#include "textflag.h"

// func svl() int64
TEXT ·svl(SB), NOSPLIT, $0-8
	WORD $0x04bf5820 // rdsvl x0, #1
	MOVD R0, ret+0(FP)
	RET

// func probePeak(iters int64, src *float32)
// 4 independent FMOPA tiles per iteration from resident Z registers:
// pure outer-product throughput, no memory traffic.
TEXT ·probePeak(SB), NOSPLIT, $0-16
	MOVD iters+0(FP), R0
	MOVD src+8(FP), R1
	WORD $0xd503477f // smstart
	WORD $0xc00800ff // zero {za}
	WORD $0x2598e3e0 // ptrue p0.s
	WORD $0xa540a020 // ld1w z0, p0/z, [x1]
	WORD $0xa541a021 // ld1w z1, p0/z, [x1, #1, mul vl]
	WORD $0xa542a022 // ld1w z2, p0/z, [x1, #2, mul vl]
	WORD $0xa543a023 // ld1w z3, p0/z, [x1, #3, mul vl]
peak:
	WORD $0x80810000 // fmopa za0.s, p0/m, p0/m, z0.s, z1.s
	WORD $0x80820001 // fmopa za1.s, p0/m, p0/m, z0.s, z2.s
	WORD $0x80830002 // fmopa za2.s, p0/m, p0/m, z0.s, z3.s
	WORD $0x80810043 // fmopa za3.s, p0/m, p0/m, z2.s, z1.s
	SUBS $1, R0, R0
	BNE  peak
	WORD $0xd503467f // smstop
	RET

// func probeLoad(iters int64, a, b *float32)
// 2 vector loads + 4 FMOPA per iteration: the load-to-compute ratio of a
// GEMM inner loop that keeps two A vectors resident.
TEXT ·probeLoad(SB), NOSPLIT, $0-24
	MOVD iters+0(FP), R0
	MOVD a+8(FP), R1
	MOVD b+16(FP), R2
	WORD $0xd503477f // smstart
	WORD $0xc00800ff // zero {za}
	WORD $0x2598e3e0 // ptrue p0.s
	WORD $0xa542a022 // ld1w z2, p0/z, [x1, #2, mul vl]
	WORD $0xa543a023 // ld1w z3, p0/z, [x1, #3, mul vl]
load:
	WORD $0xa540a020 // ld1w z0, p0/z, [x1]
	WORD $0xa540a041 // ld1w z1, p0/z, [x2]
	WORD $0x80810000 // fmopa za0.s, z0, z1
	WORD $0x80820001 // fmopa za1.s, z0, z2
	WORD $0x80830002 // fmopa za2.s, z0, z3
	WORD $0x80810043 // fmopa za3.s, z2, z1
	SUBS $1, R0, R0
	BNE  load
	WORD $0xd503467f // smstop
	RET

// func outerK(a, b *float32, k int64, out *float32)
// za0 += Σ_p a[p·SVL/4 ..] ⊗ b[p·SVL/4 ..]; the full SVL×SVL f32 tile is
// stored row-major (row stride = SVL/4 floats) to out.
TEXT ·outerK(SB), NOSPLIT, $0-32
	MOVD a+0(FP), R0
	MOVD b+8(FP), R1
	MOVD k+16(FP), R2
	MOVD out+24(FP), R3
	WORD $0xd503477f // smstart
	WORD $0xc00800ff // zero {za}
	WORD $0x2598e3e0 // ptrue p0.s
acc:
	WORD $0xa540a000 // ld1w z0, p0/z, [x0]
	WORD $0x04205020 // addvl x0, x0, #1
	WORD $0xa540a021 // ld1w z1, p0/z, [x1]
	WORD $0x04215021 // addvl x1, x1, #1
	WORD $0x80810000 // fmopa za0.s, p0/m, p0/m, z0.s, z1.s
	SUBS $1, R2, R2
	BNE  acc
	// store the tile: rows w12+imm for w12 in {0,4,8,12}, imm 0..3
	MOVW $0, R12
	WORD $0xe0bf0060 // st1w {za0h.s[w12, 0]}, p0, [x3]
	WORD $0x04235023 // addvl x3, x3, #1
	WORD $0xe0bf0061
	WORD $0x04235023
	WORD $0xe0bf0062
	WORD $0x04235023
	WORD $0xe0bf0063
	WORD $0x04235023
	MOVW $4, R12
	WORD $0xe0bf0060
	WORD $0x04235023
	WORD $0xe0bf0061
	WORD $0x04235023
	WORD $0xe0bf0062
	WORD $0x04235023
	WORD $0xe0bf0063
	WORD $0x04235023
	MOVW $8, R12
	WORD $0xe0bf0060
	WORD $0x04235023
	WORD $0xe0bf0061
	WORD $0x04235023
	WORD $0xe0bf0062
	WORD $0x04235023
	WORD $0xe0bf0063
	WORD $0x04235023
	MOVW $12, R12
	WORD $0xe0bf0060
	WORD $0x04235023
	WORD $0xe0bf0061
	WORD $0x04235023
	WORD $0xe0bf0062
	WORD $0x04235023
	WORD $0xe0bf0063
	WORD $0x04235023
	WORD $0xd503467f // smstop
	RET

// func probeN(iters int64, src *float32, variant int64)
// variant 1: one FMOPA per iteration (single-tile latency chain)
// variant 2: two tiles; variant 16: sixteen FMOPA, four per tile (issue rate)
TEXT ·probeN(SB), NOSPLIT, $0-24
	MOVD iters+0(FP), R0
	MOVD src+8(FP), R1
	MOVD variant+16(FP), R2
	WORD $0xd503477f // smstart
	WORD $0xc00800ff // zero {za}
	WORD $0x2598e3e0 // ptrue p0.s
	WORD $0xa540a020 // ld1w z0, [x1]
	WORD $0xa541a021 // ld1w z1, [x1,#1]
	WORD $0xa542a022 // ld1w z2, [x1,#2]
	WORD $0xa543a023 // ld1w z3, [x1,#3]
	CMP  $1, R2
	BEQ  one
	CMP  $2, R2
	BEQ  two
sixteen:
	WORD $0x80810000 // za0 z0z1
	WORD $0x80820001 // za1 z0z2
	WORD $0x80830002 // za2 z0z3
	WORD $0x80810043 // za3 z2z1
	WORD $0x80820040 // za0 z2z2? -> fmopa za0, z2, z2: enc: base 80800000 | zm<<16 | zn<<5 | tile: za0 z2,z2 = 0x80820040
	WORD $0x80830041 // za1 z2,z3
	WORD $0x80810062 // za2 z3,z1
	WORD $0x80800023 // za3 z1,z0
	WORD $0x80800000 // za0 z0,z0
	WORD $0x80810021 // za1 z1,z1
	WORD $0x80820042 // za2 z2,z2
	WORD $0x80830063 // za3 z3,z3
	WORD $0x80830020 // za0 z1,z3
	WORD $0x80800041 // za1 z2,z0
	WORD $0x80810002 // za2 z0,z1
	WORD $0x80820023 // za3 z1,z2
	SUBS $1, R0, R0
	BNE  sixteen
	B    done
two:
	WORD $0x80810000 // za0
	WORD $0x80820041 // za1 z2,z2
	SUBS $1, R0, R0
	BNE  two
	B    done
one:
	WORD $0x80810000 // za0
	SUBS $1, R0, R0
	BNE  one
done:
	WORD $0xd503467f // smstop
	RET

// func zakernel(kc int64, ap, bp, c *float32, ldc4 int64)
// C[32×32] = Σ_p ap[p·32..] ⊗ bp[p·32..]  (beta=0). ap/bp are packed panels
// (32 values per k step, zero-padded). ldc4 = C row stride in bytes.
// ZA layout: za0 = rows 0-15 × cols 0-15, za1 = rows 0-15 × cols 16-31,
// za2 = rows 16-31 × cols 0-15, za3 = rows 16-31 × cols 16-31.
TEXT ·zakernel(SB), NOSPLIT, $0-40
	MOVD kc+0(FP), R0
	MOVD ap+8(FP), R1
	MOVD bp+16(FP), R2
	MOVD c+24(FP), R3
	MOVD ldc4+32(FP), R5
	MOVD $16, R4
	WORD $0xd503477f // smstart
	WORD $0xc00800ff // zero {za}
	WORD $0x2598e3e0 // ptrue p0.s
zaloop:
	WORD $0xa540a020 // ld1w z0, p0/z, [x1]        (a rows 0-15)
	WORD $0xa541a021 // ld1w z1, p0/z, [x1, #1]    (a rows 16-31)
	WORD $0x04215041 // addvl x1, x1, #2
	WORD $0xa540a042 // ld1w z2, p0/z, [x2]        (b cols 0-15)
	WORD $0xa541a043 // ld1w z3, p0/z, [x2, #1]    (b cols 16-31)
	WORD $0x04225042 // addvl x2, x2, #2
	WORD $0x80820000 // fmopa za0.s, z0, z2
	WORD $0x80830001 // fmopa za1.s, z0, z3
	WORD $0x80820022 // fmopa za2.s, z1, z2
	WORD $0x80830023 // fmopa za3.s, z1, z3
	SUBS $1, R0, R0
	BNE  zaloop
	// store 32x32 C block: za0/za1 = rows 0-15, za2/za3 = rows 16-31
	MOVW $0, R12
	WORD $0xe0bf0060 // st1w {za0h.s[w12, 0]}, p0, [x3]
	WORD $0xe0a40064 // st1w {za1h.s[w12, 0]}, p0, [x3, x4, lsl #2]
	ADD  R5, R3, R3
	WORD $0xe0bf0061 // st1w {za0h.s[w12, 1]}, p0, [x3]
	WORD $0xe0a40065 // st1w {za1h.s[w12, 1]}, p0, [x3, x4, lsl #2]
	ADD  R5, R3, R3
	WORD $0xe0bf0062 // st1w {za0h.s[w12, 2]}, p0, [x3]
	WORD $0xe0a40066 // st1w {za1h.s[w12, 2]}, p0, [x3, x4, lsl #2]
	ADD  R5, R3, R3
	WORD $0xe0bf0063 // st1w {za0h.s[w12, 3]}, p0, [x3]
	WORD $0xe0a40067 // st1w {za1h.s[w12, 3]}, p0, [x3, x4, lsl #2]
	ADD  R5, R3, R3
	MOVW $4, R12
	WORD $0xe0bf0060 // st1w {za0h.s[w12, 0]}, p0, [x3]
	WORD $0xe0a40064 // st1w {za1h.s[w12, 0]}, p0, [x3, x4, lsl #2]
	ADD  R5, R3, R3
	WORD $0xe0bf0061 // st1w {za0h.s[w12, 1]}, p0, [x3]
	WORD $0xe0a40065 // st1w {za1h.s[w12, 1]}, p0, [x3, x4, lsl #2]
	ADD  R5, R3, R3
	WORD $0xe0bf0062 // st1w {za0h.s[w12, 2]}, p0, [x3]
	WORD $0xe0a40066 // st1w {za1h.s[w12, 2]}, p0, [x3, x4, lsl #2]
	ADD  R5, R3, R3
	WORD $0xe0bf0063 // st1w {za0h.s[w12, 3]}, p0, [x3]
	WORD $0xe0a40067 // st1w {za1h.s[w12, 3]}, p0, [x3, x4, lsl #2]
	ADD  R5, R3, R3
	MOVW $8, R12
	WORD $0xe0bf0060 // st1w {za0h.s[w12, 0]}, p0, [x3]
	WORD $0xe0a40064 // st1w {za1h.s[w12, 0]}, p0, [x3, x4, lsl #2]
	ADD  R5, R3, R3
	WORD $0xe0bf0061 // st1w {za0h.s[w12, 1]}, p0, [x3]
	WORD $0xe0a40065 // st1w {za1h.s[w12, 1]}, p0, [x3, x4, lsl #2]
	ADD  R5, R3, R3
	WORD $0xe0bf0062 // st1w {za0h.s[w12, 2]}, p0, [x3]
	WORD $0xe0a40066 // st1w {za1h.s[w12, 2]}, p0, [x3, x4, lsl #2]
	ADD  R5, R3, R3
	WORD $0xe0bf0063 // st1w {za0h.s[w12, 3]}, p0, [x3]
	WORD $0xe0a40067 // st1w {za1h.s[w12, 3]}, p0, [x3, x4, lsl #2]
	ADD  R5, R3, R3
	MOVW $12, R12
	WORD $0xe0bf0060 // st1w {za0h.s[w12, 0]}, p0, [x3]
	WORD $0xe0a40064 // st1w {za1h.s[w12, 0]}, p0, [x3, x4, lsl #2]
	ADD  R5, R3, R3
	WORD $0xe0bf0061 // st1w {za0h.s[w12, 1]}, p0, [x3]
	WORD $0xe0a40065 // st1w {za1h.s[w12, 1]}, p0, [x3, x4, lsl #2]
	ADD  R5, R3, R3
	WORD $0xe0bf0062 // st1w {za0h.s[w12, 2]}, p0, [x3]
	WORD $0xe0a40066 // st1w {za1h.s[w12, 2]}, p0, [x3, x4, lsl #2]
	ADD  R5, R3, R3
	WORD $0xe0bf0063 // st1w {za0h.s[w12, 3]}, p0, [x3]
	WORD $0xe0a40067 // st1w {za1h.s[w12, 3]}, p0, [x3, x4, lsl #2]
	ADD  R5, R3, R3
	MOVW $0, R12
	WORD $0xe0bf0068 // st1w {za2h.s[w12, 0]}, p0, [x3]
	WORD $0xe0a4006c // st1w {za3h.s[w12, 0]}, p0, [x3, x4, lsl #2]
	ADD  R5, R3, R3
	WORD $0xe0bf0069 // st1w {za2h.s[w12, 1]}, p0, [x3]
	WORD $0xe0a4006d // st1w {za3h.s[w12, 1]}, p0, [x3, x4, lsl #2]
	ADD  R5, R3, R3
	WORD $0xe0bf006a // st1w {za2h.s[w12, 2]}, p0, [x3]
	WORD $0xe0a4006e // st1w {za3h.s[w12, 2]}, p0, [x3, x4, lsl #2]
	ADD  R5, R3, R3
	WORD $0xe0bf006b // st1w {za2h.s[w12, 3]}, p0, [x3]
	WORD $0xe0a4006f // st1w {za3h.s[w12, 3]}, p0, [x3, x4, lsl #2]
	ADD  R5, R3, R3
	MOVW $4, R12
	WORD $0xe0bf0068 // st1w {za2h.s[w12, 0]}, p0, [x3]
	WORD $0xe0a4006c // st1w {za3h.s[w12, 0]}, p0, [x3, x4, lsl #2]
	ADD  R5, R3, R3
	WORD $0xe0bf0069 // st1w {za2h.s[w12, 1]}, p0, [x3]
	WORD $0xe0a4006d // st1w {za3h.s[w12, 1]}, p0, [x3, x4, lsl #2]
	ADD  R5, R3, R3
	WORD $0xe0bf006a // st1w {za2h.s[w12, 2]}, p0, [x3]
	WORD $0xe0a4006e // st1w {za3h.s[w12, 2]}, p0, [x3, x4, lsl #2]
	ADD  R5, R3, R3
	WORD $0xe0bf006b // st1w {za2h.s[w12, 3]}, p0, [x3]
	WORD $0xe0a4006f // st1w {za3h.s[w12, 3]}, p0, [x3, x4, lsl #2]
	ADD  R5, R3, R3
	MOVW $8, R12
	WORD $0xe0bf0068 // st1w {za2h.s[w12, 0]}, p0, [x3]
	WORD $0xe0a4006c // st1w {za3h.s[w12, 0]}, p0, [x3, x4, lsl #2]
	ADD  R5, R3, R3
	WORD $0xe0bf0069 // st1w {za2h.s[w12, 1]}, p0, [x3]
	WORD $0xe0a4006d // st1w {za3h.s[w12, 1]}, p0, [x3, x4, lsl #2]
	ADD  R5, R3, R3
	WORD $0xe0bf006a // st1w {za2h.s[w12, 2]}, p0, [x3]
	WORD $0xe0a4006e // st1w {za3h.s[w12, 2]}, p0, [x3, x4, lsl #2]
	ADD  R5, R3, R3
	WORD $0xe0bf006b // st1w {za2h.s[w12, 3]}, p0, [x3]
	WORD $0xe0a4006f // st1w {za3h.s[w12, 3]}, p0, [x3, x4, lsl #2]
	ADD  R5, R3, R3
	MOVW $12, R12
	WORD $0xe0bf0068 // st1w {za2h.s[w12, 0]}, p0, [x3]
	WORD $0xe0a4006c // st1w {za3h.s[w12, 0]}, p0, [x3, x4, lsl #2]
	ADD  R5, R3, R3
	WORD $0xe0bf0069 // st1w {za2h.s[w12, 1]}, p0, [x3]
	WORD $0xe0a4006d // st1w {za3h.s[w12, 1]}, p0, [x3, x4, lsl #2]
	ADD  R5, R3, R3
	WORD $0xe0bf006a // st1w {za2h.s[w12, 2]}, p0, [x3]
	WORD $0xe0a4006e // st1w {za3h.s[w12, 2]}, p0, [x3, x4, lsl #2]
	ADD  R5, R3, R3
	WORD $0xe0bf006b // st1w {za2h.s[w12, 3]}, p0, [x3]
	WORD $0xe0a4006f // st1w {za3h.s[w12, 3]}, p0, [x3, x4, lsl #2]
	ADD  R5, R3, R3
	WORD $0xd503467f // smstop
	RET
