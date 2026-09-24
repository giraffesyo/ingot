#include "textflag.h"

TEXT dlopen_trampoline<>(SB),NOSPLIT,$0-0
	JMP ingot_dlopen(SB)
TEXT dlsym_trampoline<>(SB),NOSPLIT,$0-0
	JMP ingot_dlsym(SB)
TEXT dlerror_trampoline<>(SB),NOSPLIT,$0-0
	JMP ingot_dlerror(SB)

GLOBL ·dlopenABI0(SB), NOPTR|RODATA, $8
DATA ·dlopenABI0(SB)/8, $dlopen_trampoline<>(SB)
GLOBL ·dlsymABI0(SB), NOPTR|RODATA, $8
DATA ·dlsymABI0(SB)/8, $dlsym_trampoline<>(SB)
GLOBL ·dlerrorABI0(SB), NOPTR|RODATA, $8
DATA ·dlerrorABI0(SB)/8, $dlerror_trampoline<>(SB)

// func callC(fn uintptr, args *[8]uintptr, stack uintptr) uintptr
// Also stores d0 (a floating-point return) into args[0].
TEXT ·callC(SB),NOSPLIT|NOFRAME,$0-32
	MOVD	fn+0(FP), R9
	MOVD	args+8(FP), R10
	MOVD	stack+16(FP), R11
	MOVD	RSP, R19	// C preserves x19-x28 (x28 = g)
	MOVD	R30, R20
	MOVD	R10, R21
	MOVD	R11, RSP
	MOVD	0(R10), R0
	MOVD	8(R10), R1
	MOVD	16(R10), R2
	MOVD	24(R10), R3
	MOVD	32(R10), R4
	MOVD	40(R10), R5
	MOVD	48(R10), R6
	MOVD	56(R10), R7
	BL	(R9)
	FMOVD	F0, 0(R21)
	MOVD	R19, RSP
	MOVD	R20, R30
	MOVD	R0, ret+24(FP)
	RET
