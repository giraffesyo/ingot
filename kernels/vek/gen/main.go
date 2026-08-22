// Command gen emits kernels/vek/vek_arm64.s: NEON f32 elementwise kernels.
//
// The Go arm64 assembler lacks vector-float mnemonics (FADD.4S etc.), so they
// are emitted as WORD with encodings verified against the system assembler.
// Each kernel processes a multiple-of-4 element count; the Go wrapper in vek.go
// handles the scalar tail. Extra float32 parameters (alpha, beta, lo, hi, s)
// follow (dst, src, n) at FP offsets 56, 60.
//
//	go run ./kernels/vek/gen > kernels/vek/vek_arm64.s
package main

import (
	"fmt"
	"math"
	"os"
	"strings"
)

// NEON .4S encodings (Q=1, sz=0), verified against clang.
func fadd(d, n, m int) uint32 { return 0x4E20D400 | u(m)<<16 | u(n)<<5 | u(d) }
func fsub(d, n, m int) uint32 { return 0x4EA0D400 | u(m)<<16 | u(n)<<5 | u(d) }
func fmul(d, n, m int) uint32 { return 0x6E20DC00 | u(m)<<16 | u(n)<<5 | u(d) }
func fdiv(d, n, m int) uint32 { return 0x6E20FC00 | u(m)<<16 | u(n)<<5 | u(d) }
func fmax(d, n, m int) uint32 { return 0x4E20F400 | u(m)<<16 | u(n)<<5 | u(d) }
func fmin(d, n, m int) uint32 { return 0x4EA0F400 | u(m)<<16 | u(n)<<5 | u(d) }
func fmla(d, n, m int) uint32 { return 0x4E20CC00 | u(m)<<16 | u(n)<<5 | u(d) } // Vd += Vn*Vm
func orr(d, n int) uint32     { return 0x4EA01C00 | u(n)<<16 | u(n)<<5 | u(d) } // move Vd=Vn
func frintn(d, n int) uint32  { return 0x4E218800 | u(n)<<5 | u(d) }            // round to nearest even
func fcvtns(d, n int) uint32  { return 0x4E21A800 | u(n)<<5 | u(d) }            // f32 → s32 (nearest)
func shl23(d, n int) uint32   { return 0x4F375400 | u(n)<<5 | u(d) }            // Vd.4S = Vn.4S << 23
func addi(d, n, m int) uint32 { return 0x4EA08400 | u(m)<<16 | u(n)<<5 | u(d) } // integer add .4S
func fneg(d, n int) uint32    { return 0x6EA0F800 | u(n)<<5 | u(d) }
func fabs(d, n int) uint32    { return 0x4EA0F800 | u(n)<<5 | u(d) }
func vand(d, n, m int) uint32 { return 0x4E201C00 | u(m)<<16 | u(n)<<5 | u(d) } // Vd = Vn & Vm
func vorr(d, n, m int) uint32 { return 0x4EA01C00 | u(m)<<16 | u(n)<<5 | u(d) } // Vd = Vn | Vm
func dupW(d, wn int) uint32   { return 0x4E040C00 | u(wn)<<5 | u(d) }           // Vd.4S=dup(Wn)
func movi0(d int) uint32      { return 0x4F000400 | u(d) }                      // Vd.4S=0
func u(x int) uint32          { return uint32(x) }
func fbits(f float64) uint32  { return math.Float32bits(float32(f)) }

// fmlaElem: FMLA Vd.4S += Vn.4S * Vm.S[idx].
func fmlaElem(d, n, m, idx int) uint32 {
	l := uint32(idx & 1)
	h := uint32(idx >> 1)
	return 0x4F801000 | l<<21 | u(m>>4)<<20 | u(m&15)<<16 | h<<11 | u(n)<<5 | u(d)
}

type gen struct{ b strings.Builder }

func (g *gen) w(f string, a ...any) { fmt.Fprintf(&g.b, f+"\n", a...) }

// dupConst broadcasts a compile-time float constant into vector vd.
func (g *gen) dupConst(vd int, f float64, name string) {
	g.w("\tMOVD $%d, R9 // %s = %g", int32(fbits(f)), name, f)
	g.w("\tWORD $0x%08X // dup v%d.4s, w9", dupW(vd, 9), vd)
}

// dupArg broadcasts a float32 argument at FP+off into vector vd.
func (g *gen) dupArg(vd, off int, name string) {
	g.w("\tMOVWU %s+%d(FP), R9", name, off)
	g.w("\tWORD $0x%08X // dup v%d.4s, w9 (%s)", dupW(vd, 9), vd, name)
}

func (g *gen) movi0(vd int) {
	g.w("\tWORD $0x%08X // movi v%d.4s, #0", movi0(vd), vd)
}

// binary emits func name_asm(dst, a, b []float32, n int); n a multiple of 4.
func (g *gen) binary(name string, enc func(d, n, m int) uint32) {
	g.w("// func %s_asm(dst, a, b []float32, n int)", name)
	g.w("TEXT ·%s_asm(SB), NOSPLIT, $0-80", name)
	g.w("\tMOVD dst_base+0(FP), R0")
	g.w("\tMOVD a_base+24(FP), R1")
	g.w("\tMOVD b_base+48(FP), R2")
	g.w("\tMOVD n+72(FP), R3")
	g.w("loop16:")
	g.w("\tCMP $16, R3")
	g.w("\tBLT loop4")
	g.w("\tVLD1.P 64(R1), [V0.S4, V1.S4, V2.S4, V3.S4]")
	g.w("\tVLD1.P 64(R2), [V4.S4, V5.S4, V6.S4, V7.S4]")
	for i := 0; i < 4; i++ {
		g.w("\tWORD $0x%08X // %s v%d,v%d,v%d", enc(i, i, i+4), name, i, i, i+4)
	}
	g.w("\tVST1.P [V0.S4, V1.S4, V2.S4, V3.S4], 64(R0)")
	g.w("\tSUB $16, R3")
	g.w("\tB loop16")
	g.w("loop4:")
	g.w("\tCMP $4, R3")
	g.w("\tBLT done")
	g.w("\tVLD1.P 16(R1), [V0.S4]")
	g.w("\tVLD1.P 16(R2), [V4.S4]")
	g.w("\tWORD $0x%08X", enc(0, 0, 4))
	g.w("\tVST1.P [V0.S4], 16(R0)")
	g.w("\tSUB $4, R3")
	g.w("\tB loop4")
	g.w("done:")
	g.w("\tRET")
	g.w("")
}

// unary emits func name_asm(dst, src []float32, n int, <extra floats>).
// body(v, s) transforms input vector v in place using scratch vector s and the
// const/arg vectors set up by prep.
func (g *gen) unary(name string, frame int, prep func(*gen), body func(*gen, int, int)) {
	g.w("// func %s_asm(dst, src []float32, n int, ...)", name)
	g.w("TEXT ·%s_asm(SB), NOSPLIT, $0-%d", name, frame)
	g.w("\tMOVD dst_base+0(FP), R0")
	g.w("\tMOVD src_base+24(FP), R1")
	g.w("\tMOVD n+48(FP), R3")
	if prep != nil {
		prep(g)
	}
	g.w("loop16:")
	g.w("\tCMP $16, R3")
	g.w("\tBLT loop4")
	g.w("\tVLD1.P 64(R1), [V0.S4, V1.S4, V2.S4, V3.S4]")
	for i := 0; i < 4; i++ {
		body(g, i, 16+i)
	}
	g.w("\tVST1.P [V0.S4, V1.S4, V2.S4, V3.S4], 64(R0)")
	g.w("\tSUB $16, R3")
	g.w("\tB loop16")
	g.w("loop4:")
	g.w("\tCMP $4, R3")
	g.w("\tBLT done")
	g.w("\tVLD1.P 16(R1), [V0.S4]")
	body(g, 0, 16)
	g.w("\tVST1.P [V0.S4], 16(R0)")
	g.w("\tSUB $4, R3")
	g.w("\tB loop4")
	g.w("done:")
	g.w("\tRET")
	g.w("")
}

// unaryN is unary with nvec (1, 2 or 4) vectors per main-loop iteration; the
// body gets (v, s): input/output vector v (0..nvec-1) and a scratch base s =
// 16+v. Fewer vectors per iteration leave more registers for constants.
func (g *gen) unaryN(name string, frame, nvec int, prep func(*gen), body func(*gen, int, int)) {
	g.w("// func %s_asm(dst, src []float32, n int, ...)", name)
	g.w("TEXT ·%s_asm(SB), NOSPLIT, $0-%d", name, frame)
	g.w("\tMOVD dst_base+0(FP), R0")
	g.w("\tMOVD src_base+24(FP), R1")
	g.w("\tMOVD n+48(FP), R3")
	if prep != nil {
		prep(g)
	}
	if nvec > 1 {
		g.w("loopN:")
		g.w("\tCMP $%d, R3", 4*nvec)
		g.w("\tBLT loop4")
		g.w("\tVLD1.P %d(R1), [%s]", 16*nvec, vregListS(0, nvec))
		for i := 0; i < nvec; i++ {
			body(g, i, 16+i)
		}
		g.w("\tVST1.P [%s], %d(R0)", vregListS(0, nvec), 16*nvec)
		g.w("\tSUB $%d, R3", 4*nvec)
		g.w("\tB loopN")
	}
	g.w("loop4:")
	g.w("\tCMP $4, R3")
	g.w("\tBLT done")
	g.w("\tVLD1.P 16(R1), [V0.S4]")
	body(g, 0, 16)
	g.w("\tVST1.P [V0.S4], 16(R0)")
	g.w("\tSUB $4, R3")
	g.w("\tB loop4")
	g.w("done:")
	g.w("\tRET")
	g.w("")
}

func vregListS(base, n int) string {
	parts := make([]string, n)
	for i := range parts {
		parts[i] = fmt.Sprintf("V%d.S4", base+i)
	}
	return strings.Join(parts, ", ")
}

func main() {
	g := &gen{}
	g.w("// Code generated by kernels/vek/gen. DO NOT EDIT.")
	g.w("")
	g.w("//go:build arm64")
	g.w("")
	g.w(`#include "textflag.h"`)
	g.w("")

	g.binary("add", fadd)
	g.binary("sub", fsub)
	g.binary("mul", fmul)
	g.binary("div", fdiv)
	g.binary("maxpair", fmax)
	g.binary("minpair", fmin)

	// relu: v = max(v, 0)
	g.unary("relu", 56, func(g *gen) { g.movi0(28) }, func(g *gen, v, s int) {
		g.w("\tWORD $0x%08X // fmax v%d,v%d,v28", fmax(v, v, 28), v, v)
	})

	// hardswish: t = clamp(x/6+0.5, 0, 1); out = x*t
	g.unary("hardswish", 56, func(g *gen) {
		g.dupConst(28, 1.0/6.0, "1/6")
		g.dupConst(29, 0.5, "0.5")
		g.movi0(30)
		g.dupConst(31, 1.0, "1.0")
	}, func(g *gen, v, s int) {
		g.w("\tWORD $0x%08X // mov v%d,v29", orr(s, 29), s)
		g.w("\tWORD $0x%08X // fmla v%d+=v%d*v28", fmla(s, v, 28), s, v)
		g.w("\tWORD $0x%08X // fmax v%d,v%d,v30", fmax(s, s, 30), s, s)
		g.w("\tWORD $0x%08X // fmin v%d,v%d,v31", fmin(s, s, 31), s, s)
		g.w("\tWORD $0x%08X // fmul v%d,v%d,v%d", fmul(v, v, s), v, v, s)
	})

	// hardsigmoid(alpha, beta): clamp(alpha*x+beta, 0, 1)
	g.unary("hardsigmoid", 64, func(g *gen) {
		g.dupArg(28, 56, "alpha")
		g.dupArg(29, 60, "beta")
		g.movi0(30)
		g.dupConst(31, 1.0, "1.0")
	}, func(g *gen, v, s int) {
		g.w("\tWORD $0x%08X // mov v%d,v29", orr(s, 29), s)
		g.w("\tWORD $0x%08X // fmla v%d+=v%d*v28", fmla(s, v, 28), s, v)
		g.w("\tWORD $0x%08X // fmax v%d,v%d,v30", fmax(s, s, 30), s, s)
		g.w("\tWORD $0x%08X // fmin v%d,v%d,v31", fmin(s, s, 31), s, s)
		g.w("\tWORD $0x%08X // mov v%d,v%d", orr(v, s), v, s)
	})

	// clip(lo, hi)
	g.unary("clip", 64, func(g *gen) {
		g.dupArg(30, 56, "lo")
		g.dupArg(31, 60, "hi")
	}, func(g *gen, v, s int) {
		g.w("\tWORD $0x%08X // fmax v%d,v%d,v30", fmax(v, v, 30), v, v)
		g.w("\tWORD $0x%08X // fmin v%d,v%d,v31", fmin(v, v, 31), v, v)
	})

	// leakyrelu(alpha): max(v,0) + alpha*min(v,0)
	g.unary("leakyrelu", 60, func(g *gen) {
		g.dupArg(28, 56, "alpha")
		g.movi0(30)
	}, func(g *gen, v, s int) {
		g.w("\tWORD $0x%08X // fmin v%d=min(v%d,0)", fmin(s, v, 30), s, v)
		g.w("\tWORD $0x%08X // fmax v%d=max(v%d,0)", fmax(v, v, 30), v, v)
		g.w("\tWORD $0x%08X // fmla v%d+=v%d*v28", fmla(v, s, 28), v, s)
	})

	// addscalar(s), mulscalar(s): extra float at FP+56
	g.unary("addscalar", 60, func(g *gen) { g.dupArg(28, 56, "s") }, func(g *gen, v, s int) {
		g.w("\tWORD $0x%08X // fadd v%d,v%d,v28", fadd(v, v, 28), v, v)
	})
	g.unary("mulscalar", 60, func(g *gen) { g.dupArg(28, 56, "s") }, func(g *gen, v, s int) {
		g.w("\tWORD $0x%08X // fmul v%d,v%d,v28", fmul(v, v, 28), v, v)
	})

	// exp / sigmoid: see expBody.
	g.unary("exp", 56, expPrep, func(g *gen, v, s int) { expBody(g, v, s, s+4, s+12) })
	// silu: x * sigmoid(x). v4..v7 hold the input copy (free: consts live in
	// v8-v15/v24-v27, scratch in v16-v23/v28-v31).
	g.unary("silu", 56, expPrep, func(g *gen, v, s int) {
		g.w("\tWORD $0x%08X // v%d = x (copy)", orr(4+v, v), 4+v)
		g.w("\tWORD $0x%08X // fneg v%d", fneg(v, v), v)
		expBody(g, v, s, s+4, s+12)
		g.w("\tWORD $0x%08X // fadd v%d += 1", fadd(v, v, 27), v)
		g.w("\tWORD $0x%08X // fdiv v%d = 1/v%d", fdiv(v, 27, v), v, v)
		g.w("\tWORD $0x%08X // fmul v%d *= x", fmul(v, v, 4+v), v)
	})
	g.unary("sigmoid", 56, expPrep, func(g *gen, v, s int) {
		g.w("\tWORD $0x%08X // fneg v%d", fneg(v, v), v)
		expBody(g, v, s, s+4, s+12)
		g.w("\tWORD $0x%08X // fadd v%d += 1", fadd(v, v, 27), v)
		g.w("\tWORD $0x%08X // fdiv v%d = 1/v%d", fdiv(v, 27, v), v, v)
	})

	g.axpy()
	g.dot()

	// erf: two vectors per iteration (register budget), see erfBody.
	g.unaryN("erf", 56, 2, erfPrep, func(g *gen, v, s int) {
		erfBody(g, v, 4+v, s, s+4, s+12, s+2)
	})
	// gelu(x) = 0.5·x·(1+erf(x/√2)): one vector per iteration.
	g.unaryN("gelu", 56, 1, func(g *gen) {
		erfPrep(g)
		g.dupConst(1, 0.70710678118654752, "1/sqrt2")
		g.dupConst(31, 0.5, "0.5")
	}, func(g *gen, v, s int) {
		g.w("\tWORD $0x%08X // v5 = x/sqrt2", fmul(5, v, 1))
		erfBody(g, 5, 4, s, s+4, s+12, s+2)
		g.w("\tWORD $0x%08X // v5 += 1", fadd(5, 5, 27))
		g.w("\tWORD $0x%08X // v%d = x*(1+erf)", fmul(v, v, 5), v)
		g.w("\tWORD $0x%08X // v%d *= 0.5", fmul(v, v, 31), v)
	})

	// Depthwise row kernels: square stride-1 taps, and the even/odd sub-kernels
	// the stride-2 path uses after de-interleaving columns (3x3 → 3x2 + 3x1,
	// 5x5 → 5x3 + 5x2).
	for _, k := range dwShapes {
		g.dwconv(k[0], k[1])
	}

	os.Stdout.WriteString(g.b.String())
}

// dwShapes are the (KH, KW) depthwise row kernels generated.
var dwShapes = [][2]int{{3, 3}, {5, 5}, {3, 2}, {3, 1}, {5, 3}, {5, 2}}

// dwconv emits func dwconvKxKs1_asm(dst, src []float32, wpacked []float32, ncols, W int):
// one interior output row of a stride-1, dilation-1, KxK depthwise conv, adding
// into dst (pre-filled with bias by the caller). src points at the top-left
// input element for output column 0 (all taps in-bounds). wpacked holds K*K
// weights one per lane, padded to a multiple of 4, loaded into V25.. once.
// ncols is a multiple of 4.
func (g *gen) dwconv(KH, KW int) {
	name := fmt.Sprintf("dwconv%dx%ds1", KH, KW)
	nw := KH * KW
	nreg := (nw + 3) / 4
	g.w("// func %s_asm(dst, src []float32, wpacked []float32, ncols, W int)", name)
	g.w("TEXT ·%s_asm(SB), NOSPLIT, $0-88", name)
	g.w("	MOVD dst_base+0(FP), R0")
	g.w("	MOVD src_base+24(FP), R1")
	g.w("	MOVD wpacked_base+48(FP), R2")
	g.w("	MOVD ncols+72(FP), R3")
	g.w("	MOVD W+80(FP), R4")
	g.w("	LSL $2, R4, R4 // row stride in bytes")
	// VLD1 loads at most 4 registers per list; split the weight preload.
	for base := 0; base < nreg; base += 4 {
		cnt := min(4, nreg-base)
		if base+cnt < nreg {
			g.w("	VLD1.P %d(R2), [%s]", cnt*16, vregList(25+base, cnt))
		} else {
			g.w("	VLD1 (R2), [%s]", vregList(25+base, cnt))
		}
	}
	g.w("loop:")
	g.w("	CMP $4, R3")
	g.w("	BLT done")
	g.w("	VLD1 (R0), [V0.S4] // acc = dst[c..c+4]")
	for kh := 0; kh < KH; kh++ {
		if kh == 0 {
			g.w("	MOVD R1, R5")
		} else {
			g.w("	ADD R4, R5, R5")
		}
		g.w("	MOVD R5, R6")
		for kw := 0; kw < KW; kw++ {
			t := kh*KW + kw
			if kw > 0 {
				g.w("	ADD $4, R6, R6")
			}
			g.w("	VLD1 (R6), [V1.S4]")
			g.w("	WORD $0x%08X // fmla v0 += v1 * v%d.s[%d] (w%d)", fmlaElem(0, 1, 25+t/4, t%4), 25+t/4, t%4, t)
		}
	}
	g.w("	VST1.P [V0.S4], 16(R0)")
	g.w("	ADD $16, R1, R1")
	g.w("	SUB $4, R3, R3")
	g.w("	B loop")
	g.w("done:")
	g.w("	RET")
	g.w("")
}

func vregList(base, n int) string {
	parts := make([]string, n)
	for i := 0; i < n; i++ {
		parts[i] = fmt.Sprintf("V%d.S4", base+i)
	}
	return strings.Join(parts, ", ")
}

// axpy emits func axpy_asm(dst, src []float32, n int, a float32): dst += a*src.
// dst is loaded, FMLA-accumulated with a broadcast, and stored.
func (g *gen) axpy() {
	g.w("// func axpy_asm(dst, src []float32, n int, a float32): dst += a*src")
	g.w("TEXT ·axpy_asm(SB), NOSPLIT, $0-60")
	g.w("	MOVD dst_base+0(FP), R0")
	g.w("	MOVD src_base+24(FP), R1")
	g.w("	MOVD n+48(FP), R3")
	g.dupArg(28, 56, "a")
	g.w("loop16:")
	g.w("	CMP $16, R3")
	g.w("	BLT loop4")
	g.w("	VLD1 (R0), [V0.S4, V1.S4, V2.S4, V3.S4]")
	g.w("	VLD1.P 64(R1), [V4.S4, V5.S4, V6.S4, V7.S4]")
	for i := 0; i < 4; i++ {
		g.w("	WORD $0x%08X // fmla v%d += v%d*v28", fmla(i, i+4, 28), i, i+4)
	}
	g.w("	VST1.P [V0.S4, V1.S4, V2.S4, V3.S4], 64(R0)")
	g.w("	SUB $16, R3")
	g.w("	B loop16")
	g.w("loop4:")
	g.w("	CMP $4, R3")
	g.w("	BLT done")
	g.w("	VLD1 (R0), [V0.S4]")
	g.w("	VLD1.P 16(R1), [V4.S4]")
	g.w("	WORD $0x%08X // fmla v0 += v4*v28", fmla(0, 4, 28))
	g.w("	VST1.P [V0.S4], 16(R0)")
	g.w("	SUB $4, R3")
	g.w("	B loop4")
	g.w("done:")
	g.w("	RET")
	g.w("")
}

// Exp constants (Cephes expf): x clamped to [lo,hi]; n = round(x·log2e);
// r = x − n·ln2 (hi/lo split); e^r ≈ 1 + r + r²·P(r); result = 2^n·e^r by
// adding n<<23 to the float bits. Inputs below lo saturate at the smallest
// normal instead of flushing to 0, above hi at ~2.3e38 instead of +Inf.
var expConsts = []struct {
	reg  int
	val  float64
	name string
}{
	{8, -87.33654, "lo"}, {9, 88.37626, "hi"}, {10, 1.44269504088896341, "log2e"},
	{11, -0.693359375, "-ln2hi"}, {12, 2.12194440e-4, "-ln2lo"},
	{13, 1.9875691500e-4, "p0"}, {14, 1.3981999507e-3, "p1"}, {15, 8.3334519073e-3, "p2"},
	{24, 4.1665795894e-2, "p3"}, {25, 1.6666665459e-1, "p4"}, {26, 5.0000001201e-1, "p5"},
	{27, 1.0, "1.0"},
}

func expPrep(g *gen) {
	for _, c := range expConsts {
		g.dupConst(c.reg, c.val, c.name)
	}
}

// expBody: v ← exp(v) using scratch s, t, u and the expConsts registers.
func expBody(g *gen, v, s, t, u int) {
	g.w("\tWORD $0x%08X // fmax v%d,lo", fmax(v, v, 8), v)
	g.w("\tWORD $0x%08X // fmin v%d,hi", fmin(v, v, 9), v)
	g.w("\tWORD $0x%08X // v%d = x*log2e", fmul(u, v, 10), u)
	g.w("\tWORD $0x%08X // frintn v%d (n)", frintn(u, u), u)
	g.w("\tWORD $0x%08X // r = x - n*ln2hi", fmla(v, u, 11))
	g.w("\tWORD $0x%08X // r -= n*ln2lo", fmla(v, u, 12))
	g.w("\tWORD $0x%08X // v%d = p1", orr(s, 14), s)
	g.w("\tWORD $0x%08X // s += p0*r", fmla(s, 13, v))
	g.w("\tWORD $0x%08X // v%d = p2", orr(t, 15), t)
	g.w("\tWORD $0x%08X // t += s*r", fmla(t, s, v))
	g.w("\tWORD $0x%08X // v%d = p3", orr(s, 24), s)
	g.w("\tWORD $0x%08X // s += t*r", fmla(s, t, v))
	g.w("\tWORD $0x%08X // v%d = p4", orr(t, 25), t)
	g.w("\tWORD $0x%08X // t += s*r", fmla(t, s, v))
	g.w("\tWORD $0x%08X // v%d = p5", orr(s, 26), s)
	g.w("\tWORD $0x%08X // s += t*r  (P(r))", fmla(s, t, v))
	g.w("\tWORD $0x%08X // t = r*r", fmul(t, v, v))
	g.w("\tWORD $0x%08X // v = r+1", fadd(v, v, 27))
	g.w("\tWORD $0x%08X // v += P*r^2", fmla(v, s, t))
	g.w("\tWORD $0x%08X // n → int", fcvtns(u, u))
	g.w("\tWORD $0x%08X // n << 23", shl23(u, u))
	g.w("\tWORD $0x%08X // v = v * 2^n (int add)", addi(v, v, u))
}

// dot emits func dot_asm(a, b []float32, n int, out []float32): out[0:16] =
// 16 partial sums of a[i]*b[i] (4 accumulators × 4 lanes); the Go wrapper
// adds them. n is a multiple of 16.
func (g *gen) dot() {
	g.w("// func dot_asm(a, b []float32, n int, out []float32)")
	g.w("TEXT ·dot_asm(SB), NOSPLIT, $0-80")
	g.w("\tMOVD a_base+0(FP), R0")
	g.w("\tMOVD b_base+24(FP), R1")
	g.w("\tMOVD n+48(FP), R3")
	g.w("\tMOVD out_base+56(FP), R2")
	for i := 16; i < 20; i++ {
		g.movi0(i)
	}
	g.w("loop16:")
	g.w("\tCMP $16, R3")
	g.w("\tBLT done")
	g.w("\tVLD1.P 64(R0), [V0.S4, V1.S4, V2.S4, V3.S4]")
	g.w("\tVLD1.P 64(R1), [V4.S4, V5.S4, V6.S4, V7.S4]")
	for i := 0; i < 4; i++ {
		g.w("\tWORD $0x%08X // fmla v%d += v%d*v%d", fmla(16+i, i, 4+i), 16+i, i, 4+i)
	}
	g.w("\tSUB $16, R3")
	g.w("\tB loop16")
	g.w("done:")
	g.w("\tVST1 [V16.S4, V17.S4, V18.S4, V19.S4], (R2)")
	g.w("\tRET")
	g.w("")
}

// Erf: Abramowitz–Stegun 7.1.26, |err| ≤ 1.5e-7 absolute:
//
//	erf(x) = sign(x)·(1 − t·(a1 + t(a2 + t(a3 + t(a4 + t·a5))))·e^{−x²}),
//	t = 1/(1+p|x|).
//
// Register plan (2 vectors/iter): data v0,v1; |x| v4,v5; scratch s=16/17,
// t=20/21, u=28/29, u2=18/19; exp consts v8-15,v24-27; erf consts P=v2,
// A1=v3, A2=v6, A3=v7, A4=v22, A5=v23, SIGN=v30.
func erfPrep(g *gen) {
	expPrep(g)
	g.dupConst(2, 0.3275911, "p")
	g.dupConst(3, 0.254829592, "a1")
	g.dupConst(6, -0.284496736, "a2")
	g.dupConst(7, 1.421413741, "a3")
	g.dupConst(22, -1.453152027, "a4")
	g.dupConst(23, 1.061405429, "a5")
	g.w("\tMOVD $%d, R9 // sign mask (0x80000000)", math.MinInt32)
	g.w("\tWORD $0x%08X // dup v30.4s, w9", dupW(30, 9))
}

// erfBody: x ← erf(x), using w (gets |x| then the sign), s, t, u, u2 as scratch.
func erfBody(g *gen, x, w, s, t, u, u2 int) {
	g.w("\tWORD $0x%08X // w = |x|", fabs(w, x))
	g.w("\tWORD $0x%08X // u = x^2", fmul(u, w, w))
	g.w("\tWORD $0x%08X // u = -x^2", fneg(u, u))
	expBody(g, u, s, t, u2) // u = e^{-x^2}
	g.w("\tWORD $0x%08X // t = 1", orr(t, 27))
	g.w("\tWORD $0x%08X // t += p*|x|", fmla(t, w, 2))
	g.w("\tWORD $0x%08X // t = 1/t", fdiv(t, 27, t))
	g.w("\tWORD $0x%08X // s = a5", orr(s, 23))
	g.w("\tWORD $0x%08X // u2 = a4", orr(u2, 22))
	g.w("\tWORD $0x%08X // u2 += s*t", fmla(u2, s, t))
	g.w("\tWORD $0x%08X // s = a3", orr(s, 7))
	g.w("\tWORD $0x%08X // s += u2*t", fmla(s, u2, t))
	g.w("\tWORD $0x%08X // u2 = a2", orr(u2, 6))
	g.w("\tWORD $0x%08X // u2 += s*t", fmla(u2, s, t))
	g.w("\tWORD $0x%08X // s = a1", orr(s, 3))
	g.w("\tWORD $0x%08X // s += u2*t", fmla(s, u2, t))
	g.w("\tWORD $0x%08X // s *= t", fmul(s, s, t))
	g.w("\tWORD $0x%08X // s *= e^{-x^2}", fmul(s, s, u))
	g.w("\tWORD $0x%08X // w = sign(x)", vand(w, x, 30))
	g.w("\tWORD $0x%08X // x = 1 - s", fsub(x, 27, s))
	g.w("\tWORD $0x%08X // x |= sign", vorr(x, x, w))
}
