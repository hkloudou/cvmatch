//go:build gc && !purego

#include "textflag.h"

// AVX2 kernels: bit-identical to the generic Go loops (see
// impl_simd_amd64.go). Complex64 values are interleaved (re, im) float32
// pairs; the butterfly evaluates t1 = q*wr, t2 = swap(q)*wi and
// addsub(t1, t2) = (t1e-t2e, t1o+t2o), which reproduces the scalar
// complex multiply's two products and single rounded add/sub per
// component (the imaginary sum is commuted; IEEE addition commutes
// bit-exactly).

// func cpuid(leaf, sub uint32) (eax, ebx, ecx, edx uint32)
TEXT ·cpuid(SB), NOSPLIT, $0-24
	MOVL leaf+0(FP), AX
	MOVL sub+4(FP), CX
	CPUID
	MOVL AX, eax+8(FP)
	MOVL BX, ebx+12(FP)
	MOVL CX, ecx+16(FP)
	MOVL DX, edx+20(FP)
	RET

// func xgetbv() uint64
TEXT ·xgetbv(SB), NOSPLIT, $0-8
	XORL CX, CX
	XGETBV
	SHLQ $32, DX
	ORQ  DX, AX
	MOVQ AX, ret+0(FP)
	RET
