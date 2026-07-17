//go:build gc && !purego

#include "textflag.h"

// NEON kernels for arm64 — the twins of the AVX2
// kernels in simd_amd64.s, bit-identical to the generic Go loops (see
// kernels.S in this directory for the annotated source). Go's assembler
// has no un-fused vector FP arithmetic, so each body below is a generated
// WORD stream; this template owns the TEXT frames and every argument
// load/store so `go vet` still checks frame offsets. The bodies use only
// x0-x15 and v0-v31, never the stack, and never fmla/fmls — every float
// op is a distinct IEEE single-rounded instruction.

// func FFTStages(a []complex64, tw []complex64, inverse bool)
TEXT ·FFTStages(SB), NOSPLIT, $0-49
	MOVD  a_base+0(FP), R0
	MOVD  a_len+8(FP), R1
	MOVD  tw_base+24(FP), R2
	MOVBU inverse+48(FP), R3
	// GENBODY: FFTStages
	RET

// func FFTColsBfly(p, q []complex64, w complex64)
TEXT ·FFTColsBfly(SB), NOSPLIT, $0-56
	MOVD  p_base+0(FP), R0
	MOVD  p_len+8(FP), R1
	MOVD  q_base+24(FP), R2
	FMOVS w_real+48(FP), F0
	FMOVS w_imag+52(FP), F1
	// GENBODY: FFTColsBfly
	RET

// func FFTCols4(r0, r1, r2, r3 []complex64, w1, w2a, w2b complex64)
TEXT ·FFTCols4(SB), NOSPLIT, $0-120
	MOVD  r0_base+0(FP), R0
	MOVD  r1_base+24(FP), R1
	MOVD  r2_base+48(FP), R2
	MOVD  r3_base+72(FP), R3
	MOVD  r0_len+8(FP), R4
	FMOVS w1_real+96(FP), F0
	FMOVS w1_imag+100(FP), F1
	FMOVS w2a_real+104(FP), F2
	FMOVS w2a_imag+108(FP), F3
	FMOVS w2b_real+112(FP), F4
	FMOVS w2b_imag+116(FP), F5
	// GENBODY: FFTCols4
	RET

// func MulConj(spec, tspec []complex64)
TEXT ·MulConj(SB), NOSPLIT, $0-48
	MOVD spec_base+0(FP), R0
	MOVD spec_len+8(FP), R1
	MOVD tspec_base+24(FP), R2
	// GENBODY: MulConj
	RET

// func NormRow(rrow []float32, crow []float32, wt *float64, stride, n, cn int,
//	mean *[4]float64, invArea, eps, templNorm float64)
TEXT ·NormRow(SB), NOSPLIT, $0-112
	MOVD  rrow_base+0(FP), R0
	MOVD  crow_base+24(FP), R1
	MOVD  wt+48(FP), R2
	MOVD  stride+56(FP), R3
	MOVD  n+64(FP), R4
	MOVD  cn+72(FP), R5
	MOVD  mean+80(FP), R6
	FMOVD invArea+88(FP), F0
	FMOVD eps+96(FP), F1
	FMOVD templNorm+104(FP), F2
	// GENBODY: NormRow
	RET

// func PackRows2(z []complex64, ra, rb []uint8, step int)
TEXT ·PackRows2(SB), NOSPLIT, $0-80
	MOVD z_base+0(FP), R0
	MOVD z_len+8(FP), R1
	MOVD ra_base+24(FP), R2
	MOVD ra_len+32(FP), R3
	MOVD rb_base+48(FP), R4
	MOVD rb_len+56(FP), R5
	MOVD step+72(FP), R6
	// GENBODY: PackRows2
	RET

// func PackRows1(z []complex64, ra []uint8, step int)
TEXT ·PackRows1(SB), NOSPLIT, $0-56
	MOVD z_base+0(FP), R0
	MOVD z_len+8(FP), R1
	MOVD ra_base+24(FP), R2
	MOVD ra_len+32(FP), R3
	MOVD step+48(FP), R4
	// GENBODY: PackRows1
	RET

// func Untangle(sa, sb, z []complex64, n, k0, k1 int)
TEXT ·Untangle(SB), NOSPLIT, $0-96
	MOVD sa_base+0(FP), R0
	MOVD sb_base+24(FP), R1
	MOVD z_base+48(FP), R2
	MOVD n+72(FP), R3
	MOVD k0+80(FP), R4
	MOVD k1+88(FP), R5
	// GENBODY: Untangle
	RET

// func CombineLow(z, sa, sb []complex64)
TEXT ·CombineLow(SB), NOSPLIT, $0-72
	MOVD z_base+0(FP), R0
	MOVD z_len+8(FP), R1
	MOVD sa_base+24(FP), R2
	MOVD sb_base+48(FP), R3
	// GENBODY: CombineLow
	RET

// func CombineHigh(z, sa, sb []complex64, n, hw int)
TEXT ·CombineHigh(SB), NOSPLIT, $0-88
	MOVD z_base+0(FP), R0
	MOVD sa_base+24(FP), R1
	MOVD sb_base+48(FP), R2
	MOVD n+72(FP), R3
	MOVD hw+80(FP), R4
	// GENBODY: CombineHigh
	RET

// func EmitRe(dst []float32, z []complex64, add bool)
TEXT ·EmitRe(SB), NOSPLIT, $0-49
	MOVD  dst_base+0(FP), R0
	MOVD  dst_len+8(FP), R1
	MOVD  z_base+24(FP), R2
	MOVBU add+48(FP), R3
	// GENBODY: EmitRe
	RET

// func EmitIm(dst []float32, z []complex64, add bool)
TEXT ·EmitIm(SB), NOSPLIT, $0-49
	MOVD  dst_base+0(FP), R0
	MOVD  dst_len+8(FP), R1
	MOVD  z_base+24(FP), R2
	MOVBU add+48(FP), R3
	// GENBODY: EmitIm
	RET

// func MinMaxRow(row []float32) (minV, maxV float32, minI, maxI int)
TEXT ·MinMaxRow(SB), NOSPLIT, $0-48
	MOVD row_base+0(FP), R0
	MOVD row_len+8(FP), R1
	// GENBODY: MinMaxRow
	FMOVS F0, minV+24(FP)
	FMOVS F1, maxV+28(FP)
	VMOV  V2.S[0], R9
	MOVD  R9, minI+32(FP)
	VMOV  V3.S[0], R10
	MOVD  R10, maxI+40(FP)
	RET

// func RGBAToGray(dst, src []uint8)
TEXT ·RGBAToGray(SB), NOSPLIT, $0-48
	MOVD dst_base+0(FP), R0
	MOVD dst_len+8(FP), R1
	MOVD src_base+24(FP), R2
	// GENBODY: RGBAToGray
	RET

// func SlideCols1(colSum []int32, colSum2 []int64, rsub, radd []uint8)
TEXT ·SlideCols1(SB), NOSPLIT, $0-96
	MOVD colSum_base+0(FP), R0
	MOVD colSum_len+8(FP), R1
	MOVD colSum2_base+24(FP), R2
	MOVD rsub_base+48(FP), R3
	MOVD radd_base+72(FP), R4
	// GENBODY: SlideCols1
	RET

// func SlideSpill1(wt, q2 []float64, lo, hi []int32, lo2, hi2 []int64,
//	s0, s2 int64) (ns0, ns2 int64)
TEXT ·SlideSpill1(SB), NOSPLIT, $0-176
	MOVD wt_base+0(FP), R0
	MOVD wt_len+8(FP), R1
	MOVD q2_base+24(FP), R2
	MOVD lo_base+48(FP), R3
	MOVD hi_base+72(FP), R4
	MOVD lo2_base+96(FP), R5
	MOVD hi2_base+120(FP), R6
	MOVD s0+144(FP), R7
	MOVD s2+152(FP), R8
	// GENBODY: SlideSpill1
	MOVD R7, ns0+160(FP)
	MOVD R8, ns2+168(FP)
	RET

// func SlideCols4(colSum []int32, colSum2 []int64, rsub, radd []uint8, cn int)
TEXT ·SlideCols4(SB), NOSPLIT, $0-104
	MOVD colSum_base+0(FP), R0
	MOVD colSum2_base+24(FP), R1
	MOVD colSum2_len+32(FP), R2
	MOVD rsub_base+48(FP), R3
	MOVD radd_base+72(FP), R4
	MOVD cn+96(FP), R5
	// GENBODY: SlideCols4
	RET
