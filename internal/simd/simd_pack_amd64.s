//go:build gc && cvmatch_asm

#include "textflag.h"
// Data-movement kernels: byte->complex packing (two real rows per
// complex FFT), spectrum untangle/combine for the 2-for-1 real FFT, and
// the inverse-transform result emit.

// Part of the AVX2 kernel set — bit-identical to the generic Go loops.
// Shared conventions (see simd_amd64.s for the CPU-detection entry):
// complex64 values are interleaved (re, im) float32 pairs; the butterfly
// evaluates t1 = q*wr, t2 = swap(q)*wi and addsub(t1, t2) =
// (t1e-t2e, t1o+t2o), which reproduces the scalar complex multiply's two
// products and single rounded add/sub per component (the imaginary sum
// is commuted; IEEE addition commutes bit-exactly). Never VFMADD*: every
// multiply and add rounds separately, exactly like the scalar code.

// func PackRows2(z []complex64, ra, rb []uint8, step int)
// z[i] = (float32(ra[i*step]), float32(rb[i*step])). Exact conversions,
// pure data movement. step is 1 or 4.
TEXT ·PackRows2(SB), NOSPLIT, $0-80
	MOVQ z_base+0(FP), DI
	MOVQ z_len+8(FP), DX
	MOVQ ra_base+24(FP), SI
	MOVQ rb_base+48(FP), BX
	MOVQ step+72(FP), CX
	XORQ R10, R10
	MOVQ DX, R11
	ANDQ $-4, R11
	CMPQ CX, $4
	JEQ  pk2_clamp4
	JMP  pk2_s1

pk2_clamp4:
	// Each stride-4 iteration gathers with a full 16-byte load, which
	// reads 3 bytes past the last used byte. Never read past the row
	// slice (the image may end exactly at a page boundary): allow only
	// floor(len/16) vector iterations per row, tail handles the rest.
	MOVQ ra_len+32(FP), AX
	SHRQ $4, AX
	SHLQ $2, AX
	CMPQ AX, R11
	CMOVQLT AX, R11
	MOVQ rb_len+56(FP), AX
	SHRQ $4, AX
	SHLQ $2, AX
	CMPQ AX, R11
	CMOVQLT AX, R11

pk2_s4:
	CMPQ R10, R11
	JGE  pk2_tail
	// gather 4 bytes at stride 4 from each row via PSHUFB
	VMOVDQU    (SI), X1
	VMOVDQU    (BX), X2
	VPSHUFB    pkshuf4<>(SB), X1, X1  // bytes 0,4,8,12 -> lanes 0..3
	VPSHUFB    pkshuf4<>(SB), X2, X2
	VPMOVZXBD  X1, X1
	VPMOVZXBD  X2, X2
	VCVTDQ2PS  X1, X1                 // 4 floats (re)
	VCVTDQ2PS  X2, X2                 // 4 floats (im)
	VUNPCKLPS  X2, X1, X3             // re0 im0 re1 im1
	VUNPCKHPS  X2, X1, X4             // re2 im2 re3 im3
	VMOVUPS    X3, (DI)(R10*8)
	VMOVUPS    X4, 16(DI)(R10*8)
	ADDQ       $16, SI
	ADDQ       $16, BX
	ADDQ       $4, R10
	JMP        pk2_s4

pk2_s1:
	CMPQ R10, R11
	JGE  pk2_tail
	VPMOVZXBD  (SI), X1               // 4 bytes -> 4 dwords
	VPMOVZXBD  (BX), X2
	VCVTDQ2PS  X1, X1
	VCVTDQ2PS  X2, X2
	VUNPCKLPS  X2, X1, X3
	VUNPCKHPS  X2, X1, X4
	VMOVUPS    X3, (DI)(R10*8)
	VMOVUPS    X4, 16(DI)(R10*8)
	ADDQ       $4, SI
	ADDQ       $4, BX
	ADDQ       $4, R10
	JMP        pk2_s1

pk2_tail:
	CMPQ R10, DX
	JGE  pk2_done
	MOVBLZX (SI), AX
	VCVTSI2SSL AX, X1, X1
	MOVBLZX (BX), AX
	VCVTSI2SSL AX, X2, X2
	VMOVSS  X1, (DI)(R10*8)
	VMOVSS  X2, 4(DI)(R10*8)
	ADDQ    CX, SI
	ADDQ    CX, BX
	INCQ    R10
	JMP     pk2_tail

pk2_done:
	VZEROUPPER
	RET

// func PackRows1(z []complex64, ra []uint8, step int)
TEXT ·PackRows1(SB), NOSPLIT, $0-56
	MOVQ z_base+0(FP), DI
	MOVQ z_len+8(FP), DX
	MOVQ ra_base+24(FP), SI
	MOVQ step+48(FP), CX
	XORQ R10, R10
	MOVQ DX, R11
	ANDQ $-4, R11
	VXORPS X2, X2, X2
	CMPQ CX, $4
	JEQ  pk1_clamp4
	JMP  pk1_s1

pk1_clamp4:
	// Same in-bounds clamp as PackRows2 (16-byte gather per iteration).
	MOVQ ra_len+32(FP), AX
	SHRQ $4, AX
	SHLQ $2, AX
	CMPQ AX, R11
	CMOVQLT AX, R11

pk1_s4:
	CMPQ R10, R11
	JGE  pk1_tail
	VMOVDQU    (SI), X1
	VPSHUFB    pkshuf4<>(SB), X1, X1
	VPMOVZXBD  X1, X1
	VCVTDQ2PS  X1, X1
	VUNPCKLPS  X2, X1, X3
	VUNPCKHPS  X2, X1, X4
	VMOVUPS    X3, (DI)(R10*8)
	VMOVUPS    X4, 16(DI)(R10*8)
	ADDQ       $16, SI
	ADDQ       $4, R10
	JMP        pk1_s4

pk1_s1:
	CMPQ R10, R11
	JGE  pk1_tail
	VPMOVZXBD  (SI), X1
	VCVTDQ2PS  X1, X1
	VUNPCKLPS  X2, X1, X3
	VUNPCKHPS  X2, X1, X4
	VMOVUPS    X3, (DI)(R10*8)
	VMOVUPS    X4, 16(DI)(R10*8)
	ADDQ       $4, SI
	ADDQ       $4, R10
	JMP        pk1_s1

pk1_tail:
	CMPQ R10, DX
	JGE  pk1_done
	MOVBLZX (SI), AX
	VCVTSI2SSL AX, X1, X1
	VMOVSS  X1, (DI)(R10*8)
	MOVL    $0, 4(DI)(R10*8)
	ADDQ    CX, SI
	INCQ    R10
	JMP     pk1_tail

pk1_done:
	VZEROUPPER
	RET

DATA pkshuf4<>+0(SB)/8, $0x808080800c080400
DATA pkshuf4<>+8(SB)/8, $0x8080808080808080
GLOBL pkshuf4<>(SB), RODATA|NOPTR, $16

// func Untangle(sa, sb, z []complex64, n, k0, k1 int)
// sa[k] = 0.5*(zk.re+zn.re, zk.im-zn.im), sb[k] = 0.5*(zk.im+zn.im,
// zn.re-zk.re), zn = z[n-k]. k in [k0, k1), k0 >= 1. Negation by xor and
// x-(-y) == x+y are IEEE-exact, so every element gets the scalar op
// sequence: one rounded add/sub, one rounded multiply by 0.5.
TEXT ·Untangle(SB), NOSPLIT, $0-96
	MOVQ sa_base+0(FP), DI
	MOVQ sb_base+24(FP), SI
	MOVQ z_base+48(FP), DX
	MOVQ n+72(FP), R8
	MOVQ k0+80(FP), R10
	MOVQ k1+88(FP), R11
	MOVL $0x80000000, AX
	MOVD AX, X0
	VPBROADCASTD X0, Y15              // sign mask
	VBROADCASTSS half32<>(SB), Y14    // 0.5f
	MOVQ R11, R12
	SUBQ R10, R12
	ANDQ $-4, R12
	ADDQ R10, R12                     // vector end

ut_loop:
	CMPQ R10, R12
	JGE  ut_tail
	VMOVUPS   (DX)(R10*8), Y1         // zk: z[k..k+3]
	MOVQ      R8, R13
	SUBQ      R10, R13                // n-k
	VMOVUPS   -24(DX)(R13*8), Y2      // z[n-k-3..n-k]
	VPERMPD   $0x1B, Y2, Y2           // reverse -> zn for k..k+3
	VXORPS    Y15, Y2, Y3             // -zn
	VADDSUBPS Y3, Y1, Y4              // (zk.re+zn.re, zk.im-zn.im)
	VMULPS    Y14, Y4, Y4
	VMOVUPS   Y4, (DI)(R10*8)         // sa[k..k+3]
	VPERMILPS $0xB1, Y1, Y5           // zk swapped (im, re)
	VPERMILPS $0xB1, Y2, Y6           // zn swapped
	VXORPS    Y15, Y5, Y5             // -zk swapped
	VADDSUBPS Y5, Y6, Y7              // (zn.im+zk.im, zn.re-zk.re)
	VMULPS    Y14, Y7, Y7
	VMOVUPS   Y7, (SI)(R10*8)         // sb[k..k+3]
	ADDQ      $4, R10
	JMP       ut_loop

ut_tail:
	CMPQ R10, R11
	JGE  ut_done
	MOVQ      R8, R13
	SUBQ      R10, R13
	VMOVSD    (DX)(R10*8), X1         // zk
	VMOVSD    (DX)(R13*8), X2         // zn
	VXORPS    X15, X2, X3
	VADDSUBPS X3, X1, X4
	VMULPS    X14, X4, X4
	VMOVSD    X4, (DI)(R10*8)
	VPERMILPS $0xB1, X1, X5
	VPERMILPS $0xB1, X2, X6
	VXORPS    X15, X5, X5
	VADDSUBPS X5, X6, X7
	VMULPS    X14, X7, X7
	VMOVSD    X7, (SI)(R10*8)
	INCQ      R10
	JMP       ut_tail

ut_done:
	VZEROUPPER
	RET

DATA half32<>+0(SB)/4, $0.5
GLOBL half32<>(SB), RODATA|NOPTR, $4

// func CombineLow(z, sa, sb []complex64)
// z[k] = (sa.re - sb.im, sa.im + sb.re): exactly one addsub per element.
TEXT ·CombineLow(SB), NOSPLIT, $0-72
	MOVQ z_base+0(FP), DI
	MOVQ z_len+8(FP), DX
	MOVQ sa_base+24(FP), SI
	MOVQ sb_base+48(FP), BX
	XORQ R10, R10
	MOVQ DX, R11
	ANDQ $-4, R11

cl_loop:
	CMPQ R10, R11
	JGE  cl_tail
	VMOVUPS   (SI)(R10*8), Y1
	VMOVUPS   (BX)(R10*8), Y2
	VPERMILPS $0xB1, Y2, Y2           // (sb.im, sb.re)
	VADDSUBPS Y2, Y1, Y3              // (sa.re-sb.im, sa.im+sb.re)
	VMOVUPS   Y3, (DI)(R10*8)
	ADDQ      $4, R10
	JMP       cl_loop

cl_tail:
	CMPQ R10, DX
	JGE  cl_done
	VMOVSD    (SI)(R10*8), X1
	VMOVSD    (BX)(R10*8), X2
	VPERMILPS $0xB1, X2, X2
	VADDSUBPS X2, X1, X3
	VMOVSD    X3, (DI)(R10*8)
	INCQ      R10
	JMP       cl_tail

cl_done:
	VZEROUPPER
	RET

// func CombineHigh(z, sa, sb []complex64, n, hw int)
// z[k] = (sa[m].re + sb[m].im, sb[m].re - sa[m].im), m = n-k, k in [hw, n).
TEXT ·CombineHigh(SB), NOSPLIT, $0-88
	MOVQ z_base+0(FP), DI
	MOVQ sa_base+24(FP), SI
	MOVQ sb_base+48(FP), BX
	MOVQ n+72(FP), R8
	MOVQ hw+80(FP), R10
	MOVL $0x80000000, AX
	MOVD AX, X0
	VPBROADCASTD X0, Y15
	MOVQ R8, R12
	SUBQ R10, R12
	ANDQ $-4, R12
	ADDQ R10, R12                     // vector end

ch_loop:
	CMPQ R10, R12
	JGE  ch_tail
	MOVQ      R8, R13
	SUBQ      R10, R13                // m = n-k
	VMOVUPS   -24(SI)(R13*8), Y1      // sa[m-3..m]
	VMOVUPS   -24(BX)(R13*8), Y2      // sb[m-3..m]
	VPERMPD   $0x1B, Y1, Y1           // reverse -> sa[m] for k..k+3
	VPERMPD   $0x1B, Y2, Y2
	VPERMILPS $0xB1, Y2, Y2           // (sb.im, sb.re)
	VXORPS    Y15, Y1, Y1             // -sa
	VADDSUBPS Y1, Y2, Y3              // (sb.im+sa.re, sb.re-sa.im)
	VMOVUPS   Y3, (DI)(R10*8)
	ADDQ      $4, R10
	JMP       ch_loop

ch_tail:
	CMPQ R10, R8
	JGE  ch_done
	MOVQ      R8, R13
	SUBQ      R10, R13
	VMOVSD    (SI)(R13*8), X1
	VMOVSD    (BX)(R13*8), X2
	VPERMILPS $0xB1, X2, X2
	VXORPS    X15, X1, X1
	VADDSUBPS X1, X2, X3
	VMOVSD    X3, (DI)(R10*8)
	INCQ      R10
	JMP       ch_tail

ch_done:
	VZEROUPPER
	RET

// func EmitRe(dst []float32, z []complex64, add bool)
TEXT ·EmitRe(SB), NOSPLIT, $0-49
	MOVQ    dst_base+0(FP), DI
	MOVQ    dst_len+8(FP), DX
	MOVQ    z_base+24(FP), SI
	MOVBLZX add+48(FP), CX
	XORQ    R10, R10
	MOVQ    DX, R11
	ANDQ    $-8, R11

er_loop:
	CMPQ R10, R11
	JGE  er_tail
	VMOVUPS  (SI)(R10*8), Y1
	VMOVUPS  32(SI)(R10*8), Y2
	VSHUFPS  $0x88, Y2, Y1, Y3        // re lanes, 128-bit interleaved
	VPERMPD  $0xD8, Y3, Y3            // fix lane order
	TESTQ    CX, CX
	JZ       er_store
	VADDPS   (DI)(R10*4), Y3, Y3

er_store:
	VMOVUPS  Y3, (DI)(R10*4)
	ADDQ     $8, R10
	JMP      er_loop

er_tail:
	CMPQ R10, DX
	JGE  er_done
	VMOVSS (SI)(R10*8), X1
	TESTQ  CX, CX
	JZ     er_tstore
	VADDSS (DI)(R10*4), X1, X1

er_tstore:
	VMOVSS X1, (DI)(R10*4)
	INCQ   R10
	JMP    er_tail

er_done:
	VZEROUPPER
	RET

// func EmitIm(dst []float32, z []complex64, add bool)
TEXT ·EmitIm(SB), NOSPLIT, $0-49
	MOVQ    dst_base+0(FP), DI
	MOVQ    dst_len+8(FP), DX
	MOVQ    z_base+24(FP), SI
	MOVBLZX add+48(FP), CX
	XORQ    R10, R10
	MOVQ    DX, R11
	ANDQ    $-8, R11

ei_loop:
	CMPQ R10, R11
	JGE  ei_tail
	VMOVUPS  (SI)(R10*8), Y1
	VMOVUPS  32(SI)(R10*8), Y2
	VSHUFPS  $0xDD, Y2, Y1, Y3        // im lanes
	VPERMPD  $0xD8, Y3, Y3
	TESTQ    CX, CX
	JZ       ei_store
	VADDPS   (DI)(R10*4), Y3, Y3

ei_store:
	VMOVUPS  Y3, (DI)(R10*4)
	ADDQ     $8, R10
	JMP      ei_loop

ei_tail:
	CMPQ R10, DX
	JGE  ei_done
	VMOVSS 4(SI)(R10*8), X1
	TESTQ  CX, CX
	JZ     ei_tstore
	VADDSS (DI)(R10*4), X1, X1

ei_tstore:
	VMOVSS X1, (DI)(R10*4)
	INCQ   R10
	JMP    ei_tail

ei_done:
	VZEROUPPER
	RET
